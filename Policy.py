###Class that models a policy for exploration or evaluation
import numpy
import scipy.sparse
import sklearn.model_selection
import sklearn.tree
import sklearn.ensemble
import sklearn.linear_model
# from sklearn.externals import joblib
import joblib
import os
import sys
import Settings
import GammaDP
import scipy.linalg
import itertools

#UniformGamma(...) computes a Gamma_pinv matrix for uniform exploration
#num_candidates: (int) Number of candidates, m
#ranking_size: (int) Size of slate, l
#allow_repetitions: (bool) If True, repetitions were allowed in the ranking
def UniformGamma(num_candidates, ranking_size, allow_repetitions):
    validDocs=ranking_size
    if not allow_repetitions:
        validDocs=min(ranking_size, num_candidates)
                
    gamma=numpy.empty((num_candidates*validDocs, num_candidates*validDocs), dtype=numpy.float64)
    if num_candidates==1:
        gamma.fill(1.0)
    else:
        #First set all the off-diagonal blocks
        if allow_repetitions:
            gamma.fill(1.0/(num_candidates*num_candidates))
        else:
            gamma.fill(1.0/(num_candidates*(num_candidates-1)))
            #Correct the diagonal of each off-diagonal block: Pairwise=0
            for p in range(1,validDocs):
                diag=numpy.diagonal(gamma, offset=p*num_candidates)
                diag.setflags(write=True)
                diag.fill(0)
                        
                diag=numpy.diagonal(gamma, offset=-p*num_candidates)
                diag.setflags(write=True)
                diag.fill(0)
                        
        #Now correct the diagonal blocks: Diagonal matrix with marginals = 1/m
        for j in range(validDocs):
            currentStart=j*num_candidates
            currentEnd=(j+1)*num_candidates
            gamma[currentStart:currentEnd, currentStart:currentEnd]=0
            numpy.fill_diagonal(gamma, 1.0/num_candidates)
    
    gammaInv=scipy.linalg.pinv(gamma)
    return (num_candidates, gammaInv)
    
    
#NonUniformGamma(...) computes a Gamma_pinv matrix for non-uniform exploration
#num_candidates: (int) Number of candidates, m
#decay: (double) Decay factor. Doc Selection Prob \propto exp2(-decay * floor[ log2(rank) ])
#ranking_size: (int) Size of slate, l
#allow_repetitions: (bool) If True, repetitions were allowed in the ranking
def NonUniformGamma(num_candidates, decay, ranking_size, allow_repetitions):
    validDocs=ranking_size
    if not allow_repetitions:
        validDocs=min(ranking_size, num_candidates)

    multinomial=numpy.arange(1, num_candidates+1, dtype=numpy.float64)
    multinomial=numpy.exp2((-decay)*numpy.floor(numpy.log2(multinomial)))
    
    for i in range(1,num_candidates):
        prevVal=multinomial[i-1]
        currVal=multinomial[i]
        if numpy.isclose(currVal, prevVal):
            multinomial[i]=prevVal
    
    gamma=None
    if num_candidates==1:
        gamma=numpy.ones((num_candidates*validDocs, num_candidates*validDocs), dtype=numpy.longdouble)
    else:
        if allow_repetitions:
            offDiagonal=numpy.outer(multinomial, multinomial)
            gamma=numpy.tile(offDiagonal, (validDocs, validDocs))
            for j in range(validDocs):
                currentStart=j*num_candidates
                currentEnd=(j+1)*num_candidates
                gamma[currentStart:currentEnd, currentStart:currentEnd]=numpy.diag(multinomial)
        else:
            gammaVals=GammaDP.GammaCalculator(multinomial.tolist(), validDocs)
            gamma=numpy.diag(numpy.ravel(gammaVals.unitMarginals))
            
            for p in range(validDocs):
                for q in range(p+1, validDocs):
                    pairMarginals=gammaVals.pairwiseMarginals[(p,q)]
                    currentRowStart=p*num_candidates
                    currentRowEnd=(p+1)*num_candidates
                    currentColumnStart=q*num_candidates
                    currentColumnEnd=(q+1)*num_candidates
                    gamma[currentRowStart:currentRowEnd, currentColumnStart:currentColumnEnd]=pairMarginals
                    gamma[currentColumnStart:currentColumnEnd, currentRowStart:currentRowEnd]=pairMarginals.T
    
    normalizer=numpy.sum(multinomial, dtype=numpy.longdouble)
    multinomial=multinomial/normalizer
   
    gammaInv=scipy.linalg.pinv(gamma)
    return (num_candidates, multinomial, gammaInv)
             
class Policy:
    #dataset: (Datasets) Must be initialized using Datasets.loadTxt(...)/loadNpz(...)
    #allow_repetitions: (bool) If true, the policy predicts rankings with repeated documents
    def __init__(self, dataset, allow_repetitions):
        self.dataset=dataset
        self.allowRepetitions=allow_repetitions
        self.name=None
        ###All sub-classes of Policy should supply a predict method
        ###Requires: (int) query_id; (int) ranking_size.
        ###Returns: list[int],length=min(ranking_size,docsPerQuery[query_id]) ranking
           
                
class DeterministicPolicy(Policy):
    #model_type: (str) Model class to use for scoring documents
    def __init__(self, dataset, model_type, regress_gains=False, weighted_ls=False, hyper_params=None, anti=False):
        """
        anti = True, means the policy would learn to predict relevance and return the least relevance documents
        """
        Policy.__init__(self, dataset, False)
        self.modelType=model_type
        self.hyperParams={'alpha': (numpy.logspace(-3,2,num=6,base=10)).tolist()}
        if hyper_params is not None:
            self.hyperParams=hyper_params
        
        self.regressGains=regress_gains
        self.weighted=weighted_ls
        self.anti=anti
        
        self.treeDepths={'max_depth': list(range(3,21,3))}
        
        #Must call train(...) to set all these members
        #before using DeterministicPolicy objects elsewhere
        self.featureList=None
        if self.modelType=='tree':
            self.tree=None
        else:
            self.policyParams=None
            
        #These members are set by predictAll(...) method
        self.savedRankingsSize=None
        self.savedRankings=None
        
        print("DeterministicPolicy:init [INFO] Dataset", dataset.name, flush=True)
    
    #feature_list: list[int],length=unmaskedFeatures; List of features that should be used for training
    #name: (str) String to help identify this DeterministicPolicy object henceforth
    def train(self, feature_list, name):
        self.featureList=feature_list
        self.name=name+'-'+self.modelType
        modelFile=Settings.DATA_DIR+self.dataset.name+'_'+self.name
        if 'alpha' not in self.hyperParams:
            #Expecting hyper-params for GBRT; Add those hyper-params to the model file name
            modelFile=modelFile+'ensemble-'+str(self.hyperParams['ensemble'])+'_lr-'+str(self.hyperParams['lr'])+'_subsample-'+str(self.hyperParams['subsample'])+'_leaves-'+str(self.hyperParams['leaves'])
            
        if self.modelType=='tree' or self.modelType=='gbrt':
            modelFile+='.z'
        else:
            modelFile+='.npz'
            
        self.savedRankingsSize=None
        self.savedRankings=None
        
        # adjust to for in case tree cannot be load
        rerun_model = False
        if os.path.exists(modelFile):
            if self.modelType=='tree' or self.modelType=='gbrt':
                try:
                    self.tree=joblib.load(modelFile)
                    print("DeterministicPolicy:train [INFO] Using precomputed policy", modelFile, flush=True)
                except:
                    print("Policy file found, but cannot be loaded.")
                    rerun_model = True
                    
            else:
                with numpy.load(modelFile) as npFile:
                    self.policyParams=npFile['policyParams']
                print("DeterministicPolicy:train [INFO] Using precomputed policy", modelFile, flush=True)
                print("DeterministicPolicy:train [INFO] PolicyParams", self.policyParams,flush=True)
        else:
            rerun_model = True
                
        if rerun_model:
            numQueries=len(self.dataset.features)
        
            allFeatures=None
            allTargets=None
            print("DeterministicPolicy:train [INFO] Constructing features and targets", flush=True)
                
            if self.dataset.mask is None:
                allFeatures=scipy.sparse.vstack(self.dataset.features, format='csc')
                allTargets=numpy.hstack(self.dataset.relevances)
            else:
                temporaryFeatures=[]
                temporaryTargets=[]
                for currentQuery in range(numQueries):
                    temporaryFeatures.append(self.dataset.features[currentQuery][self.dataset.mask[currentQuery], :])
                    temporaryTargets.append(self.dataset.relevances[currentQuery][self.dataset.mask[currentQuery]])
                
                allFeatures=scipy.sparse.vstack(temporaryFeatures, format='csc')
                allTargets=numpy.hstack(temporaryTargets)
        
            if self.regressGains:
                allTargets=numpy.exp2(allTargets)-1.0
            
            allSampleWeights=None
            fitParams=None
            if self.weighted:
                allSampleWeights=numpy.array(self.dataset.docsPerQuery, dtype=numpy.float64)
                allSampleWeights=numpy.reciprocal(allSampleWeights)
                allSampleWeights=numpy.repeat(allSampleWeights, self.dataset.docsPerQuery)    
                fitParams={'sample_weight': allSampleWeights}
            
            #Restrict features to only the unmasked features
            if self.featureList is not None:
                print("DeterministicPolicy:train [INFO] Masking unused features. Remaining feature size", 
                    len(feature_list), flush=True)
                allFeatures = allFeatures[:, self.featureList]
        
            print("DeterministicPolicy:train [INFO] Beginning training", self.modelType, flush=True)
            if self.modelType=='tree':
#                 treeCV=sklearn.model_selection.GridSearchCV(sklearn.tree.DecisionTreeRegressor(criterion="mse",
#                                                         splitter="random", min_samples_split=4, 
#                                                         min_samples_leaf=4),
#                                 param_grid=self.treeDepths,
#                                 scoring=None, n_jobs=-2,
#                                 iid=True, cv=5, refit=True, verbose=0, pre_dispatch="1*n_jobs",
#                                 error_score='raise', return_train_score=False)
                            
                    
                # presort; fit_params; iid deprecated
                treeCV=sklearn.model_selection.GridSearchCV(sklearn.tree.DecisionTreeRegressor(criterion="mse",
                                                        splitter="random", min_samples_split=4, 
                                                        min_samples_leaf=4),
                                param_grid=self.treeDepths,
                                scoring=None, n_jobs=-2, cv=5, refit=True, verbose=0, pre_dispatch="1*n_jobs",
                                error_score='raise', return_train_score=False)

                treeCV.fit(allFeatures, allTargets)
                self.tree=treeCV.best_estimator_
                print("DeterministicPolicy:train [INFO] Done. Best depth", 
                            treeCV.best_params_['max_depth'], flush=True)
                
                joblib.dump(self.tree, modelFile, compress=9, protocol=-1)
            
            elif self.modelType=='lasso':
                lassoCV=sklearn.model_selection.GridSearchCV(sklearn.linear_model.Lasso(fit_intercept=False,
                                                        normalize=False, precompute=False, copy_X=False, 
                                                        max_iter=3000, tol=1e-4, warm_start=False, positive=False,
                                                        random_state=None, selection='random'),
                                param_grid=self.hyperParams,
                                scoring=None,  n_jobs=-2,
                                iid=True, cv=5, refit=True, verbose=0, pre_dispatch="1*n_jobs",
                                error_score='raise', return_train_score=False)
                                
                lassoCV.fit(allFeatures, allTargets)
                self.policyParams=lassoCV.best_estimator_.coef_
                print("DeterministicPolicy:train [INFO] Done. CVAlpha", lassoCV.best_params_['alpha'], flush=True)
                print("DeterministicPolicy:train [INFO] PolicyParams", self.policyParams,flush=True)
                numpy.savez_compressed(modelFile, policyParams=self.policyParams)
        
            elif self.modelType == 'ridge':
                ridgeCV=sklearn.model_selection.GridSearchCV(sklearn.linear_model.Ridge(fit_intercept=False,
                                                                                    normalize=False, copy_X=False,
                                                                                    max_iter=3000, tol=1e-4, random_state=None),
                                                         param_grid=self.hyperParams,
                                                         n_jobs=-2,
                                                         iid=True, cv=3, refit=True, verbose=0, pre_dispatch='1*n_jobs')
                ridgeCV.fit(allFeatures, allTargets)
                self.policyParams=ridgeCV.best_estimator_.coef_
                print("DeterministicPolicy:train [INFO] Done. CVAlpha", ridgeCV.best_params_['alpha'], flush=True)
            elif self.modelType=='gbrt':
                tree=sklearn.ensemble.GradientBoostingRegressor(learning_rate=self.hyperParams['lr'],
                            n_estimators=self.hyperParams['ensemble'], subsample=self.hyperParams['subsample'], max_leaf_nodes=self.hyperParams['leaves'], 
                            max_features=1.0)
                tree.fit(allFeatures, allTargets, sample_weight=allSampleWeights)
                self.tree=tree
                print("DeterministicPolicy:train [INFO] Done.", flush=True)
                joblib.dump(self.tree, modelFile, compress=9, protocol=-1)
            
            else:
                print("DeterministicPolicy:train [ERR] %s not supported." % self.modelType, flush=True)
                sys.exit(0)
    
    #query_id: (int) Query ID in self.dataset
    #ranking_size: (int) Size of ranking. Returned ranking length is min(ranking_size,docsPerQuery[query_id])
    #                       Use ranking_size=-1 to rank all available documents for query_id
    def predict(self, query_id, ranking_size):
        if self.savedRankingsSize is not None and self.savedRankingsSize==ranking_size:
            return self.savedRankings[query_id]
        
        allowedDocs=self.dataset.docsPerQuery[query_id]
        validDocs=ranking_size
        if ranking_size <= 0 or validDocs > allowedDocs:
            validDocs=allowedDocs
        
        currentFeatures=None
        if self.dataset.mask is None:
            if self.featureList is not None:
                currentFeatures=self.dataset.features[query_id][:, self.featureList]
            else:
                currentFeatures=self.dataset.features[query_id]
            
        else:
            currentFeatures=self.dataset.features[query_id][self.dataset.mask[query_id], :]
            if self.featureList is not None:
                currentFeatures=currentFeatures[:, self.featureList]
        
        allDocScores=None
        if self.modelType=='tree':
            allDocScores=self.tree.predict(currentFeatures)
        elif self.modelType=='gbrt':
            allDocScores=self.tree.predict(currentFeatures.toarray())
        else:
            allDocScores=currentFeatures.dot(self.policyParams)
            
        tieBreaker=numpy.random.random(allDocScores.size)
        if self.anti:
            # predict the least optimal ranking
            sortedDocScores=numpy.lexsort((tieBreaker,-allDocScores))[::-1][0:validDocs]
        else:
            sortedDocScores=numpy.lexsort((tieBreaker,-allDocScores))[0:validDocs]
            
        if self.dataset.mask is None:
            return sortedDocScores
        else:
            return self.dataset.mask[query_id][sortedDocScores]
    
    #ranking_size: (int) Size of ranking. Returned ranking length is min(ranking_size,docsPerQuery[query_id])
    #                       Use ranking_size=-1 to rank all available documents for query_id
    def predictAll(self, ranking_size):
        if self.savedRankingsSize is not None and self.savedRankingsSize==ranking_size:
            return
            
        numQueries=len(self.dataset.features)
        predictedRankings=[]
        for i in range(numQueries):
            predictedRankings.append(self.predict(i, ranking_size))
                
            if i%100==0:
                print(".", end="", flush=True)
                
        self.savedRankingsSize=ranking_size
        self.savedRankings=predictedRankings
        print("", flush=True)
        print("DeterministicPolicy:predictAll [INFO] Generated all predictions for %s using policy: " %
                self.dataset.name, self.name, flush=True)
        
    #num_allowed_docs: (int) Filters the dataset where the max docs per query is num_allowed_docs.
    #                        Uses policyParams to rank and filter the original document set.
    def filterDataset(self, num_allowed_docs):
        self.savedRankingsSize=None
        self.savedRankings=None
        
        numQueries=len(self.dataset.docsPerQuery)
        
        self.dataset.name=self.dataset.name+'-filt('+self.name+'-'+str(num_allowed_docs)+')'
        
        newMask = []
        for i in range(numQueries):
            producedRanking=self.predict(i, num_allowed_docs)
            self.dataset.docsPerQuery[i]=numpy.shape(producedRanking)[0]
            newMask.append(producedRanking)
            if i%100==0:
                print(".", end="", flush=True)
                
        self.dataset.mask=newMask
        print("", flush=True)
        print("DeterministicPolicy:filteredDataset [INFO] New Name", self.dataset.name, "\t MaxNumDocs", num_allowed_docs, flush=True)

        
class UniformPolicy(Policy):
    def __init__(self, dataset, allow_repetitions):
        Policy.__init__(self, dataset, allow_repetitions)
        self.name='Unif-'
        if allow_repetitions:
            self.name+='Rep'
        else:
            self.name+='NoRep'
    
        #These members are set on-demand by setupGamma(...)
        self.gammas=None
        self.gammaRankingSize=None
        
        print("UniformPolicy:init [INFO] Dataset: %s AllowRepetitions:" % dataset.name,
                        allow_repetitions, flush=True)
    
    #ranking_size: (int) Size of ranking.
    def setupGamma(self, ranking_size):
        if self.gammaRankingSize is not None and self.gammaRankingSize==ranking_size:
            print("UniformPolicy:setupGamma [INFO] Gamma has been pre-computed for this ranking_size. Size of Gamma cache:", len(self.gammas), flush=True)
            return
        
        gammaFile=Settings.DATA_DIR+self.dataset.name+'_'+self.name+'_'+str(ranking_size)+'.z'
        rerun = False
        if os.path.exists(gammaFile):
            try: 
                self.gammas=joblib.load(gammaFile)
                self.gammaRankingSize=ranking_size
                print("UniformPolicy:setupGamma [INFO] Using precomputed gamma", gammaFile, flush=True)
            except:
                rerun = True
        else:
            rerun = True
                
            
        if rerun:
            self.gammas={}
            self.gammaRankingSize=ranking_size
            
            candidateSet=set(self.dataset.docsPerQuery)
            n_jobs = 8
            responses=joblib.Parallel(n_jobs=n_jobs, verbose=50)(joblib.delayed(UniformGamma)(i, ranking_size, self.allowRepetitions) for i in candidateSet)
                
#             responses=joblib.Parallel(n_jobs=-2, verbose=50)(joblib.delayed(UniformGamma)(i, ranking_size, self.allowRepetitions) for i in candidateSet)
            
            for tup in responses:
                self.gammas[tup[0]]=tup[1]
            
            joblib.dump(self.gammas, gammaFile, compress=9, protocol=-1)
            print("", flush=True)
            print("UniformPolicy:setupGamma [INFO] Finished creating Gamma_pinv cache. Size", len(self.gammas), flush=True)

    def predict(self, query_id, ranking_size):
        allowedDocs=self.dataset.docsPerQuery[query_id]    
        
        validDocs=ranking_size
        if ranking_size < 0 or ((not self.allowRepetitions) and (validDocs > allowedDocs)):
            validDocs=allowedDocs
            
        producedRanking=None
        if self.allowRepetitions:
            producedRanking=numpy.random.choice(allowedDocs, size=validDocs,
                                replace=True)
        else:
            producedRanking=numpy.random.choice(allowedDocs, size=validDocs,
                                replace=False)
                                
        if self.dataset.mask is None:
            return producedRanking
        else:
            return self.dataset.mask[query_id][producedRanking]
        

class NonUniformPolicy(Policy):
    def __init__(self, deterministic_policy, dataset, allow_repetitions, decay):
        Policy.__init__(self, dataset, allow_repetitions)
        self.decay = decay
        self.policy = deterministic_policy
        self.name='NonUnif-'
        if allow_repetitions:
            self.name+='Rep'
        else:
            self.name+='NoRep'
        self.name += '(' + deterministic_policy.name + ';' + str(decay) + ')'
        
        #These members are set on-demand by setupGamma
        self.gammas=None
        self.multinomials=None
        self.gammaRankingSize=None
        
        print("NonUniformPolicy:init [INFO] Dataset: %s AllowRepetitions:" % dataset.name,
                        allow_repetitions, "\t Decay:", decay, flush=True)
    
    
    def setupGamma(self, ranking_size):
        if self.gammaRankingSize is not None and self.gammaRankingSize==ranking_size:
            print("NonUniformPolicy:setupGamma [INFO] Gamma has been pre-computed for this ranking_size. Size of Gamma cache:", len(self.gammas), flush=True)
            return
        
        gammaFile=Settings.DATA_DIR+self.dataset.name+'_'+self.name+'_'+str(ranking_size)+'.z'
        rerun = False
        if os.path.exists(gammaFile):
            try:
                self.gammas, self.multinomials=joblib.load(gammaFile)
                self.gammaRankingSize=ranking_size
                print("NonUniformPolicy:setupGamma [INFO] Using precomputed gamma", gammaFile, flush=True)
            except:
                rerun = True
        else:
            rerun = True
            
        if rerun:
            self.gammas={}
            self.multinomials={}
            self.gammaRankingSize=ranking_size
            
            candidateSet=set(self.dataset.docsPerQuery)
            n_jobs=8
            responses=joblib.Parallel(n_jobs=n_jobs, verbose=50)(joblib.delayed(NonUniformGamma)(i, self.decay, ranking_size, self.allowRepetitions) for i in candidateSet)
            
            for tup in responses:
                self.gammas[tup[0]]=tup[2]
                self.multinomials[tup[0]]=tup[1]
            
            joblib.dump((self.gammas, self.multinomials), gammaFile, compress=9, protocol=-1)
            print("", flush=True)
            print("NonUniformPolicy:setupGamma [INFO] Finished creating Gamma_pinv cache. Size", len(self.gammas), flush=True)

        self.policy.predictAll(-1)

    def predict(self, query_id, ranking_size):
        allowedDocs=self.dataset.docsPerQuery[query_id]    
        underlyingRanking=self.policy.predict(query_id, -1)
            
        validDocs=ranking_size
        if ranking_size < 0 or ((not self.allowRepetitions) and (validDocs > allowedDocs)):
            validDocs=allowedDocs
            
        currentDistribution=self.multinomials[allowedDocs]
        producedRanking=None
        if self.allowRepetitions:
            producedRanking=numpy.random.choice(allowedDocs, size=validDocs,
                                replace=True, p=currentDistribution)
        else:
            producedRanking=numpy.random.choice(allowedDocs, size=validDocs,
                                replace=False, p=currentDistribution)
                                
        return underlyingRanking[producedRanking]
        
        
if __name__=="__main__":
    import Settings
    import Datasets
    import scipy.stats
    
    M=100
    L=10
    resetSeed=387
    
    mslrData=Datasets.Datasets()
    mslrData.loadNpz(Settings.DATA_DIR+'MSLR/mslr')
    
    anchorURLFeatures, bodyTitleDocFeatures=Settings.get_feature_sets("MSLR")
    
    numpy.random.seed(resetSeed)
    detLogger=DeterministicPolicy(mslrData, 'tree')
    detLogger.train(anchorURLFeatures, 'url')
    
    detLogger.filterDataset(M)
    
    filteredDataset=detLogger.dataset
    del mslrData
    del detLogger
    
    uniform=UniformPolicy(filteredDataset, False)
    uniform.setupGamma(L)
    del uniform
    
    numpy.random.seed(resetSeed)
    loggingPolicyTree=DeterministicPolicy(filteredDataset, 'tree')
    loggingPolicyTree.train(anchorURLFeatures, 'url')
            
    numpy.random.seed(resetSeed)
    targetPolicyTree=DeterministicPolicy(filteredDataset, 'tree')
    targetPolicyTree.train(bodyTitleDocFeatures, 'body')
    
    numpy.random.seed(resetSeed)
    loggingPolicyLinear=DeterministicPolicy(filteredDataset, 'lasso')
    loggingPolicyLinear.train(anchorURLFeatures, 'url')
    
    numpy.random.seed(resetSeed)
    targetPolicyLinear=DeterministicPolicy(filteredDataset, 'lasso')
    targetPolicyLinear.train(bodyTitleDocFeatures, 'body')
    
    numQueries=len(filteredDataset.docsPerQuery)
    
    TTtau=[]
    TToverlap=[]
    TLtau=[]
    TLoverlap=[]
    LTtau=[]
    LToverlap=[]
    LLtau=[]
    LLoverlap=[]
    LogLogtau=[]
    LogLogoverlap=[]
    TargetTargettau=[]
    TargetTargetoverlap=[]
    
    def computeTau(ranking1, ranking2):
        rank1set=set(ranking1)
        rank2set=set(ranking2)
        documents=rank1set | rank2set
        rankingSize=len(rank1set)
        
        newRanking1=numpy.zeros(len(documents), dtype=numpy.int)
        newRanking2=numpy.zeros(len(documents), dtype=numpy.int)
        
        for docID, doc in enumerate(documents):
            if doc not in rank1set:
                newRanking1[docID]=rankingSize + 1
                newRanking2[docID]=ranking2.index(doc)
            elif doc not in rank2set:
                newRanking2[docID]=rankingSize + 1
                newRanking1[docID]=ranking1.index(doc)
            else:
                newRanking1[docID]=ranking1.index(doc)
                newRanking2[docID]=ranking2.index(doc)
            
        return scipy.stats.kendalltau(newRanking1, newRanking2)[0], 1.0*len(rank1set&rank2set)/rankingSize
    
    numpy.random.seed(resetSeed)    
    for currentQuery in range(numQueries):
        if filteredDataset.docsPerQuery[currentQuery]<4:
            continue
            
        logTreeRanking=loggingPolicyTree.predict(currentQuery, L).tolist()
        logLinearRanking=loggingPolicyLinear.predict(currentQuery, L).tolist()
        
        targetTreeRanking=targetPolicyTree.predict(currentQuery, L).tolist()
        targetLinearRanking=targetPolicyLinear.predict(currentQuery, L).tolist()
        
        tau, overlap=computeTau(logTreeRanking, targetTreeRanking)
        TTtau.append(tau)
        TToverlap.append(overlap)
        
        tau, overlap=computeTau(logTreeRanking, targetLinearRanking)
        TLtau.append(tau)
        TLoverlap.append(overlap)
        
        tau, overlap=computeTau(logLinearRanking, targetTreeRanking)
        LTtau.append(tau)
        LToverlap.append(overlap)
        
        tau, overlap=computeTau(logLinearRanking, targetLinearRanking)
        LLtau.append(tau)
        LLoverlap.append(overlap)
        
        tau, overlap=computeTau(logLinearRanking, logTreeRanking)
        LogLogtau.append(tau)
        LogLogoverlap.append(overlap)
        
        tau, overlap=computeTau(targetLinearRanking, targetTreeRanking)
        TargetTargettau.append(tau)
        TargetTargetoverlap.append(overlap)
        
        if len(TTtau) % 100 == 0:
            print(".", end="", flush=True)
    
    TTtau=numpy.array(TTtau)
    TLtau=numpy.array(TLtau)
    LTtau=numpy.array(LTtau)
    LLtau=numpy.array(LLtau)
    LogLogtau=numpy.array(LogLogtau)
    TargetTargettau=numpy.array(TargetTargettau)
    
    TToverlap=numpy.array(TToverlap)
    TLoverlap=numpy.array(TLoverlap)
    LToverlap=numpy.array(LToverlap)
    LLoverlap=numpy.array(LLoverlap)
    LogLogoverlap=numpy.array(LogLogoverlap)
    TargetTargetoverlap=numpy.array(TargetTargetoverlap)
    
    print("", flush=True)    
    print("TTtau", numpy.amax(TTtau), numpy.amin(TTtau), numpy.mean(TTtau), numpy.std(TTtau), numpy.median(TTtau), len(numpy.where(TTtau > 0.99)[0]))
    print("TToverlap", numpy.amax(TToverlap), numpy.amin(TToverlap), numpy.mean(TToverlap), numpy.std(TToverlap), numpy.median(TToverlap), len(numpy.where(TToverlap > 0.99)[0]))
    print("TLtau", numpy.amax(TLtau), numpy.amin(TLtau), numpy.mean(TLtau), numpy.std(TLtau), numpy.median(TLtau), len(numpy.where(TLtau > 0.99)[0]))
    print("TLoverlap", numpy.amax(TLoverlap), numpy.amin(TLoverlap), numpy.mean(TLoverlap), numpy.std(TLoverlap), numpy.median(TLoverlap), len(numpy.where(TLoverlap > 0.99)[0]))
    print("LTtau", numpy.amax(LTtau), numpy.amin(LTtau), numpy.mean(LTtau), numpy.std(LTtau), numpy.median(LTtau), len(numpy.where(LTtau > 0.99)[0]))
    print("LToverlap", numpy.amax(LToverlap), numpy.amin(LToverlap), numpy.mean(LToverlap), numpy.std(LToverlap), numpy.median(LToverlap), len(numpy.where(LToverlap > 0.99)[0]))
    print("LLtau", numpy.amax(LLtau), numpy.amin(LLtau), numpy.mean(LLtau), numpy.std(LLtau), numpy.median(LLtau), len(numpy.where(LLtau > 0.99)[0]))
    print("LLoverlap", numpy.amax(LLoverlap), numpy.amin(LLoverlap), numpy.mean(LLoverlap), numpy.std(LLoverlap), numpy.median(LLoverlap), len(numpy.where(LLoverlap > 0.99)[0]))
    print("LogLogtau", numpy.amax(LogLogtau), numpy.amin(LogLogtau), numpy.mean(LogLogtau), numpy.std(LogLogtau), numpy.median(LogLogtau), len(numpy.where(LogLogtau > 0.99)[0]))
    print("LogLogoverlap", numpy.amax(LogLogoverlap), numpy.amin(LogLogoverlap), numpy.mean(LogLogoverlap), numpy.std(LogLogoverlap), numpy.median(LogLogoverlap), len(numpy.where(LogLogoverlap > 0.99)[0]))
    print("TargetTargettau", numpy.amax(TargetTargettau), numpy.amin(TargetTargettau), numpy.mean(TargetTargettau), numpy.std(TargetTargettau), numpy.median(TargetTargettau), len(numpy.where(TargetTargettau > 0.99)[0]))
    print("TargetTargetoverlap", numpy.amax(TargetTargetoverlap), numpy.amin(TargetTargetoverlap), numpy.mean(TargetTargetoverlap), numpy.std(TargetTargetoverlap), numpy.median(TargetTargetoverlap), len(numpy.where(TargetTargetoverlap > 0.99)[0]))