# Off-policy evaluation for ranking in information retrieval
Based on the work in [1], we further explore off-policy evaluation of slate recommenders/ranker.

These python scripts and classes run semi-synthetic experiments on the MSLR-10K datasets
to study off-policy estimators for the slate bandit problem (combinatorial contextual bandits).

#### How to run Evaluation experiments:

**Setup:** All dependencies can be found in `environment.yml `

The `run5_script_0.s` bash script is a sample script to the experiments we have done in cluster jobs.`Run.ipynb` contains all scripts to the entrie suite of experiments reported. Alternatively, you can call the python scripts directly, e.g. for running evaluation for `DM_ridge` metrics under anti-optimal stochastic logging policy and optimal deterministic target policy.

**Run:** 

```
python Parallel.py \
-t 1 -lf All -la True \
-ef All -e tree \
-m 100 -l 10 -v ERR \
-a DM_ridge \
--start 0 --stop 5 \
&> eval.log.ERR.100.10.tree.All.Log.All.anti.DM_ridge

```

Refer `Parallel.py::main` for examples on how to set up other variants of experiments

**Main Changes**

We highlight the major changes in the code we made based on the original repo as follows:

* Add stochastic target policy and change the `IPS` and `SN_IPS` estimator for stochastic target policy accordingly. (The previous estimators implementation only accept deterministic target policy. )
* Add anti-optimal, sub-optimal and optimal policy for both target and logging policy for experiments.



#### Data:

MSLR-10K has 10K queries, each with up to 908 judged documents on relevance scale of {0, 1, 2, 3, 4}
The feature dimension of <query, document> is 136.

Refer `Datasets.py::main` for how to read in these datasets
    Download and uncompress the dataset files in the `../Project/`
    MSLR: https://www.microsoft.com/en-us/research/project/mslr/



#### Result log 

All result logs are saved in `log` folder.



#### Other

`PI_estimate_analysis.ipynb` contains a toy example to compare $PI$ estimator and $IW$ estimator.

`make_the_plot.ipynb` parses all the logs after evaluation and generate figures in the report.

​    

[1] Off policy evaluation for slate recommendation, https://arxiv.org/abs/1605.04812 ; https://nips.cc/Conferences/2017/Schedule?showEvent=9146