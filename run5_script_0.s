#!/bin/sh

#SBATCH --job-name=run50batch_bao
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=50GB
#SBATCH --time=72:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=gw1107@nyu.edu

singularity exec $nv \
	    --overlay /scratch/sm7582/prince/overlay-50G-10M.ext3:ro \
	    /scratch/work/public/singularity/cuda10.0-cudnn7-devel-ubuntu18.04.sif \
	    /bin/bash -c "
source /ext3/env.sh
conda activate bao
python Parallel.py -t 0 -ef url -m 100 -l 10 -v ERR -e tree -a OnPolicy --start 0 --stop 25 &> eval.log.ERR.100.10.tree.url.LogUniform.OnPolicy
python Parallel.py -t 0 -ef url -m 100 -l 10 -v ERR -e tree -a IPS_SN --start 0 --stop 25 &> eval.log.ERR.100.10.tree.url.LogUniform.IPS_SN
"