#!/bin/bash
### name of the job
#SBATCH --job-name=cryoB
### number of nodes
#SBATCH --nodes=1
### Specify the number of tasks per core. Values greater than one turn hyperthreading on
#SBATCH --ntasks-per-core=1
### Specify the number of tasks per node. Defines the number of MPI ranks per node
#SBATCH --ntasks-per-node=1
### Specify the number of cpus per task. Defines the number of OpenMP threads per MPI rank
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
### type of partition/qos
#SBATCH --account=baies
#SBATCH --partition=baies
# request 1 GPU
#SBATCH --gres=gpu:1
#SBATCH --array=1-144%48
#SBATCH --output=slurm_OUT/slurm-%A_%a.out
#SBATCH --exclude=maestro-3484,maestro-3480

# activate conda
source ~/SCRATCH/miniconda3/bin/activate
# activate conda environment
conda activate cryo-EM

# 0. get directory test
jobdir=`sed -n ${SLURM_ARRAY_TASK_ID}p jobdir_list`
# go there and do your job
cd $jobdir

# 1. run script
bash go.sh
