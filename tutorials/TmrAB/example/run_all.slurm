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
#SBATCH --exclude=maestro-3484,maestro-3480
#SBATCH --array=1-24
#SBATCH --output=slurm_OUT/slurm-%A_%a.out

# various dir
SCRIPTS_=/pasteur/appa/scratch/mbonomi/cryoSBI/src/cryo_sbi/utils
DATA_=/pasteur/appa/scratch/mbonomi/cryoSBI/tutorials/TmrAB/data

# activate conda
source ~/SCRATCH/miniconda3/bin/activate
# activate conda environment
conda activate cryo-EM

# 0. get directory test
jobdir=`sed -n ${SLURM_ARRAY_TASK_ID}p jobdir_list`
# go there and do your job
cd $jobdir

# 1. train image embedding
pretrain_image_embed --image_config simulation_parameters.json --training_mode synthetic --embedding SPATIAL_CRYO --batch_size 256 --embedding_dim 16 --simulation_batch_size 1024 --epochs 100 --lr 2e-4 > log.pretrain

# 2. check embedding
python ${SCRIPTS_}/compare_embeddings.py --pretrained_weights pretrained_image_embed.pt --image_config simulation_parameters.json --real_images ${DATA_}/1p5mln-TmrAB-stack-128-small.mrcs --embedding SPATIAL_CRYO --embedding_dim 16 --n_synthetic 20000 --n_real 20000 > log.check

# 3. train flow
train_nle_model --image_config_file simulation_parameters.json --train_config_file training_parameters_nle.json --epochs 200 --estimator_file tutorial_estimator.pt --n_workers 4 --train_device cuda --pretrained_embedding_path pretrained_image_embed.pt --freeze_embedding True --loss_file tutorial.loss --simulation_batch_size 2048 > log.train

# 4. do inference
ln -s Tmr*.pt TmrAB.pt
run_inference_real_data --image-stack ${DATA_}/1p5mln-TmrAB-stack-128-small.mrcs --models_file TmrAB.pt --estimator_file tutorial_estimator.pt --train_config_file training_parameters_nle.json --image_config_file simulation_parameters.json --output-file results.pt > log.inference
