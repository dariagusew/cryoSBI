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
#SBATCH --account=cosb
#SBATCH --partition=cosb
# request 1 GPU
#SBATCH --gres=gpu:1
#SBATCH --exclude=maestro-3484,maestro-3480
#SBATCH --exclusive

# activate conda
source ~/SCRATCH/miniconda3/bin/activate
# activate conda environment
conda activate cryo-EM

# we have 897 micrographs
python extract_patches_to_mrc.py --in_dir /pasteur/helix/scratch/mbonomi/cryoSBI/tutorials/ice/17may18h_leginon/rawdata --out_train ice_train.mrc --out_val ice_val.mrc --box_size 128 --pixel_size 1.693 --mic_pixel_size 1.096 --num_patches 400 --split_by patch --device cuda
# train diffusion
python train_noise_GAN.py --train_mrc ice_train.mrc --val_mrc ice_val.mrc --max_epochs 300 --batch_size 128 --lambda_nps 10.0 --lambda_fm 1.0 --augment --device cuda  --steps_per_epoch 50 --gen_channels 8 --disc_channels 8
# validate
python visualize_gan_output.py --model_path noise_gan.pt --mrc_path ice_val.mrc --device cuda 
