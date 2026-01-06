#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BASE_PATH=/projects/dynaplix/people/zvh378/cryoSBI/tutorials/HSP90_hparams/


# Step 1: Pretrain ResNet18 FFT Filter
echo "=== Started pretraining ResNet18 FFT Filter ==="

pretrain_image_embed \
    --embedding RESNET18_FFT_FILTER \
    --epochs 100 \
    --batch_size 256 \
    --device cuda:1 \
    --output pretrained_resnet18_fft_filter.pt \
    --embedding_dim 16 \
    --image_config "${BASE_PATH}simulation_parameters.json" \
    --lr 0.0002 \
    --simulation_batch_size 1024

echo "=== Finished pretraining ResNet18 FFT Filter ==="


# Step 2: Train NLE model 
echo "=== Started training NLE ==="

train_nle_model \
    --image_config_file "${BASE_PATH}simulation_parameters.json" \
    --train_config_file training_parameters_nle.json\
    --epochs 200 \
    --estimator_file tutorial_estimator.pt \
    --loss_file tutorial.loss \
    --n_workers 4 \
    --train_device cuda:1 \
    --pretrained_embedding_path pretrained_resnet18_fft_filter.pt \
    --freeze_embedding True \
    --saving_freq 100 \
    --simulation_batch_size 2048 \
    
echo "=== Finished training NLE ==="
