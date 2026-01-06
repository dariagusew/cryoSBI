#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"


BASE_PATH=/projects/dynaplix/people/zvh378/cryoSBI/tutorials/HSP90_hparams_pref_orient/


# Step 1: Pretrain ResNet18
echo "=== Started pretraining ResNet18 ==="

pretrain_image_embed \
    --embedding RESNET18 \
    --epochs 100 \
    --batch_size 256 \
    --device cuda:0 \
    --output pretrained_resnet18.pt \
    --embedding_dim 16 \
    --image_config "${BASE_PATH}simulation_parameters.json" \
    --lr 0.0002 \
    --simulation_batch_size 1024

echo "=== Finished pretraining ResNet18 ==="


# Step 2: Train NLE model 
echo "=== Started training NLE ==="

train_nle_model \
    --image_config_file "${BASE_PATH}simulation_parameters.json" \
    --train_config_file training_parameters_nle.json\
    --epochs 200 \
    --estimator_file tutorial_estimator.pt \
    --loss_file tutorial.loss \
    --n_workers 4 \
    --train_device cuda:0 \
    --pretrained_embedding_path pretrained_resnet18.pt \
    --freeze_embedding True \
    --saving_freq 100 \
    --simulation_batch_size 2048 \
    
echo "=== Finished training NLE ==="
