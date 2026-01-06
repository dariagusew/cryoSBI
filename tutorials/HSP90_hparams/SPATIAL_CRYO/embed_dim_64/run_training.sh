#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BASE_PATH=/projects/dynaplix/people/zvh378/cryoSBI/tutorials/HSP90_hparams/


# Step 1: Pretrain SPATIAL_CRYO
echo "=== Started pretraining SPATIAL_CRYO ==="

pretrain_image_embed \
    --embedding SPATIAL_CRYO \
    --epochs 100 \
    --batch_size 256 \
    --device cuda:0 \
    --output pretrained_spatial_cryo.pt \
    --embedding_dim 64 \
    --image_config "${BASE_PATH}simulation_parameters.json" \
    --lr 0.0002 \
    --simulation_batch_size 1024

echo "=== Finished pretraining SPATIAL_CRYO ==="


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
    --pretrained_embedding_path pretrained_spatial_cryo.pt \
    --freeze_embedding True \
    --saving_freq 100 \
    --simulation_batch_size 2048 \
    
echo "=== Finished training NLE ==="
