#!/bin/bash
set -e

# Step 1: Pretrain SPATIAL_CRYO_FFT_FILTER
echo "=== Started pretraining SPATIAL_CRYO_FFT_FILTER ==="

pretrain_spatial_cryo \
    --embedding SPATIAL_CRYO_FFT_FILTER \
    --epochs 100 \
    --batch_size 256 \
    --device cuda:1 \
    --output pretrained_spatial_cryo_filter.pt \
    --embedding_dim 256 \
    --image_config simulation_parameters.json \
    --lr 0.0002 \
    --simulation_batch_size 2048

echo "=== Finished pretraining SPATIAL_CRYO_FFT_FILTER ==="


# Step 2: Train NLE model 
echo "=== Started training NLE ==="

train_nle_model \
    --image_config_file simulation_parameters.json \
    --train_config_file training_parameters_nle.json\
    --epochs 200 \
    --estimator_file tutorial_estimator \
    --loss_file tutorial.loss \
    --n_workers 4 \
    --train_device cuda:1 \
    --pretrained_embedding_path pretrained_spatial_cryo_filter.pt \
    --freeze_embedding True \
    --saving_freq 100 \
    --simulation_batch_size 4096 \
    
echo "=== Finished training NLE ==="
