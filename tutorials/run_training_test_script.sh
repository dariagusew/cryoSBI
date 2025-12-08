#!/bin/bash
set -e

# Step 1: Pretrain ResNet18
echo "=== Started pretraining ResNet18 ==="

pretrain_resnet18 \
    --embedding RESNET18 \
    --epochs 1 \
    --batch_size 256 \
    --device cuda:0 \
    --output pretrained_resnet18.pt \
    --embedding_dim 256 \
    --image_config simulation_parameters.json \
    --lr 0.0002 \
    --simulation_batch_size 1024

echo "=== Finished pretraining ResNet18 ==="


# Step 2: Train NLE model 
echo "=== Started training NLE ==="

train_nle_model \
    --image_config_file simulation_parameters.json \
    --train_config_file training_parameters_nle.json\
    --epochs 1 \
    --estimator_file tutorial_estimator \
    --loss_file tutorial.loss \
    --n_workers 4 \
    --train_device cuda:0 \
    --pretrained_embedding_path pretrained_resnet18.pt \
    --freeze_embedding True \
    --saving_freq 100 \
    --simulation_batch_size 2048 \
    
echo "=== Finished training NLE ==="
