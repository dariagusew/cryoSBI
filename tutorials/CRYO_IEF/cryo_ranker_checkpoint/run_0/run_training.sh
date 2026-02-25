#!/usr/bin/env bash
# set -e

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# cd "$SCRIPT_DIR"


# #Step 1: Pretrain CryoIEFBaseSafe
# echo "=== Started pretraining CryoIEFBaseSafe ==="

# pretrain_image_embed \
#     --embedding CryoIEFBaseSafe \
#     --epochs 100 \
#     --batch_size 256 \
#     --device cuda:0 \
#     --output pretrained_cryoief.pt \
#     --embedding_dim 768 \
#     --image_config "simulation_parameters.json" \
#     --lr 0.0002 \
#     --simulation_batch_size 1024

# echo "=== Finished pretraining CryoIEFBaseSafe ==="


# # Step 2: Train NLE model 
 echo "=== Started training NLE ==="

train_nle_model \
    --image_config_file "simulation_parameters.json" \
    --train_config_file training_parameters_nle.json\
    --epochs 200 \
    --estimator_file tutorial_estimator.pt \
    --loss_file tutorial.loss \
    --n_workers 4 \
    --train_device cuda:0 \
    --saving_freq 100 \
    --simulation_batch_size 2048 \
    --pretrained_embedding_path ../pretrained_cryoief.pt \
    --freeze_embedding True \

 echo "=== Finished training NLE ==="
