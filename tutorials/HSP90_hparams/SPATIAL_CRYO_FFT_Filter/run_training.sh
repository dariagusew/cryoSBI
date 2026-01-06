#!/bin/bash

BASE_PATH="/projects/dynaplix/people/zvh378/cryoSBI/tutorials/HSP90_hparams/"

echo "=== Started training SPATIAL_CRYO_FFT_Filter embed_dim=16 ==="

bash "${BASE_PATH}SPATIAL_CRYO_FFT_Filter/embed_dim_16/run_training.sh"

echo "=== Started training SPATIAL_CRYO_FFT_Filter embed_dim=64 ==="

bash "${BASE_PATH}SPATIAL_CRYO_FFT_Filter/embed_dim_64/run_training.sh"


echo "=== Started training SPATIAL_CRYO_FFT_Filter embed_dim=265 ==="

bash "${BASE_PATH}SPATIAL_CRYO_FFT_Filter/embed_dim_256/run_training.sh"

echo "All trainings finished"

