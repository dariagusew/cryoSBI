#!/bin/bash

BASE_PATH="/projects/dynaplix/people/zvh378/cryoSBI/tutorials/HSP90_hparams/"

echo "=== Started inference SPATIAL_CRYO embed_dim=16 ==="

bash "${BASE_PATH}SPATIAL_CRYO/embed_dim_16/run_inference.sh"

echo "=== Started inference SPATIAL_CRYO embed_dim=64 ==="

bash "${BASE_PATH}SPATIAL_CRYO/embed_dim_64/run_inference.sh"


echo "=== Started inference SPATIAL_CRYO embed_dim=265 ==="

bash "${BASE_PATH}SPATIAL_CRYO/embed_dim_256/run_inference.sh"

echo "Inference finished"

