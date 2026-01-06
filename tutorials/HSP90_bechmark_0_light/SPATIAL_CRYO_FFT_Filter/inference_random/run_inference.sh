#!/bin/bash
set -e

echo "=== Started inference 10k ==="

infer_populations \
    --models_file ../../hsp90_models-small.pt \
    --train_config_file ../training_parameters_nle.json \
    --image_config_file simulation_parameters_ensemble.json \
    --estimator_file ../tutorial_estimator \
    --device cuda:1 \
    --population_steps 5 \
    --num_sim 10000 \
    --verbose \
    --output_prefix 10k \
    --use_random

echo "=== Finished inference 10k ==="

echo "=== Started inference 100k ==="

infer_populations \
   --models_file ../../hsp90_models-small.pt \
   --train_config_file ../training_parameters_nle.json \
   --image_config_file simulation_parameters_ensemble.json \
   --estimator_file ../tutorial_estimator \
   --device cuda:1 \
   --population_steps 5 \
   --num_sim 100000 \
   --verbose \
   --output_prefix 100k \
   --use_random

echo "=== Finished inference 100k ==="

echo "=== Started inference 1M ==="

infer_populations \
    --models_file ../../hsp90_models-small.pt \
    --train_config_file ../training_parameters_nle.json \
    --image_config_file simulation_parameters_ensemble.json \
    --estimator_file ../tutorial_estimator \
    --device cuda:1 \
    --population_steps 5 \
    --num_sim 1000000 \
    --verbose \
    --output_prefix 1M \
    --use_random

echo "=== Finished inference 1M ==="
