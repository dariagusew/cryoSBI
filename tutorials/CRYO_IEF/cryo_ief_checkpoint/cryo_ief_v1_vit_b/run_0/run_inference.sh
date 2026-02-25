#!/usr/bin/env bash
# set -e

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# cd "$SCRIPT_DIR"

# for inference in inference1 inference2 inference3
# do 
#     mkdir -p "$inference"
#     cd "$inference"

echo "=== Started inference 10k ==="

infer_populations \
    --models_file ../../../hsp90_models-small.pt \
    --train_config_file training_parameters_nle.json \
    --image_config_file simulation_parameters_ensemble.json \
    --estimator_file tutorial_estimator.pt \
    --device cuda:0 \
    --population_steps 5 \
    --num_sim 10000 \
    --verbose \
    --output_prefix 10k \

echo "=== Finished inference 10k ==="

    echo "=== Started inference 100k ==="

    infer_populations \
        --models_file ../../../hsp90_models-small.pt \
        --train_config_file training_parameters_nle.json \
        --image_config_file simulation_parameters_ensemble.json \
        --estimator_file tutorial_estimator.pt \
        --device cuda:0 \
        --population_steps 5 \
        --num_sim 100000 \
        --verbose \
        --output_prefix 100k \

    echo "=== Finished inference 100k ==="

    echo "=== Started inference 1M ==="

    infer_populations \
        --models_file ../../../hsp90_models-small.pt \
        --train_config_file ../training_parameters_nle.json \
        --image_config_file ../../../simulation_parameters_ensemble.json \
        --estimator_file ../tutorial_estimator.pt \
        --device cuda:0 \
        --population_steps 5 \
        --num_sim 1000000 \
        --verbose \
        --output_prefix 1M \

    echo "=== Finished inference 1M ==="
    cd ..
done 
