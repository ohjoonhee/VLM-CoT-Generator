#!/bin/bash

export FIREWORKS_MODEL="accounts/fireworks/models/qwen3-vl-235b-a22b-thinking"


python -u src/inference/base/vlm/fireworks.py \
    --output_dir output/distill \
    --dataset_name ohjoonhee/Visual-CoT-4k \
    --split train \