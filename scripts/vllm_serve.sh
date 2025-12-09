#!/bin/bash

vllm serve Qwen/Qwen3-VL-32B-Thinking-FP8 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --gpu-memory-utilization 0.95 \
    --enforce-eager \
    --max-model-len 8192 \
    --quantization fp8
    # --reasoning-parser deepseek_r1 \