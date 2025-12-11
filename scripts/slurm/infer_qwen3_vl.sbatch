#!/bin/bash
#SBATCH -J launch
#SBATCH -p gpu-4farm
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH -o logs/%x_%j.log
#SBATCH -e logs/%x_%j.log
#SBATCH --account=gpu

set -a
source .env
set +a


# Serve vllm text thinking model
export MODEL="Qwen/Qwen3-VL-235B-A22B-Thinking-FP8"

# vllm serve $MODEL --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser deepseek_r1 --gpu-memory-utilization 0.4 &

uv run vllm serve $MODEL \
    --port 10630 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.95 \
    --enforce-eager \
    --max-model-len 32768 \
    --quantization fp8 &
    # --reasoning-parser deepseek_r1 \

# Save the Process ID (PID) of the background job
VLLM_PID=$!
echo "VLLM server started with PID: $VLLM_PID"

# A more robust approach is to use a health check loop.
# This loop checks a port/endpoint until the server is ready.
# This assumes vllm is serving on localhost:10630 and has a /health endpoint.
SERVER_READY=0
MAX_ATTEMPTS=30
ATTEMPT=0
echo "Checking VLLM server readiness..."
while [ $SERVER_READY -eq 0 ] && [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    if curl -s http://localhost:10630/health > /dev/null; then
        SERVER_READY=1
        echo "VLLM server is READY."
    else
        ATTEMPT=$((ATTEMPT + 1))
        echo "Attempt $ATTEMPT/$MAX_ATTEMPTS: Server not ready yet. Waiting 2 seconds..."
        sleep 30
    fi
done

if [ $SERVER_READY -eq 0 ]; then
    echo "ERROR: VLLM server failed to start or become ready after $MAX_ATTEMPTS attempts."
    # Kill the background process and exit the script
    kill $VLLM_PID
    exit 1
fi

echo "VLLM is ready. Running inference script..."

uv run python src/infer/vlm/qwen3vl.py --model "$MODEL" --port 10630 --output_dir "output" --system_prompt_path "configs/prompts/think_first_v0.txt"
uv run python src/infer/vlm/qwen3vl.py --model "$MODEL" --port 10630 --output_dir "output" --system_prompt_path "configs/prompts/think_first_v0.txt" --dataset_name "DreamMr/HR-Bench" --split "hrbench_4k"
uv run python src/infer/vlm/qwen3vl.py --model "$MODEL" --port 10630 --output_dir "output" --system_prompt_path "configs/prompts/think_first_v0.txt" --dataset_name "DreamMr/HR-Bench" --split "hrbench_8k"
uv run python src/infer/vlm/qwen3vl.py --model "$MODEL" --port 10630 --output_dir "output" --system_prompt_path "configs/prompts/think_first_v0.txt" --dataset_name "jonathan-roberts1/zerobench" --split "zerobench" --question_column "question_text" --image_column "question_images_decoded"


echo "Inference complete. Stopping VLLM server (PID: $VLLM_PID)..."
kill $VLLM_PID

echo "Script finished."
