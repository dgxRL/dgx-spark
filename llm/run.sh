#!/bin/bash
#
# vLLM Server Runner for NVIDIA DGX Spark
# This script runs vLLM using NVIDIA's optimized container for Grace Blackwell GB10
#

# Configuration
MODEL="${1:-Qwen/Qwen3-VL-30B-A3B-Instruct-FP8}"
MAX_LEN="${2:-32768}"
PORT="${3:-8888}"

# Clear system memory caches for optimal performance
echo "Clearing system memory caches..."
if sudo -n sysctl -w vm.drop_caches=3 > /dev/null 2>&1; then
    echo "✓ Memory caches cleared"
else
    echo "Note: Could not clear memory caches (may require sudo password)"
    echo "      Run manually: sudo sysctl -w vm.drop_caches=3"
fi
echo ""

echo "Starting vLLM server..."
echo "Model: $MODEL"
echo "Max length: $MAX_LEN"
echo "Port: $PORT (mapped to container port 8000)"
echo ""
echo "Access the API at: http://localhost:$PORT"
echo "Press Ctrl+C to stop"
echo ""

docker run -it --gpus all -p ${PORT}:8000 \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/.cache/vllm:/root/.cache/vllm \
    --rm vllm-custom:latest \
    vllm serve "$MODEL" --max-model-len "$MAX_LEN"

