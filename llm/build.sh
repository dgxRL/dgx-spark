#!/bin/bash
#
# Build Script for vLLM Docker Image on NVIDIA DGX Spark
# This builds a custom vLLM container optimized for Grace Blackwell GB10
#

set -e

echo "======================================"
echo "vLLM Docker Image Builder"
echo "======================================"
echo ""
echo "This will build a custom vLLM image using:"
echo "  - NVIDIA's base vLLM container (25.09)"
echo "  - Latest vLLM source from GitHub"
echo "  - CUDA 13 support for Grace Blackwell"
echo ""
echo "Build time: ~5-10 minutes"
echo "Image size: ~10GB"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Build cancelled."
    exit 1
fi

echo ""
echo "Building vllm-custom:25.09..."
echo ""

docker build -t vllm-custom:25.09 .

echo ""
echo "======================================"
echo "Build complete!"
echo "======================================"
echo ""
echo "Image: vllm-custom:25.09"
echo ""
echo "Next steps:"
echo "  1. Run a model:  ./run.sh"
echo "  2. Or customize: ./run.sh <MODEL_NAME> <MAX_LENGTH> <PORT>"
echo ""
echo "Examples:"
echo "  ./run.sh nvidia/Llama-3.1-8B-Instruct-FP8 8192 8888"
echo "  ./run.sh mistralai/Mistral-7B-Instruct-v0.3 8192 9000"
echo ""
