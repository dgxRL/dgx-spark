#!/bin/bash

# Launch ComfyUI with DGX Spark optimizations
cd "$(dirname "$0")"

# Activate virtual environment
source comfyui-env/bin/activate

# Set DGX Spark environment variables
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH}

# Optimize for Grace Blackwell unified memory
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

# Launch ComfyUI
cd ComfyUI/
python main.py \
    --listen 0.0.0.0 \
    --port 8188 \
    --fp16-vae \
    --use-pytorch-cross-attention \
    --disable-xformers \
    --config ../config_dgx_spark.yaml \
    "$@"
