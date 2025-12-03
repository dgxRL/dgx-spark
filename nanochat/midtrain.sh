#!/bin/bash

# This script runs the midtraining (fine-tuning) for the Nanochat model on Nvidia DGX Spark.
# It sets up all necessary environment variables and activates the virtual environment
# to ensure training can run even if the user has logged out and back in.
#
# Credit: Andrej Karpathy - https://github.com/karpathy/nanochat
#
# Author: Jason Cox
# Date: 2025-10-25
# https://github.com/jasonacox/dgx-spark

# Check if we're in the nanochat directory and navigate to it
if [ -d "nanochat" ]; then
    cd nanochat
fi

if [ ! -f "pyproject.toml" ] || [ ! -d ".venv" ]; then
    echo "Error: This script must be run from the dgx-spark/nanochat directory."
    echo "Make sure you've run prepare.sh first."
    exit 1
fi

# Setup CUDA environment variables for Grace Blackwell GB10
export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH}

# Ensure Rust/Cargo is available
export PATH="$HOME/.cargo/bin:$PATH"
source "$HOME/.cargo/env" 2>/dev/null || true

# Activate the Python virtual environment
echo "Activating Python virtual environment..."
source .venv/bin/activate

# Verify CUDA installation
echo "Verifying CUDA installation..."
nvcc --version
if [ $? -ne 0 ]; then
    echo "Error: CUDA not found. Please ensure CUDA 13.0 is properly installed."
    exit 1
fi

# Verify GPU availability
echo "Checking GPU availability..."
nvidia-smi
if [ $? -ne 0 ]; then
    echo "Error: nvidia-smi failed. Please check GPU drivers and installation."
    exit 1
fi

# Verify wandb is configured
echo "Checking wandb configuration..."
wandb status
if [ $? -ne 0 ]; then
    echo "Warning: wandb not configured. You may need to run 'wandb login' first."
    echo "Continuing anyway..."
fi

# Set optimized settings for DGX Spark GB10
export PYTORCH_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

# Download identity conversations file if it doesn't exist
if [ ! -f "$HOME/.cache/nanochat/identity_conversations.jsonl" ]; then
    echo "Downloading identity conversations dataset..."
    mkdir -p "$HOME/.cache/nanochat"
    curl -L -o "$HOME/.cache/nanochat/identity_conversations.jsonl" https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
fi

echo "Starting midtraining (fine-tuning) on DGX Spark Grace Blackwell GB10..."
echo "Configuration:"
echo "  - Fine-tuning pretrained model for conversational AI"
echo "  - Optimized for single GB10 GPU with 128GB unified memory"
echo "  - Using unified memory architecture"
echo ""

# Run midtraining with DGX Spark optimized settings
torchrun --standalone --nproc_per_node=1 -m scripts.mid_train -- --run="nanochat-midtrain" 

echo ""
echo "Midtraining complete!"
echo "Fine-tuned models can be found in: ~/.cache/nanochat/mid_checkpoints/"
echo ""
echo "Next step: Start a chat session with your trained model!"
echo "Run: ./chat.sh"