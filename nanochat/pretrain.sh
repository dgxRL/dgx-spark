#!/bin/bash

# This script runs the pretraining for the Nanochat model on Nvidia DGX Spark.
# It sets up all necessary environment variables and activates the virtual environment
# to ensure training can run even if the user has logged out and back in.
#
# Author: Jason Cox
# Date: 2025-10-25
# https://github.com/jasonacox/dgx-spark

# Check if we're in the nanochat directory
if [ ! -f "pyproject.toml" ] || [ ! -d ".venv" ]; then
    echo "Error: This script must be run from the nanochat directory."
    echo "Make sure you've run setup.sh first and are in the nanochat folder."
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
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

echo "Starting pretraining on DGX Spark Grace Blackwell GB10..."
echo "Configuration:"
echo "  - Model depth: 20 layers (~1.9B parameters)"
echo "  - Device batch size: 32 (optimized for 128GB unified memory)"
echo "  - Training optimized for single GB10 GPU"
echo "  - Using unified memory architecture"
echo ""

# Run pretraining with DGX Spark optimized settings
torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --depth=20 \
    --run="nanochat-dgx-spark" \
    --device_batch_size=32 \
    --sample_every=100

echo ""
echo "Pretraining complete!"
echo "Models can be found in: ~/.cache/nanochat/base_checkpoints/"
echo ""
echo "Next step: Run midtraining (fine-tuning) for conversational AI capabilities!"
echo "Run: ./midtrain.sh"
echo ""
echo "Or to chat with your current model:"
echo "  1. Activate the environment: source .venv/bin/activate"
echo "  2. Start the web interface: python -m scripts.chat_web"
echo "  3. Open the displayed URL in your browser"