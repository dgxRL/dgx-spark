#!/bin/bash

# This script runs the Reinforcement Learning (RL) training for the NanoChat model on Nvidia DGX Spark.
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

# Set optimized settings for DGX Spark GB10
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

echo "Starting Reinforcement Learning (RL) training on DGX Spark Grace Blackwell GB10..."
echo "Configuration:"
echo "  - GRPO training loop with GSM8K math problem rewards"
echo "  - Simplified REINFORCE-like approach"
echo "  - Helps mitigate hallucinations and improve performance"
echo "  - Optimized for single GB10 GPU with 128GB unified memory"
echo ""

# Run RL training with DGX Spark optimized settings
torchrun --standalone --nproc_per_node=1 -m scripts.chat_rl

echo ""
echo "Running RL evaluation on GSM8K..."
torchrun --standalone --nproc_per_node=1 -m scripts.chat_eval -- -i rl -a GSM8K

echo ""
echo "Reinforcement Learning training complete!"
echo "RL models can be found in: ~/.cache/nanochat/rl_checkpoints/"
echo ""
echo "🎉 Congratulations! Your Nanochat model is fully trained!"
echo ""
echo "Your model now has:"
echo "  ✅ Base language understanding (pretraining)"
echo "  ✅ Conversational abilities (midtraining)"
echo "  ✅ Safety and quality improvements (SFT)"
echo "  ✅ Reduced hallucinations (RL)"
echo ""
echo "Start chatting with your fully trained model:"
echo "Run: ./chat.sh"