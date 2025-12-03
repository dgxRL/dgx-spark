#!/bin/bash

# This script prepares the dataset and tokenizer for training the Nanochat model.
# It assumes that the environment has already been set up using setup.sh.
#
# Usage:
#   ./prepare.sh              # Full setup including dataset download
#   ./prepare.sh --setup-only # Only setup environment, skip dataset download
#
# Credit: Andrej Karpathy - https://github.com/karpathy/nanochat
#
# Author: Jason Cox
# Date: 2025-10-25
# https://github.com/jasonacox/dgx-spark

# Parse command line arguments
SETUP_ONLY=false
if [ "$1" == "--setup-only" ]; then
    SETUP_ONLY=true
fi

# Check to make sure CUDA 13 is installed
# look for Build cuda_13.0.r13.0/compiler.36424714_0
if ! nvcc --version | grep -q "cuda_13"; then
    echo "Error: CUDA 13.0 is not installed. Please run setup.sh first."
    exit 1
fi

# Pull the nanochat repository if it doesn't exist
echo "Cloning NanoChat repository..."
if [ -d "nanochat" ]; then
    echo "nanochat directory already exists. Skipping clone."
    cd nanochat
else
    git clone https://github.com/karpathy/nanochat.git
    cd nanochat
fi

# Update requirements and switch to CUDA 13.0 - pyproject.toml
# Credit: Alexander Falk - https://github.com/karpathy/nanochat/discussions/28#discussioncomment-14756733
echo "Update pyproject.toml for CUDA 13.0 compatibility..."
sed -i 's/"torch>=2.8.0"/"torch>=2.9.0"/g' pyproject.toml
sed -i '/tiktoken>=0.11.0/a\    "triton>=3.5.0",' pyproject.toml
sed -i 's|pytorch-cu128|pytorch-cu130|g' pyproject.toml
sed -i 's|cu128|cu130|g' pyproject.toml
sed -i 's|# target torch to cuda 12.8 or CPU|# target torch to cuda 13.0 or CPU|' pyproject.toml

# Setup CUDA environment variables BEFORE installing PyTorch
echo "Setting up CUDA 13.0 environment..."
export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH}

# install uv (if not already installed)
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"
    
    # Add to shell profile for persistence
    SHELL_RC="$HOME/.bashrc"
    if [ -n "$ZSH_VERSION" ]; then
        SHELL_RC="$HOME/.zshrc"
    fi
    
    if ! grep -q '.local/bin' "$SHELL_RC" 2>/dev/null; then
        echo "" >> "$SHELL_RC"
        echo '# Added by nanochat prepare.sh' >> "$SHELL_RC"
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_RC"
        echo "Added ~/.local/bin to PATH in $SHELL_RC"
        echo "Run 'source $SHELL_RC' or start a new terminal to use uv globally"
    fi
else
    # Ensure uv is in PATH even if already installed
    export PATH="$HOME/.local/bin:$PATH"
fi

# Check if existing .venv has CPU-only PyTorch
if [ -d ".venv" ]; then
    EXISTING_TORCH=$(.venv/bin/python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "none")
    if [[ "$EXISTING_TORCH" =~ \+cpu ]] || [[ "$EXISTING_TORCH" == "none" ]] || [[ ! "$EXISTING_TORCH" =~ cu130 ]]; then
        echo "Detected incompatible PyTorch installation (version: $EXISTING_TORCH)"
        echo "Removing old virtual environment to ensure CUDA 13.0 support..."
        rm -rf .venv
    fi
fi

# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies with GPU support (CUDA 13.0)
echo "Installing dependencies with CUDA 13.0 support..."
uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# Install wandb for experiment tracking
echo "Installing wandb for experiment tracking..."
uv pip install wandb

# Install Rust / Cargo
if ! command -v cargo &> /dev/null; then
    echo "Installing Rust/Cargo..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    
    # Add to shell profile for persistence
    SHELL_RC="$HOME/.bashrc"
    if [ -n "$ZSH_VERSION" ]; then
        SHELL_RC="$HOME/.zshrc"
    fi
    
    if ! grep -q '.cargo/env' "$SHELL_RC" 2>/dev/null; then
        echo "" >> "$SHELL_RC"
        echo '# Added by nanochat prepare.sh' >> "$SHELL_RC"
        echo 'source "$HOME/.cargo/env"' >> "$SHELL_RC"
        echo "Added Rust/Cargo to PATH in $SHELL_RC"
    fi
else
    source "$HOME/.cargo/env" 2>/dev/null || true
fi
# Build the rustbpe Tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# Exit here if --setup-only flag was provided
if [ "$SETUP_ONLY" = true ]; then
    echo ""
    echo "Environment setup complete!"
    echo ""
    exit 0
fi

# Download and prepare the dataset
python -m nanochat.dataset -n 240

# Train the tokenizer
python -m scripts.tok_train --max_chars=2000000000
python -m scripts.tok_eval

# Download evaluation bundle
curl -L -o eval_bundle.zip https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
unzip -q eval_bundle.zip
rm eval_bundle.zip
mv eval_bundle "$HOME/.cache/nanochat"

# Log in to wandb (virtual environment is already active)
echo "Logging in to wandb..."
wandb login

echo "Setup complete! Environment is ready for training."
echo ""
echo "To start training, run: ./pretrain.sh"
echo "Models will be saved to: ~/.cache/nanochat/base_checkpoints/"
