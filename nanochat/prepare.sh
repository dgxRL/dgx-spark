#!/bin/bash

# This script prepares the dataset and tokenizer for training the Nanochat model.
# It assumes that the environment has already been set up using setup.sh.
#
# Author: Jason Cox
# Date: 2025-10-25
# https://github.com/jasonacox/dgx-spark

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
echo "Update pyproject.toml for CUDA 13.0 compatibility..."
sed -i 's/"torch>=2.8.0"/"torch>=2.9.0"/g' pyproject.toml
sed -i '/tiktoken>=0.11.0/a\    "triton>=3.5.0",' pyproject.toml
sed -i 's|pytorch-cu128|pytorch-cu130|g' pyproject.toml
sed -i 's|cu128|cu130|g' pyproject.toml
sed -i 's|# target torch to cuda 12.8 or CPU|# target torch to cuda 13.0 or CPU|' pyproject.toml

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# Install Rust / Cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
# Build the rustbpe Tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

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

# Log in to wandb
echo "Logging in to wandb..."
wandb login

echo "Setup complete! Environment is ready for training."
echo ""
echo "To start training, run: ./pretrain.sh"
echo "Models will be saved to: ~/.cache/nanochat/base_checkpoints/"
