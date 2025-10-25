#!/bin/bash

# This script sets up the environment for training the NanoChat model on Nvidia DGX Spark.
# It installs the necessary dependencies, including CUDA Toolkit 13.0, sets up the Python virtual environment,
# builds the tokenizer, downloads the dataset, and prepares everything for training.
# Use pretrain.sh to start the actual training process.

git clone https://github.com/karpathy/nanochat.git
cd nanochat

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

# Setup script for installing CUDA Toolkit on Ubuntu 24.04 ARM64
# Download CUDA repository pin if not already present
if [ ! -f cuda-ubuntu2404.pin ]; then
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-ubuntu2404.pin
fi
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
if [ ! -f cuda-repo-ubuntu2404-13-0-local_13.0.2-580.95.05-1_arm64.deb ]; then
    wget https://developer.download.nvidia.com/compute/cuda/13.0.2/local_installers/cuda-repo-ubuntu2404-13-0-local_13.0.2-580.95.05-1_arm64.deb
fi
sudo dpkg -i cuda-repo-ubuntu2404-13-0-local_13.0.2-580.95.05-1_arm64.deb
sudo cp /var/cuda-repo-ubuntu2404-13-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-0

# Setup environment variables
# assuming CUDA 13.0 is installed at /usr/local/cuda-13.0
export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH}

# Add the above to ~/.bashrc for future sessions
echo 'export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas' >> ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda-13.0' >> ~/.bashrc
echo 'export PATH=/usr/local/cuda-13.0/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH}' >> ~/.bashrc
echo "CUDA Toolkit installation and environment setup complete."

# Verify installation
nvcc --version
nvidia-smi

# Log in to wandb
echo "Logging in to wandb..."
wandb login

echo "Setup complete! Environment is ready for training."
echo ""
echo "To start training, run: ./pretrain.sh"
echo "Models will be saved to: ~/.cache/nanochat/base_checkpoints/"
