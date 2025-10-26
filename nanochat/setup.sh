#!/bin/bash

# This script sets up the Spark DGX environment for training the NanoChat model.
# It installs the necessary dependencies, including CUDA Toolkit 13.0, sets up the Python virtual environment,
# builds the tokenizer, downloads the dataset, and prepares everything for training.
# Use pretrain.sh to start the actual training process.
#
# Author: Jason Cox
# Date: 2025-10-25
# https://github.com/jasonacox/dgx-spark

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

# Notify user that setup is complete and ready to run prepare.sh
echo "Setup complete! Environment is ready for preparing the dataset and tokenizer."
echo ""
echo "To prepare the dataset and tokenizer, run: ./prepare.sh"
