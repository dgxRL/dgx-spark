#!/bin/bash
#
# ImageGen Setup Script for Nvidia DGX Spark
# This script will install ComfyUI and dependencies optimized for Grace Blackwell GB10
#
# Author: Jason Cox
# Date: 2025-10-25
# https://github.com/jasonacox/dgx-spark

set -e  # Exit on any error

echo "🎨 Setting up ImageGen (ComfyUI) for DGX Spark Grace Blackwell GB10..."
echo ""

# Check if we're on DGX Spark with proper CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ Error: nvidia-smi not found. Please ensure GPU drivers are installed."
    exit 1
fi

if ! nvidia-smi | grep -q "GB10"; then
    echo "⚠️  Warning: This setup is optimized for Grace Blackwell GB10 GPU."
    echo "   Continuing anyway, but performance may vary."
fi

# Verify CUDA 13.0
echo "🔍 Checking CUDA installation..."
if ! nvcc --version | grep -q "V13.0"; then
    echo "❌ Error: CUDA 13.0 not found. Please install CUDA 13.0 for DGX Spark."
    exit 1
fi

# Setup environment variables for Grace Blackwell
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH}

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p models/{checkpoints,vae,loras,controlnet,upscale_models,embeddings,clip_vision,clip,unet,diffusion_models,gligen,style_models,photomaker}
mkdir -p input
mkdir -p output
mkdir -p temp

# Check Python version
echo "🐍 Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python3 not found. Please install Python 3.8 or later."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if ! python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
    echo "❌ Error: Python 3.8 or later required. Found Python $PYTHON_VERSION"
    exit 1
fi

# Create and activate Python virtual environment
echo "🔧 Creating Python virtual environment..."
if [ -d "comfyui-env" ]; then
    echo "   Virtual environment already exists. Skipping creation."
else
    python3 -m venv comfyui-env
fi

echo "🔌 Activating virtual environment..."
source comfyui-env/bin/activate

# Upgrade pip and install base packages
echo "📦 Upgrading pip and installing base packages..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 13.0 support optimized for ARM64
echo "🔥 Installing PyTorch with CUDA 13.0 support for ARM64..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Verify PyTorch CUDA installation
echo "✅ Verifying PyTorch CUDA installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Clone ComfyUI repository
echo "📥 Cloning ComfyUI repository..."
if [ -d "ComfyUI" ]; then
    echo "   ComfyUI directory already exists. Updating..."
    cd ComfyUI/
    git pull origin master
else
    git clone https://github.com/comfyanonymous/ComfyUI.git
    cd ComfyUI/
fi

# Install ComfyUI requirements
echo "📋 Installing ComfyUI requirements..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    # Install essential packages if requirements.txt is missing
    echo "   Installing essential packages..."
    pip install pillow opencv-python requests tqdm psutil
fi

# Install additional packages for enhanced functionality
echo "🚀 Installing additional packages for DGX Spark optimization..."
pip install accelerate xformers safetensors transformers diffusers

# Download and install ComfyUI Manager (for easy extension management)
echo "🔧 Installing ComfyUI Manager..."
cd custom_nodes/
if [ -d "ComfyUI-Manager" ]; then
    echo "   ComfyUI Manager already exists. Updating..."
    cd ComfyUI-Manager/
    git pull origin main
    cd ..
else
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git
fi
cd ..

# Create symbolic links for easier model management
echo "🔗 Creating model directory links..."
rm -rf models
ln -sf ../models models

# Create configuration for DGX Spark optimization
echo "⚙️  Creating DGX Spark optimization config..."
cat > extra_model_paths.yaml << EOF
# DGX Spark Grace Blackwell optimization paths
base_path: ../models/

checkpoints: checkpoints/
vae: vae/
loras: loras/
controlnet: controlnet/
upscale_models: upscale_models/
embeddings: embeddings/
clip_vision: clip_vision/
clip: clip/
unet: unet/
diffusion_models: diffusion_models/
gligen: gligen/
style_models: style_models/
photomaker: photomaker/
EOF

# Set up memory optimization for 128GB unified memory
echo "🧠 Configuring memory optimization for 128GB unified memory..."
cat > config_dgx_spark.yaml << EOF
# DGX Spark Grace Blackwell GB10 optimizations
gpu_dtype: float16  # Use FP16 for memory efficiency
force_fp16: true
cpu_mode: false
attention_implementation: xformers  # Use xformers for efficiency
vram_management: auto  # Let ComfyUI manage VRAM automatically
preview_format: jpeg  # Efficient preview format
preview_size: 512  # Reasonable preview size
max_vram_usage: 0.95  # Use up to 95% of available memory
enable_args_parsing: true
EOF

cd ..

# Create launcher script with DGX Spark optimizations
echo "🚀 Creating optimized launcher script..."
cat > launch_comfyui.sh << 'EOF'
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
EOF

chmod +x launch_comfyui.sh

# Print setup completion message
echo ""
echo "🎉 ImageGen setup complete!"
echo ""
echo "Next steps:"
echo "1. Download models with: ./models.sh"
echo "2. Start ComfyUI with: ./start.sh"
echo ""
echo "DGX Spark optimizations applied:"
echo "✅ CUDA 13.0 support for Grace Blackwell GB10"
echo "✅ ARM64 native PyTorch installation"
echo "✅ Memory optimization for 128GB unified memory"
echo "✅ FP16 precision for efficient inference"
echo "✅ ComfyUI Manager for easy extension management"
echo ""
echo "ComfyUI will be available at: http://your-dgx-spark-ip:8188"

