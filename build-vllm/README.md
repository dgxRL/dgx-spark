# Build vLLM for NVIDIA DGX Spark

**Building vLLM from source for DGX Spark (Grace Blackwell GB10)**

> **Note:** This is an alternative to the Docker-based approach. Building from source is more complex but gives you more control. For most users, the Docker method in `../llm/README.md` is recommended.

**Under Development**

## Prerequisites

- NVIDIA DGX Spark with Grace Blackwell GB10
- CUDA 13.0 or higher
- Python 3.12
- `uv` package manager

## Installation Steps

### 1. Install uv Package Manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create Python Virtual Environment

```bash
uv venv .vllm --python 3.12
source .vllm/bin/activate
```

### 3. Install PyTorch with CUDA 13 Support

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

### 4. Install Required Dependencies

```bash
# Install flashinfer, xgrammar, and triton
uv pip install xgrammar triton flashinfer-python --prerelease=allow

# Install Python development headers (required for compilation)
sudo apt install python3-dev
```

### 5. Clone and Build vLLM

```bash
# Clone vLLM repository with submodules
git clone --recursive https://github.com/vllm-project/vllm.git
cd vllm

# Use existing PyTorch installation
python3 use_existing_torch.py

# Install build requirements
uv pip install -r requirements/build.txt

# Build and install vLLM
uv pip install --no-build-isolation -e .
```

### 6. Set Environment Variables

```bash
# CUDA architecture for Grace Blackwell GB10
export TORCH_CUDA_ARCH_LIST=12.1a  # Options: 12.0f, 12.1a for Spark

# CUDA paths
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**Tip:** Add these to your `~/.bashrc` for persistence:
```bash
cat >> ~/.bashrc << 'EOF'
export TORCH_CUDA_ARCH_LIST=12.1a
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
EOF
source ~/.bashrc
```

### 7. Clear System Memory (Recommended)

Before running large models, clear system caches:

```bash
sudo sysctl -w vm.drop_caches=3
```

## Running Models

### Example: GPT-OSS-120B

**Setup Tiktoken Encodings:**

```bash
# Create directory for encodings
mkdir -p tiktoken_encodings

# Download required encoding files
wget -O tiktoken_encodings/o200k_base.tiktoken \
  "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
wget -O tiktoken_encodings/cl100k_base.tiktoken \
  "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"

# Set encoding path
export TIKTOKEN_ENCODINGS_BASE=${PWD}/tiktoken_encodings
```

**Start vLLM Server:**

```bash
# Enable MXFP8 activation for MoE (faster, but slightly less accurate)
export VLLM_USE_FLASHINFER_MXFP4_MOE=1

# Run vLLM server
uv run vllm serve "openai/gpt-oss-120b" \
  --async-scheduling \
  --port 8000 \
  --host 0.0.0.0 \
  --trust-remote-code \
  --swap-space 16 \
  --max-model-len 32000 \
  --tensor-parallel-size 1 \
  --max-num-seqs 1024 \
  --gpu-memory-utilization 0.7
```

**Key Parameters:**
- `--async-scheduling` - Enable asynchronous request scheduling
- `--swap-space 16` - CPU swap space in GB for offloading
- `--max-model-len 32000` - Maximum context length
- `--max-num-seqs 1024` - Maximum concurrent sequences
- `--gpu-memory-utilization 0.7` - Use 70% of GPU memory (adjust as needed)

## Troubleshooting

### Triton Backend Errors

If you encounter Triton backend compilation errors:

```bash
# Remove existing triton kernels
rm -rf vllm/triton_kernels

# Install Triton from main branch
git clone https://github.com/triton-lang/triton.git
cd triton
pip install -e python
cd ..
```

### CUDA Out of Memory

Try these adjustments:
- Reduce `--gpu-memory-utilization` (e.g., from 0.7 to 0.6)
- Reduce `--max-num-seqs` (e.g., from 1024 to 512)
- Reduce `--max-model-len` (e.g., from 32000 to 16000)
- Clear memory caches: `sudo sysctl -w vm.drop_caches=3`

### Build Failures

**Missing Python headers:**
```bash
sudo apt install python3-dev build-essential
```

**CMake errors:**
```bash
uv pip install --upgrade cmake ninja
```

**Out of memory during build:**
```bash
export MAX_JOBS=4  # Limit parallel compilation
```

## Additional Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM GitHub Repository](https://github.com/vllm-project/vllm)
- [Grace Blackwell Architecture](https://www.nvidia.com/en-us/data-center/grace-blackwell/)

## Alternative: Docker-Based Installation

For a simpler, more reliable installation, see the Docker-based approach in `../llm/README.md` which uses NVIDIA's pre-built vLLM container.
