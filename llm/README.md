# LLM - Run Large Language Models on DGX Spark

**Deploy and chat with state-of-the-art open source models using vLLM**

This project helps you deploy powerful open source language models on the DGX Spark, taking advantage of its massive 128GB unified memory and Grace Blackwell GB10 GPU to run large models efficiently with vLLM.

## Overview

The DGX Spark's large memory footprint and powerful GB10 GPU makes it perfect for running large language models. This guide shows you how to use vLLM (a high-performance LLM inference engine) to deploy and interact with popular open source models optimized for the Grace Blackwell architecture.

### Why Docker?

**Important:** The DGX Spark with Grace Blackwell GB10 uses **CUDA 13.0**, which creates compatibility challenges:
- Standard PyPI packages are compiled for CUDA 12 and will fail with library errors
- Building vLLM from source is complex and fragile on this platform
- **NVIDIA provides a pre-built vLLM container** specifically for Grace Blackwell systems

**Solution:** Use NVIDIA's official vLLM container (`nvcr.io/nvidia/vllm:25.09-py3`) which includes:
- Pre-configured CUDA 13.0 environment
- Optimized PyTorch build for Grace Blackwell
- All required system dependencies
- Proven compatibility with DGX Spark

### Quick Helper Scripts

This directory includes several helper scripts to simplify the process:
- **`build.sh`** - Interactive script to build the Docker image
- **`run.sh`** - Run vLLM server with a model (supports custom models/ports)
- **`test_api.py`** - Test the vLLM API is working correctly

## Prerequisites

Before starting, ensure you have:

1. **Docker installed** on your DGX Spark system
2. **NVIDIA Container Toolkit** configured (for GPU access)
3. **Internet connection** to pull the NVIDIA container and download models
4. **Sufficient disk space** for the container image (~10GB) and model weights

### Verify Docker and GPU Access

```bash
# Check Docker is installed
docker --version

# Verify NVIDIA runtime is available
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

If `nvidia-smi` displays your GPU information, you're ready to proceed.

## Quick Start

Get vLLM running in under 5 minutes:

### Option 1: Using Helper Scripts (Easiest)

```bash
# Build the Docker image
./build.sh

# Run vLLM server with default model
./run.sh

# Or specify a different model, max length, and port
./run.sh nvidia/Llama-3.1-8B-Instruct-FP8 8192 8888

# In another terminal, test the API
./test_api.py
```

### Option 2: Manual Setup

Create the Dockerfile

```bash
FROM nvcr.io/nvidia/vllm:25.09-py3

WORKDIR /workspace

RUN git clone https://github.com/vllm-project/vllm.git
RUN cd vllm && \ 
    python use_existing_torch.py && \
    pip install -r requirements/build.txt && \
    pip install --no-build-isolation -e .

EXPOSE 8000

CMD ["/bin/bash"]
```

Build the custom vLLM image (this takes 5-10 minutes)

```bash
docker build -t vllm-custom:25.09 .
```

Run vLLM server with a model

```bash
docker run -it --gpus all -p 8888:8000 \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/vllm:/root/.cache/vllm \
  --rm vllm-custom:25.09 \
  vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 --max-model-len 32768
```

The server will start and automatically download the model on first run. Once you see "Application startup complete", vLLM is ready to serve requests at `http://localhost:8888`.

**Example models:**
```bash
# Qwen 30B vision-language model (FP8 quantized)
docker run -it --gpus all -p 8888:8000 \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/vllm:/root/.cache/vllm \
  --rm vllm-custom:25.09 \
  vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 --max-model-len 32768

# Llama 3.1 8B (FP8 quantized)
docker run -it --gpus all -p 8888:8000 \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/vllm:/root/.cache/vllm \
  --rm vllm-custom:25.09 \
  vllm serve nvidia/Llama-3.1-8B-Instruct-FP8 --max-model-len 8192
```

**Docker run options explained:**
- `--gpus all` - Gives container access to all GPUs
- `-p 8888:8000` - Maps host port 8888 to container port 8000
- `--ulimit memlock=-1` - Removes memory lock limit (required for GPU operations)
- `--ulimit stack=67108864` - Sets stack size to 64MB (prevents stack overflow)
- `-v ~/.cache/huggingface:/root/.cache/huggingface` - Persists HuggingFace model cache (models stay after container stops)
- `-v ~/.cache/vllm:/root/.cache/vllm` - Persists vLLM cache
- `--rm` - Automatically removes container when stopped
- `vllm-custom:25.09` - The image we built
- `vllm serve` - Command to start the OpenAI-compatible API server
- `--max-model-len` - Maximum sequence length (context window)

## Using the vLLM Server

### OpenAI-Compatible API

Once the server is running, you can interact with it using OpenAI's Python client or curl:

**Python example:**
```python
from openai import OpenAI

# Point to your local vLLM server
client = OpenAI(
    base_url="http://localhost:8888/v1",
    api_key="dummy"  # vLLM doesn't require authentication
)

# Chat completion
response = client.chat.completions.create(
    model="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
    messages=[
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    max_tokens=512,
    temperature=0.7
)

print(response.choices[0].message.content)
```

**cURL example:**
```bash
curl http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 100
  }'
```

### Interactive Shell

For debugging or experimentation, you can start a bash shell inside the container:

```bash
docker run -it --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --rm vllm-custom:25.09 /bin/bash
```

Then run vLLM commands manually:
```bash
# Inside container
vllm serve nvidia/Llama-3.1-8B-Instruct-FP8 --max-model-len 8192
```

## Recommended Models for DGX Spark

The DGX Spark's 128GB unified memory allows running very large models. Here are recommended models:

### Large Models (20B+ parameters)

```bash
# Qwen 30B Vision-Language (FP8) - ~15GB
docker run -it --gpus all -p 8888:8000 \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/vllm:/root/.cache/vllm \
  --rm vllm-custom:25.09 \
  vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 --max-model-len 32768

# Llama 3.1 70B (requires quantization)
docker run -it --gpus all -p 8888:8000 \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/vllm:/root/.cache/vllm \
  --rm vllm-custom:25.09 \
  vllm serve meta-llama/Llama-3.1-70B-Instruct --quantization awq --max-model-len 8192
```

### Medium Models (7B-20B parameters)

```bash
# Llama 3.1 8B Instruct (FP8) - ~4GB
docker run -it --gpus all -p 8888:8000 \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/vllm:/root/.cache/vllm \
  --rm vllm-custom:25.09 \
  vllm serve nvidia/Llama-3.1-8B-Instruct-FP8 --max-model-len 8192

# Mistral 7B Instruct
docker run -it --gpus all -p 8888:8000 \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/vllm:/root/.cache/vllm \
  --rm vllm-custom:25.09 \
  vllm serve mistralai/Mistral-7B-Instruct-v0.3 --max-model-len 8192
```

## Advanced Configuration

### Custom vLLM Server Options

```bash
docker run -it --gpus all -p 8888:8000 \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/vllm:/root/.cache/vllm \
  --rm vllm-custom:25.09 \
  vllm serve <MODEL_NAME> \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --trust-remote-code
```

**Key options:**
- `--max-model-len`: Maximum context length (affects memory usage)
- `--gpu-memory-utilization`: Fraction of GPU memory to use (0.0-1.0)
- `--tensor-parallel-size`: Number of GPUs to use for tensor parallelism
- `--dtype`: Data type for model weights (auto, float16, bfloat16, float32)
- `--trust-remote-code`: Allow running custom model code from HuggingFace

### Memory Optimization

**Clear system caches before running large models:**

```bash
# Clear page cache, dentries, and inodes
sudo sysctl -w vm.drop_caches=3
```

This is **automatically done** by the `server.sh` and `run.sh` helper scripts. Manual clearing is recommended when:
- Switching between different models
- Running very large models (70B+)
- Experiencing out-of-memory errors
- System has been running for a long time

### Persistent Container

To keep the container running across sessions:

```bash
# Start container in background
docker run -d --name vllm-server --gpus all -p 8888:8000 \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/vllm:/root/.cache/vllm \
  vllm-custom:25.09 \
  vllm serve nvidia/Llama-3.1-8B-Instruct-FP8 --max-model-len 8192

# Check logs
docker logs -f vllm-server

# Stop container
docker stop vllm-server

# Start again
docker start vllm-server

# Remove container
docker rm vllm-server
```

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Setup Instructions](#detailed-setup-instructions)
- [Using the vLLM Server](#using-the-vllm-server)
- [Recommended Models](#recommended-models-for-dgx-spark)
- [Advanced Configuration](#advanced-configuration)
- [Troubleshooting](#troubleshooting)

## Troubleshooting

### Docker Issues

**Problem: "docker: command not found"**
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
# Log out and back in for group changes to take effect
```

**Problem: "could not select device driver"**

NVIDIA Container Toolkit is not installed:
```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

**Problem: "permission denied" when running docker**
```bash
# Add your user to docker group
sudo usermod -aG docker $USER
# Log out and back in
```

### Build Issues

**Problem: "Failed to download base image"**

Check your internet connection and try again. The base image is ~10GB.

**Problem: Build fails during vLLM compilation**

The build may take 10+ minutes. If it fails:
```bash
# Clean up and rebuild
docker system prune -a
docker build --no-cache -t vllm-custom:25.09 .
```

### Runtime Issues

**Problem: "CUDA out of memory"**

Reduce `--gpu-memory-utilization` or `--max-model-len`:
```bash
docker run -it --gpus all -p 8888:8000 \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/vllm:/root/.cache/vllm \
  --rm vllm-custom:25.09 \
  vllm serve nvidia/Llama-3.1-8B-Instruct-FP8 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.7
```

**Problem: "Model download fails"**

Some models require HuggingFace authentication:
```bash
# First, get a token from https://huggingface.co/settings/tokens
# Then pass it to the container
docker run -it --gpus all -p 8888:8000 \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e HUGGING_FACE_HUB_TOKEN=your_token_here \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/vllm:/root/.cache/vllm \
  --rm vllm-custom:25.09 \
  vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 8192
```

**Problem: Server starts but model doesn't load**

Check container logs for specific errors:
```bash
# If running in background with -d
docker logs -f vllm-server
```

**Problem: "Application startup complete" but can't connect**

Verify the port mapping is correct:
```bash
# Check if container is running
docker ps

# Test connection
curl http://localhost:8888/v1/models
```

### Performance Issues

**Problem: Inference is slow**

Try these optimizations:
- Use FP8 quantized models (e.g., models with "FP8" in the name)
- Increase `--gpu-memory-utilization` to 0.9 if you have memory available
- Reduce `--max-model-len` if you don't need long context
- **Clear system memory caches before starting the server** (done automatically by `server.sh` and `run.sh`)

**Clearing memory caches manually:**

For optimal performance, especially when switching between models, clear the system memory caches:

```bash
sudo sysctl -w vm.drop_caches=3
```

This frees up page cache, dentries, and inodes. The `server.sh` and `run.sh` scripts automatically attempt to do this when starting new containers.

**Why clear caches?**
- Frees RAM from previous model loads
- Provides maximum available memory for new model
- Can prevent out-of-memory errors with large models
- Improves initial model loading time

**Problem: First request is very slow**

This is normal - vLLM needs to:
1. Download the model (first time only)
2. Load model into GPU memory
3. Compile CUDA kernels (first time only)

Subsequent requests will be much faster.

## Additional Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [NVIDIA vLLM Container Documentation](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm)
- [HuggingFace Models](https://huggingface.co/models)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference) (vLLM is compatible)

## Contributing

Found an issue or want to improve this guide? Please submit a pull request or open an issue on GitHub.

## License

This project is provided as-is for running LLMs on NVIDIA DGX Spark systems.

