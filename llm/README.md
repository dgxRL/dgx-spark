# LLM - Run Large Language Models on DGX Spark

**Deploy and chat with state-of-the-art open source models**

This project helps you deploy powerful open source language models on the DGX Spark, taking advantage of its massive 128GB unified memory to run models that would normally require multiple GPUs.

## Overview

The DGX Spark's large memory footprint makes it perfect for running large language models that would typically require distributed setups. This project provides simple tools to deploy and interact with popular open source LLMs optimized for the Grace Blackwell architecture.

## Features

- ✅ Support for popular models (Llama, Mistral, CodeLlama, and more)
- ✅ Simple CLI interface for quick interactions
- ✅ Web-based chatbot for user-friendly conversations
- ✅ Optimized for DGX Spark's 128GB unified memory
- ✅ ARM64 native performance
- ✅ Single GPU deployment (no distributed complexity)

## Supported Models

### Large Models (70B+ parameters)
- **Llama 2 70B**: Meta's flagship conversational AI model
- **Code Llama 70B**: Specialized for code generation and programming
- **Mixtral 8x7B**: Mixture of experts model with excellent performance

### Medium Models (7B-30B parameters)
- **Llama 2 13B**: Balanced performance and efficiency
- **Mistral 7B**: High-quality general purpose model
- **CodeLlama 13B**: Code-focused model for development tasks

## Quick Start

```bash
cd llm
./setup.sh      # Install dependencies and configure environment
./deploy.sh     # Choose and deploy a model
./chat.sh       # Start chatting with your model
```

## System Requirements

- **Hardware**: Nvidia DGX Spark with Grace Blackwell GB10 GPU
- **Memory**: 128GB unified memory (enables large model deployment)
- **OS**: Ubuntu 24.04 ARM64
- **Storage**: 50-200GB depending on model size
- **Network**: High-speed connection for model downloads

## What You'll Get

After setup, you'll have:
- 🤖 **Production-ready LLM deployment** optimized for DGX Spark
- 💬 **Multiple interaction methods** (CLI, web interface, API)
- ⚡ **High-performance inference** using unified memory architecture
- 🔧 **Easy model switching** between different LLMs
- 📊 **Performance monitoring** and optimization tools

## Usage Examples

### CLI Chat
```bash
./chat.sh --model llama2-70b
> What is quantum computing?
> Can you write a Python function to sort a list?
```

### Web Interface
```bash
./web.sh --model mistral-7b --port 8080
# Open http://your-dgx-spark-ip:8080
```

### API Server
```bash
./api.sh --model codellama-13b --port 8000
# OpenAI-compatible API endpoint
```

## Model Management

```bash
./models.sh list              # Show available models
./models.sh download llama2   # Download a specific model
./models.sh remove mistral    # Remove a model to free space
./models.sh optimize llama2   # Optimize model for DGX Spark
```

## Performance

The DGX Spark's unified memory architecture provides significant advantages:

- **No memory transfers**: Direct GPU access to full 128GB
- **Larger context windows**: Support for longer conversations
- **Faster inference**: Reduced latency from unified architecture
- **Better throughput**: Single GPU eliminates distributed overhead

## Getting Started

1. **Prerequisites**: Ensure your DGX Spark is set up with CUDA 13.0
2. **Setup**: Run `./setup.sh` to configure the environment
3. **Choose Model**: Select based on your use case and memory requirements
4. **Deploy**: Use `./deploy.sh` to download and configure your chosen model
5. **Chat**: Start interacting with `./chat.sh` or `./web.sh`

## Coming Soon

- 🔄 **Model fine-tuning** tools for custom domains
- 🎯 **RAG integration** for knowledge-augmented responses
- 🔌 **API integrations** with popular frameworks
- 📈 **Benchmarking suite** for performance evaluation
- 🛠️ **Custom model support** for proprietary models

---

**Note**: This project is currently under development. Check back soon for the complete implementation optimized for DGX Spark's Grace Blackwell architecture.

## Status: 🚧 Coming Soon

This project is in active development. The implementation will focus on:
- Native ARM64 optimization
- Grace Blackwell memory architecture utilization
- Simple deployment and management tools
- Production-ready inference servers

**Estimated availability**: Q4 2025

**Follow development**: Watch this repository for updates and early access releases.