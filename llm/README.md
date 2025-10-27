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

| Model | Parameters | Specialization | Description |
|-------|------------|----------------|-------------|
| **Llama 2 70B** | 70B+ | Conversational AI | Meta's flagship conversational AI model |
| **Code Llama 70B** | 70B+ | Code Generation | Specialized for code generation and programming |
| **Mixtral 8x7B** | 70B+ | General Purpose | Mixture of experts model with excellent performance |
| **Llama 2 Vision 13B** | 13B | Balanced | Balanced performance and efficiency |
| **Mistral 7B** | 7B | General Purpose | High-quality general purpose model |
| **CodeLlama 13B** | 13B | Development | Code-focused model for development tasks |

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

## Coming Soon

- 🔄 **Model fine-tuning** tools for custom domains
- 🎯 **RAG integration** for knowledge-augmented responses
- 🔌 **API integrations** with popular frameworks
- 📈 **Benchmarking suite** for performance evaluation
- 🛠️ **Custom model support** for proprietary models

---

**Note**: This project is currently under development. Check back soon for the complete implementation optimized for DGX Spark's Grace Blackwell architecture.
