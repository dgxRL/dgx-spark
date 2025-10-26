# DGX Spark AI Development Hub

A collection of projects, tools, and resources specifically designed for the **Nvidia DGX Spark** AI supercomputer platform.

**Quick Start:**
```bash
git clone https://github.com/jasonacox/dgx-spark.git
cd dgx-spark
```

## About the Nvidia DGX Spark

The **[Nvidia DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)** was designed to be a personal experimentation laboratory for AI exploration, featuring the **Grace Blackwell architecture** with **128GB unified memory**. The design eliminates costly CPU-GPU transfers and enables larger model training, while the **ARM-based Grace CPU** delivers exceptional energy efficiency for extended development and training workloads.

- **GPU**: Grace Blackwell GB10 with 6,144 CUDA cores
- **Memory**: 128GB LPDDR5x Unified Memory Architecture with 273 GB/s of bandwidth
- **CPU**: 20-core ARM processor (10 Cortex-X925 performance cores and 10 Cortex-A725 efficiency cores)
- **OS**: Ubuntu 24.04 ARM64 optimized for AI workloads
- **Power**: Idle at 40–45W with 120–130W under GPU load (240W max)

## Projects to Explore

### 🤖 Nanochat - Train Your Own LLM from Scratch

**Train a 1.9 billion parameter ChatGPT-like model on DGX Spark**

[Nanochat](https://github.com/karpathy/nanochat) is a complete implementation for training Large Language Models from scratch, specifically optimized for the DGX Spark's unique architecture. Based on Andrej Karpathy's educational framework, it provides a full pipeline from tokenization to a functional ChatGPT-like web interface.

- A functional **1.9B parameter transformer model**
- **ChatGPT-like web interface** to interact with your model  
- Performance that **outperforms GPT-2** on benchmarks
- Complete understanding of **modern LLM training pipeline**
- A model that's **fully yours** - no API dependencies

**[→ Get started with Nanochat](./nanochat/README.md)**

### 🗣️ LLM - Run Large Language Models

**Deploy and chat with state-of-the-art open source models**

The large memory footprint of the DGX Spark means that you can run large LLMs that would normally require multiple GPUs. This project helps you deploy powerful open source language models with simple CLI and web-based chat interfaces.

- Support for popular models like Llama, Mistral,Qwen, and more
- Simple CLI interface for quick interactions
- Web-based chatbot for user-friendly conversations
- Optimized for DGX Spark's 128GB unified memory

**[→ Get started with LLM](./llm/README.md)**

### 🎨 ImageGen - Generate High-Quality Images

**Create stunning images and videos with AI**

This project helps you set up the DGX Spark to run high-fidelity open source models capable of generating and editing professional-quality images and videos. Get a complete ComfyUI setup running on your DGX Spark for creative AI workflows.

- High-resolution image generation
- Video creation and editing
- Style transfer and artistic effects
- ComfyUI workflow management
- GPU-accelerated processing

**[→ Get started with ImageGen](./imagegen/README.md)**

## Contributing

We welcome contributions that leverage the unique capabilities of the DGX Spark platform:

- **New Projects**: AI/ML projects optimized for Grace Blackwell architecture
- **Optimizations**: Improvements that take advantage of unified memory
- **Documentation**: Better guides and tutorials for DGX Spark development
- **Tools**: Utilities that simplify development on the platform

## Resources

- **[Nvidia DGX Spark Documentation](https://docs.nvidia.com/dgx/)**
- **[Grace Blackwell Architecture Guide](https://www.nvidia.com/en-us/data-center/grace-blackwell/)**
- **[ARM Development Resources](https://developer.arm.com/)**
- **[CUDA for ARM](https://developer.nvidia.com/cuda-gpus)**

## Future Projects

Coming soon to the DGX Spark development hub:

- 🔬 **Scientific Computing**: Optimized numerical computing workflows for research
- � **Computer Vision**: Advanced image and video processing pipelines
- 📊 **Data Science**: Large-scale analytics and visualization tools
- 🎮 **Reinforcement Learning**: Game AI and robotics training environments
- 🧬 **Bioinformatics**: Genomics and protein folding applications
- 🌐 **Edge AI**: Deployment tools for ARM-based edge devices

---

**Built for the future of AI computing. Optimized for Grace Blackwell. Designed for developers.**

