# ImageGen - Generate High-Quality Images and Videos

**Create stunning images and videos with AI on DGX Spark**

This project helps you set up the DGX Spark to run high-fidelity open source models capable of generating professional-quality images and videos. Get a complete ComfyUI setup running on your DGX Spark for creative AI workflows.

## Overview

Transform your DGX Spark into a powerful creative AI workstation. The Grace Blackwell architecture's unified memory and high-performance compute make it ideal for image and video generation tasks that traditionally require expensive cloud services or complex multi-GPU setups.

## Features

- ✅ **High-resolution image generation** (up to 4K and beyond)
- ✅ **Video creation and editing** with AI-powered workflows
- ✅ **Style transfer and artistic effects** for creative projects
- ✅ **ComfyUI workflow management** with visual node editor
- ✅ **GPU-accelerated processing** optimized for Grace Blackwell
- ✅ **Real-time preview and iteration** with unified memory advantages

## Supported Models

### Image Generation
- **Stable Diffusion XL**: High-quality image generation with fine control
- **SDXL Turbo**: Fast iteration for real-time creative workflows
- **ControlNet**: Precise control over image composition and style
- **IP-Adapter**: Image prompt conditioning for style consistency

### Video Generation
- **Stable Video Diffusion**: Transform images into smooth video sequences
- **AnimateDiff**: Create animated sequences from static images
- **Video ControlNet**: Precise control over video generation

### Specialized Models
- **Real-ESRGAN**: AI-powered image upscaling and enhancement
- **CodeFormer**: Face restoration and enhancement
- **GFPGAN**: Portrait enhancement and restoration
- **Background Removal**: Automatic subject isolation

## Quick Start

```bash
cd imagegen
./setup.sh      # Install ComfyUI and dependencies
./models.sh     # Download base models and checkpoints
./start.sh      # Launch ComfyUI interface
```

Access the interface at: `http://your-dgx-spark-ip:8188`

## System Requirements

- **Hardware**: Nvidia DGX Spark with Grace Blackwell GB10 GPU
- **Memory**: 128GB unified memory (enables large model combinations)
- **OS**: Ubuntu 24.04 ARM64
- **Storage**: 100-500GB for models and generated content
- **Network**: High-speed connection for model downloads

## What You'll Get

After setup, you'll have:
- 🎨 **Professional image generation studio** with AI models
- 🎬 **Video creation capabilities** for motion graphics and animation
- 🖼️ **Image enhancement tools** for photo restoration and upscaling
- 🎯 **Precise control systems** for consistent artistic output
- 🔄 **Workflow automation** with ComfyUI's node-based system

## Use Cases

### Creative Projects
- **Concept art and illustration** for games and media
- **Marketing materials** and social media content
- **Product visualization** and mockups
- **Artistic exploration** and style development

### Professional Applications
- **Photo restoration** and enhancement services
- **Content creation** for digital marketing
- **Video production** with AI-assisted workflows
- **Prototyping** visual concepts and designs

### Educational and Research
- **AI art education** and experimentation
- **Computer vision research** with generative models
- **Creative coding** and algorithmic art
- **Style transfer studies** and artistic analysis

## DGX Spark Advantages

### Unified Memory Benefits
- **Load multiple large models** simultaneously without swapping
- **Real-time preview** of high-resolution generations
- **Batch processing** of multiple images efficiently
- **Complex workflows** with multiple AI models in sequence

### Performance Advantages
- **Faster generation times** with optimized memory access
- **Higher resolution outputs** without memory constraints
- **Smooth real-time interaction** in ComfyUI interface
- **Efficient model switching** between different generators

## Sample Workflows

### Image Generation Pipeline
```
Text Prompt → CLIP Encoder → SDXL Base → Refiner → Upscaler → Final Image
```

### Video Creation Pipeline
```
Input Image → Stable Video Diffusion → Temporal Smoothing → Enhancement → Final Video
```

### Style Transfer Workflow
```
Content Image + Style Reference → ControlNet → Style Transfer → Output Image
```

## Getting Started

1. **Prerequisites**: Ensure your DGX Spark has CUDA 13.0 and sufficient storage
2. **Setup**: Run `./setup.sh` to install ComfyUI and core dependencies
3. **Models**: Use `./models.sh` to download your preferred AI models
4. **Launch**: Start the interface with `./start.sh`
5. **Create**: Begin generating images and videos through the web interface

## Model Management

```bash
./models.sh list                    # Show installed models
./models.sh install sdxl-base       # Install specific model
./models.sh install video-models    # Install video generation suite
./models.sh update                  # Update to latest model versions
./models.sh cleanup                 # Remove unused models to free space
```

## Performance Optimization

### For Image Generation
- **Batch sizes**: Optimize for 128GB memory capacity
- **Model precision**: Use FP16 for faster generation
- **Caching**: Keep frequently used models in memory

### For Video Generation
- **Frame optimization**: Process multiple frames simultaneously
- **Memory management**: Leverage unified memory for smooth playback
- **Temporal consistency**: Use DGX Spark's memory for better frame coherence

## Coming Soon

- 🎪 **Custom model training** tools for personalized styles
- 🔌 **API integration** for programmatic image generation
- 📱 **Mobile-optimized interface** for remote creative work
- 🤖 **Automated workflow templates** for common use cases
- 🎨 **Advanced style controls** and artistic filters

---

**Note**: This project is currently under development. The implementation will provide a complete creative AI suite optimized for the DGX Spark's unique capabilities.

## Status: 🚧 Coming Soon

This project is in active development. The implementation will focus on:
- Native ARM64 ComfyUI optimization
- Grace Blackwell memory utilization for large models
- Professional workflow templates
- High-performance generation pipelines

**Estimated availability**: Q1 2026

**Follow development**: Watch this repository for updates and early access releases.

## Preview

While the full implementation is in development, you can prepare by:
1. Exploring ComfyUI documentation and workflows
2. Familiarizing yourself with Stable Diffusion models
3. Planning your creative projects and use cases
4. Setting up storage for large model files and generated content