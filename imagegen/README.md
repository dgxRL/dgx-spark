# ImageGen - Generate High-Quality Images and Videos

**Create stunning images and videos with AI on DGX Spark**

This project provides a complete ComfyUI setup for the DGX Spark, optimized to run high-fidelity open source models capable of generating professional-quality images and videos. Transform your DGX Spark into a powerful creative AI workstation.

## Overview

The Grace Blackwell architecture's unified memory and high-performance compute make the DGX Spark ideal for image and video generation tasks that traditionally require expensive cloud services or complex multi-GPU setups. Our implementation provides a complete creative AI suite optimized for the platform's unique capabilities.

## Supported Models

### Base Models (`./models.sh install base` - 4GB)
Essential starter models for all workflows:
- **Stable Diffusion v1.5** - General image generation, LoRA/ControlNet compatible
- **SD VAE** - Enhanced image quality and color accuracy  
- **4x ESRGAN** - Basic upscaling for detail enhancement

### SDXL Models (`./models.sh install sdxl` - 13.5GB)
High-resolution professional generation:
- **SDXL Base & Refiner** - 1024x1024 native resolution with superior detail
- **SDXL VAE** - Optimized for high-quality FP16 processing

### Flux Models (`./models.sh install flux` - 29GB)
Cutting-edge AI generation:
- **Flux Schnell** - Ultra-fast generation with advanced text understanding
- **Text Encoders** - Enhanced prompt interpretation and adherence

### ControlNet Models (`./models.sh install controlnet` - 4.2GB)
Guided generation tools:
- **Canny, Depth, OpenPose** - Edge, 3D structure, and pose control

### Video Models (`./models.sh install video` - 10GB)
Motion and animation:
- **Stable Video Diffusion XT** - Image-to-video with smooth motion synthesis

### Upscale Models (`./models.sh install upscale` - 2.1GB)
Image enhancement:
- **ESRGAN (2x/4x)** - Standard upscaling for general use
- **LDSR** - Highest quality latent diffusion super-resolution

### Model Installation Commands

```bash
# Install specific model sets
./models.sh install base       # Essential starter models (~4GB)
./models.sh install sdxl       # High-quality SDXL suite (~13.5GB)
./models.sh install flux       # Cutting-edge Flux models (~29GB)
./models.sh install controlnet # Guided generation models (~4.2GB)
./models.sh install video      # Video generation (~10GB)
./models.sh install upscale    # Additional upscaling models (~2.1GB)

# Install everything
./models.sh install all        # Complete model collection (~62GB+)

# Management commands
./models.sh list              # Show installed models by category
./models.sh usage             # Display storage usage
./models.sh remove            # Interactive removal tool
```

### Model Recommendations by Use Case

**Getting Started**: `base` - Essential models for learning
**High Quality Images**: `base` + `sdxl` - Professional image generation  
**Cutting-Edge Generation**: `base` + `flux` - Latest AI capabilities
**Guided Creation**: `base` + `controlnet` - Precise creative control
**Video Content**: `base` + `video` - Motion and animation
**Complete Studio**: `all` - Everything for professional workflows

### Storage Requirements

- **Minimal Setup**: 4GB (base models only)
- **Professional Setup**: 18GB (base + sdxl)
- **Cutting-Edge Setup**: 33GB (base + flux)
- **Complete Setup**: 62GB+ (all models)

*Note: The DGX Spark's 128GB unified memory can easily handle multiple large models simultaneously, enabling complex workflows that would require model swapping on traditional systems.*

## Quick Start

```bash
cd imagegen
./setup.sh                 # Install ComfyUI and dependencies
./models.sh install base   # Download essential starter models
./start.sh                 # Launch ComfyUI interface
```

**Access the interface at:** `http://your-dgx-spark-ip:8188`

### Alternative Setup Options

```bash
# Install comprehensive model suite
./models.sh install all     # Download all model sets (requires significant storage)

# Install specific model categories
./models.sh install sdxl       # SDXL models for high-quality generation
./models.sh install controlnet # ControlNet for guided generation
./models.sh install video      # Video generation models

# Production deployment
./start.sh --daemon --log comfyui.log  # Run as background service

# Management commands
./models.sh list           # List all installed models
./models.sh usage          # Check storage usage
./start.sh --status        # Check service status
```

## System Requirements

- **Hardware**: Nvidia DGX Spark with Grace Blackwell GB10 GPU
- **Memory**: 128GB unified memory (fully utilized for optimal performance)
- **OS**: Ubuntu 24.04 ARM64
- **Storage**: 100-500GB for models and generated content
- **Network**: High-speed connection for model downloads

## What You'll Get

After setup, you'll have:
- 🎨 **Professional image generation studio** with ComfyUI visual interface
- 🎬 **Video creation capabilities** for motion graphics and animation
- 🖼️ **Image enhancement tools** for photo restoration and upscaling
- 🎯 **Precise control systems** with ControlNet for consistent artistic output
- 🔄 **Workflow automation** with ComfyUI's node-based visual system
- 📁 **Organized model management** with easy installation and updates
- ⚙️ **Production-ready deployment** with daemon mode and logging

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

## Model Management

The `models.sh` script provides comprehensive model management:

```bash
# Installation commands
./models.sh install base       # Essential starter models (SD 1.5, VAE, upscaler)
./models.sh install sdxl       # Stable Diffusion XL models
./models.sh install controlnet # ControlNet guidance models
./models.sh install video      # Video generation models
./models.sh install upscale    # Additional upscaling models
./models.sh install all        # Everything (requires 200+ GB storage)

# Management commands
./models.sh list              # Show installed models by category
./models.sh usage             # Display storage usage
./models.sh remove            # Interactive removal tool
./models.sh update            # Update existing models
```

## Advanced Usage

### ComfyUI Daemon Management

```bash
# Start as daemon with logging
./start.sh --daemon --log comfyui.log

# Check daemon status
./start.sh --status

# Stop daemon
./start.sh --stop

# Restart daemon
./start.sh --restart

# Custom port and host
./start.sh --port 8080 --host 192.168.1.100
```

### Performance Optimization

The setup automatically optimizes for DGX Spark:

- **FP16 Precision**: Memory-efficient inference by default
- **Unified Memory**: Leverages 128GB for large model combinations
- **CUDA 13.0**: Native support for Grace Blackwell architecture
- **ARM64 Optimization**: Native ARM64 PyTorch installation

### Memory Configuration

For different workloads, you can adjust settings:

```bash
# High VRAM mode (default for DGX Spark)
./start.sh --high-vram

# Low VRAM mode (if needed)
./start.sh --low-vram

# CPU mode (not recommended)
./start.sh --cpu

# Force FP32 (uses more memory)
./start.sh --fp32
```

## Troubleshooting

### Common Issues

1. **CUDA not found**: Ensure CUDA 13.0 is installed via setup
2. **Out of memory**: Use `--low-vram` flag or smaller models
3. **Model not loading**: Check model file integrity with `./models.sh list`
4. **Slow generation**: Verify GPU is being used in ComfyUI settings

### Getting Help

- Check ComfyUI logs when running in daemon mode
- Use `./start.sh --status` to verify service status
- Monitor storage with `./models.sh usage`
- Verify CUDA with `nvidia-smi` command

## Development and Customization

### Custom Workflows

ComfyUI supports custom workflows through:
- **Node-based interface**: Drag and drop workflow creation
- **Custom nodes**: Install extensions via ComfyUI Manager
- **API mode**: Programmatic workflow execution
- **Batch processing**: Automated generation pipelines

### Extension Management

The setup includes ComfyUI Manager for easy extension installation:
1. Access ComfyUI web interface
2. Click "Manager" button
3. Browse and install custom nodes
4. Restart ComfyUI to activate new extensions

---

**Status: ✅ Ready to Use**

This implementation is complete and production-ready, featuring:
- Native ARM64 optimization for DGX Spark
- Grace Blackwell memory utilization
- Professional workflow templates
- High-performance generation pipelines

**Get started now**: Run `./setup.sh` to begin your creative AI journey on DGX Spark!

