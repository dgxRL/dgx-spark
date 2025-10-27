# ImageGen - Generate High-Quality Images and Videos

**Create stunning images and videos with AI on DGX Spark**

This project provides a complete ComfyUI setup for the DGX Spark, optimized to run high-fidelity open source models capable of generating professional-quality images and videos. Transform your DGX Spark into a powerful creative AI workstation.

## Overview

The Grace Blackwell architecture's unified memory and high-performance compute make the DGX Spark ideal for image and video generation tasks that traditionally require expensive cloud services or complex multi-GPU setups. Our implementation provides a complete creative AI suite optimized for the platform's unique capabilities.

## Supported Models

| Model Category | Size | Models Included | Description |
|----------------|------|-----------------|-------------|
| **Base Models** (`./models.sh install base`) | 4GB | • Stable Diffusion v1.5<br>• SD VAE<br>• 4x ESRGAN | Essential starter models for all workflows. General image generation, LoRA/ControlNet compatible, with enhanced image quality and basic upscaling. |
| **SDXL Models** (`./models.sh install sdxl`) | 13.5GB | • SDXL Base & Refiner<br>• SDXL VAE | High-resolution professional generation with 1024x1024 native resolution, superior detail, and optimized FP16 processing. |
| **Flux Models** (`./models.sh install flux`) | 29GB | • Flux Schnell<br>• Text Encoders | Cutting-edge AI generation with ultra-fast generation, advanced text understanding, and enhanced prompt interpretation. |
| **ControlNet Models** (`./models.sh install controlnet`) | 4.2GB | • Canny<br>• Depth<br>• OpenPose | Guided generation tools for edge detection, 3D structure control, and pose control. |
| **Video Models** (`./models.sh install video`) | 10GB | • Stable Video Diffusion XT | Motion and animation with image-to-video conversion and smooth motion synthesis. |
| **Upscale Models** (`./models.sh install upscale`) | 2.1GB | • ESRGAN (2x/4x)<br>• LDSR | Image enhancement with standard upscaling for general use and highest quality latent diffusion super-resolution. |

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

| Description | Command |
|-------------|---------|
| Essential starter models (SD 1.5, VAE, upscaler) | `./models.sh install base` |
| Stable Diffusion XL models | `./models.sh install sdxl` |
| ControlNet guidance models | `./models.sh install controlnet` |
| Video generation models | `./models.sh install video` |
| Additional upscaling models | `./models.sh install upscale` |
| Everything (requires 200+ GB storage) | `./models.sh install all` |
| Show installed models by category | `./models.sh list` |
| Display storage usage | `./models.sh usage` |
| Interactive removal tool | `./models.sh remove` |
| Update existing models | `./models.sh update` |

## Advanced Usage

### ComfyUI Daemon Management
| Description | Command |
|-------------|---------|
| Start as daemon with logging | `./start.sh --daemon --log comfyui.log` |
| Check daemon status | `./start.sh --status` |
| Stop daemon | `./start.sh --stop` |
| Restart daemon | `./start.sh --restart` |
| Custom port and host | `./start.sh --port 8080 --host 192.168.1.100` |


### Memory Configuration

For different workloads, you can adjust settings:

| Description | Command |
|-------------|---------|
| High VRAM mode (default for DGX Spark) | `./start.sh --high-vram` |
| Low VRAM mode (if needed) | `./start.sh --low-vram` |
| CPU mode (not recommended) | `./start.sh --cpu` |
| Force FP32 (uses more memory) | `./start.sh --fp32` |

## Troubleshooting

TBD


