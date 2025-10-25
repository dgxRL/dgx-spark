# NanoChat for Nvidia DGX Spark

> "The best ChatGPT that $100 can buy" - Andrej Karpathy

This project provides a complete setup for training a Large Language Model (LLM) from scratch using the NanoChat framework specifically optimized for the **Nvidia DGX Spark** platform. The setup leverages the unique capabilities of the DGX Spark's Grace Blackwell GB10 GPU architecture with 128GB unified memory, making it ideal for training the 1.9B parameter model despite the GB10's moderate compute power.

## Overview

NanoChat is a **full-stack implementation of an LLM like ChatGPT** in a single, clean, minimal, hackable, dependency-lite codebase. Created by Andrej Karpathy, it's designed to be:

- **Fully Yours**: Completely configurable, tweakable, hackable, and trained by you from start to end
- **End-to-End**: Covers tokenization, pretraining, finetuning, evaluation, inference, and web serving
- **Accessible**: Designed to run meaningful models on budgets under $1000
- **Educational**: Serves as the capstone project for the LLM101n course by Eureka Labs

Unlike massive production LLMs that cost millions to train, NanoChat focuses on **micro models** that are accessible for learning and experimentation while still producing functional ChatGPT-like conversational AI.

## Why DGX Spark is Perfect for NanoChat

The Nvidia DGX Spark's unique architecture makes it exceptionally well-suited for training the NanoChat 1.9B parameter model:

- **Grace Blackwell GB10 GPU**: Optimized for AI workloads with unified memory architecture
- **128GB Unified Memory**: Massive memory capacity allows handling large models that would OOM on traditional GPUs
- **ARM-based Platform**: Energy-efficient architecture ideal for extended training sessions
- **Single GPU Design**: Perfect match for NanoChat's educational focus and streamlined workflow

The GB10's **large memory capacity compensates for moderate compute**, making it ideal for the memory-intensive transformer training process while maintaining reasonable training times.

## What You'll Get

After training completes on your DGX Spark, you'll have:
- A **functional ChatGPT-like web interface** to chat with your model
- A model with **~1.9 billion parameters** perfectly sized for the GB10's 128GB memory
- Performance that **outperforms GPT-2** on various benchmarks
- A complete understanding of the LLM training pipeline optimized for Grace Blackwell architecture
- Full ownership and control over your AI model

**Note**: These micro models are intentionally smaller than modern LLMs like GPT-4. They may make mistakes, be somewhat naive, and hallucinate - "a bit like children" as Karpathy describes. But they're **fully yours** and perfectly matched to the DGX Spark's capabilities.

## Quick Start

To set up the environment and begin training an LLM from scratch, simply run:

```bash
./setup.sh
```

This single command will handle all setup steps and initiate the training process with default parameters including a model depth of 20 layers.

## What the Setup Script Does

The `setup.sh` script performs the following comprehensive setup:

### 1. **Repository Setup**
- Clones the original NanoChat repository from Karpathy's GitHub
- Updates the project configuration for CUDA 13.0 compatibility
- Modifies `pyproject.toml` to use appropriate PyTorch and CUDA versions

### 2. **Python Environment Configuration**
- Installs `uv` package manager (if not present)
- Creates a local virtual environment (`.venv`)
- Installs all project dependencies via `uv sync`
- Activates the virtual environment

### 3. **Tokenizer Build**
- Installs Rust/Cargo toolchain
- Builds the high-performance `rustbpe` tokenizer using Maturin
- Optimized for fast text processing during training

### 4. **Dataset Preparation**
- Downloads training dataset (240 million tokens by default)
- Trains a custom tokenizer on the dataset
- Evaluates tokenizer performance
- Downloads evaluation bundle for model assessment

### 5. **CUDA Toolkit Installation**
- Installs CUDA Toolkit 13.0 optimized for DGX Spark's ARM64 architecture
- Configures environment variables for Grace Blackwell GB10 integration
- Adds permanent CUDA paths to shell configuration
- Verifies GPU and CUDA installation on the unified memory system

### 6. **Training Initialization**
- Sets up Weights & Biases (wandb) logging
- Launches training optimized for single GB10 GPU
- **Default configuration: 20-layer transformer model** (perfect for 128GB unified memory)
- Uses batch sizes optimized for Grace Blackwell architecture

## Training Configuration

The training configuration is optimized for the DGX Spark's Grace Blackwell GB10 GPU:

- **Model Depth**: 20 transformer layers (~1.9B parameters)
- **Memory Utilization**: Designed to leverage the full 128GB unified memory capacity
- **Training Approach**: Single GPU training optimized for GB10 architecture
- **Dataset**: 38 billion tokens (automatically downloaded and prepared)
- **Run Name**: "nanochat-dgx"
- **Device Batch Size**: 32 (optimized for GB10's memory bandwidth)
- **Sampling Frequency**: Every 100 steps

### DGX Spark Advantages

The unique characteristics of the DGX Spark platform provide several benefits for NanoChat training:

1. **Unified Memory Architecture**: Eliminates CPU-GPU memory transfers, reducing bottlenecks
2. **Large Memory Capacity**: 128GB allows training larger models without memory constraints
3. **Energy Efficiency**: ARM-based Grace CPU reduces power consumption during long training runs
4. **Simplified Setup**: Single GPU eliminates distributed training complexity

Our **depth=20 configuration** is specifically chosen to maximize the DGX Spark's memory advantage while providing excellent model performance.

## System Requirements

- **Hardware**: Nvidia DGX Spark with Grace Blackwell GB10 GPU
- **Memory**: 128GB unified memory (fully utilized for optimal training)
- **OS**: Ubuntu 24.04 ARM64
- **CUDA**: 13.0 (automatically installed and optimized for Grace Blackwell)
- **Storage**: Several GB for datasets, models, and dependencies

## After Training: Chat with Your LLM

Once training completes, you can interact with your model through a **ChatGPT-like web interface**:

1. **Activate the environment**:
   ```bash
   source .venv/bin/activate
   ```

2. **Start the web server**:
   ```bash
   python -m scripts.chat_web
   ```

3. **Access the interface**: Visit the displayed URL (e.g., `http://your-server-ip:8000/`)

4. **Start chatting**: Ask your model to write stories, explain concepts, or have conversations!

### What to Expect

Your model will be capable of:
- ✅ Creative writing (stories, poems)
- ✅ Basic question answering
- ✅ Simple conversations
- ✅ Code explanations (basic level)

Keep in mind:
- ⚠️ May hallucinate or make factual errors
- ⚠️ Less sophisticated than modern LLMs
- ⚠️ Performance similar to early GPT-2 models
- ⚠️ "Childlike" behavior - naive but charming

### Performance Evaluation

After training, check your model's `report.md` file for detailed metrics including:
- **CORE benchmarks**: Overall language understanding
- **ARC-Challenge/Easy**: Reasoning capabilities  
- **GSM8K**: Mathematical problem solving
- **HumanEval**: Code generation abilities
- **MMLU**: Multi-domain knowledge
- **ChatCORE**: Conversational abilities

## Customizing Training

To modify training parameters for your DGX Spark, you can edit the final training command in `setup.sh`:

```bash
torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --depth=20 \
    --run="nanochat-dgx" \
    --device_batch_size=32 \
    --sample_every=100
```

### Available Parameters:
- `--depth`: Number of transformer layers (default: 20, optimized for GB10's 128GB memory)
- `--run`: Experiment name for logging
- `--device_batch_size`: Batch size optimized for Grace Blackwell memory bandwidth
- `--sample_every`: Steps between sample generations

### DGX Spark Optimization Notes

The 128GB unified memory of the DGX Spark allows for:
- **Larger batch sizes** without memory constraints
- **Deeper models** than typically possible on traditional GPUs
- **Simplified memory management** due to unified architecture
- **No memory fragmentation** issues common in multi-GPU setups

## Monitoring Progress

The script integrates with Weights & Biases for experiment tracking. After running the setup:

1. Visit [wandb.ai](https://wandb.ai) and log in
2. Navigate to your project dashboard
3. Monitor training metrics, loss curves, and generated samples

## Troubleshooting

### Grace Blackwell GB10 Issues
- Verify GPU availability: `nvidia-smi`
- Check CUDA 13.0 installation: `nvcc --version`
- Ensure Grace Blackwell drivers are properly installed
- Monitor unified memory usage during training

### DGX Spark Specific Issues
- Verify ARM64 compatibility of all packages
- Ensure CUDA paths are configured for Grace architecture
- Check that unified memory is being utilized effectively

### Dependency Issues
- Ensure the virtual environment is activated
- Re-run `uv sync` if ARM64 packages are missing
- Verify Rust/Cargo installation for ARM architecture

## Philosophy & Design

NanoChat is intentionally designed as a **"strong baseline"** rather than an exhaustively configurable framework:

- **No giant configuration objects** or complex abstractions
- **Single, cohesive, minimal codebase** (~330KB, ~8K lines)
- **Maximally hackable and forkable** for educational purposes
- **Production-ready pipeline** from tokenization to deployment
- **Budget-conscious** - meaningful models under $1000

This makes NanoChat on DGX Spark perfect for:
- 🎓 **Learning** how LLMs work end-to-end on modern Grace Blackwell architecture
- 🔬 **Research** on unified memory optimization for transformer training
- 🛠️ **Experimentation** with models that leverage large memory capacity
- 🏗️ **Building** custom AI applications on ARM-based platforms
- 📚 **Teaching** LLM concepts with cutting-edge hardware

## Next Steps

Once training is complete, you can:

1. **Evaluate the model** using the provided evaluation scripts
2. **Generate text samples** to assess model quality  
3. **Fine-tune** the model on specific datasets for specialized tasks
4. **Customize personality** through synthetic data generation ([Guide](https://github.com/karpathy/nanochat/discussions/139))
5. **Export** the model for inference applications optimized for Grace Blackwell
6. **Experiment** with larger models that take advantage of the 128GB unified memory

## DGX Spark Platform Advantages

The Nvidia DGX Spark offers unique benefits for LLM training:

**Architecture Benefits**:
- ✅ **Grace Blackwell GB10 GPU** - Latest generation AI acceleration
- ✅ **128GB Unified Memory** - No CPU-GPU memory bottlenecks
- ✅ **ARM-based Grace CPU** - Energy efficient for long training runs
- ✅ **Single GPU Design** - Simplified setup and debugging

**Training Advantages**:
- ✅ **Memory Abundance** - Handle larger models without OOM errors
- ✅ **Unified Memory Architecture** - Faster data loading and processing
- ✅ **Reduced Complexity** - No distributed training coordination needed
- ✅ **Optimal for 1.9B Models** - Perfect sweet spot for GB10 capabilities

**Development Benefits**:
- ✅ **Educational Focus** - Single GPU simplifies learning
- ✅ **ARM Ecosystem** - Future-proof platform experience
- ✅ **Energy Efficiency** - Lower power consumption than multi-GPU setups


**Note**: This setup is specifically optimized for the Nvidia DGX Spark platform with Grace Blackwell GB10 GPU and 128GB unified memory. The configuration takes full advantage of the unique architecture to provide an optimal NanoChat training experience.
