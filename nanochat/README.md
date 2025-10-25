# NanoChat for Nvidia DGX Spark

> "The best ChatGPT that $100 can buy" - Andrej Karpathy

This project provides a complete setup for training a Large Language Model (LLM) from scratch using the NanoChat framework specifically optimized for the **Nvidia DGX Spark** platform. The setup leverages the unique capabilities of the DGX Spark's Grace Blackwell GB10 GPU architecture with 128GB unified memory, making it ideal for training the 1.9B parameter model despite the GB10's moderate compute power.

## Overview

NanoChat is a **full-stack implementation of an LLM like ChatGPT** in a single, clean, minimal, hackable, dependency-lite codebase. Created by Andrej Karpathy, it's designed to be:

- **Fully Yours**: Completely configurable, tweakable, hackable, and trained by you from start to end
- **End-to-End**: Covers tokenization, pretraining, finetuning, evaluation, inference, and web serving
- **Accessible**: Designed to run meaningful models on modest hardware budgets
- **Educational**: Perfect for learning how modern LLMs actually work

## Pre-Training

To set up the environment and begin training an LLM from scratch, simply run the scripts below. This will pretrain a model with default parameters including a model depth of 20 layers. The training set consists of text from many webpages, and for this part we will use the [FineWeb-EDU](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) dataset, specifically the sample-100B version from [karpathy/fineweb-edu-100b-shuffle](https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle).

```bash
# Configure the DGX Spark for CUDA 13 - Required to support the GB10 GPU
./setup.sh

# Download NanoChat and required training data
./prepare.sh

# Run pretraining - It is recommended that you run this in screen or tmux terminal as
# training can take several days and a disconnect would cause the training to stop.
./pretrain.sh
```

<img width="1067" height="1099" alt="Screenshot 2025-10-25 at 3 52 46 PM" src="https://github.com/user-attachments/assets/eab9dbbf-e9e1-44c6-a8c1-fbe08e5864db" />

Once this completes, the base model will be able to generate tokens based on input prompts. However, it will not be able to chat properly. That will require introducing a user/assistant chat format, which is done in midtraining.

## Midtraining

Next up is midtraining. This stage will fine-tune the model based on [smol-SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk). The training process will be the same as pretraining, but the dataset now becomes conversations, and the model adapts itself to the new special tokens that structure multi-turn conversation objects. Each conversation now looks something like this, loosely following the [OpenAI Harmony chat format](https://github.com/openai/harmony):

```
<|bos|>
<|user_start|>What is the color of the sky?<|user_end|>
<|assistant_start|>Red. Wait, possibly blue. I'm not sure.<|assistant_end|>
<|user_start|>lol<|user_end|>
<|assistant_start|>...etcetc
```

Where `<|example|>` represents special tokens, following the format of OpenAI special tokens. The midtraining stage teaches the model several key capabilities: learning special tokens for multi-turn conversations, adapting from internet document distribution to conversation patterns, and crucially learning to handle multiple choice questions (since small models don't naturally acquire this skill from web data alone). Additionally, the model learns to use tools like Python interpreters through special tokens, enabling it to solve mathematical problems and perform evaluations on common benchmarks like MMLU and GSM8K.

```bash
./midtrain.sh
```

## Chat

Once midtraining completes, you can chat with your model through a **ChatGPT-like interface**:

```bash
# Simple script with environment setup
./chat.sh

# Alternative manual methods:
# Command Line interface
python -m scripts.chat_cli

# Web-based interface
python -m scripts.chat_web
```

**For web interface**: Visit the displayed URL (e.g., `http://your-server-ip:8000/`)

## Supervised Finetuning (SFT)

Following midtraining is the Supervised Fine-tuning (SFT) stage, which performs additional fine-tuning on curated, high-quality conversations. This introduces safety training. SFT addresses a key domain mismatch by formatting examples to match test-time conditions - stretching and padding data rows individually rather than concatenating them for training efficiency as done in pre/mid-training stages. This formatting alignment provides another performance boost by ensuring the model trains on data that mirrors its actual inference usage patterns.

```bash
./sft.sh
```

## Reinforcement Learning (RL)

The final stage is Reinforcement Learning (RL), which provides modest performance gains and helps mitigate issues like hallucinations. Using GSM8K's objective math problem rewards, the model runs a simplified GRPO training loop that samples completions, rewards correct answers, and trains on high-reward responses. This implementation removes complexity like trust regions, PPO ratios, and z-score normalization, resulting in a REINFORCE-like approach that retains group relative advantage calculation from rewards.

```bash
./rl.sh
```

## Congratulations!

You have completed the training of your model! You may now go back to the [Chat section](#chat) above and interact with your fully trained model.

## Why DGX Spark is Perfect for NanoChat

The Nvidia DGX Spark's unique architecture makes it exceptionally well-suited for training the NanoChat 1.9B parameter model:

- **Grace Blackwell GB10 GPU**: Optimized for AI workloads with unified memory architecture
- **128GB Unified Memory**: Allows for larger models than typical high-end consumer-grade GPUs
- **ARM-based Platform**: Energy-efficient architecture ideal for extended training sessions

**Note**: These micro models are intentionally smaller than modern LLMs like GPT-4. They may make mistakes, be somewhat naive, and hallucinate - "a bit like children" as Karpathy describes. But they're **fully yours** and perfectly matched to the DGX Spark's capabilities.

## System Requirements

- **Hardware**: Nvidia DGX Spark with Grace Blackwell GB10 GPU
- **Memory**: 128GB unified memory (fully utilized for optimal training)
- **OS**: Ubuntu 24.04 ARM64
- **CUDA**: 13.0 (automatically installed and optimized for Grace Blackwell)
- **Storage**: Several GB for datasets, models, and dependencies


## Customizing Training

To modify training parameters for your DGX Spark, you can edit the final training command in `pretrain.sh`:

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

