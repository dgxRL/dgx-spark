# Nanochat for Nvidia DGX Spark

> "The best ChatGPT that $100 can buy" - Andrej Karpathy

This project provides a complete setup for training a Large Language Model (LLM) from scratch using the [Nanochat framework](https://github.com/karpathy/nanochat) created by [Andrej Karpathy](https://github.com/karpathy). This version has been specifically optimized for the **Nvidia DGX Spark** platform. The setup leverages the unique capabilities of the DGX Spark's Grace Blackwell GB10 GPU architecture with 128GB unified memory, making it ideal for training the 1.9B parameter model despite the GB10's moderate compute power.

## Overview

Nanochat is a **full-stack implementation of an LLM like ChatGPT** in a single, clean, minimal, hackable, dependency-lite codebase. 

- **Fully Yours**: Completely configurable, tweakable, hackable, and trained by you from start to end
- **End-to-End**: Covers tokenization, pretraining, finetuning, evaluation, inference, and web serving
- **Accessible**: Designed to run meaningful models on modest hardware budgets
- **Educational**: Perfect for learning how modern LLMs actually work

## Pre-Training

To set up the environment and begin training an LLM from scratch, simply run the scripts below. This will pretrain a model with default parameters including a model depth of 20 layers. The training set consists of text from many webpages, and for this part we will use the [FineWeb-EDU](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) dataset, specifically the sample-100B version from [karpathy/fineweb-edu-100b-shuffle](https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle).

```bash
# Configure the DGX Spark for CUDA 13 - Required to support the GB10 GPU
./setup.sh

# Download Nanochat and required training data
./prepare.sh

# Run pretraining - It is recommended that you run this in screen or tmux terminal as
# training can take several days and a disconnect would cause the training to stop.
./pretrain.sh
```

<img width="1067" height="1099" alt="Screenshot 2025-10-25 at 3 52 46 PM" src="https://github.com/user-attachments/assets/eab9dbbf-e9e1-44c6-a8c1-fbe08e5864db" />

Once this completes, the base model will be able to generate tokens based on input prompts. However, it will not be able to chat properly. That will require introducing a user/assistant chat format, which is done in midtraining.

## Midtraining

Next up is midtraining. This stage fine-tunes the model on conversational data, teaching it to engage in multi-turn dialogues. The training uses several datasets including [smol-SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk), MMLU, GSM8K, and custom identity conversations.

**Data Format**: Conversations are stored in JSONL files, where each line contains a JSON object with a `messages` array following the OpenAI chat format:

```json
{
  "messages": [
    {"role": "user", "content": "What is the color of the sky?"},
    {"role": "assistant", "content": "The sky is typically blue during the day."},
    {"role": "user", "content": "Why is it blue?"},
    {"role": "assistant", "content": "The sky appears blue due to Rayleigh scattering..."}
  ]
}
```

The tokenizer converts these conversations into a token sequence with special tokens:

```
<|bos|>
<|user_start|>What is the color of the sky?<|user_end|>
<|assistant_start|>The sky is typically blue during the day.<|assistant_end|>
<|user_start|>Why is it blue?<|user_end|>
<|assistant_start|>The sky appears blue due to Rayleigh scattering...<|assistant_end|>
```

The midtraining stage teaches the model several key capabilities:
- **Multi-turn conversations**: Learning special tokens that structure dialogue turns
- **Distribution shift**: Adapting from internet documents to conversational patterns  
- **Multiple choice questions**: Handling QA formats (via MMLU dataset)
- **Tool use**: Using Python interpreters through special tokens for math (via GSM8K dataset)
- **Identity**: Learning model-specific information from custom identity conversations

You can customize your model's personality by editing `~/.cache/nanochat/identity_conversations.jsonl`. See the included `view_identity_data.py` script to view and manage this data.

```bash
./midtrain.sh
```

## Chat

Once midtraining completes, you can chat with your model through a **ChatGPT-like interface**:

```bash
# Web-based interface (recommended)
./chat.sh

# Command-line interface
./chat_cli.sh
```

**For web interface**: Visit the displayed URL (e.g., `http://your-server-ip:8000/`)

Both scripts automatically detect and use your most advanced trained model (RL > SFT > Mid > Base).

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

_These micro models are intentionally smaller than modern LLMs like GPT-4. They may make mistakes, are somewhat naive, and hallucinate - "a bit like children" as Karpathy describes. But they're **fully yours** and perfectly matched to the DGX Spark's capabilities._

## Sharing Your Model

Once training is complete, you can share your model on HuggingFace Hub:

```bash
# 1. Prepare models for upload
./hf_prepare.sh --author your-hf-username

# 2. Install HuggingFace CLI and login
pip install huggingface_hub
huggingface-cli login

# 3. Upload to HuggingFace (dry run first to preview)
python upload_to_hf.py --username your-hf-username --dry-run
python upload_to_hf.py --username your-hf-username
```

This will organize your checkpoints, create model cards, and upload them to HuggingFace. See [HUGGINGFACE_UPLOAD.md](HUGGINGFACE_UPLOAD.md) for detailed instructions.

Example Models:

* https://huggingface.co/jasonacox/nanochat-1.8B-pretrain
* https://huggingface.co/jasonacox/nanochat-1.8B-midtrain
* https://huggingface.co/jasonacox/nanochat-1.8B-sft

---

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

## Credits and Thanks

 * Andrej Karpathy for Nanochat - https://github.com/karpathy/nanochat
 * Alexander Falk for the Fix for DGX Spark - https://github.com/karpathy/nanochat/discussions/28#discussioncomment-14756733 and https://forums.developer.nvidia.com/t/anyone-got-nanochat-training-working-on-the-dgx-spark/348537/8 

---

"The best way to predict the future is to build it yourself."
