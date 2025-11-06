#!/usr/bin/env python3
"""
Test script to verify trained NanoChat models can be loaded and generate text.

This script demonstrates how to:
- Load a trained model from local HF-prepared directory OR from HuggingFace Hub
- Use the Engine class for efficient text generation
- Generate responses with proper dtype handling (BFloat16)

Usage:
    source nanochat/.venv/bin/activate
    
    # Load from local hf_models directory:
    python hf_test.py
    python hf_test.py --model hf_models/nanochat-1.8B-midtrain
    
    # Load from HuggingFace Hub:
    python hf_test.py --model jasonacox/nanochat-1.8B-pretrain
    python hf_test.py --model jasonacox/nanochat-1.8B-midtrain
    
    # Customize generation:
    python hf_test.py --model jasonacox/nanochat-1.8B-sft --prompt "What is AI?" --max-tokens 200

https://github.com/jasonacox/dgx-spark/
Date: 2025-11-05
"""
import sys
import os
import glob
import argparse

# Add the nanochat repo to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nanochat'))

from nanochat.checkpoint_manager import build_model
from nanochat.common import compute_init, autodetect_device_type
from nanochat.engine import Engine
import torch
from contextlib import nullcontext

def download_from_huggingface(repo_id, cache_dir=None):
    """Download model from HuggingFace Hub"""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface_hub not installed. Install with:")
        print("  pip install huggingface_hub")
        sys.exit(1)
    
    print(f"Downloading model from HuggingFace: {repo_id}")
    
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "nanochat", "hf_downloads")
    
    try:
        model_path = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            local_dir=os.path.join(cache_dir, repo_id.replace("/", "_")),
            local_dir_use_symlinks=False
        )
        print(f"✓ Downloaded to: {model_path}")
        return model_path
    except Exception as e:
        print(f"Error downloading from HuggingFace: {e}")
        print("\nMake sure:")
        print("  1. The repository exists and is public (or you're logged in)")
        print("  2. You have internet connection")
        print("  3. For private repos, run: huggingface-cli login")
        sys.exit(1)

def load_nanochat_model(model_path, device, phase="eval"):
    """Load a NanoChat model from a directory"""
    # Find the checkpoint step from the model file
    checkpoint_files = glob.glob(os.path.join(model_path, "model_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No model checkpoint found in {model_path}")
    step = int(os.path.basename(checkpoint_files[0]).split("_")[-1].split(".")[0])
    
    print(f"Loading model from: {model_path}")
    print(f"Checkpoint step: {step}")
    
    # Build the model directly from the directory
    model, tokenizer, meta = build_model(model_path, step, device, phase=phase)
    return model, tokenizer, meta

def main():
    parser = argparse.ArgumentParser(description='Test NanoChat models from local or HuggingFace')
    parser.add_argument('--model', type=str, default='hf_models/nanochat-1.8B-midtrain',
                        help='Local path or HuggingFace repo (e.g., jasonacox/nanochat-1.8B-pretrain)')
    parser.add_argument('--prompt', type=str, default='Hello, how are you?',
                        help='Text prompt for the model')
    parser.add_argument('--max-tokens', type=int, default=100,
                        help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=50,
                        help='Top-k sampling parameter')
    args = parser.parse_args()
    
    # Initialize device
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    ptdtype = torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()
    
    # Determine if model_path is local or HuggingFace repo
    model_path = args.model
    
    # Check if it looks like a HuggingFace repo (contains /)
    if "/" in model_path and not os.path.exists(model_path):
        # Treat as HuggingFace repo
        model_path = download_from_huggingface(model_path)
    elif not os.path.exists(model_path):
        # Try as relative path from script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(script_dir, model_path)
        if os.path.exists(full_path):
            model_path = full_path
        else:
            print(f"Error: Model path not found: {model_path}")
            print(f"Tried: {full_path}")
            sys.exit(1)
    
    # Load the model
    model, tokenizer, meta = load_nanochat_model(model_path, device, phase="eval")

    # Load the model
    model, tokenizer, meta = load_nanochat_model(model_path, device, phase="eval")

    # Create Engine for generation
    engine = Engine(model, tokenizer)

    # Generate text
    prompt = args.prompt
    tokens = tokenizer.encode(prompt)

    print(f"\nPrompt: {prompt}")
    print(f"Response: ", end="", flush=True)

    # Generate response
    with autocast_ctx:
        for token_column, token_masks in engine.generate(
            tokens, 
            num_samples=1, 
            max_tokens=args.max_tokens, 
            temperature=args.temperature, 
            top_k=args.top_k
        ):
            token = token_column[0]  # pop the batch dimension
            token_text = tokenizer.decode([token])
            print(token_text, end="", flush=True)

    print()  # New line at the end

if __name__ == "__main__":
    main()
