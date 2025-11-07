#!/usr/bin/env python3
"""
Interactive CLI chat with NanoChat models from local or HuggingFace Hub.

This script provides an interactive chat interface where you can:
- Chat with trained NanoChat models
- Load models from local directories or HuggingFace Hub
- Use efficient text generation with proper dtype handling (BFloat16)

Prerequisites:
    Before running this script, ensure NanoChat is set up:
    1. Download prepare.sh:
       wget https://raw.githubusercontent.com/jasonacox/dgx-spark/main/nanochat/prepare.sh
       chmod +x prepare.sh
    2. Run prepare.sh with --setup-only to configure NanoChat environment:
       ./prepare.sh --setup-only
    3. Activate the virtual environment:
       source nanochat/.venv/bin/activate

Usage:
    # Interactive chat with local model:
    python hf_test.py --model hf_models/nanochat-1.8B-sft
    
    # Interactive chat with HuggingFace model:
    python hf_test.py --model jasonacox/nanochat-1.8B-sft
    python hf_test.py --model jasonacox/nanochat-1.8B-rl
    
    # Single prompt (non-interactive):
    python hf_test.py --model jasonacox/nanochat-1.8B-sft --prompt "What is your name?"
    
    # Customize generation:
    python hf_test.py --model jasonacox/nanochat-1.8B-sft --max-tokens 200 --temperature 0.7

https://github.com/jasonacox/dgx-spark/
Date: 2025-11-06
"""
import sys
import os
import glob
import argparse

def check_nanochat_setup(nanochat_path):
    """Check if NanoChat is properly set up and provide instructions if not"""
    if not os.path.exists(nanochat_path):
        print("❌ Error: NanoChat not found")
        print(f"\nExpected location: {nanochat_path}")
        print("\nPlease run prepare.sh to set up NanoChat:")
        print("  wget https://raw.githubusercontent.com/jasonacox/dgx-spark/main/nanochat/prepare.sh")
        print("  chmod +x prepare.sh")
        print("  ./prepare.sh --setup-only")
        print("\nThis will:")
        print("  1. Clone the NanoChat repository")
        print("  2. Patch it for DGX Spark (CUDA 13.0) if needed")
        print("  3. Install all dependencies in a virtual environment")
        sys.exit(1)
    
    # Check if virtual environment exists
    venv_path = os.path.join(nanochat_path, '.venv')
    if not os.path.exists(venv_path):
        print("❌ Error: NanoChat virtual environment not found")
        print(f"\nExpected location: {venv_path}")
        print("\nPlease run prepare.sh to set up the environment:")
        print("  ./prepare.sh --setup-only")
        print("\nOr manually set up the environment:")
        print(f"  cd {nanochat_path}")
        print("  uv venv")
        print("  uv sync --extra gpu")
        sys.exit(1)
    
    # Check if running inside the virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if not in_venv:
        print("❌ Error: Not running in NanoChat virtual environment")
        print("\nPlease activate the virtual environment first:")
        print("  source nanochat/.venv/bin/activate")
        print("\nThen run this script again.")
        sys.exit(1)
    
    # Add nanochat to Python path
    sys.path.insert(0, nanochat_path)

def download_from_huggingface(repo_id, cache_dir=None):
    """Download model from HuggingFace Hub"""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface_hub not installed. Install with:")
        print("  pip install huggingface_hub")
        sys.exit(1)
    
    print(f"📥 Downloading model from HuggingFace: {repo_id}")
    
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
    # Import here after nanochat is in path and potentially patched
    from nanochat.checkpoint_manager import load_checkpoint
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.tokenizer import RustBPETokenizer
    import torch
    
    # Find the checkpoint step from the model file
    checkpoint_files = glob.glob(os.path.join(model_path, "model_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No model checkpoint found in {model_path}")
    step = int(os.path.basename(checkpoint_files[0]).split("_")[-1].split(".")[0])
    
    print(f"📂 Loading model from: {model_path}")
    print(f"   Checkpoint step: {step}")
    
    # Check if tokenizer exists in model directory (HuggingFace models)
    tokenizer_dir = os.path.join(model_path, "tokenizer")
    if os.path.exists(tokenizer_dir):
        # Load tokenizer from model directory
        tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)
    else:
        # Use default tokenizer from nanochat cache
        from nanochat.tokenizer import get_tokenizer
        tokenizer = get_tokenizer()
    
    # Load checkpoint and build model
    model_data, optimizer_data, meta_data = load_checkpoint(model_path, step, device, load_optimizer=False)
    
    if device.type in {"cpu", "mps"}:
        # Convert bfloat16 tensors to float for CPU inference
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }
    
    # Fix torch compile issue
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    
    # Build model
    model_config_kwargs = meta_data["model_config"]
    print(f"   Model config: {model_config_kwargs}")
    model_config = GPTConfig(**model_config_kwargs)
    
    with torch.device("meta"):
        model = GPT(model_config)
    
    # Load the model state
    model.to_empty(device=device)
    model.init_weights()
    model.load_state_dict(model_data, strict=True, assign=True)
    
    # Put the model in eval mode
    if phase == "eval":
        model.eval()
    else:
        model.train()
    
    # Sanity check: compatibility between model and tokenizer
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"]
    
    return model, tokenizer, meta_data

def main():
    parser = argparse.ArgumentParser(description='Interactive chat with NanoChat models from local or HuggingFace')
    parser.add_argument('--model', type=str, default='jasonacox/nanochat-1.8B-rl',
                        help='Local path or HuggingFace repo (e.g., jasonacox/nanochat-1.8B-rl)')
    parser.add_argument('--prompt', type=str, default='',
                        help='Single prompt for non-interactive mode')
    parser.add_argument('--max-tokens', type=int, default=256,
                        help='Maximum tokens to generate per response')
    parser.add_argument('--temperature', type=float, default=0.6,
                        help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=50,
                        help='Top-k sampling parameter')
    args = parser.parse_args()
    
    # Check NanoChat setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nanochat_path = os.path.join(script_dir, 'nanochat')
    check_nanochat_setup(nanochat_path)
    
    # Now import after potential patching
    from nanochat.common import compute_init, autodetect_device_type
    from nanochat.engine import Engine
    import torch
    from contextlib import nullcontext
    
    # Initialize device
    print("🔧 Initializing device...")
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
            print(f"❌ Error: Model path not found: {model_path}")
            print(f"   Tried: {full_path}")
            sys.exit(1)
    
    # Load the model
    print("🤖 Loading NanoChat model...")
    model, tokenizer, meta = load_nanochat_model(model_path, device, phase="eval")

    # Create Engine for generation
    engine = Engine(model, tokenizer)
    
    # Special tokens for the chat state machine
    bos = tokenizer.get_bos_token_id()
    user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
    assistant_start, assistant_end = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")
    
    # Initialize conversation with BOS token
    conversation_tokens = [bos]
    
    # Check if we're in prompt mode (single query and exit)
    if args.prompt:
        # Single prompt mode
        print(f"\n💬 User: {args.prompt}")
        
        # Add User message to the conversation
        conversation_tokens.append(user_start)
        conversation_tokens.extend(tokenizer.encode(args.prompt))
        conversation_tokens.append(user_end)
        
        # Kick off the assistant
        conversation_tokens.append(assistant_start)
        
        # Generate response
        print("\n🤖 Assistant: ", end="", flush=True)
        response_tokens = []
        
        with autocast_ctx:
            for token_column, token_masks in engine.generate(
                conversation_tokens, 
                num_samples=1, 
                max_tokens=args.max_tokens, 
                temperature=args.temperature, 
                top_k=args.top_k
            ):
                token = token_column[0]
                response_tokens.append(token)
                
                if token == assistant_end:
                    break
                
                token_text = tokenizer.decode([token])
                print(token_text, end="", flush=True)
        
        print("\n")
        return
    
    # Interactive mode
    # Print welcome message
    print("\n" + "="*60)
    print("🤖 NanoChat Interactive CLI")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Settings: max_tokens={args.max_tokens}, temperature={args.temperature}, top_k={args.top_k}")
    print("\nCommands:")
    print("  Type your message and press Enter to chat")
    print("  Type 'exit' or 'quit' to end the conversation")
    print("  Type 'clear' to start a new conversation")
    print("  Press Ctrl+C to exit")
    print("="*60)
    
    # Print ready message
    print("\n🤖 Assistant: Ready.")

    # Interactive chat loop
    try:
        while True:
            # Get user input
            try:
                user_input = input("\n💬 You: ").strip()
            except EOFError:
                print("\n\nGoodbye! 👋")
                break
            
            # Handle commands
            if not user_input:
                continue
            if user_input.lower() in ['exit', 'quit']:
                print("\nGoodbye! 👋")
                break
            if user_input.lower() == 'clear':
                conversation_tokens = [bos]
                os.system('clear' if os.name != 'nt' else 'cls')
                print("="*60)
                print("🤖 NanoChat Interactive CLI")
                print("="*60)
                print("Conversation cleared.\n")
                continue
            
            # Add User message to the conversation
            conversation_tokens.append(user_start)
            conversation_tokens.extend(tokenizer.encode(user_input))
            conversation_tokens.append(user_end)
            
            # Kick off the assistant
            conversation_tokens.append(assistant_start)
            
            # Generate response
            print("\n🤖 Assistant: ", end="", flush=True)
            response_tokens = []
            
            with autocast_ctx:
                for token_column, token_masks in engine.generate(
                    conversation_tokens, 
                    num_samples=1, 
                    max_tokens=args.max_tokens, 
                    temperature=args.temperature, 
                    top_k=args.top_k
                ):
                    token = token_column[0]  # pop the batch dimension
                    response_tokens.append(token)
                    
                    # Check if we hit the assistant_end token
                    if token == assistant_end:
                        break
                    
                    # Only print if it's not the end token
                    token_text = tokenizer.decode([token])
                    print(token_text, end="", flush=True)
            
            print()  # New line after response
            
            # Ensure that the assistant end token is the last token
            # so even if generation ends due to max tokens, we append it to the end
            if response_tokens[-1] != assistant_end:
                response_tokens.append(assistant_end)
            conversation_tokens.extend(response_tokens)
            
    except KeyboardInterrupt:
        print("\n\nGoodbye! 👋")
    except Exception as e:
        print(f"\n\n❌ Error during chat: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
