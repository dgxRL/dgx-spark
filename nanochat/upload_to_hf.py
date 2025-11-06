#!/usr/bin/env python3
"""
Upload NanoChat models to HuggingFace Hub

This script automates the upload of prepared NanoChat models to HuggingFace.

Usage:
    python upload_to_hf.py --username your-hf-username [--model-dir ./hf_models]

Requirements:
    pip install huggingface_hub
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo, login
except ImportError:
    print("Error: huggingface_hub not installed")
    print("Install with: pip install huggingface_hub")
    sys.exit(1)


def upload_model(model_path: Path, repo_id: str, token: str = None):
    """Upload a model directory to HuggingFace Hub"""
    
    api = HfApi(token=token)
    
    print(f"\n📤 Uploading {model_path.name} to {repo_id}")
    
    try:
        # Create repository
        print(f"  Creating repository: {repo_id}")
        create_repo(repo_id, repo_type="model", exist_ok=True, token=token)
        
        # Upload all files
        print(f"  Uploading files from {model_path}")
        api.upload_folder(
            folder_path=str(model_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload {model_path.name} model"
        )
        
        print(f"  ✓ Successfully uploaded to https://huggingface.co/{repo_id}")
        return True
        
    except Exception as e:
        print(f"  ✗ Failed to upload {model_path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Upload NanoChat models to HuggingFace Hub"
    )
    parser.add_argument(
        "--username",
        required=True,
        help="Your HuggingFace username"
    )
    parser.add_argument(
        "--model-dir",
        default="./hf_models",
        help="Directory containing prepared models (default: ./hf_models)"
    )
    parser.add_argument(
        "--token",
        help="HuggingFace API token (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=["pretrain", "midtrain", "sft", "rl"],
        help="Specific phases to upload (default: all available)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading"
    )
    
    args = parser.parse_args()
    
    # Get token
    token = args.token or os.environ.get("HF_TOKEN")
    if not token and not args.dry_run:
        print("No HuggingFace token provided. Attempting interactive login...")
        try:
            login()
        except Exception as e:
            print(f"Login failed: {e}")
            print("\nPlease either:")
            print("  1. Run: huggingface-cli login")
            print("  2. Set HF_TOKEN environment variable")
            print("  3. Use --token argument")
            sys.exit(1)
    
    # Find models to upload
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}")
        print("Run hf_prepare.sh first to prepare models")
        sys.exit(1)
    
    # Determine which phases to upload
    if args.phases:
        phases = args.phases
    else:
        # Find all prepared model directories
        phases = []
        for phase in ["pretrain", "midtrain", "sft", "rl"]:
            phase_dirs = list(model_dir.glob(f"*-{phase}"))
            if phase_dirs:
                phases.append(phase)
    
    if not phases:
        print("No prepared models found in", model_dir)
        print("Run hf_prepare.sh first")
        sys.exit(1)
    
    print(f"\n🚀 Uploading NanoChat models to HuggingFace")
    print(f"Username: {args.username}")
    print(f"Models directory: {model_dir}")
    print(f"Phases to upload: {', '.join(phases)}")
    
    if args.dry_run:
        print("\n[DRY RUN - No actual uploads will be performed]")
    
    # Upload each model
    success_count = 0
    total_count = 0
    
    for phase in phases:
        # Find the model directory for this phase
        phase_dirs = list(model_dir.glob(f"*-{phase}"))
        
        if not phase_dirs:
            print(f"\n⚠️  No {phase} model found, skipping")
            continue
        
        model_path = phase_dirs[0]
        model_name = model_path.name
        repo_id = f"{args.username}/{model_name}"
        
        total_count += 1
        
        if args.dry_run:
            print(f"\n[DRY RUN] Would upload:")
            print(f"  Local: {model_path}")
            print(f"  Remote: https://huggingface.co/{repo_id}")
            success_count += 1
        else:
            if upload_model(model_path, repo_id, token):
                success_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Upload Summary:")
    print(f"  Successful: {success_count}/{total_count}")
    
    if args.dry_run:
        print(f"\nThis was a dry run. To actually upload, remove --dry-run flag")
    elif success_count == total_count:
        print(f"\n✓ All models uploaded successfully!")
        print(f"\nView your models at:")
        for phase in phases:
            phase_dirs = list(model_dir.glob(f"*-{phase}"))
            if phase_dirs:
                model_name = phase_dirs[0].name
                print(f"  https://huggingface.co/{args.username}/{model_name}")
    else:
        print(f"\n⚠️  Some uploads failed. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
