#!/bin/bash

# Script to prepare NanoChat models for Hugging Face upload
# This organizes model checkpoints into a HuggingFace-compatible structure
#
# Author: Jason Cox
# Date: 2025-11-05
# https://github.com/jasonacox/dgx-spark

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}=== NanoChat HuggingFace Upload Preparation ===${NC}"
echo ""

# Model configuration
MODEL_NAME="nanochat-1.8B"
AUTHOR_NAME=${HF_USERNAME:-"your-username"}
OUTPUT_DIR="./hf_models"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --author)
            AUTHOR_NAME="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --name NAME       Model name (default: nanochat-dgx-spark)"
            echo "  --author AUTHOR   HuggingFace username (default: your-username)"
            echo "  --output DIR      Output directory (default: ./hf_models)"
            echo "  --help            Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  HF_USERNAME       Your HuggingFace username"
            echo "  HF_TOKEN          Your HuggingFace API token"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if output directory exists and prompt to clear
if [ -d "$OUTPUT_DIR" ]; then
    echo -e "${YELLOW}Output directory already exists: $OUTPUT_DIR${NC}"
    echo -n "Clear it before proceeding? (y/N): "
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Removing existing directory...${NC}"
        rm -rf "$OUTPUT_DIR"
        echo -e "${GREEN}✓ Directory cleared${NC}"
        echo ""
    else
        echo -e "${YELLOW}Continuing with existing directory (files may be overwritten)${NC}"
        echo ""
    fi
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to prepare a model for upload
prepare_model() {
    local phase=$1
    local checkpoint_dir=$2
    local description=$3
    
    if [ ! -d "$checkpoint_dir" ]; then
        echo -e "${YELLOW}⚠️  Skipping $phase - no checkpoints found${NC}"
        return
    fi
    
    echo -e "${GREEN}📦 Preparing $phase model...${NC}"
    
    local model_dir="$OUTPUT_DIR/${MODEL_NAME}-${phase}"
    mkdir -p "$model_dir"
    
    # Find the latest checkpoint
    local latest_checkpoint=$(find "$checkpoint_dir" -name "model_*.pt" | sort -V | tail -1)
    
    if [ -z "$latest_checkpoint" ]; then
        echo -e "${YELLOW}⚠️  No model files found in $checkpoint_dir${NC}"
        return
    fi
    
    local checkpoint_base=$(dirname "$latest_checkpoint")
    local step_num=$(basename "$latest_checkpoint" | grep -oP '\d+' | head -1)
    
    echo "  Found checkpoint at step $step_num"
    
    # Copy model files
    cp "$checkpoint_base/model_${step_num}.pt" "$model_dir/"
    cp "$checkpoint_base/meta_${step_num}.json" "$model_dir/"
    
    # Copy tokenizer
    if [ -d "$HOME/.cache/nanochat/tokenizer" ]; then
        cp -r "$HOME/.cache/nanochat/tokenizer" "$model_dir/"
    fi
    
    # Create README.md
    cat > "$model_dir/README.md" << EOF
---
license: mit
base_model: nanochat
tags:
  - nanochat
  - llm
  - dgx-spark
  - grace-blackwell
  - from-scratch
language:
  - en
pipeline_tag: text-generation
---

# ${MODEL_NAME}-${phase}

${description}

## Model Details

- **Model Type:** GPT-style transformer trained from scratch
- **Parameters:** ~1.9 billion
- **Training Phase:** ${phase}
- **Architecture:** 20 layers, 1280 embedding dimension
- **Hardware:** NVIDIA DGX Spark (Grace Blackwell GB10)
- **Framework:** [NanoChat](https://github.com/karpathy/nanochat)
- **Training Precision:** BFloat16

## Training Details

- **GPU:** NVIDIA Grace Blackwell GB10
- **Memory:** 128GB unified memory
- **CUDA:** 13.0
- **Optimization:** Muon optimizer for matrix parameters, AdamW for others
- **Checkpoint Step:** $step_num

## Usage

### Prerequisites

\`\`\`bash
# Clone the NanoChat repository
git clone https://github.com/karpathy/nanochat.git
cd nanochat

# Install dependencies
uv sync --extra gpu
\`\`\`

### Loading the Model

\`\`\`python
from nanochat.checkpoint_manager import load_model_from_dir
import torch

# Load model
model, tokenizer, meta = load_model_from_dir(
    "path/to/model/directory",
    device="cuda",
    phase="eval"
)

# Generate text
prompt = "Hello, how are you?"
tokens = tokenizer.encode(prompt)
tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to("cuda")

# Generate response
with torch.no_grad():
    generated = model.generate(
        tokens,
        max_new_tokens=100,
        temperature=0.8,
        top_k=50
    )

# Decode and print
response = tokenizer.decode(generated[0].tolist())
print(response)
\`\`\`

## Training Pipeline

This model was trained using the DGX Spark optimized training pipeline:

1. **Pretraining:** Base language model on FineWeb-EDU dataset
2. **Midtraining:** Fine-tuned on conversational data (SmolTalk)
3. **SFT:** Supervised fine-tuning on curated conversations
4. **RL:** Reinforcement learning with GRPO

## Limitations

- This is a micro-model (1.9B parameters) - smaller than commercial LLMs
- May make factual errors or hallucinate
- Limited knowledge cutoff from training data
- Best suited for educational purposes and experimentation

## Citation

\`\`\`bibtex
@misc{${MODEL_NAME},
  author = {${AUTHOR_NAME}},
  title = {${MODEL_NAME}-${phase}},
  year = {2025},
  publisher = {HuggingFace},
  howpublished = {\url{https://huggingface.co/${AUTHOR_NAME}/${MODEL_NAME}-${phase}}}
}
\`\`\`

## Acknowledgments

- Andrej Karpathy for [NanoChat](https://github.com/karpathy/nanochat)
- NVIDIA DGX Spark platform
- FineWeb-EDU and SmolTalk datasets

## License

MIT License - Free to use for research and educational purposes
EOF

    # Create config.json (HuggingFace model card metadata)
    cat > "$model_dir/config.json" << EOF
{
  "model_type": "nanochat",
  "architecture": "gpt",
  "n_layer": 20,
  "n_head": 10,
  "n_kv_head": 10,
  "n_embd": 1280,
  "vocab_size": 65536,
  "sequence_len": 2048,
  "phase": "${phase}",
  "checkpoint_step": ${step_num},
  "torch_dtype": "bfloat16"
}
EOF

    # Create .gitattributes for LFS
    cat > "$model_dir/.gitattributes" << EOF
*.pt filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.safetensors filter=lfs diff=lfs merge=lfs -text
EOF

    echo -e "${GREEN}✓ Prepared $phase model in $model_dir${NC}"
    echo ""
}

# Prepare each training phase
prepare_model "pretrain" "$HOME/.cache/nanochat/base_checkpoints/d20" \
    "Base pretrained model on FineWeb-EDU dataset. This model has learned basic language patterns but hasn't been trained for conversation."

prepare_model "midtrain" "$HOME/.cache/nanochat/mid_checkpoints/d20" \
    "Midtrained model fine-tuned for conversational interactions. Trained on SmolTalk dataset with special tokens for multi-turn conversations."

prepare_model "sft" "$HOME/.cache/nanochat/sft_checkpoints/d20" \
    "Supervised fine-tuned model with safety training and high-quality conversation data. Optimized for better response quality and safety."

prepare_model "rl" "$HOME/.cache/nanochat/rl_checkpoints/d20" \
    "Final model with reinforcement learning (GRPO). Improved performance on math problems and reduced hallucinations."

# Create upload instructions
cat > "$OUTPUT_DIR/UPLOAD_INSTRUCTIONS.md" << EOF
# Uploading to HuggingFace

## Prerequisites

1. Install HuggingFace CLI:
   \`\`\`bash
   pip install huggingface_hub
   \`\`\`

2. Login to HuggingFace:
   \`\`\`bash
   huggingface-cli login
   \`\`\`
   
   Or set your token:
   \`\`\`bash
   export HF_TOKEN="your_token_here"
   \`\`\`

## Upload Each Model

### Option 1: Using HuggingFace CLI

\`\`\`bash
cd ${OUTPUT_DIR}

# Upload pretrain model
huggingface-cli repo create ${MODEL_NAME}-pretrain --type model
cd ${MODEL_NAME}-pretrain
git lfs install
git init
git add .
git commit -m "Initial commit: Pretrained NanoChat model"
git remote add origin https://huggingface.co/${AUTHOR_NAME}/${MODEL_NAME}-pretrain
git push -u origin main

# Repeat for other phases (midtrain, sft, rl)
\`\`\`

### Option 2: Using Python API

\`\`\`python
from huggingface_hub import HfApi, create_repo

api = HfApi()

# Create and upload pretrain model
repo_id = "${AUTHOR_NAME}/${MODEL_NAME}-pretrain"
create_repo(repo_id, repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path="${OUTPUT_DIR}/${MODEL_NAME}-pretrain",
    repo_id=repo_id,
    repo_type="model"
)

# Repeat for other phases
\`\`\`

### Option 3: Web Upload

1. Go to https://huggingface.co/new
2. Create a new model repository for each phase
3. Use "Add file" → "Upload files" to upload the contents of each directory

## Model Repository Structure

Each model will have:
- \`model_XXXXXX.pt\` - Model weights
- \`meta_XXXXXX.json\` - Model metadata
- \`tokenizer/\` - Tokenizer files
- \`README.md\` - Model card with details
- \`config.json\` - Model configuration
- \`.gitattributes\` - Git LFS configuration

## Recommended Repository Names

- \`${AUTHOR_NAME}/${MODEL_NAME}-pretrain\`
- \`${AUTHOR_NAME}/${MODEL_NAME}-midtrain\`
- \`${AUTHOR_NAME}/${MODEL_NAME}-sft\`
- \`${AUTHOR_NAME}/${MODEL_NAME}-rl\`

## Tips

- Use Git LFS for large files (already configured in .gitattributes)
- Update README.md with specific performance metrics if available
- Add example usage code in the README
- Tag your models appropriately for discoverability
EOF

echo -e "${BLUE}=== Preparation Complete ===${NC}"
echo ""
echo "Models prepared in: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Review the generated README.md files"
echo "  2. Read $OUTPUT_DIR/UPLOAD_INSTRUCTIONS.md"
echo "  3. Upload to HuggingFace using one of the methods described"
echo ""
echo -e "${GREEN}✓ Ready for upload!${NC}"
