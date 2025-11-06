# Uploading NanoChat Models to HuggingFace

This guide explains how to organize and upload your trained NanoChat models to HuggingFace Hub.

## Quick Start

### Step 1: Prepare Models for Upload

```bash
# Basic usage (uses default name: nanochat-dgx-spark)
./hf_prepare.sh --author your-hf-username

# Custom model name
./hf_prepare.sh --author your-hf-username --name my-custom-nanochat

# Custom output directory
./hf_prepare.sh --author your-hf-username --output ~/my-models
```

This will create a `hf_models` directory with properly structured models ready for upload.

### Step 2: Upload to HuggingFace

#### Option A: Automated Upload (Recommended)

```bash
# Install required package
pip install huggingface_hub

# Dry run (see what would be uploaded)
python upload_to_hf.py --username your-hf-username --dry-run

# Actual upload
python upload_to_hf.py --username your-hf-username

# Upload specific phases only
python upload_to_hf.py --username your-hf-username --phases midtrain sft
```

#### Option B: Manual Upload via Web

1. Go to https://huggingface.co/new
2. Create a new model repository (e.g., `nanochat-dgx-spark-midtrain`)
3. Clone the repository:
   ```bash
   git clone https://huggingface.co/your-username/nanochat-dgx-spark-midtrain
   cd nanochat-dgx-spark-midtrain
   ```
4. Copy your prepared model files:
   ```bash
   cp -r ../hf_models/nanochat-dgx-spark-midtrain/* .
   ```
5. Install Git LFS and commit:
   ```bash
   git lfs install
   git add .
   git commit -m "Upload NanoChat midtrain model"
   git push
   ```

#### Option C: Manual Upload via CLI

```bash
# Install HuggingFace CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Create and upload repository
cd hf_models/nanochat-dgx-spark-midtrain
huggingface-cli repo create nanochat-dgx-spark-midtrain --type model
git init
git lfs install
git add .
git commit -m "Initial commit"
git remote add origin https://huggingface.co/your-username/nanochat-dgx-spark-midtrain
git push -u origin main
```

## Model Structure

Each prepared model includes:

```
nanochat-dgx-spark-{phase}/
├── model_XXXXXX.pt         # Model weights (LFS)
├── meta_XXXXXX.json        # Model metadata
├── tokenizer/              # Tokenizer files
│   ├── tokenizer.bin
│   └── ...
├── README.md               # Model card
├── config.json             # HuggingFace config
└── .gitattributes          # Git LFS configuration
```

## Training Phases

Your models are organized by training phase:

1. **pretrain** - Base model trained on FineWeb-EDU
   - Basic language understanding
   - Token prediction capabilities
   - ~1.9B parameters

2. **midtrain** - Fine-tuned for conversations
   - Multi-turn conversation support
   - Special tokens for user/assistant format
   - Trained on SmolTalk dataset

3. **sft** - Supervised fine-tuning
   - High-quality conversation data
   - Safety training
   - Improved response quality

4. **rl** - Reinforcement learning
   - GRPO optimization
   - Reduced hallucinations
   - Improved math performance

## Model Card Information

Each model includes a comprehensive README.md with:

- Model architecture and parameters
- Training details and hardware
- Usage examples
- Limitations and caveats
- Citations and acknowledgments

You can customize these by editing the README.md files in the prepared directories.

## Tips and Best Practices

1. **Use Git LFS** - Already configured in .gitattributes for .pt files
2. **Update README** - Add your specific metrics and examples
3. **Tag appropriately** - Use tags for discoverability
4. **Include examples** - Show how to load and use your model
5. **Document limitations** - Be clear about model capabilities

## Repository Naming

Recommended naming convention:
- `your-username/nanochat-dgx-spark-pretrain`
- `your-username/nanochat-dgx-spark-midtrain`
- `your-username/nanochat-dgx-spark-sft`
- `your-username/nanochat-dgx-spark-rl`

Or shorter:
- `your-username/nanochat-pretrain`
- `your-username/nanochat-mid`
- `your-username/nanochat-sft`
- `your-username/nanochat-rl`

## Troubleshooting

### Large File Upload Issues
```bash
# Ensure Git LFS is installed
git lfs install

# Track large files
git lfs track "*.pt"
git lfs track "*.pth"
```

### Authentication Issues
```bash
# Set your token
export HF_TOKEN="your_token_here"

# Or login interactively
huggingface-cli login
```

### Upload Speed
- HuggingFace uploads can be slow for large files
- Consider uploading during off-peak hours
- Use `--resume` flag if upload is interrupted

## Example: Complete Upload Workflow

```bash
```bash
# Step 1: Prepare models
./hf_prepare.sh --author jasonacox --name nanochat-dgx-spark

# 2. Review generated files
ls -lh hf_models/

# 3. Test with dry run
python upload_to_hf.py --username jasonacox --dry-run

# 4. Upload all models
python upload_to_hf.py --username jasonacox

# 5. Verify on HuggingFace
# Visit: https://huggingface.co/jasonacox
```

## Support

- HuggingFace Documentation: https://huggingface.co/docs/hub
- Git LFS Guide: https://git-lfs.github.com/
- NanoChat Repository: https://github.com/karpathy/nanochat
