# LLM - Run Large Language Models on DGX Spark

**Deploy and chat with state-of-the-art open source models using vLLM**

This project helps you deploy powerful open source language models on the DGX Spark, taking advantage of its massive 128GB unified memory and Grace Blackwell GB10 GPU to run large models efficiently with vLLM.

## Overview

The DGX Spark's large memory footprint and powerful GB10 GPU makes it perfect for running large language models. This guide shows you how to use vLLM (a high-performance LLM inference engine) to deploy and interact with popular open source models optimized for the Grace Blackwell architecture.

### CUDA 13 Requirement

**Important:** The DGX Spark with Grace Blackwell GB10 uses **CUDA 13.0**, while most pre-built PyTorch and vLLM packages are compiled for CUDA 12. This means you cannot simply `pip install vllm` and expect it to work.

**Why this matters:**
- Standard PyPI packages will fail with `ImportError: libcudart.so.12: cannot open shared object file`
- You must install PyTorch compiled specifically for CUDA 13.0 from PyTorch's wheel repository
- **You must build vLLM from source** against CUDA 13.0 to ensure compatibility
- The installation order is critical: PyTorch (CUDA 13) → Build vLLM from source

This guide includes the correct installation steps to ensure compatibility with CUDA 13. If you've used vLLM on other systems before, note that the installation process requires building from source for DGX Spark.

## Quick Start

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install PyTorch for CUDA 13
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# 3. Set up CUDA 13 environment and clone vLLM
export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=/usr/local/cuda-13.0/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH}
pip install --upgrade pip setuptools wheel cmake ninja packaging
git clone https://github.com/vllm-project/vllm.git

# 4. Build vLLM with CUDA 13
cd vllm
# Modify pyproject.toml to ensure PyTorch 2.9.x with CUDA 13 is used
sed -i 's/"torch == 2.8.0",/"torch >= 2.9.0",/' pyproject.toml
export VLLM_BUILD_CUDA_EXT=1
pip install -e . --no-build-isolation
cd ..

# 5. Reinstall PyTorch with CUDA 13 and rebuild vLLM extensions
pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
cd vllm
pip install -e . --no-build-isolation
cd ..

# 6. Run a model with vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --dtype bfloat16

# 7. In another terminal (with venv activated):
source venv/bin/activate
python chat.py --model meta-llama/Llama-2-7b-chat-hf
```

## Table of Contents

- [Installation](#installation)
- [Running Models](#running-models)
- [Chat Interface](#chat-interface)
- [Supported Models](#supported-models)
- [Performance Tips](#performance-tips)
- [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- NVIDIA DGX Spark with Grace Blackwell GB10
- CUDA 13.0 or higher
- Python 3.10 or higher
- Build tools and development libraries:
  ```bash
  sudo apt-get update
  sudo apt-get install -y python3-dev build-essential cmake ninja-build git libnuma-dev
  ```

### Step 1: Create Virtual Environment

It's recommended to use a virtual environment to isolate dependencies:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# You should see (venv) in your prompt
```

**Note:** You'll need to activate the virtual environment each time you start a new terminal session:
```bash
source venv/bin/activate
```

### Step 2: Install PyTorch with CUDA 13

Install PyTorch compiled for CUDA 13.0 (required for DGX Spark):

```bash
# Install PyTorch with CUDA 13 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
```

### Step 3: Build vLLM from Source for CUDA 13

Since pre-built vLLM packages are compiled for CUDA 12, you need to build vLLM from source for CUDA 13:

```bash
# Set CUDA 13 environment variables
export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=/usr/local/cuda-13.0/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH}

# Install build dependencies
pip install --upgrade pip setuptools wheel
pip install cmake ninja packaging

# Clone vLLM repository
git clone https://github.com/vllm-project/vllm.git
cd vllm

# IMPORTANT: Modify pyproject.toml to use PyTorch with CUDA 13
# Edit the file to change the torch dependency to allow 2.9.x with CUDA 13
sed -i 's/"torch == 2.8.0",/"torch >= 2.9.0",/' pyproject.toml

# Build and install with CUDA extensions
export VLLM_BUILD_CUDA_EXT=1
pip install -e . --no-build-isolation

# Return to parent directory
cd ..
```

**Note:** Building vLLM from source may take 15-30 minutes depending on your system. The `sed` command modifies vLLM's `pyproject.toml` to use PyTorch 2.9.x instead of the hardcoded 2.8.0, ensuring compatibility with CUDA 13.

**Manual edit alternative:** If you prefer to edit manually, open `vllm/pyproject.toml` and change line 9 from:
```python
"torch == 2.8.0",
```
to:
```python
"torch >= 2.9.0",
```

**Common build issues:**
- If you see `python3-dev` or development header errors, install: `sudo apt-get install python3-dev`
- If you see CMake errors, ensure CMake 3.21+ is installed: `pip install --upgrade cmake`
- If build fails with memory errors, try: `export MAX_JOBS=4` before `pip install`

**Add to your shell profile** (optional but recommended):
```bash
# Add to ~/.bashrc or ~/.bash_profile
cat >> ~/.bashrc << 'EOF'
export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=/usr/local/cuda-13.0/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH}
EOF
source ~/.bashrc
```

### Step 4: Rebuild vLLM with PyTorch CUDA 13

After building vLLM, reinstall PyTorch with CUDA 13 and rebuild vLLM's extensions:

```bash
# Reinstall PyTorch with CUDA 13
pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Rebuild vLLM extensions with the correct PyTorch version
cd vllm
pip install -e . --no-build-isolation
cd ..

# Verify CUDA is available
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

You should see:
```
PyTorch: 2.9.1+cu130
CUDA available: True
```

**Why this is necessary:** By default, vLLM hardcodes PyTorch 2.8.0 for ARM64 platforms, which may not have the correct CUDA support. After modifying the dependency constraint and reinstalling PyTorch with CUDA 13, we rebuild vLLM's extensions to ensure they're compiled against the correct PyTorch version with CUDA 13 support.

### Step 5: Install Additional Tools

```bash
# Install OpenAI Python client for API access
pip install openai

# Install Gradio for web interface (optional)
pip install gradio

# Install Hugging Face CLI for model downloads
pip install huggingface_hub
```

### Step 6: Configure Hugging Face Access

Some models require authentication:

```bash
# Login to Hugging Face
huggingface-cli login

# Enter your token when prompted
# Get a token from: https://huggingface.co/settings/tokens
```

## Running Models

### Method 1: vLLM OpenAI-Compatible Server

The easiest way to run models is using vLLM's OpenAI-compatible API server:

```bash
# Start the server
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --port 8000
```

**Server Parameters:**
- `--model`: HuggingFace model ID or local path
- `--dtype`: Data type (bfloat16 recommended for GB10)
- `--max-model-len`: Maximum sequence length
- `--port`: API server port (default: 8000)
- `--tensor-parallel-size`: Number of GPUs (1 for DGX Spark)

The server provides OpenAI-compatible endpoints:
- `http://localhost:8000/v1/completions`
- `http://localhost:8000/v1/chat/completions`
- `http://localhost:8000/v1/models`

### Method 2: Direct Python API

For more control, use vLLM's Python API:

```python
from vllm import LLM, SamplingParams

# Initialize model
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    dtype="bfloat16",
    max_model_len=4096
)

# Configure sampling
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256
)

# Generate
prompts = ["Tell me about the NVIDIA DGX Spark"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

## Chat Interface

### CLI Chat Tool

Create a simple chat script (`chat.py`):

```python
#!/usr/bin/env python3
"""
Simple CLI chat interface for vLLM models.
Usage: python chat.py --model meta-llama/Llama-2-7b-chat-hf
"""
import argparse
from openai import OpenAI

def main():
    parser = argparse.ArgumentParser(description='Chat with vLLM models')
    parser.add_argument('--model', type=str, required=True,
                       help='Model name or path')
    parser.add_argument('--base-url', type=str, default='http://localhost:8000/v1',
                       help='vLLM server URL')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Sampling temperature')
    parser.add_argument('--max-tokens', type=int, default=256,
                       help='Maximum tokens to generate')
    args = parser.parse_args()
    
    # Initialize OpenAI client pointing to vLLM server
    client = OpenAI(
        api_key="EMPTY",  # vLLM doesn't require API key
        base_url=args.base_url
    )
    
    print("="*60)
    print("🤖 Chat with LLM (type 'exit' to quit)")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Settings: temperature={args.temperature}, max_tokens={args.max_tokens}\n")
    
    conversation = []
    
    while True:
        try:
            user_input = input("\n💬 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! 👋")
            break
        
        if not user_input:
            continue
        if user_input.lower() in ['exit', 'quit']:
            print("\nGoodbye! 👋")
            break
        
        # Add user message to conversation
        conversation.append({"role": "user", "content": user_input})
        
        # Get response from model
        response = client.chat.completions.create(
            model=args.model,
            messages=conversation,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        
        assistant_message = response.choices[0].message.content
        conversation.append({"role": "assistant", "content": assistant_message})
        
        print(f"\n🤖 Assistant: {assistant_message}")

if __name__ == "__main__":
    main()
```

Make it executable:

```bash
chmod +x chat.py
```

### Running the Chat Interface

1. **Start the vLLM server** (in one terminal):
```bash
# Activate virtual environment
source venv/bin/activate

# Start server
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --dtype bfloat16
```

2. **Start the chat client** (in another terminal):
```bash
# Activate virtual environment in new terminal
source venv/bin/activate

# Run chat interface
python chat.py --model meta-llama/Llama-2-7b-chat-hf
```

### Web Interface with Gradio

For a web-based chat interface, create `web_chat.py`:

```python
#!/usr/bin/env python3
"""
Web-based chat interface using Gradio.
Usage: python web_chat.py --model meta-llama/Llama-2-7b-chat-hf
"""
import argparse
import gradio as gr
from openai import OpenAI

def create_chat_interface(model_name, base_url="http://localhost:8000/v1"):
    client = OpenAI(api_key="EMPTY", base_url=base_url)
    
    def chat(message, history, temperature, max_tokens):
        # Convert Gradio history format to OpenAI format
        messages = []
        for h in history:
            messages.append({"role": "user", "content": h[0]})
            messages.append({"role": "assistant", "content": h[1]})
        messages.append({"role": "user", "content": message})
        
        # Get response
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    # Create Gradio interface
    with gr.Blocks(title=f"Chat with {model_name}") as demo:
        gr.Markdown(f"# 🤖 Chat with {model_name}")
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=500)
                msg = gr.Textbox(
                    placeholder="Type your message here...",
                    show_label=False
                )
                with gr.Row():
                    submit = gr.Button("Send")
                    clear = gr.Button("Clear")
            
            with gr.Column(scale=1):
                temperature = gr.Slider(
                    minimum=0.0, maximum=2.0, value=0.7, step=0.1,
                    label="Temperature"
                )
                max_tokens = gr.Slider(
                    minimum=64, maximum=2048, value=256, step=64,
                    label="Max Tokens"
                )
        
        def respond(message, chat_history, temp, tokens):
            bot_message = chat(message, chat_history, temp, tokens)
            chat_history.append((message, bot_message))
            return "", chat_history
        
        msg.submit(respond, [msg, chatbot, temperature, max_tokens], [msg, chatbot])
        submit.click(respond, [msg, chatbot, temperature, max_tokens], [msg, chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)
    
    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Web chat interface')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--share', action='store_true',
                       help='Create public sharing link')
    args = parser.parse_args()
    
    demo = create_chat_interface(args.model)
    demo.launch(server_port=args.port, share=args.share)
```

Run the web interface:

```bash
# Make sure venv is activated
source venv/bin/activate

# Start vLLM server first, then:
python web_chat.py --model meta-llama/Llama-2-7b-chat-hf
```

Access at: `http://localhost:7860`

## Supported Models

### Recommended Models for DGX Spark

| Model | Parameters | Memory | Use Case |
|-------|------------|--------|----------|
| **Llama-2-7b-chat** | 7B | ~14GB | General chat, fast responses |
| **Llama-2-13b-chat** | 13B | ~26GB | Better reasoning, still fast |
| **Llama-2-70b-chat** | 70B | ~140GB* | Maximum quality |
| **CodeLlama-13b** | 13B | ~26GB | Code generation |
| **CodeLlama-34b** | 34B | ~68GB | Advanced coding |
| **Mistral-7B-Instruct** | 7B | ~14GB | Efficient general purpose |
| **Mixtral-8x7B-Instruct** | 47B | ~94GB | High performance MoE |
| **Yi-34B-Chat** | 34B | ~68GB | Multilingual chat |
| **Qwen-72B-Chat** | 72B | ~144GB* | Advanced reasoning |

*May require quantization or optimized memory settings

### HuggingFace Model IDs

```bash
# Llama 2 models
meta-llama/Llama-2-7b-chat-hf
meta-llama/Llama-2-13b-chat-hf
meta-llama/Llama-2-70b-chat-hf

# Code Llama
codellama/CodeLlama-13b-Instruct-hf
codellama/CodeLlama-34b-Instruct-hf

# Mistral
mistralai/Mistral-7B-Instruct-v0.2
mistralai/Mixtral-8x7B-Instruct-v0.1

# Other popular models
01-ai/Yi-34B-Chat
Qwen/Qwen-72B-Chat
```

## Performance Tips

### Memory Optimization

For models that exceed available memory:

```bash
# Use 8-bit quantization
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-70b-chat-hf \
    --dtype bfloat16 \
    --quantization awq \
    --max-model-len 2048
```

### Enable Flash Attention

vLLM automatically uses flash attention when available:

```bash
export VLLM_FLASH_ATTN_ENABLE=1
```

### GPU Memory Utilization

Control GPU memory usage:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-13b-chat-hf \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9  # Use 90% of available memory
```

### Optimize for Throughput

```bash
# Increase batch size for better throughput
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --dtype bfloat16 \
    --max-num-seqs 256  # Process more sequences in parallel
```

## Troubleshooting

### ModuleNotFoundError: No module named 'vllm.vllm_flash_attn'

If you get this error or `Segmentation fault (core dumped)`:

```bash
# Check if PyTorch has CUDA support
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

If it shows `CUDA available: False` or PyTorch version without `+cu130`:

```bash
# Fix: Reinstall PyTorch with CUDA 13 and rebuild vLLM extensions
source venv/bin/activate
pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Rebuild vLLM extensions with correct PyTorch
cd vllm
pip install -e . --no-build-isolation
cd ..

# Verify fix
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Why this happens:** vLLM's installation may downgrade PyTorch to a CPU-only version. When you reinstall PyTorch, vLLM's compiled CUDA extensions become incompatible and must be rebuilt.

### CUDA Version Issues

If you encounter `ImportError: libcudart.so.12: cannot open shared object file`, this means vLLM is looking for CUDA 12 but you have CUDA 13:

```bash
# Check CUDA version
nvcc --version

# Solution: Rebuild vLLM from source for CUDA 13
cd ~/Code/dgx-spark  # or wherever your project is
source venv/bin/activate

# Uninstall any existing vLLM
pip uninstall -y vllm

# Set CUDA 13 environment variables
export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=/usr/local/cuda-13.0/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH}

# Clone and build vLLM
git clone https://github.com/vllm-project/vllm.git
cd vllm
export VLLM_BUILD_CUDA_EXT=1
pip install -e .
```

**Why this happens:** Pre-built vLLM packages from PyPI are compiled against CUDA 12. Building from source ensures vLLM is compiled specifically for your CUDA 13.0 installation.

### Out of Memory Errors

Try these solutions:

1. **Reduce sequence length:**
```bash
--max-model-len 2048
```

2. **Use smaller batch size:**
```bash
--max-num-seqs 32
```

3. **Enable quantization:**
```bash
--quantization awq
```

### Model Download Issues

If model downloads are slow or fail:

```bash
# Set HuggingFace cache directory
export HF_HOME=/path/to/large/storage/.cache/huggingface

# Use mirror (if available)
export HF_ENDPOINT=https://hf-mirror.com
```

### Connection Issues

If chat client can't connect to server:

```bash
# Check if server is running
curl http://localhost:8000/v1/models

# Check firewall settings
sudo ufw allow 8000
```

## Advanced Usage

### Batch Processing

Process multiple prompts efficiently:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-chat-hf", dtype="bfloat16")
sampling_params = SamplingParams(temperature=0.7, max_tokens=256)

prompts = [
    "Explain quantum computing",
    "Write a Python function to sort a list",
    "What is the capital of France?"
]

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Output: {output.outputs[0].text}\n")
```

### Custom System Prompts

For chat models, use proper formatting:

```python
def format_llama2_prompt(system_prompt, user_message):
    return f"""<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_message} [/INST]"""

prompt = format_llama2_prompt(
    "You are a helpful AI assistant specialized in programming.",
    "How do I reverse a string in Python?"
)
```

### Multi-Turn Conversations

```python
conversation = """<s>[INST] What is Python? [/INST] Python is a high-level programming language...</s>
<s>[INST] How do I install it? [/INST]"""

# Continue the conversation
outputs = llm.generate([conversation], sampling_params)
```

## Resources

- **vLLM Documentation**: https://docs.vllm.ai/
- **HuggingFace Models**: https://huggingface.co/models
- **TinyLLM Project**: https://github.com/jasonacox/TinyLLM
- **DGX Spark Guide**: https://github.com/jasonacox/dgx-spark

## Contributing

Contributions welcome! Please submit issues and pull requests.

## License

MIT License - see LICENSE file for details.

## Credits

- Based on [TinyLLM](https://github.com/jasonacox/TinyLLM) by Jason Cox
- Powered by [vLLM](https://github.com/vllm-project/vllm)
- Optimized for NVIDIA DGX Spark (Grace Blackwell GB10)