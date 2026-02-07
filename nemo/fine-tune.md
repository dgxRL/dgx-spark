# source: https://build.nvidia.com/spark/pytorch-fine-tune/instructions


# pull docker image
docker pull nvcr.io/nvidia/pytorch:25.11-py3

# run docker

docker run \
  --ipc=host \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v ${PWD}:/workspace -w /workspace \
  --gpus all \
  --ulimit memlock=-1 \
  -it --ulimit stack=67108864 \
  --entrypoint /usr/bin/bash \
  --rm nvcr.io/nvidia/pytorch:25.11-py3



Copy
# Clone the repository
git clone https://github.com/NVIDIA-NeMo/Automodel.git

# Navigate to the repository
cd Automodel

uv sync

---
# Initialize virtual environment
uv venv --system-site-packages

uv pip install scikit-build-core setuptools wheel

# Install packages with uv
uv sync --inexact --frozen --all-extras \
  --no-install-package torch \
  --no-install-package torchvision \
  --no-install-package triton \
  --no-install-package nvidia-cublas-cu12 \
  --no-install-package nvidia-cuda-cupti-cu12 \
  --no-install-package nvidia-cuda-nvrtc-cu12 \
  --no-install-package nvidia-cuda-runtime-cu12 \
  --no-install-package nvidia-cudnn-cu12 \
  --no-install-package nvidia-cufft-cu12 \
  --no-install-package nvidia-cufile-cu12 \
  --no-install-package nvidia-curand-cu12 \
  --no-install-package nvidia-cusolver-cu12 \
  --no-install-package nvidia-cusparse-cu12 \
  --no-install-package nvidia-cusparselt-cu12 \
  --no-install-package nvidia-nccl-cu12 \
  --no-install-package transformer-engine \
  --no-install-package nvidia-modelopt \
  --no-install-package nvidia-modelopt-core \
  --no-install-package flash-attn \
  --no-install-package transformer-engine-cu12 \
  --no-install-package transformer-engine-torch

# Install bitsandbytes
CMAKE_ARGS="-DCOMPUTE_BACKEND=cuda -DCOMPUTE_CAPABILITY=80;86;87;89;90" \
CMAKE_BUILD_PARALLEL_LEVEL=8 \
uv pip install --no-deps git+https://github.com/bitsandbytes-foundation/bitsandbytes.git@50be19c39698e038a1604daf3e1b939c9ac1c342

# verify
uv run --frozen --no-sync python -c "import nemo_automodel; print('✅ NeMo AutoModel ready')"

# Check available examples
ls -la examples/


# Run basic LLM fine-tuning example
export HF_TOKEN=

# Add this to your ~/.bashrc or Dockerfile
export UV_HTTP_TIMEOUT=300
export UV_HTTP_RETRIES=5

uv pip install --force-reinstall   torch==2.10.0+cu130   torchvision==0.25.0+cu130   --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match

uv run --frozen --no-sync \
examples/llm_finetune/finetune.py \
-c examples/llm_finetune/qwen/qwen3_8b_squad_spark.yaml \
--model.pretrained_model_name_or_path Qwen/Qwen3-8B \
--step_scheduler.local_batch_size 1 \
--step_scheduler.max_steps 20 \
--packed_sequence.packed_sequence_size 1024 \
--model.attn_implementation="sdpa"


uv run --frozen --no-sync \
examples/llm_finetune/finetune.py \
-c examples/llm_finetune/qwen/qwen3_8b_squad_spark.yaml \
--model.pretrained_model_name_or_path Qwen/Qwen3-8B \
--step_scheduler.local_batch_size 1 \
--step_scheduler.max_steps 20 \
++packed_sequence.packed_sequence_size=0 \
--model.attn_implementation="sdpa"


# ----------------
# python version fix
# 1️⃣ Remove old venv if exists
rm -rf /workspace/Automodel/.venv

# 2️⃣ Install prerequisites
apt update && apt install -y software-properties-common curl ca-certificates git build-essential

# 3️⃣ Add Deadsnakes PPA for Python 3.11
add-apt-repository -y ppa:deadsnakes/ppa
apt update

# 4️⃣ Install Python 3.11 and dev tools
apt install -y python3.11 python3.11-venv python3.11-dev python3-pip

# 5️⃣ Make Python 3.11 the default
update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
update-alternatives --set python /usr/bin/python3.11

# 6️⃣ Verify Python
python --version

# 7️⃣ Create a fresh virtual environment
cd /workspace/Automodel
python -m venv .venv
source .venv/bin/activate

# 8️⃣ Upgrade pip
pip install --upgrade pip

# 9️⃣ Install PyTorch + CUDA 12.1
pip install --force-reinstall \
  torch==2.2.2+cu121 \
  torchvision==0.17.2+cu121 \
  torchaudio==2.2.2+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

# 10️⃣ Install Transformers & NeMo
pip install transformers==4.41.0
pip install nemo-toolkit[llm]

# 11️⃣ Disable FlashAttention for stability on DGX
export DISABLE_FLASH_ATTN=1
export TORCHDYNAMO_DISABLE=1

# 12️⃣ Verify installation
python - <<EOF
import sys
print("Python:", sys.version)
import torch
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
from transformers import PreTrainedModel
import nemo_automodel
print("NeMo automodel imports OK")
EOF