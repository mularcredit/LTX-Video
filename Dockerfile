FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime

# Set CUDA architectures for ALL common RunPod GPUs
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0"
ENV CUDA_LAUNCH_BLOCKING=1
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV TORCH_FORCE_CUDA=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install PyTorch with CUDA support
RUN pip install --upgrade pip

# Force reinstall PyTorch with correct CUDA architecture
RUN pip uninstall -y torch torchvision torchaudio || true
RUN pip install --no-cache-dir \
    torch==2.5.0 \
    torchvision==0.20.0 \
    torchaudio==2.5.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
RUN pip install --no-cache-dir \
    runpod \
    huggingface_hub \
    diffusers \
    transformers \
    accelerate \
    sentencepiece \
    imageio \
    imageio-ffmpeg \
    opencv-python-headless \
    einops \
    tiktoken \
    protobuf

COPY . .

# Verify CUDA setup on build
RUN python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'PyTorch Version: {torch.__version__}')"

CMD ["python", "-u", "handler.py"]
