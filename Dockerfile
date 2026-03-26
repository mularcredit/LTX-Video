FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime

ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0"
ENV CUDA_LAUNCH_BLOCKING=1
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# DO NOT reinstall torch — base image already has correct cu124 wheels
RUN pip install --no-cache-dir \
    runpod>=1.7.0 \
    huggingface_hub \
    "diffusers>=0.31.0" \
    "transformers>=4.44.0" \
    "accelerate>=0.33.0" \
    sentencepiece \
    imageio \
    imageio-ffmpeg \
    opencv-python-headless \
    einops \
    tiktoken \
    protobuf

COPY . .

CMD ["python", "-u", "handler.py"]
