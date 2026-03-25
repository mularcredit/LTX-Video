FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime


ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
ENV CUDA_LAUNCH_BLOCKING=1
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

WORKDIR /app

RUN apt-get update && apt-get install -y git curl ffmpeg

RUN pip install --no-cache-dir \
    runpod \
    huggingface_hub \
    torch \
    torchvision \
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

CMD ["python", "handler.py"]
