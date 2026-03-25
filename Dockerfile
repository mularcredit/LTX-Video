FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime

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
    einops

COPY . .

CMD ["python", "handler.py"]
