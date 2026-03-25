FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y git curl && \
    pip install uv

COPY . .

RUN uv sync --frozen
RUN pip install runpod huggingface_hub

CMD ["python", "handler.py"]
