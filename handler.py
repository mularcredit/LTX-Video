import runpod
import torch
import os
from huggingface_hub import hf_hub_download

# Download model on first run
def download_models():
    os.makedirs("/app/models", exist_ok=True)
    hf_hub_download(
        repo_id="Lightricks/LTX-Video",
        filename="ltx-2.3-22b-distilled.safetensors",
        local_dir="/app/models"
    )

def handler(job):
    input_data = job["input"]
    prompt = input_data.get("prompt", "")
    height = input_data.get("height", 512)
    width = input_data.get("width", 768)
    num_frames = input_data.get("num_frames", 49)
    seed = input_data.get("seed", 42)

    from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline

    pipe = LTXVideoPipeline.from_pretrained(
        "/app/models",
        torch_dtype=torch.bfloat16
    ).to("cuda")

    output = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        generator=torch.Generator().manual_seed(seed)
    )

    # Save and return video as base64
    import base64, tempfile
    from diffusers.utils import export_to_video

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        export_to_video(output.frames[0], f.name, fps=24)
        with open(f.name, "rb") as video_file:
            video_b64 = base64.b64encode(video_file.read()).decode("utf-8")

    return {"video_base64": video_b64}

# Download models before starting
download_models()
runpod.serverless.start({"handler": handler})
