import runpod
import torch
import os
import base64
import tempfile

pipe = None

def load_pipeline():
    global pipe
    if pipe is None:
        from diffusers import LTXPipeline
        from diffusers.utils import export_to_video
        print("Loading LTX-Video pipeline...")
        pipe = LTXPipeline.from_pretrained(
            "Lightricks/LTX-Video",
            cache_dir="/runpod-volume/models",
            torch_dtype=torch.bfloat16
        )
        pipe.to("cuda")
        print("Model loaded!")
    return pipe

def handler(job):
    from diffusers.utils import export_to_video

    pipeline = load_pipeline()
    input_data = job["input"]

    prompt = input_data.get("prompt", "")
    negative_prompt = input_data.get("negative_prompt", "worst quality, inconsistent motion, blurry")
    height = input_data.get("height", 480)
    width = input_data.get("width", 704)
    num_frames = input_data.get("num_frames", 121)
    num_inference_steps = input_data.get("num_inference_steps", 50)
    seed = input_data.get("seed", 42)

    video = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator("cuda").manual_seed(seed)
    ).frames[0]

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        export_to_video(video, f.name, fps=24)
        with open(f.name, "rb") as video_file:
            video_b64 = base64.b64encode(video_file.read()).decode("utf-8")
        os.unlink(f.name)

    return {"video_base64": video_b64}

runpod.serverless.start({"handler": handler})
