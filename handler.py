import runpod
import torch
import os
import base64
import tempfile

# ── CUDA env vars ──────────────────────────────────────────────────────────────
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_CUDA_ARCH_LIST'] = '7.0 7.5 8.0 8.6 8.9 9.0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# ── Fix: cuDNN "No execution plans" on LTX attention ──────────────────────────
# Must be set BEFORE any model is loaded
torch.backends.cuda.enable_cudnn_sdp(False)   # kills the failing cuDNN backend
torch.backends.cuda.enable_flash_sdp(True)    # use Flash Attention if available
torch.backends.cuda.enable_math_sdp(True)     # fallback math kernel always on

# ── CUDA sanity check ─────────────────────────────────────────────────────────
print("Checking CUDA setup...")
if torch.cuda.is_available():
    print(f"✅ CUDA available | GPU: {torch.cuda.get_device_name(0)} | "
          f"CC: {torch.cuda.get_device_capability(0)} | "
          f"cuDNN: {torch.backends.cudnn.version()}")
    test = torch.tensor([1.0]).cuda()
    del test
    torch.cuda.empty_cache()
    print("✅ CUDA tensor test passed")
else:
    raise RuntimeError("❌ CUDA not available")

pipe = None

def load_pipeline():
    global pipe
    if pipe is None:
        from diffusers import LTXPipeline
        print("Loading LTX-Video pipeline...")
        pipe = LTXPipeline.from_pretrained(
            "Lightricks/LTX-Video",
            cache_dir="/runpod-volume/models",
            torch_dtype=torch.bfloat16,   # bfloat16 is safer than float16 on Ampere+
        )
        pipe.to("cuda")
        # Optional: shave VRAM with attention slicing if you hit OOM
        # pipe.enable_attention_slicing()
        print("✅ Pipeline loaded on GPU")
    return pipe

def handler(job):
    from diffusers.utils import export_to_video

    pipeline = load_pipeline()
    inp = job["input"]

    prompt           = inp.get("prompt", "")
    negative_prompt  = inp.get("negative_prompt", "worst quality, inconsistent motion, blurry")
    height           = inp.get("height", 480)
    width            = inp.get("width", 704)
    num_frames       = inp.get("num_frames", 121)
    num_steps        = inp.get("num_inference_steps", 50)
    seed             = inp.get("seed", 42)

    print(f"Generating: '{prompt[:60]}...' | {width}x{height} | {num_frames}f | {num_steps} steps")

    result = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_steps,
        generator=torch.Generator("cuda").manual_seed(seed),
    )

    video = result.frames[0]

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmp_path = f.name

    export_to_video(video, tmp_path, fps=24)

    with open(tmp_path, "rb") as vf:
        video_b64 = base64.b64encode(vf.read()).decode("utf-8")

    os.unlink(tmp_path)
    torch.cuda.empty_cache()

    return {"video_base64": video_b64}

runpod.serverless.start({"handler": handler})
