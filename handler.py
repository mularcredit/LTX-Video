import runpod
import torch
import os
import base64
import tempfile

# Force CUDA environment variables
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_CUDA_ARCH_LIST'] = '7.0 7.5 8.0 8.6 8.9 9.0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['TORCH_FORCE_CUDA'] = '1'

# Verify CUDA is working before loading model
print("Checking CUDA setup...")
if torch.cuda.is_available():
    print(f"✅ CUDA is available")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Compute Capability: {torch.cuda.get_device_capability(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    
    # Test CUDA tensor creation
    try:
        test = torch.tensor([1.0]).cuda()
        print("✅ CUDA tensor test passed")
        del test
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ CUDA test failed: {e}")
        raise
else:
    print("❌ CUDA is NOT available")
    raise RuntimeError("CUDA not available")

pipe = None

def load_pipeline():
    global pipe
    if pipe is None:
        from diffusers import LTXPipeline
        print("Loading LTX-Video pipeline...")
        pipe = LTXPipeline.from_pretrained(
            "Lightricks/LTX-Video",
            cache_dir="/runpod-volume/models",
            torch_dtype=torch.bfloat16
        )
        print("Moving pipeline to CUDA...")
        pipe.to("cuda")
        print("✅ Model loaded on GPU!")
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

    print(f"Generating video for: {prompt[:50]}...")
    
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

    # Clean up
    torch.cuda.empty_cache()
    
    return {"video_base64": video_b64}

runpod.serverless.start({"handler": handler})
