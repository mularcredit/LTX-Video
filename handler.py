import runpod
import torch
import os
import base64
import tempfile

# Set CUDA environment variables BEFORE any torch operations
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_CUDA_ARCH_LIST'] = '7.0;7.5;8.0;8.6;8.9'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

pipe = None

def check_gpu():
    """Debug function to check GPU availability and compute capability"""
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        
        # Test CUDA tensor creation
        try:
            test_tensor = torch.zeros(1).cuda()
            print("CUDA tensor test passed")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"CUDA test failed: {e}")
            raise
    else:
        print("WARNING: CUDA not available")
        raise RuntimeError("CUDA not available")

def load_pipeline():
    global pipe
    if pipe is None:
        from diffusers import LTXPipeline
        from diffusers.utils import export_to_video
        
        # Check GPU before loading
        check_gpu()
        
        print("Loading LTX-Video pipeline...")
        try:
            pipe = LTXPipeline.from_pretrained(
                "Lightricks/LTX-Video",
                cache_dir="/runpod-volume/models",
                torch_dtype=torch.bfloat16
            )
            print("Pipeline loaded, moving to CUDA...")
            pipe.to("cuda")
            
            # Verify pipeline is on CUDA
            print(f"Pipeline device: {next(pipe.parameters()).device}")
            print("Model loaded successfully!")
            
        except RuntimeError as e:
            if "no kernel image" in str(e):
                print("CUDA kernel error detected. Available GPUs:")
                os.system("nvidia-smi")
                print(f"Environment variables: TORCH_CUDA_ARCH_LIST={os.environ.get('TORCH_CUDA_ARCH_LIST')}")
            raise
    return pipe

def handler(job):
    try:
        from diffusers.utils import export_to_video
        
        # Force CUDA memory cleanup before processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        pipeline = load_pipeline()
        input_data = job["input"]

        prompt = input_data.get("prompt", "")
        negative_prompt = input_data.get("negative_prompt", "worst quality, inconsistent motion, blurry")
        height = input_data.get("height", 480)
        width = input_data.get("width", 704)
        num_frames = input_data.get("num_frames", 121)
        num_inference_steps = input_data.get("num_inference_steps", 50)
        seed = input_data.get("seed", 42)

        print(f"Generating video for prompt: {prompt[:50]}...")
        
        # Set generator and ensure CUDA device
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        video_output = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            generator=generator
        )
        
        video = video_output.frames[0]
        
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            export_to_video(video, f.name, fps=24)
            with open(f.name, "rb") as video_file:
                video_b64 = base64.b64encode(video_file.read()).decode("utf-8")
            os.unlink(f.name)
        
        # Clean up CUDA memory
        del video_output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {"video_base64": video_b64}
        
    except Exception as e:
        print(f"Error in handler: {e}")
        # Print CUDA memory info on error
        if torch.cuda.is_available():
            print(f"CUDA memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
            print(f"CUDA memory cached: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
        raise

# Start the serverless handler
runpod.serverless.start({"handler": handler})
