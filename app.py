from fastapi import FastAPI
from pydantic import BaseModel
from handler import generate_video

app = FastAPI()

class VideoRequest(BaseModel):
    prompt: str
    negative_prompt: str = "worst quality, inconsistent motion, blurry"
    height: int = 480
    width: int = 704
    num_frames: int = 121
    num_inference_steps: int = 50
    seed: int = 42

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/generate")
def generate(request: VideoRequest):
    video_b64 = generate_video(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        height=request.height,
        width=request.width,
        num_frames=request.num_frames,
        num_inference_steps=request.num_inference_steps,
        seed=request.seed
    )
    return {"video_base64": video_b64}
