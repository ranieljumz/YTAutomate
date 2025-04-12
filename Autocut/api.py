from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import os

from autocut6 import create_composite_video_ffmpeg

app = FastAPI()

class VideoRequest(BaseModel):
    image_paths: list[str]
    audio_path: str
    output_path: str
	
	
	
@app.post("/create_video")
async def create_video(request: VideoRequest):
    try:
        logging.info(f'Generating speechs for {request.audio_path}')
        # Validate inputs
        if not request.image_paths:
            raise HTTPException(status_code=400, detail="At least one image path is required")
        if not os.path.exists(request.audio_path):
            raise HTTPException(status_code=400, detail="Audio file not found")
        
        # Create video
        create_composite_video_ffmpeg(
            image_paths=request.image_paths,
            audio_path=request.audio_path,
            output_path=request.output_path
        )
        
        return {"status": "success", "output_path": request.output_path}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
      
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9168)