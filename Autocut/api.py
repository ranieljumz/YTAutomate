from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import os
import datetime             # Import the datetime module

from autocut8 import create_composite_video_ffmpeg

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
        
        # --- Modification Start ---
        # 1. Get the base path components from the constant template
        output_dir = os.path.dirname(request.output_path)
        base_filename_with_ext = os.path.basename(request.output_path)
        base_filename, extension = os.path.splitext(base_filename_with_ext) # ("output_", ".mp4")

        # 2. Get the current datetime and format it as MMDDHHMMSS
        now = datetime.datetime.now()
        timestamp = now.strftime("%m%d%H%M%S") # Example: 0419251407

        # 3. Construct the new filename
        new_filename = f"{base_filename}{timestamp}{extension}" # "output_0419251407.mp4"

        # 4. Construct the full modified output path
        modified_output_path = os.path.join(output_dir, new_filename)
        # Example: "/home/raniel/YTAutomate/Autocut/outputs/output_0419251407.mp4"

        # 5. (Optional but recommended) Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        logging.info(f"Generated dynamic output path: {modified_output_path}")
        # --- Modification End ---

        # Create video using the *modified* path
        create_composite_video_ffmpeg(
            image_paths=request.image_paths,
            audio_path=request.audio_path,
            output_path=modified_output_path # Pass the generated path
        )

        # Return success with the *modified* path
        return {"status": "success", "output_path": modified_output_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
      
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9168)