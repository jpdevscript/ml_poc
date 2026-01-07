"""
FastAPI Backend for PD Measurement Application.
Provides endpoints for face detection guidance and PD calculation.
"""

import io
import os
import base64
from typing import Optional, List

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pd_service import get_pd_service

# Create FastAPI app
app = FastAPI(
    title="PD Measurement API",
    description="API for measuring Pupillary Distance using camera and reference card",
    version="1.0.0"
)

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve debug images as static files
inputs_dir = 'inputs'
os.makedirs(inputs_dir, exist_ok=True)
app.mount("/debug-images", StaticFiles(directory=inputs_dir), name="debug-images")


class Base64ImageRequest(BaseModel):
    """Request with base64 encoded image."""
    image: str


def decode_base64_image(base64_str: str) -> np.ndarray:
    """Decode base64 image string to numpy array."""
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    
    img_bytes = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Failed to decode image")
    
    return image


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "PD Measurement API"}


@app.post("/api/detect-face")
async def detect_face(request: Base64ImageRequest):
    """Detect face and return guidance for positioning."""
    try:
        image = decode_base64_image(request.image)
        service = get_pd_service()
        result = service.detect_face(image)
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.post("/api/measure-pd")
async def measure_pd(request: Base64ImageRequest):
    """Measure PD from captured image with card on forehead."""
    try:
        image = decode_base64_image(request.image)
        service = get_pd_service()
        result = service.measure_pd(image)
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Measurement failed: {str(e)}")


class MultiFrameImageRequest(BaseModel):
    """Request with multiple base64 encoded images for statistical averaging."""
    images: List[str]
    method: Optional[str] = None  # 'card' (default) or 'iris'


@app.post("/api/measure-pd-multi")
async def measure_pd_multi(request: MultiFrameImageRequest):
    """Measure PD from multiple frames with statistical averaging."""
    try:
        if len(request.images) < 2:
            raise HTTPException(status_code=400, detail="At least 2 images required for multi-frame averaging")
        
        if len(request.images) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 images allowed")
        
        # Decode all images
        images = []
        for i, img_str in enumerate(request.images):
            try:
                image = decode_base64_image(img_str)
                images.append(image)
            except Exception as e:
                print(f"[API] Failed to decode image {i}: {e}")
                # Skip invalid images
                continue
        
        if len(images) < 2:
            raise HTTPException(status_code=400, detail="Less than 2 valid images after decoding")
        
        service = get_pd_service()
        result = service.measure_pd_multi(images, method=request.method)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-frame measurement failed: {str(e)}")


@app.get("/api/debug-images/{session_id}")
async def list_debug_images(session_id: str):
    """List all debug images for a measurement session, organized by frame."""
    session_dir = os.path.join(inputs_dir, session_id)
    
    if not os.path.exists(session_dir):
        raise HTTPException(status_code=404, detail="Session not found")
    
    frames = []
    root_images = []
    
    for item in sorted(os.listdir(session_dir)):
        item_path = os.path.join(session_dir, item)
        
        if os.path.isdir(item_path) and item.startswith('frame_'):
            # This is a frame subdirectory
            frame_images = []
            for filename in sorted(os.listdir(item_path)):
                if filename.endswith(('.jpg', '.png', '.jpeg')):
                    frame_images.append({
                        'name': filename,
                        'url': f"/debug-images/{session_id}/{item}/{filename}"
                    })
            frames.append({
                'frame_id': item,
                'images': frame_images
            })
        elif item.endswith(('.jpg', '.png', '.jpeg')):
            # Root level images
            root_images.append({
                'name': item,
                'url': f"/debug-images/{session_id}/{item}"
            })
    
    # Flatten all images for backward compatibility
    all_images = root_images.copy()
    for frame in frames:
        all_images.extend(frame['images'])
    
    return {
        "session_id": session_id, 
        "images": all_images,
        "frames": frames,
        "root_images": root_images
    }


@app.post("/api/measure-pd-file")
async def measure_pd_file(file: UploadFile = File(...)):
    """Measure PD from uploaded file."""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        service = get_pd_service()
        result = service.measure_pd(image)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Measurement failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("  PD Measurement API Server")
    print("="*60)
    print("\n  Starting server on http://0.0.0.0:8000")
    print("  API docs: http://localhost:8000/docs")
    print("\n" + "="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
