"""
FastAPI Backend for PD Measurement Application.
Provides endpoints for face detection guidance and PD calculation.
"""

import io
import os
import base64
import hmac
import hashlib
import time
from typing import Optional, List

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

from pd_service import get_pd_service

# Security configuration from environment
TOKEN_SECRET = os.getenv("TOKEN_SECRET", "dev-secret-change-in-production")
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "*")
TOKEN_EXPIRY_SECONDS = 60

# Rate limiting configuration
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "20"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds

# Simple in-memory rate limiter (use Redis in production for multi-instance)
rate_limit_store: dict = {}


def validate_token(token: str) -> tuple[bool, str]:
    """Validate HMAC-signed token."""
    if not token:
        return False, "Missing token"
    
    parts = token.split(".")
    if len(parts) != 3:
        return False, "Invalid token format"
    
    timestamp_str, nonce, provided_signature = parts
    
    try:
        timestamp = int(timestamp_str)
    except ValueError:
        return False, "Invalid timestamp"
    
    # Check expiry
    now = int(time.time())
    if now - timestamp > TOKEN_EXPIRY_SECONDS:
        return False, "Token expired"
    
    # Verify signature
    payload = f"{timestamp_str}.{nonce}"
    expected_signature = hmac.new(
        TOKEN_SECRET.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()
    
    if not hmac.compare_digest(provided_signature, expected_signature):
        return False, "Invalid signature"
    
    return True, ""


def check_rate_limit(client_ip: str) -> tuple[bool, int]:
    """Check if client IP has exceeded rate limit. Returns (allowed, remaining)."""
    now = time.time()
    
    # Clean old entries
    if client_ip in rate_limit_store:
        rate_limit_store[client_ip] = [
            t for t in rate_limit_store[client_ip]
            if now - t < RATE_LIMIT_WINDOW
        ]
    else:
        rate_limit_store[client_ip] = []
    
    # Check limit
    request_count = len(rate_limit_store[client_ip])
    if request_count >= RATE_LIMIT_REQUESTS:
        return False, 0
    
    # Add new request
    rate_limit_store[client_ip].append(now)
    return True, RATE_LIMIT_REQUESTS - request_count - 1


# Create FastAPI app
app = FastAPI(
    title="PD Measurement API",
    description="API for measuring Pupillary Distance using camera and reference card",
    version="1.0.0"
)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Middleware to validate tokens, origin, and rate limits on /api/* routes."""
    
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        
        # Skip validation for non-API routes (health check, static files, docs)
        if not path.startswith("/api/"):
            return await call_next(request)
        
        # Skip validation for OPTIONS requests (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)
        
        # Get client IP (handle proxy headers)
        client_ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        if not client_ip:
            client_ip = request.client.host if request.client else "unknown"
        
        # 1. Rate limiting
        allowed, remaining = check_rate_limit(client_ip)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
                headers={"Retry-After": str(RATE_LIMIT_WINDOW)}
            )
        
        # 2. Origin validation (if ALLOWED_ORIGIN is set)
        if ALLOWED_ORIGIN != "*":
            origin = request.headers.get("Origin", "")
            # Allow requests without Origin header (direct API calls from same server)
            if origin and origin != ALLOWED_ORIGIN:
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Origin not allowed"}
                )
        
        # 3. Token validation (if TOKEN_SECRET is set)
        if TOKEN_SECRET and TOKEN_SECRET != "dev-secret-change-in-production":
            token = request.headers.get("X-Request-Token", "")
            valid, error = validate_token(token)
            if not valid:
                return JSONResponse(
                    status_code=401,
                    content={"detail": f"Authentication failed: {error}"}
                )
        
        response = await call_next(request)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response


# Add security middleware (before CORS)
app.add_middleware(SecurityMiddleware)

# Add CORS middleware for frontend
# Use specific origin in production for security
allowed_origins = [ALLOWED_ORIGIN] if ALLOWED_ORIGIN != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
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
