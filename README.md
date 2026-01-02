# PD Measurement Backend

FastAPI backend for Pupillary Distance (PD) measurement using computer vision and deep learning.

## Features

- 🎯 **Face Detection** - MediaPipe Face Landmarker for iris detection
- 💳 **Card Detection** - Forehead ROI + SAM3 segmentation for calibration
- 📐 **3D Photogrammetry** - Accurate PD calculation using homography
- 🔧 **REST API** - FastAPI with CORS support

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Installation

**Using uv:**

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync
```

### Environment Setup

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your Roboflow API key
nano .env
```

Required environment variables:

- `ROBOFLOW_API_KEY` - Your Roboflow API key for SAM3 segmentation

### Running the Server

```bash
# Using uv
uv run python -m main

# Or with activated venv
source .venv/bin/activate
python -m main
```

Server starts at `http://localhost:8000`

API documentation: `http://localhost:8000/docs`

## API Endpoints

### POST `/api/measure-pd`

Measure PD from an image.

**Request:**

```json
{
  "image": "base64_encoded_image_data"
}
```

**Response:**

```json
{
  "success": true,
  "pd_mm": 63.5,
  "confidence": 0.85,
  "debug_dir": "inputs/20231231_120000",
  "details": {
    "raw_pd_px": 127.3,
    "scale_factor": 0.499,
    "camera_distance_mm": 400,
    "head_pose": { "yaw": 0.5, "pitch": 2.1, "roll": 1.0 }
  }
}
```

### GET `/api/debug-images/{session_id}`

Get debug images for a measurement session.

## Testing

Run the demo script on a test image:

```bash
# Full PD measurement
uv run python demo.py path/to/image.jpg

# Card detection only
uv run python demo.py path/to/image.jpg --card-only

# Without forehead ROI detection
uv run python demo.py path/to/image.jpg --no-forehead
```

Debug output will be saved to `debug/<timestamp>/`

## Project Structure

```
backend/
├── main.py              # FastAPI app entry point
├── pd_service.py        # PD measurement service
├── pd_engine/           # Core measurement engine
│   ├── core.py          # Main PDMeasurement class
│   ├── calibration.py   # Card detection & calibration
│   ├── measurement.py   # Iris detection
│   ├── photogrammetry.py # 3D calculations
│   ├── forehead_detection.py
│   ├── sam3_segmentation.py
│   ├── corrections.py
│   └── utils.py
├── demo.py              # CLI testing script
├── pyproject.toml       # Project dependencies
└── .env                 # Environment variables
```

## Development

### Adding Dependencies

```bash
uv add package_name
```

### CORS Configuration

By default, all origins are allowed for development. For production, update `main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-domain.com"],
    # ...
)
```

## License

rimloo
