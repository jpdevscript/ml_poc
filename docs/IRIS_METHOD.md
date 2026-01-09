# Iris PD Measurement - Backend

## Overview

Python backend for pupillary distance measurement using MediaPipe iris detection.

## Key Files

| File                          | Description                            |
| ----------------------------- | -------------------------------------- |
| `pd_engine/iris_pd_engine.py` | Main iris-based PD measurement engine  |
| `pd_engine/measurement.py`    | MediaPipe face/iris landmark detection |
| `pd_service.py`               | Multi-frame processing service         |
| `demo.py`                     | CLI demo tool                          |
| `main.py`                     | FastAPI server                         |

## Iris Method Pipeline

1. **Face Detection** - MediaPipe FaceLandmarker with iris refinement
2. **Iris Extraction** - Landmarks 468 (right), 473 (left)
3. **Depth Estimation** - `depth = 12.65mm × focal_length / iris_px`
4. **PD Calculation** - `pd = raw_pd_px × depth / focal_length`
5. **Temporal Smoothing** - IQR outlier rejection + weighted average

## Calibration Constants

```python
# pd_engine/iris_pd_engine.py
IRIS_DIAMETER_MM = 12.65  # Calibrated for MediaPipe
FOV_HORIZONTAL_DEG = 70   # Smartphone front camera
```

## API Endpoints

### POST /api/measure-pd-multi

Multi-frame PD measurement with iris method support.

```bash
curl -X POST http://localhost:8000/api/measure-pd-multi \
  -H "Content-Type: application/json" \
  -d '{"images": ["base64..."], "method": "iris"}'
```

## Usage

```bash
# Start server
python -m main

# CLI demo
python demo.py path/to/image.jpg --iris-method
```
