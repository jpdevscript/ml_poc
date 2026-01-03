# Backend Documentation - PD Measurement Engine

## Overview

The ML backend provides a medical-grade Pupillary Distance (PD) measurement system using computer vision and deep learning. It processes images of a person holding an ID-1 card on their forehead to calculate accurate PD measurements.

---

## Architecture

```
ml_poc/
├── main.py                 # FastAPI application entry point
├── pd_service.py           # High-level PD measurement service
├── pd_engine/              # Core measurement engine
│   ├── core.py             # PDMeasurement orchestrator
│   ├── measurement.py      # Iris detection (MediaPipe)
│   ├── calibration.py      # Card detection pipeline
│   ├── photogrammetry.py   # PD calculation from scale
│   ├── corrections.py      # Error corrections
│   ├── forehead_detection.py  # MediaPipe forehead ROI
│   ├── sam3_segmentation.py   # SAM3 card segmentation
│   └── utils.py            # Constants and helpers
└── docs/                   # This documentation
```

---

## API Endpoints

### POST `/api/measure-pd-multi`

Multi-frame PD measurement with statistical averaging.

**Request:**

```json
{
  "images": ["data:image/jpeg;base64,...", ...]
}
```

**Response:**

```json
{
  "success": true,
  "pd_mm": 65.5,
  "confidence": 0.87,
  "details": {
    "method": "multi_frame_average",
    "frames_total": 5,
    "frames_valid": 4,
    "std_mm": 0.42,
    "median_mm": 65.4
  }
}
```

### POST `/api/detect-face`

Face detection with guidance for positioning.

**Request:**

```json
{
  "image": "data:image/jpeg;base64,..."
}
```

**Response:**

```json
{
  "detected": true,
  "positioned": true,
  "yaw": 2.3,
  "pitch": -1.5,
  "guidance": {
    "distance": "ok",
    "position": "ok",
    "tilt": "ok"
  }
}
```

---

## Core Modules

### PDMeasurement (`core.py`)

Main orchestrator class that combines all modules.

```python
from pd_engine.core import PDMeasurement

engine = PDMeasurement()
result = engine.process_frame(image)

print(f"PD: {result.pd_final_mm:.2f} mm")
print(f"Confidence: {result.confidence:.1%}")
```

**Key Methods:**

- `process_frame(frame, debug_dir)` - Process single image
- `process_image(path)` - Load and process image file
- `process_video(path)` - Process video frames
- `get_aggregated_pd(results)` - Statistical aggregation

---

### CardCalibration (`calibration.py`)

Hybrid card detection using forehead ROI + SAM3 segmentation.

**Pipeline:**

1. **Forehead Detection** - MediaPipe face mesh identifies forehead region
2. **ROI Extraction** - Extract card region with padding
3. **SAM3 Segmentation** - Deep learning card segmentation
4. **Corner Detection** - Polygon approximation or minAreaRect
5. **Scale Calculation** - ISO ID-1 card dimensions (85.6 × 53.98 mm)

```python
from pd_engine.calibration import CardCalibration

calibration = CardCalibration(debug_dir="debug")
result = calibration.detect_card(image, debug=True)

if result.detected:
    print(f"Scale: {result.scale_factor:.4f} mm/px")
    print(f"Corners: {result.corners}")
```

---

### IrisMeasurer (`measurement.py`)

MediaPipe Face Mesh for iris center detection.

**Key Features:**

- 468+ facial landmarks at 20+ FPS
- Iris center landmarks (468, 473)
- Iris diameter calculation (±0.5mm accuracy)
- Head pose estimation (yaw, pitch, roll)

```python
from pd_engine.measurement import IrisMeasurer

measurer = IrisMeasurer()
result = measurer.measure(image)

print(f"Left iris: {result.left_iris}")
print(f"Right iris: {result.right_iris}")
print(f"Raw PD: {result.raw_pd_px:.1f} px")
print(f"Iris diameter: {result.iris_diameter_px:.1f} px")
```

---

### Photogrammetry (`photogrammetry.py`)

PD calculation using card-based scale factor.

**Algorithm:**

1. Detect card orientation (landscape vs portrait)
2. Use longer edge as card width (85.6mm)
3. Calculate scale factor: `mm/px = 85.6 / edge_px`
4. Apply calibration factor (0.984) for segmentation bias
5. PD = horizontal_pupil_distance × scale × calibration

```python
from pd_engine.photogrammetry import simple_pd_from_scale

result = simple_pd_from_scale(
    card_corners=corners,
    pupil_left_px=(x1, y1),
    pupil_right_px=(x2, y2),
    debug=True
)
print(f"PD: {result.pd_near_mm:.2f} mm")
```

---

### PDCorrector (`corrections.py`)

Error correction pipeline (mostly disabled for accuracy).

**Available Corrections:**
| Correction | Purpose | Status |
|------------|---------|--------|
| Yaw | Head rotation compensation | Enabled |
| Depth | Forehead-to-cornea offset | Disabled |
| Vergence | Eye convergence at near distance | Disabled |

**Iris-Based Camera Distance:**

```python
from pd_engine.corrections import PDCorrector

focal = PDCorrector.estimate_focal_length_px(image_width=1920)
distance = PDCorrector.estimate_camera_distance_from_iris(
    iris_diameter_px=45,
    focal_length_px=focal
)
print(f"Camera distance: {distance:.0f} mm")
```

---

## Quality Gates

### Blur Detection

Laplacian variance threshold (30.0):

- `score > 100`: Sharp image
- `score 50-100`: Acceptable
- `score < 30`: Rejected (too blurry)

### Head Pose Check

Frames rejected if:

- |yaw| > 15°
- |pitch| > 15°

---

## Configuration

### Environment Variables

```bash
# .env
ROBOFLOW_API_KEY=your_key_here
```

### Constants (`utils.py`)

```python
CARD_WIDTH_MM = 85.60      # ISO ID-1 width
CARD_HEIGHT_MM = 53.98     # ISO ID-1 height
AVERAGE_IRIS_MM = 11.7     # Human iris diameter
```

### Calibration Factor (`photogrammetry.py`)

```python
PD_CALIBRATION_FACTOR = 0.984  # Corrects ~1.6% segmentation bias
```

---

## Running the Server

```bash
cd ml_poc
source .venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**With HTTPS (for mobile):**

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 \
  --ssl-keyfile=key.pem --ssl-certfile=cert.pem
```

---

## Debug Output

Enable debug mode to save intermediate images:

```python
result = engine.process_frame(image, debug_dir="debug/session_001")
```

**Generated files:**

- `input.jpg` - Original image
- `mask.jpg` - SAM3 segmentation mask
- `corners.jpg` - Detected card corners
- `result.jpg` - Final visualization with PD measurement

---

## Testing

```bash
# Run demo on image
python demo.py path/to/image.jpg

# Run demo with debug output
python demo.py path/to/image.jpg -o debug_output
```

---

## Accuracy

| Metric             | Value                     |
| ------------------ | ------------------------- |
| Target accuracy    | ±1mm                      |
| Tested accuracy    | ±0.5mm (with calibration) |
| Calibration method | Known PD reference images |
| Repeatability      | σ < 0.5mm across frames   |
