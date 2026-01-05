# PD Measurement Algorithm

## Overview

The PD (Pupillary Distance) measurement algorithm calculates the distance between the centers of the left and right pupils using a calibration card for scale reference. This implementation follows medical-grade standards achieving MAE ±0.5mm accuracy.

---

## Algorithm Pipeline

```
┌─────────────────────────────┐
│     Input Image             │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Step 1: Quality Gates       │
│ - Blur check (Laplacian>30) │
│ - Pose check (yaw,pitch<5°) │
└──────────────┬──────────────┘
               │ Pass
               ▼
┌─────────────────────────────┐
│ Step 2: Iris Detection      │
│ - MediaPipe Face Mesh       │
│ - Landmarks 468, 473        │
│ - Iris diameter calc        │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Step 3: Card Detection      │
│ - SAM3 segmentation         │
│ - Corner detection          │
│ - Orientation (L/P)         │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Step 4: Scale Calibration   │
│ - Card width → 85.6mm       │
│ - Apply calibration factor  │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Step 5: Depth Estimation    │
│ - Z_eye from iris diameter  │
│ - Vertex correction (0mm)   │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Step 6: Asymmetry Correction│
│ - Per-eye depth for yaw     │
│ - Monocular PD calculation  │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Step 7: Weighted Median     │
│ - Multi-frame averaging     │
│ - Outlier rejection (IQR)   │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│     Result (mm)             │
│ - Total PD                  │
│ - Monocular L/R             │
│ - Far PD (distance glasses) │
└─────────────────────────────┘
```

---

## Key Formulas

### 1. Scale Factor from Card

```
Scale_raw = CARD_WIDTH_MM / card_width_px
Scale_calibrated = Scale_raw × PD_CALIBRATION_FACTOR
```

Where:

- `CARD_WIDTH_MM = 85.60` (ISO ID-1)
- `PD_CALIBRATION_FACTOR = 0.984` (corrects segmentation boundary expansion)

### 2. Eye Depth from Iris Diameter

```
Z_eye = (f_px × IRIS_DIAMETER_MM) / iris_diameter_px
```

Where:

- `IRIS_DIAMETER_MM = 11.7` (human average ±0.5mm)
- `f_px` = focal length in pixels (from EXIF or estimated at 60° FOV)

### 3. Vertex Distance Correction

```
M_corr = (Z_eye + VERTEX_DISTANCE_MM) / Z_eye
```

Where:

- `VERTEX_DISTANCE_MM = 0.0` (currently disabled - empirically not needed)

### 4. Asymmetry Correction for Head Yaw

When head is turned by angle θ, each eye is at different depth:

```
d_L = Z_eye × cos(θ) - (PD/2) × sin(θ)
d_R = Z_eye × cos(θ) + (PD/2) × sin(θ)

corr_L = (d_L + vertex) / (d_avg + vertex)
corr_R = (d_R + vertex) / (d_avg + vertex)
```

### 5. Monocular PD Calculation

```
PD_left = |nose_x - left_pupil_x| × scale × M_corr × corr_L
PD_right = |right_pupil_x - nose_x| × scale × M_corr × corr_R
PD_total = PD_left + PD_right
```

### 6. Far PD (Distance Glasses)

```
PD_far = PD_near + 3.5mm
```

Near-to-far adjustment for eye vergence when looking at distance.

---

## Weighted Median Filtering

For multi-frame measurements:

```python
# Weight calculation
for frame in valid_frames:
    confidence = frame.confidence
    blur_weight = min(blur_score / 100, 1.0)
    weight = confidence × (0.7 + 0.3 × blur_weight)

# Weighted median
sorted_values = sort_by_value(pd_values)
sorted_weights = reorder(weights)
cumulative = cumsum(sorted_weights)
median_idx = searchsorted(cumulative, total_weight / 2)
result = sorted_values[median_idx]
```

---

## Quality Gates

### Blur Detection

Laplacian variance threshold:

- `score > 100`: Sharp image
- `score 50-100`: Acceptable
- `score 30-50`: Marginal
- `score < 30`: Rejected

### Head Pose (Medical-Grade)

Strict thresholds for geometric accuracy:

- `|yaw| < 5°` (head rotation)
- `|pitch| < 5°` (head tilt forward/back)

Beyond 5°, geometric foreshortening becomes uncorrectable.

---

## Constants

```python
# ISO ID-1 Card
CARD_WIDTH_MM = 85.60
CARD_HEIGHT_MM = 53.98

# Human anatomy
IRIS_DIAMETER_MM = 11.7        # ±0.5mm
VERTEX_DISTANCE_MM = 0.0       # Tunable
NEAR_TO_FAR_ADJUSTMENT = 3.5   # mm

# Calibration
PD_CALIBRATION_FACTOR = 0.984  # Segmentation correction
```

---

## Accuracy Results

Validated against known PD = 66.0mm:

| Image | Result | Error |
| ----- | ------ | ----- |
| test1 | 66.38  | +0.38 |
| test2 | 65.33  | -0.67 |
| test5 | 66.01  | +0.01 |
| test6 | 66.26  | +0.26 |
| test7 | 65.92  | -0.08 |

**Average: 65.98mm** (MAE: 0.28mm)

---

## Output Structure

```python
MedicalGradePDResult(
    success=True,
    pd_total_mm=66.01,        # Binocular PD
    pd_left_mm=33.0,          # Monocular (left)
    pd_right_mm=33.01,        # Monocular (right)
    pd_far_mm=69.51,          # Distance glasses
    z_eye_mm=538.7,           # Camera-to-eye distance
    depth_correction=1.0,     # Magnification factor
    scale_factor=0.522,       # mm/px (calibrated)
    confidence=0.95,
    warnings=[]
)
```
