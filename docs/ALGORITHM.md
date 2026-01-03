# PD Measurement Algorithm

## Overview

The PD (Pupillary Distance) measurement algorithm calculates the distance between the centers of the left and right pupils using a calibration card for scale reference.

---

## Pipeline

```
┌─────────────────┐
│  Input Image    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Quality Gates   │─────┐
│ (blur, pose)    │     │ Reject
└────────┬────────┘     │
         │ Pass         │
         ▼              │
┌─────────────────┐     │
│ Face Detection  │◄────┘
│ (MediaPipe)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Iris Detection  │
│ (landmarks)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Card Detection  │
│ (SAM3 + edges)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Scale Calc      │
│ (orientation)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ PD = dx × scale │
│ × calibration   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Result (mm)     │
└─────────────────┘
```

---

## Step 1: Quality Gates

### Blur Detection

Uses Laplacian variance to detect motion blur:

```python
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
is_sharp = blur_score > 30.0
```

### Head Pose Check

Rejects frames with excessive head rotation:

```python
if abs(yaw) > 15.0 or abs(pitch) > 15.0:
    reject_frame()
```

---

## Step 2: Iris Detection

MediaPipe Face Mesh provides 478 landmarks including iris centers:

| Landmark | Description                  |
| -------- | ---------------------------- |
| 468      | Left iris center             |
| 473      | Right iris center            |
| 469, 471 | Left iris horizontal extent  |
| 474, 476 | Right iris horizontal extent |

```python
left_iris = landmarks[468][:2]
right_iris = landmarks[473][:2]
raw_pd_px = distance(left_iris, right_iris)
```

---

## Step 3: Card Detection

### Forehead ROI

Uses facial landmarks to identify card region:

```python
# Landmarks 10, 67, 69, 104, 108, etc.
forehead_box = expand_roi(forehead_landmarks, padding=0.15)
```

### SAM3 Segmentation

Roboflow workflow segments card from ROI:

```python
segmenter = SAM3Segmenter()
mask = segmenter.segment(roi)
```

### Corner Detection

```python
contours = cv2.findContours(mask, ...)
approx = cv2.approxPolyDP(largest_contour, epsilon, True)

if len(approx) == 4:
    corners = approx
else:
    corners = cv2.boxPoints(cv2.minAreaRect(contour))
```

---

## Step 4: Orientation Detection

Cards can be held landscape or portrait:

```python
horizontal_px = avg(top_edge, bottom_edge)
vertical_px = avg(left_edge, right_edge)

if horizontal_px >= vertical_px:
    # Landscape: horizontal edge = 85.6mm
    card_width_px = horizontal_px
else:
    # Portrait: vertical edge = 85.6mm
    card_width_px = vertical_px

scale_factor = 85.6 / card_width_px  # mm/px
```

---

## Step 5: PD Calculation

```python
# Horizontal pupil distance (invariant to head tilt)
pupil_dx = abs(right_iris[0] - left_iris[0])

# Raw PD
raw_pd = pupil_dx * scale_factor

# Calibrated PD (correct segmentation bias)
pd_mm = raw_pd * 0.984
```

---

## Calibration Factor

The segmentation model slightly expands card boundaries, causing ~1.6% over-estimation:

| Metric      | Value            |
| ----------- | ---------------- |
| Raw average | 67.05 mm         |
| Actual PD   | 66.0 mm          |
| Correction  | 0.984 (66/67.05) |

---

## Multi-Frame Averaging

Statistical averaging with outlier rejection:

```python
# IQR-based outlier rejection
q1 = percentile(values, 25)
q3 = percentile(values, 75)
iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

filtered = [v for v in values if lower <= v <= upper]

# Final result
result = median(filtered)
```

---

## Accuracy Analysis

| Source of Error       | Magnitude | Mitigation               |
| --------------------- | --------- | ------------------------ |
| Segmentation boundary | ~1.6%     | Calibration factor       |
| Head yaw              | cos(θ)    | Horizontal-only distance |
| Blur                  | Variable  | Laplacian filter         |
| Card flexion          | ~1-2%     | Aspect ratio check       |

---

## Constants

```python
# ISO ID-1 Card
CARD_WIDTH_MM = 85.60
CARD_HEIGHT_MM = 53.98
CARD_ASPECT_RATIO = 1.586

# Human anatomy
AVERAGE_IRIS_DIAMETER_MM = 11.7
BROW_TO_CORNEA_MM = 12.0

# Calibration
PD_CALIBRATION_FACTOR = 0.984
```
