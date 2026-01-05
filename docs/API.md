# API Reference

## Base URL

```
http://localhost:8000
```

---

## Endpoints

### Health Check

```http
GET /
```

**Response:**

```json
{
  "status": "ok",
  "version": "1.0.0"
}
```

---

### Multi-Frame PD Measurement

```http
POST /api/measure-pd-multi
Content-Type: application/json
```

**Request Body:**

```json
{
  "images": [
    "data:image/jpeg;base64,/9j/4AAQ...",
    "data:image/jpeg;base64,/9j/4AAQ...",
    "data:image/jpeg;base64,/9j/4AAQ..."
  ]
}
```

**Success Response (200):**

```json
{
  "success": true,
  "pd_mm": 65.5,
  "confidence": 0.87,
  "error": null,
  "debug_dir": "inputs/20260103_141523",
  "details": {
    "method": "multi_frame_average",
    "frames_total": 5,
    "frames_valid": 4,
    "frames_used": 4,
    "outliers_removed": 0,
    "std_mm": 0.42,
    "median_mm": 65.4,
    "individual_results": [
      { "frame": 0, "valid": true, "pd_mm": 65.3, "confidence": 0.85 },
      { "frame": 1, "valid": true, "pd_mm": 65.6, "confidence": 0.88 },
      {
        "frame": 2,
        "valid": false,
        "pd_mm": null,
        "confidence": 0,
        "rejection_reason": "blurry (score=23.4)"
      },
      { "frame": 3, "valid": true, "pd_mm": 65.5, "confidence": 0.86 },
      { "frame": 4, "valid": true, "pd_mm": 65.4, "confidence": 0.87 }
    ],
    "warnings": []
  }
}
```

**Error Response (200):**

```json
{
  "success": false,
  "pd_mm": null,
  "confidence": 0,
  "error": "Only 1 valid frames detected. Need at least 2.",
  "details": {
    "frames_total": 5,
    "frames_valid": 1
  }
}
```

---

### Face Detection (Guidance)

```http
POST /api/detect-face
Content-Type: application/json
```

**Request Body:**

```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQ..."
}
```

**Response:**

```json
{
  "detected": true,
  "positioned": true,
  "centerX": 0.48,
  "centerY": 0.42,
  "faceSize": 0.35,
  "leftEyeX": 0.42,
  "leftEyeY": 0.4,
  "rightEyeX": 0.55,
  "rightEyeY": 0.4,
  "yaw": 2.3,
  "pitch": -1.5,
  "roll": 0.8,
  "guidance": {
    "distance": "ok",
    "position": "ok",
    "tilt": "ok",
    "look": "ok",
    "eyes": "ok"
  }
}
```

---

## Error Codes

| Code | Meaning                            |
| ---- | ---------------------------------- |
| 200  | Success (check `success` field)    |
| 400  | Bad request (invalid image format) |
| 500  | Internal server error              |

---

## Image Format

- **Accepted formats:** JPEG, PNG (base64 encoded)
- **Prefix:** `data:image/jpeg;base64,` or `data:image/png;base64,`
- **Recommended resolution:** 720p - 1080p
- **Orientation:** Front-facing camera (mirrored)

---

## Quality Requirements

For accurate measurements:

| Requirement  | Threshold                  |
| ------------ | -------------------------- |
| Blur score   | > 30 (Laplacian variance)  |
| Head yaw     | < ±15°                     |
| Head pitch   | < ±15°                     |
| Card visible | Full ID-1 card on forehead |
| Lighting     | Even, no harsh shadows     |

---

## Rate Limits

No rate limits in development mode.

For production, implement:

- 10 requests/second per IP
- 100 requests/minute per IP
