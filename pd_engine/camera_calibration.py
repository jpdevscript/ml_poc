"""
Camera Calibration Module

Extracts camera intrinsic parameters from EXIF metadata for accurate
focal length calculation. Falls back to FOV estimation if EXIF unavailable.

Key formula:
    f_px = (f_mm × W_img_px) / W_s_mm
"""

import math
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import io
import numpy as np


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    focal_length_px: float
    focal_length_mm: Optional[float] = None
    sensor_width_mm: Optional[float] = None
    principal_point: Tuple[float, float] = (0.0, 0.0)  # (cx, cy)
    image_size: Tuple[int, int] = (0, 0)  # (width, height)
    source: str = "estimated"  # "exif", "estimated", "default"
    device_model: Optional[str] = None


# Sensor database for common smartphone front cameras
# Format: {model_pattern: (sensor_width_mm, sensor_height_mm)}
SENSOR_DATABASE = {
    # Apple iPhones (front camera)
    "iphone 14": (4.86, 3.64),
    "iphone 13": (4.86, 3.64),
    "iphone 12": (4.86, 3.64),
    "iphone 11": (4.86, 3.64),
    "iphone x": (4.86, 3.64),
    "iphone 8": (4.86, 3.64),
    "iphone 7": (4.86, 3.64),
    "iphone 6": (4.86, 3.64),
    "iphone se": (4.86, 3.64),
    
    # Samsung Galaxy (front camera)
    "samsung sm-s9": (4.22, 3.17),  # S9/S9+
    "samsung sm-g9": (4.22, 3.17),  # S8/S7
    "samsung sm-a": (4.22, 3.17),   # A series
    "samsung sm-n9": (4.22, 3.17),  # Note series
    "galaxy s2": (4.22, 3.17),
    "galaxy s": (4.22, 3.17),
    
    # Google Pixel (front camera)
    "pixel 8": (4.55, 3.41),
    "pixel 7": (4.55, 3.41),
    "pixel 6": (4.55, 3.41),
    "pixel 5": (4.55, 3.41),
    "pixel 4": (4.55, 3.41),
    "pixel 3": (4.55, 3.41),
    
    # OnePlus (front camera)
    "oneplus": (4.22, 3.17),
    
    # Xiaomi (front camera)
    "xiaomi": (4.22, 3.17),
    "redmi": (4.22, 3.17),
    "poco": (4.22, 3.17),
    
    # Default for unknown devices
    "default": (4.89, 3.67),  # Common 1/3" sensor
}


def get_sensor_size(device_model: Optional[str]) -> Tuple[float, float]:
    """
    Look up sensor dimensions for a device model.
    
    Args:
        device_model: Device model string from EXIF
        
    Returns:
        (sensor_width_mm, sensor_height_mm)
    """
    if not device_model:
        return SENSOR_DATABASE["default"]
    
    model_lower = device_model.lower()
    
    for pattern, size in SENSOR_DATABASE.items():
        if pattern in model_lower:
            return size
    
    return SENSOR_DATABASE["default"]


def extract_exif_from_bytes(image_bytes: bytes) -> Dict:
    """
    Extract EXIF data from image bytes.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Dictionary of EXIF tags
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        exif_data = img._getexif()
        
        if exif_data is None:
            return {}
        
        exif = {}
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            exif[tag] = value
        
        return exif
    except Exception as e:
        print(f"[CameraCalibration] EXIF extraction failed: {e}")
        return {}


def extract_exif_from_file(image_path: str) -> Dict:
    """
    Extract EXIF data from image file.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Dictionary of EXIF tags
    """
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        
        if exif_data is None:
            return {}
        
        exif = {}
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            exif[tag] = value
        
        return exif
    except Exception as e:
        print(f"[CameraCalibration] EXIF extraction failed: {e}")
        return {}


def calculate_focal_length_from_exif(
    exif: Dict,
    image_width_px: int,
    image_height_px: int
) -> Optional[CameraIntrinsics]:
    """
    Calculate focal length in pixels from EXIF data.
    
    Formula: f_px = (f_mm × W_img_px) / W_s_mm
    
    Args:
        exif: EXIF dictionary
        image_width_px: Image width in pixels
        image_height_px: Image height in pixels
        
    Returns:
        CameraIntrinsics or None if calculation fails
    """
    # Extract focal length in mm
    focal_length_mm = None
    
    if "FocalLength" in exif:
        fl = exif["FocalLength"]
        if hasattr(fl, "numerator") and hasattr(fl, "denominator"):
            focal_length_mm = fl.numerator / fl.denominator
        elif isinstance(fl, (int, float)):
            focal_length_mm = float(fl)
        elif isinstance(fl, tuple) and len(fl) == 2:
            focal_length_mm = fl[0] / fl[1]
    
    if focal_length_mm is None:
        return None
    
    # Get device model for sensor lookup
    device_model = exif.get("Model", exif.get("Make", None))
    sensor_width_mm, sensor_height_mm = get_sensor_size(device_model)
    
    # Check for FocalLengthIn35mmFilm (can help validate)
    fl_35mm = exif.get("FocalLengthIn35mmFilm")
    if fl_35mm:
        # 35mm film has 36mm sensor width
        # crop_factor = 36 / sensor_width
        # We can use this to refine sensor size estimate
        crop_factor = fl_35mm / focal_length_mm
        refined_sensor_width = 36.0 / crop_factor
        if 3.0 < refined_sensor_width < 10.0:  # Sanity check
            sensor_width_mm = refined_sensor_width
            sensor_height_mm = refined_sensor_width * 0.75  # 4:3 ratio
    
    # Calculate focal length in pixels
    # f_px = (f_mm × W_img_px) / W_s_mm
    focal_length_px = (focal_length_mm * image_width_px) / sensor_width_mm
    
    # Principal point at image center
    cx = image_width_px / 2.0
    cy = image_height_px / 2.0
    
    return CameraIntrinsics(
        focal_length_px=focal_length_px,
        focal_length_mm=focal_length_mm,
        sensor_width_mm=sensor_width_mm,
        principal_point=(cx, cy),
        image_size=(image_width_px, image_height_px),
        source="exif",
        device_model=device_model
    )


def estimate_focal_length_from_fov(
    image_width_px: int,
    image_height_px: int,
    fov_degrees: float = 60.0
) -> CameraIntrinsics:
    """
    Estimate focal length from assumed field of view.
    
    Most smartphone front cameras have FOV around 60-80 degrees.
    Using 60° as a reasonable default.
    
    Formula: f_px = (width / 2) / tan(FOV / 2)
    
    Args:
        image_width_px: Image width in pixels
        image_height_px: Image height in pixels
        fov_degrees: Assumed horizontal FOV in degrees
        
    Returns:
        CameraIntrinsics with estimated values
    """
    fov_radians = math.radians(fov_degrees / 2)
    focal_length_px = (image_width_px / 2) / math.tan(fov_radians)
    
    cx = image_width_px / 2.0
    cy = image_height_px / 2.0
    
    return CameraIntrinsics(
        focal_length_px=focal_length_px,
        focal_length_mm=None,
        sensor_width_mm=None,
        principal_point=(cx, cy),
        image_size=(image_width_px, image_height_px),
        source="estimated",
        device_model=None
    )


def get_camera_intrinsics(
    image_width_px: int,
    image_height_px: int,
    exif: Optional[Dict] = None,
    image_path: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    fallback_fov: float = 60.0
) -> CameraIntrinsics:
    """
    Get camera intrinsic parameters, preferring EXIF when available.
    
    Priority:
    1. EXIF from provided dict
    2. EXIF from image file
    3. EXIF from image bytes
    4. Estimated from FOV
    
    Args:
        image_width_px: Image width
        image_height_px: Image height
        exif: Pre-extracted EXIF dict
        image_path: Path to image file
        image_bytes: Raw image bytes
        fallback_fov: FOV to use if EXIF unavailable
        
    Returns:
        CameraIntrinsics
    """
    # Try EXIF sources
    if exif is None:
        if image_path:
            exif = extract_exif_from_file(image_path)
        elif image_bytes:
            exif = extract_exif_from_bytes(image_bytes)
    
    if exif:
        intrinsics = calculate_focal_length_from_exif(
            exif, image_width_px, image_height_px
        )
        if intrinsics:
            print(f"[CameraCalibration] Using EXIF: f={intrinsics.focal_length_px:.1f}px "
                  f"({intrinsics.focal_length_mm:.2f}mm, sensor={intrinsics.sensor_width_mm:.2f}mm)")
            return intrinsics
    
    # Fallback to estimated
    intrinsics = estimate_focal_length_from_fov(
        image_width_px, image_height_px, fallback_fov
    )
    print(f"[CameraCalibration] Using estimated FOV ({fallback_fov}°): f={intrinsics.focal_length_px:.1f}px")
    return intrinsics


def build_intrinsic_matrix(intrinsics: CameraIntrinsics) -> np.ndarray:
    """
    Build 3x3 camera intrinsic matrix K.
    
    K = | fx  0  cx |
        | 0  fy  cy |
        | 0   0   1 |
    
    Args:
        intrinsics: Camera intrinsic parameters
        
    Returns:
        3x3 numpy array
    """
    fx = intrinsics.focal_length_px
    fy = intrinsics.focal_length_px  # Assume square pixels
    cx, cy = intrinsics.principal_point
    
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)
