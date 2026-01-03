"""
Utility functions for PD measurement engine.
"""

import math
import numpy as np
import cv2
from typing import Tuple, Optional
import os


# ISO/IEC 7810 ID-1 Standard Card Dimensions (mm)
CARD_WIDTH_MM = 85.60
CARD_HEIGHT_MM = 53.98
CARD_ASPECT_RATIO = CARD_WIDTH_MM / CARD_HEIGHT_MM  # ~1.586

# Magnetic stripe aspect ratio (width / height)
MAGSTRIPE_ASPECT_RATIO = 85.6 / 9.52  # ~9.0

# Anatomical constants (mm)
BROW_TO_CORNEA_OFFSET_MM = 12.0  # Average distance from supraorbital ridge to cornea
EYE_CENTER_OF_ROTATION_MM = 13.0  # Distance from cornea to center of rotation

# MediaPipe Iris Landmark Indices
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473

# Thresholds
ASPECT_RATIO_TOLERANCE = 0.25  # Tolerance for card aspect ratio (1.586 ± 0.25)
MIN_CARD_AREA_RATIO = 0.01  # Card must be at least 1% of ROI area
MAX_YAW_DEGREES = 5.0  # Warning threshold for head rotation


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two 2D points."""
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def calculate_angle(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate angle between two points in degrees."""
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))


def order_corners(corners: np.ndarray) -> np.ndarray:
    """
    Order 4 corners in clockwise order starting from top-left.
    
    Args:
        corners: Array of 4 corner points
        
    Returns:
        Ordered corners: [top-left, top-right, bottom-right, bottom-left]
    """
    # Sort by y-coordinate (top to bottom)
    corners = corners.reshape(4, 2)
    sorted_by_y = corners[np.argsort(corners[:, 1])]
    
    # Top two points
    top_points = sorted_by_y[:2]
    # Bottom two points
    bottom_points = sorted_by_y[2:]
    
    # Sort top points by x (left to right)
    top_left, top_right = top_points[np.argsort(top_points[:, 0])]
    # Sort bottom points by x (left to right)
    bottom_left, bottom_right = bottom_points[np.argsort(bottom_points[:, 0])]
    
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def refine_corners_subpixel(
    gray_image: np.ndarray,
    corners: np.ndarray,
    window_size: int = 5
) -> np.ndarray:
    """
    Refine corner positions to sub-pixel accuracy using cv2.cornerSubPix.
    
    This is crucial for medical-grade accuracy - a 1px error at 50cm 
    distance can equal 0.5mm error in PD measurement.
    
    Args:
        gray_image: Grayscale image
        corners: Initial corner positions (4x2)
        window_size: Half-size of search window
        
    Returns:
        Refined corner positions with sub-pixel accuracy
    """
    # Prepare corners for cornerSubPix (needs shape Nx1x2)
    corners_input = corners.reshape(-1, 1, 2).astype(np.float32)
    
    # Termination criteria: max 100 iterations or 0.01 pixel accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)
    
    # Refine corners
    refined = cv2.cornerSubPix(
        gray_image,
        corners_input,
        winSize=(window_size, window_size),
        zeroZone=(-1, -1),
        criteria=criteria
    )
    
    return refined.reshape(-1, 2)


def calculate_camera_distance(
    card_width_px: float,
    focal_length_px: Optional[float] = None,
    image_width_px: Optional[int] = None
) -> float:
    """
    Estimate camera-to-subject distance using the card as a reference.
    
    Formula: D_camera = (card_width_mm * focal_length_px) / card_width_px
    
    If focal length is unknown, estimate it from image width (assuming ~60° FOV).
    
    Args:
        card_width_px: Detected card width in pixels
        focal_length_px: Camera focal length in pixels (optional)
        image_width_px: Image width for focal length estimation
        
    Returns:
        Estimated distance to camera in mm
    """
    if focal_length_px is None:
        if image_width_px is None:
            # Default assumption: typical selfie distance ~400mm
            return 400.0
        # Estimate focal length assuming ~60° horizontal FOV
        # focal_length ≈ image_width / (2 * tan(FOV/2))
        fov_radians = math.radians(60)
        focal_length_px = image_width_px / (2 * math.tan(fov_radians / 2))
    
    # D = (W_real * f) / W_image
    distance_mm = (CARD_WIDTH_MM * focal_length_px) / card_width_px
    
    return distance_mm


def rotation_matrix_to_euler_angles(rotation_matrix: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert a 3x3 rotation matrix to Euler angles (roll, pitch, yaw).
    
    Args:
        rotation_matrix: 3x3 rotation matrix
        
    Returns:
        Tuple of (roll, pitch, yaw) in degrees
    """
    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    
    singular = sy < 1e-6
    
    if not singular:
        roll = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        pitch = math.atan2(-rotation_matrix[2, 0], sy)
        yaw = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        roll = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        pitch = math.atan2(-rotation_matrix[2, 0], sy)
        yaw = 0
    
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


def validate_aspect_ratio(width: float, height: float, expected_ratio: float, tolerance: float = 0.1) -> bool:
    """Check if width/height ratio matches expected ratio within tolerance."""
    if height == 0:
        return False
    actual_ratio = width / height
    return abs(actual_ratio - expected_ratio) <= tolerance


def preprocess_for_card_detection(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess image for card detection.
    
    Args:
        image: BGR input image
        
    Returns:
        Tuple of (grayscale, edges)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur (5x5 kernel as per methodology)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edge detection (thresholds: 50, 150 as per methodology)
    edges = cv2.Canny(blurred, 50, 150)
    
    return gray, edges


def draw_landmarks_on_image(
    image: np.ndarray,
    iris_left: Optional[Tuple[float, float]],
    iris_right: Optional[Tuple[float, float]],
    card_corners: Optional[np.ndarray] = None,
    pd_mm: Optional[float] = None
) -> np.ndarray:
    """
    Draw visualization of detected landmarks on image.
    
    Args:
        image: Input BGR image
        iris_left: Left iris center (x, y)
        iris_right: Right iris center (x, y)
        card_corners: Detected card corners (4x2 array)
        pd_mm: Calculated PD in mm
        
    Returns:
        Annotated image
    """
    annotated = image.copy()
    
    # Draw card if detected
    if card_corners is not None:
        corners = card_corners.astype(np.int32)
        cv2.polylines(annotated, [corners], True, (0, 255, 0), 2)
        for corner in corners:
            cv2.circle(annotated, tuple(corner), 5, (0, 255, 0), -1)
    
    # Draw iris centers
    if iris_left is not None:
        cv2.circle(annotated, (int(iris_left[0]), int(iris_left[1])), 8, (255, 0, 0), -1)
        cv2.circle(annotated, (int(iris_left[0]), int(iris_left[1])), 10, (255, 255, 255), 2)
    
    if iris_right is not None:
        cv2.circle(annotated, (int(iris_right[0]), int(iris_right[1])), 8, (255, 0, 0), -1)
        cv2.circle(annotated, (int(iris_right[0]), int(iris_right[1])), 10, (255, 255, 255), 2)
    
    # Draw line between iris centers
    if iris_left is not None and iris_right is not None:
        cv2.line(
            annotated,
            (int(iris_left[0]), int(iris_left[1])),
            (int(iris_right[0]), int(iris_right[1])),
            (0, 0, 255), 2
        )
    
    # Display PD value
    if pd_mm is not None:
        cv2.putText(
            annotated,
            f"PD: {pd_mm:.1f} mm",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2
        )
    
    return annotated


# ============================================================================
# IMPROVEMENT 1: Real Camera Calibration from EXIF
# ============================================================================

def extract_camera_intrinsics_from_exif(
    image_path: Optional[str] = None,
    image_array: Optional[np.ndarray] = None
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Extract camera intrinsics and distortion from EXIF data.
    
    Args:
        image_path: Path to image file
        image_array: Optional numpy array (if image_path not available)
        
    Returns:
        Tuple of (K: 3x3 intrinsic matrix, D: distortion coefficients) or None
    """
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
        
        # Load image
        if image_path and os.path.exists(image_path):
            img = Image.open(image_path)
        elif image_array is not None:
            # Convert numpy array to PIL Image
            img = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
        else:
            return None
        
        exif = img._getexif()
        if not exif:
            return None
        
        # EXIF tag mappings
        FOCAL_LENGTH = 37386  # 0x920A
        FOCAL_LENGTH_35MM = 41989  # 0xA405
        SENSOR_WIDTH = None  # Not in standard EXIF, need database
        
        # Get focal length in mm
        focal_length_mm = exif.get(FOCAL_LENGTH)
        if focal_length_mm is None:
            # Try 35mm equivalent
            focal_length_35mm = exif.get(FOCAL_LENGTH_35MM)
            if focal_length_35mm:
                # Approximate conversion (varies by camera)
                focal_length_mm = focal_length_35mm / 1.5  # Rough estimate
        
        if focal_length_mm is None:
            return None
        
        # Get image dimensions
        width_px = img.width
        height_px = img.height
        
        # Estimate sensor width (common smartphone sensors)
        # iPhone: ~7.01mm, Samsung: ~6.17mm, Generic: ~6.0mm
        # For now, use a reasonable default based on image size
        if width_px > 3000:  # High-res camera
            sensor_width_mm = 7.0
        elif width_px > 2000:  # Mid-range
            sensor_width_mm = 6.17
        else:  # Lower res
            sensor_width_mm = 6.0
        
        # Calculate focal length in pixels
        focal_length_px = (focal_length_mm * width_px) / sensor_width_mm
        
        # Build intrinsic matrix
        K = np.array([
            [focal_length_px, 0, width_px / 2],
            [0, focal_length_px, height_px / 2],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Distortion coefficients (default to zero if not available)
        D = np.zeros(5, dtype=np.float64)
        
        return K, D
        
    except Exception as e:
        print(f"[Utils] EXIF extraction failed: {e}")
        return None


def get_sensor_width_from_model(camera_model: Optional[str]) -> float:
    """
    Get sensor width from camera model (database lookup).
    
    Args:
        camera_model: Camera model string from EXIF
        
    Returns:
        Sensor width in mm
    """
    if camera_model is None:
        return 6.0  # Default
    
    # Common smartphone sensor widths (mm)
    sensor_database = {
        'iPhone': 7.01,
        'Samsung': 6.17,
        'Google Pixel': 6.17,
        'OnePlus': 6.17,
        'Xiaomi': 6.17,
    }
    
    for brand, width in sensor_database.items():
        if brand.lower() in camera_model.lower():
            return width
    
    return 6.0  # Default fallback


# ============================================================================
# IMPROVEMENT 2: Sub-Pixel Iris Refinement
# ============================================================================

def refine_iris_centers_subpixel(
    image: np.ndarray,
    left_iris_approx: Tuple[float, float],
    right_iris_approx: Tuple[float, float],
    roi_size: int = 50
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Refine iris centers to sub-pixel accuracy using circular Hough transform.
    
    Args:
        image: BGR input image
        left_iris_approx: Approximate left iris center (x, y)
        right_iris_approx: Approximate right iris center (x, y)
        roi_size: Size of ROI around iris for detection
        
    Returns:
        Tuple of (refined_left_iris, refined_right_iris)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    def refine_single_iris(iris_approx: Tuple[float, float]) -> Tuple[float, float]:
        """Refine a single iris center."""
        x, y = int(iris_approx[0]), int(iris_approx[1])
        
        # Extract ROI
        x1 = max(0, x - roi_size)
        y1 = max(0, y - roi_size)
        x2 = min(w, x + roi_size)
        y2 = min(h, y + roi_size)
        
        roi = gray[y1:y2, x1:x2]
        
        if roi.size == 0:
            return iris_approx
        
        # Apply Gaussian blur for better circle detection
        roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)
        
        # Circular Hough transform
        circles = cv2.HoughCircles(
            roi_blur,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=8,
            maxRadius=25
        )
        
        if circles is not None and len(circles[0]) > 0:
            # Get best circle (first one)
            cx, cy, r = circles[0][0]
            
            # Convert back to global coordinates
            refined_x = cx + x1
            refined_y = cy + y1
            
            return (refined_x, refined_y)
        
        # Fallback: Use gradient-based refinement
        # Find center of mass of high-gradient regions
        edges = cv2.Canny(roi, 50, 150)
        moments = cv2.moments(edges)
        
        if moments['m00'] > 0:
            cx = moments['m10'] / moments['m00']
            cy = moments['m01'] / moments['m00']
            return (cx + x1, cy + y1)
        
        return iris_approx
    
    refined_left = refine_single_iris(left_iris_approx)
    refined_right = refine_single_iris(right_iris_approx)
    
    return refined_left, refined_right


# ============================================================================
# IMPROVEMENT 6: Card Flatness Validation
# ============================================================================

def validate_card_flatness(
    corners: np.ndarray,
    image: Optional[np.ndarray] = None
) -> Tuple[bool, float, dict]:
    """
    Validate that card is flat (not bent/warped).
    
    Args:
        corners: 4 corner points (TL, TR, BR, BL)
        image: Optional image for visualization
        
    Returns:
        Tuple of (is_flat, convexity_defect, validation_info)
    """
    corners = corners.reshape(4, 2).astype(np.float32)
    
    # Method 1: Check corner angles (should be ~90°)
    def calculate_angle_at_corner(p1, p2, p3):
        """Calculate angle at corner p2."""
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
    
    angles = []
    for i in range(4):
        p1 = corners[(i - 1) % 4]
        p2 = corners[i]
        p3 = corners[(i + 1) % 4]
        angle = calculate_angle_at_corner(p1, p2, p3)
        angles.append(angle)
    
    angle_deviation = np.std([abs(a - 90.0) for a in angles])
    
    # Method 2: Check parallelism of opposite edges
    def normalize(v):
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v
    
    top_edge = corners[1] - corners[0]
    bottom_edge = corners[2] - corners[3]
    left_edge = corners[3] - corners[0]
    right_edge = corners[2] - corners[1]
    
    top_bottom_parallelism = 1 - abs(np.dot(normalize(top_edge), normalize(bottom_edge)))
    left_right_parallelism = 1 - abs(np.dot(normalize(left_edge), normalize(right_edge)))
    avg_parallelism = (top_bottom_parallelism + left_right_parallelism) / 2
    
    # Method 3: Check convexity
    contour = corners.reshape(-1, 1, 2).astype(np.int32)
    hull = cv2.convexHull(contour)
    contour_area = cv2.contourArea(contour)
    hull_area = cv2.contourArea(hull)
    
    if hull_area > 0:
        convexity_defect = 1 - (contour_area / hull_area)
    else:
        convexity_defect = 1.0
    
    # Validation thresholds
    MAX_ANGLE_DEVIATION = 5.0  # degrees
    MAX_PARALLELISM_ERROR = 0.05  # 5%
    MAX_CONVEXITY_DEFECT = 0.05  # 5%
    
    is_flat = (
        angle_deviation < MAX_ANGLE_DEVIATION and
        avg_parallelism < MAX_PARALLELISM_ERROR and
        convexity_defect < MAX_CONVEXITY_DEFECT
    )
    
    validation_info = {
        'angle_deviation': angle_deviation,
        'parallelism_error': avg_parallelism,
        'convexity_defect': convexity_defect,
        'angles': angles
    }
    
    return is_flat, convexity_defect, validation_info
