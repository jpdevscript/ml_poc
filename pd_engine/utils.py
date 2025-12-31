"""
Utility functions for PD measurement engine.
"""

import math
import numpy as np
import cv2
from typing import Tuple, Optional


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
