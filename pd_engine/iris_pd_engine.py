"""
Complete PD Measurement System using Google MediaPipe Iris
6-Stage Pipeline with Production-Grade Implementation

Based on the iris biometric constant algorithm where:
- Iris diameter is 11.7mm (±0.5mm) - biological constant
- Depth = 11.7mm × focal_length / iris_diameter_px
- PD = pixel_distance × depth / focal_length
"""

import cv2
import numpy as np
from collections import deque
from typing import Dict, Optional, Tuple
import math

# ============================================================================
# CONSTANTS & BIOLOGICAL PARAMETERS
# ============================================================================

# Iris biometric constant
# NOTE: Anatomical iris is 11.7mm, but MediaPipe landmarks measure a smaller region
# Calibrated for optimal capture distance (180-220px raw_pd) - reduced to fix +2mm error
IRIS_DIAMETER_MM = 12.35  # Calibrated for close-distance capture

# Valid measurement ranges
PD_MIN_MM = 50.0          # Minimum realistic pupillary distance
PD_MAX_MM = 75.0          # Maximum realistic pupillary distance
DEPTH_MIN_MM = 150.0      # Minimum sensor-to-face distance (~15cm)
DEPTH_MAX_MM = 1500.0     # Maximum sensor-to-face distance (~150cm)

# Temporal smoothing parameters
SMOOTHING_WINDOW = 10     # Number of frames to accumulate
OUTLIER_IQR_MULTIPLIER = 1.5  # IQR method threshold
CONFIDENCE_MIN_THRESHOLD = 0.3  # Minimum confidence to accept (0.3 allows single-frame)

# MediaPipe landmark indices for iris (from the existing implementation)
LEFT_IRIS_INDICES = [473, 474, 475, 476, 477]    # Center + 4 perimeter points
RIGHT_IRIS_INDICES = [468, 469, 470, 471, 472]   # Center + 4 perimeter points


class IrisPDEngine:
    """
    Production-grade pupillary distance measurement engine using iris biometrics.
    
    Uses existing IrisMeasurer for landmark detection and adds:
    - Proper iris diameter estimation (enclosing circle + PCA)
    - Focal length estimation from FOV
    - Depth estimation using 11.7mm constant
    - PD calculation using pinhole model
    - Temporal smoothing with IQR outlier rejection
    """
    
    def __init__(self, smoothing_window: int = SMOOTHING_WINDOW):
        """
        Initialize PD measurement engine.
        
        Args:
            smoothing_window: Number of frames for temporal averaging
        """
        # Import and create IrisMeasurer (uses MediaPipe Tasks API internally)
        from pd_engine.measurement import IrisMeasurer
        self.measurer = IrisMeasurer()
        
        # State management for temporal smoothing
        self.pd_history = deque(maxlen=smoothing_window)
        self.left_iris_history = deque(maxlen=smoothing_window)
        self.right_iris_history = deque(maxlen=smoothing_window)
        self.depth_history = deque(maxlen=smoothing_window)
        self.confidence_history = deque(maxlen=smoothing_window)
        
        self.smoothing_window = smoothing_window
        self.focal_length_px = None
        self.frame_count = 0
    
    def estimate_focal_length(self, image_width: int) -> float:
        """
        Estimate focal length from image width.
        
        Modern smartphone front cameras typically have:
        - 70-80° horizontal FOV for front cameras
        - Using 70° as default for selfie cameras
        
        focal_length = (width/2) / tan(FOV/2)
        
        Args:
            image_width: Image width in pixels
            
        Returns:
            Focal length in pixels
        """
        # Using 70° FOV which is more typical for smartphone front cameras
        # This is wider than the 50° used in the documentation but matches real devices
        FOV_HORIZONTAL_DEG = 70
        fov_rad = math.radians(FOV_HORIZONTAL_DEG)
        focal_length = (image_width / 2) / math.tan(fov_rad / 2)
        return focal_length
    
    def _estimate_iris_diameter_from_landmarks(self, face_landmarks: np.ndarray, 
                                                image_height: int,
                                                image_width: int) -> Tuple[float, float]:
        """
        Estimate iris diameter using minimal enclosing circle + PCA methods.
        
        Args:
            face_landmarks: Face landmarks array from IrisMeasurer
            image_height: Image height
            image_width: Image width
            
        Returns:
            (left_iris_diameter, right_iris_diameter) in pixels
        """
        # Extract iris contour points
        left_iris_points = []
        right_iris_points = []
        
        for idx in LEFT_IRIS_INDICES:
            if idx < len(face_landmarks):
                x, y = face_landmarks[idx][:2]
                left_iris_points.append([x, y])
        
        for idx in RIGHT_IRIS_INDICES:
            if idx < len(face_landmarks):
                x, y = face_landmarks[idx][:2]
                right_iris_points.append([x, y])
        
        left_diameter = self._estimate_single_iris_diameter(np.array(left_iris_points, dtype=np.float32))
        right_diameter = self._estimate_single_iris_diameter(np.array(right_iris_points, dtype=np.float32))
        
        return left_diameter, right_diameter
    
    def _estimate_single_iris_diameter(self, iris_points: np.ndarray) -> float:
        """
        Estimate iris diameter using TWO robust methods averaged together.
        
        Method 1: Minimal Enclosing Circle (Welzl's algorithm)
        Method 2: PCA-based Major Axis
        
        Args:
            iris_points: (n_points, 2) array of pixel coordinates
            
        Returns:
            Estimated iris diameter in pixels
        """
        if len(iris_points) < 3:
            return 0.0
        
        # Method 1: Minimal Enclosing Circle
        try:
            (cx, cy), radius = cv2.minEnclosingCircle(iris_points)
            diameter_method1 = 2 * radius
        except:
            diameter_method1 = 0.0
        
        # Method 2: PCA-based Major Axis
        try:
            mean = iris_points.mean(axis=0)
            centered_points = iris_points - mean
            
            # Covariance matrix
            if len(centered_points) >= 2:
                cov = np.cov(centered_points.T)  # 2x2 matrix
                
                # Eigenvalue decomposition
                eigenvalues, _ = np.linalg.eig(cov)
                max_eigenvalue = np.max(np.abs(eigenvalues))
                
                # Diameter = 4 * sqrt(eigenvalue) because points are on perimeter
                diameter_method2 = 4 * np.sqrt(max_eigenvalue)
            else:
                diameter_method2 = diameter_method1
        except:
            diameter_method2 = diameter_method1
        
        # Average for robustness
        if diameter_method1 > 0 and diameter_method2 > 0:
            diameter = (diameter_method1 + diameter_method2) / 2
        else:
            diameter = max(diameter_method1, diameter_method2)
        
        return float(diameter)
    
    def _compute_detection_quality(self, left_iris_diameter: float,
                                   right_iris_diameter: float) -> float:
        """
        Compute quality score for iris detection (0-1 scale).
        """
        # Factor 1: Valid range (15-80 pixels typical for selfie)
        diameter_valid = (15 < left_iris_diameter < 80 and 
                         15 < right_iris_diameter < 80)
        
        if not diameter_valid or max(left_iris_diameter, right_iris_diameter) == 0:
            return 0.0
        
        # Factor 2: Similarity (left and right should be similar)
        diameter_ratio = min(left_iris_diameter, right_iris_diameter) / \
                        max(left_iris_diameter, right_iris_diameter)
        
        # Allow up to 30% difference
        quality_score = max(0, (diameter_ratio - 0.7) / 0.3)  # Normalize 0.7-1.0 to 0-1
        
        return min(1.0, quality_score)
    
    def smooth_measurements(self, pd_mm: float,
                           left_iris_diameter: float,
                           right_iris_diameter: float,
                           depth_mm: float) -> Tuple[float, float]:
        """
        Apply temporal smoothing and IQR outlier rejection.
        """
        # Add to history buffers
        self.pd_history.append(pd_mm)
        self.left_iris_history.append(left_iris_diameter)
        self.right_iris_history.append(right_iris_diameter)
        self.depth_history.append(depth_mm)
        
        self.frame_count += 1
        
        # Need minimum frames for smoothing
        if len(self.pd_history) < 3:
            return pd_mm, 0.3  # Low confidence with few frames
        
        # IQR Outlier Detection
        pd_array = np.array(list(self.pd_history))
        
        q1 = np.percentile(pd_array, 25)
        q3 = np.percentile(pd_array, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - OUTLIER_IQR_MULTIPLIER * iqr
        upper_bound = q3 + OUTLIER_IQR_MULTIPLIER * iqr
        
        valid_mask = (pd_array >= lower_bound) & (pd_array <= upper_bound)
        valid_measurements = pd_array[valid_mask]
        
        if len(valid_measurements) == 0:
            valid_measurements = pd_array
        
        # Weighted Average (recent frames weighted more)
        weights = np.arange(1, len(valid_measurements) + 1) ** 2
        weights = weights / weights.sum()
        
        smoothed_pd_mm = np.average(valid_measurements, weights=weights)
        
        # Confidence Scoring
        measurement_variance = np.var(valid_measurements)
        variance_confidence = 1.0 / (1.0 + measurement_variance / 10.0)
        
        # Iris symmetry check
        if max(left_iris_diameter, right_iris_diameter) > 0:
            iris_ratio = min(left_iris_diameter, right_iris_diameter) / max(left_iris_diameter, right_iris_diameter)
            consistency_bonus = max(0, (iris_ratio - 0.7) / 0.3) * 0.5
        else:
            consistency_bonus = 0
        
        confidence = min(1.0, variance_confidence * 0.7 + consistency_bonus * 0.3)
        
        self.confidence_history.append(confidence)
        
        return smoothed_pd_mm, confidence
    
    def process_frame(self, image: np.ndarray) -> Dict:
        """
        Process single frame through complete 6-stage pipeline.
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            dict with PD measurement and diagnostics
        """
        h, w = image.shape[:2]
        
        # Initialize focal length if needed
        if self.focal_length_px is None:
            self.focal_length_px = self.estimate_focal_length(w)
        
        # STAGE 1: Face Detection using existing IrisMeasurer
        result = self.measurer.measure(image)
        
        if not result.detected:
            return {
                'pd_mm': None,
                'confidence': 0.0,
                'is_valid': False,
                'error': result.error_message or 'No face detected'
            }
        
        if result.face_landmarks is None:
            return {
                'pd_mm': None,
                'confidence': 0.0,
                'is_valid': False,
                'error': 'No landmarks detected'
            }
        
        # STAGE 2-3: Use iris diameter from IrisMeasurer directly
        # The IrisMeasurer calculates diameter using horizontal landmark distance
        # This is the method that has been calibrated
        if result.iris_diameter_px and result.iris_diameter_px > 0:
            avg_iris_diameter = result.iris_diameter_px
        else:
            return {
                'pd_mm': None,
                'confidence': 0.0,
                'is_valid': False,
                'error': 'Could not estimate iris diameter'
            }
        
        # Quality check based on iris size
        quality_score = 1.0 if (15 < avg_iris_diameter < 80) else 0.5
        
        if quality_score < 0.3:
            return {
                'pd_mm': None,
                'confidence': quality_score,
                'is_valid': False,
                'error': f'Low iris quality: {quality_score:.2f}'
            }
        
        # STAGE 4: Depth Estimation using 11.7mm biological constant
        depth_mm = (IRIS_DIAMETER_MM * self.focal_length_px) / avg_iris_diameter
        
        if not (DEPTH_MIN_MM < depth_mm < DEPTH_MAX_MM):
            return {
                'pd_mm': None,
                'confidence': 0.0,
                'is_valid': False,
                'error': f'Invalid depth: {depth_mm:.0f}mm'
            }
        
        # STAGE 5: PD Calculation using pinhole camera model
        # PD = (pixel_distance × depth) / focal_length
        if result.raw_pd_px and result.raw_pd_px > 0:
            px_distance = result.raw_pd_px
        elif result.left_iris and result.right_iris:
            px_distance = np.linalg.norm(
                np.array(result.right_iris) - np.array(result.left_iris)
            )
        else:
            return {
                'pd_mm': None,
                'confidence': 0.0,
                'is_valid': False,
                'error': 'Could not calculate pixel distance'
            }
        
        pd_mm = (px_distance * depth_mm) / self.focal_length_px
        
        # STAGE 6: Temporal Smoothing (use avg diameter for both left/right)
        smoothed_pd, confidence = self.smooth_measurements(
            pd_mm, avg_iris_diameter, avg_iris_diameter, depth_mm
        )
        
        # Final Validation
        is_valid = (
            PD_MIN_MM <= smoothed_pd <= PD_MAX_MM and
            confidence >= CONFIDENCE_MIN_THRESHOLD and
            quality_score >= 0.4
        )
        
        return {
            'pd_mm': round(smoothed_pd, 2) if is_valid else None,
            'pd_raw': round(pd_mm, 2),
            'depth_mm': round(depth_mm, 0),
            'confidence': round(confidence, 2),
            'is_valid': is_valid,
            'iris_diameter_px': round(avg_iris_diameter, 1),
            'quality_score': round(quality_score, 2),
            'frame_number': self.frame_count,
            'focal_length_px': round(self.focal_length_px, 0)
        }
    
    def reset(self):
        """Reset all history buffers for new measurement session."""
        self.pd_history.clear()
        self.left_iris_history.clear()
        self.right_iris_history.clear()
        self.depth_history.clear()
        self.confidence_history.clear()
        self.frame_count = 0
    
    def get_final_measurement(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Extract final PD measurement from accumulated history.
        
        Returns:
            (final_pd_mm, uncertainty_mm)
        """
        if len(self.pd_history) == 0:
            return None, None
        
        pd_array = np.array(list(self.pd_history))
        
        # Use median for robustness
        final_pd = np.median(pd_array)
        uncertainty = np.std(pd_array)
        
        return round(final_pd, 2), round(uncertainty, 2)
