"""
Iris-Based PD Measurement Engine - Ratio Method

Uses a geometrically rigorous ratio-based algorithm:
  PD_mm = (PD_px / avg_iris_px) × HVID_mm × BIAS_CORRECTION

Based on research showing:
- HVID (Horizontal Visible Iris Diameter) = 11.7mm anatomical mean
- MediaPipe landmarks underestimate iris by ~5-10%
- Kalman filter provides superior temporal smoothing vs simple averaging

Key improvements over focal-length method:
- Focal length cancels out in ratio calculation
- Device-independent (works across laptop/mobile)
- More robust to distance variations
"""

import cv2
import numpy as np
from collections import deque
from typing import Dict, Optional, Tuple
import math

# ============================================================================
# ALGORITHM CONSTANTS
# ============================================================================

# Anatomical reference: Horizontal Visible Iris Diameter
# Global mean HVID is 11.7mm (±0.5mm) - biological constant
HVID_MM = 11.7

# Bias correction for MediaPipe landmark underestimation
# Calibrated from ground truth (3 points):
#   iris=39.6px → 63.5mm (laptop)
#   iris=21.1px → 66.5mm (laptop far)
#   iris=37.5px → 66.5mm (mobile)
# Formula: bias = BASE + (iris_px - REF) * SCALE
BIAS_BASE = 1.06            # Reduced base for better mobile fit
BIAS_REFERENCE_IRIS_PX = 35  # Reference iris size in pixels
BIAS_SCALE_FACTOR = 0.0008   # Reduced scale for more stable cross-device results

# Resolution/orientation-adaptive correction
# Laptops (landscape, width > height) tend to underestimate more than mobile
LOW_RES_THRESHOLD = 720  # pixels (height)
LOW_RES_ADDITIONAL_BIAS = 0.03  # +3% for low-res
LANDSCAPE_ADDITIONAL_BIAS = 0.02  # +2% for landscape (laptop webcams)

# Vergence correction for Far PD
# At close range (phone/laptop), eyes converge inward
# Add ~3mm to convert to "Far PD" for distance glasses
VERGENCE_CORRECTION_MM = 3.0
APPLY_VERGENCE_CORRECTION = False  # Set True for distance glasses PD

# Valid measurement ranges
PD_MIN_MM = 50.0          # Minimum realistic adult PD
PD_MAX_MM = 75.0          # Maximum realistic adult PD

# Kalman filter parameters
KALMAN_PROCESS_NOISE = 0.01     # Q: How much we expect PD to change between frames
KALMAN_MEASUREMENT_NOISE = 1.0  # R: How noisy MediaPipe measurements are

# Quality thresholds
MIN_IRIS_DIAMETER_PX = 15   # Minimum iris size for valid measurement
MAX_IRIS_DIAMETER_PX = 80   # Maximum iris size for valid measurement
IRIS_SYMMETRY_THRESHOLD = 0.7  # Min ratio between left/right iris

# Temporal smoothing
SMOOTHING_WINDOW = 10  # Number of frames for history
CONFIDENCE_MIN_THRESHOLD = 0.3

# MediaPipe landmark indices
# For iris width, use horizontal perimeter landmarks (not center)
LEFT_IRIS_CENTER = 473
RIGHT_IRIS_CENTER = 468
# Horizontal diameter landmarks
LEFT_IRIS_LEFT = 474    # Left edge of left iris
LEFT_IRIS_RIGHT = 476   # Right edge of left iris
RIGHT_IRIS_LEFT = 469   # Left edge of right iris  
RIGHT_IRIS_RIGHT = 471  # Right edge of right iris


class KalmanFilter1D:
    """
    Simple 1D Kalman filter for PD smoothing.
    
    Much more robust than simple averaging at rejecting
    MediaPipe's characteristic "jitter noise".
    """
    
    def __init__(self, process_noise: float = KALMAN_PROCESS_NOISE, 
                 measurement_noise: float = KALMAN_MEASUREMENT_NOISE):
        self.Q = process_noise      # Process noise covariance
        self.R = measurement_noise  # Measurement noise covariance
        
        # State
        self.x = None  # Estimated value
        self.P = 1.0   # Error covariance
        
    def reset(self):
        """Reset filter state."""
        self.x = None
        self.P = 1.0
    
    def update(self, measurement: float) -> float:
        """
        Process a new measurement and return filtered value.
        
        Args:
            measurement: Raw PD measurement
            
        Returns:
            Filtered PD value
        """
        if self.x is None:
            # First measurement - initialize
            self.x = measurement
            self.P = 1.0
            return self.x
        
        # Prediction step
        # State prediction: x_pred = x (assuming PD doesn't change)
        x_pred = self.x
        P_pred = self.P + self.Q
        
        # Update step
        # Kalman gain
        K = P_pred / (P_pred + self.R)
        
        # State update
        self.x = x_pred + K * (measurement - x_pred)
        
        # Covariance update
        self.P = (1 - K) * P_pred
        
        return self.x


class IrisPDEngine:
    """
    Production-grade pupillary distance measurement using ratio method.
    
    Algorithm:
        PD_mm = (PD_px / avg_iris_px) × HVID_mm × BIAS_CORRECTION
    
    Where:
        - PD_px: Distance between iris centers (landmarks 468, 473)
        - avg_iris_px: Average horizontal iris diameter (landmarks 469/471, 474/476)
        - HVID_mm: Anatomical constant (11.7mm)
        - BIAS_CORRECTION: Factor to compensate for MediaPipe underestimation
    """
    
    def __init__(self, smoothing_window: int = SMOOTHING_WINDOW):
        """
        Initialize PD measurement engine.
        
        Args:
            smoothing_window: Number of frames for history tracking
        """
        # Import and create IrisMeasurer
        from pd_engine.measurement import IrisMeasurer
        self.measurer = IrisMeasurer()
        
        # Kalman filter for temporal smoothing
        self.kalman = KalmanFilter1D(
            process_noise=KALMAN_PROCESS_NOISE,
            measurement_noise=KALMAN_MEASUREMENT_NOISE
        )
        
        # History for diagnostics and confidence calculation
        self.pd_history = deque(maxlen=smoothing_window)
        self.iris_history = deque(maxlen=smoothing_window)
        self.confidence_history = deque(maxlen=smoothing_window)
        
        self.smoothing_window = smoothing_window
        self.frame_count = 0
    
    def _extract_iris_measurements(self, landmarks_px: np.ndarray, 
                                    image_height: int) -> Tuple[float, float, float, float, float]:
        """
        Extract iris measurements from landmarks.
        
        Uses horizontal perimeter landmarks for width (not enclosing circle).
        
        Args:
            landmarks_px: Array of landmark pixel coordinates
            image_height: Image height for resolution check
            
        Returns:
            (left_iris_width, right_iris_width, pd_px, left_center, right_center)
        """
        # Left iris horizontal width: landmarks 474 (left edge) and 476 (right edge)
        left_iris_left = landmarks_px[LEFT_IRIS_LEFT][:2]
        left_iris_right = landmarks_px[LEFT_IRIS_RIGHT][:2]
        left_iris_width = np.linalg.norm(np.array(left_iris_right) - np.array(left_iris_left))
        
        # Right iris horizontal width: landmarks 469 (left edge) and 471 (right edge)  
        right_iris_left = landmarks_px[RIGHT_IRIS_LEFT][:2]
        right_iris_right = landmarks_px[RIGHT_IRIS_RIGHT][:2]
        right_iris_width = np.linalg.norm(np.array(right_iris_right) - np.array(right_iris_left))
        
        # Pupillary distance: between iris centers 468 and 473
        left_center = landmarks_px[LEFT_IRIS_CENTER][:2]
        right_center = landmarks_px[RIGHT_IRIS_CENTER][:2]
        pd_px = np.linalg.norm(np.array(right_center) - np.array(left_center))
        
        return left_iris_width, right_iris_width, pd_px, left_center, right_center
    
    def _compute_quality_score(self, left_iris_width: float, 
                                right_iris_width: float) -> float:
        """
        Compute quality score based on iris detection quality.
        
        Args:
            left_iris_width: Left iris width in pixels
            right_iris_width: Right iris width in pixels
            
        Returns:
            Quality score 0-1
        """
        # Check valid range
        if not (MIN_IRIS_DIAMETER_PX < left_iris_width < MAX_IRIS_DIAMETER_PX):
            return 0.0
        if not (MIN_IRIS_DIAMETER_PX < right_iris_width < MAX_IRIS_DIAMETER_PX):
            return 0.0
        
        # Check symmetry (left and right should be similar)
        if max(left_iris_width, right_iris_width) == 0:
            return 0.0
            
        symmetry_ratio = min(left_iris_width, right_iris_width) / max(left_iris_width, right_iris_width)
        
        if symmetry_ratio < IRIS_SYMMETRY_THRESHOLD:
            return 0.3  # Low quality but still usable
        
        # Higher symmetry = higher quality
        quality = (symmetry_ratio - IRIS_SYMMETRY_THRESHOLD) / (1.0 - IRIS_SYMMETRY_THRESHOLD)
        return min(1.0, 0.5 + quality * 0.5)  # Range 0.5-1.0 for symmetric
    
    def _calculate_pd(self, pd_px: float, avg_iris_px: float, 
                      image_height: int, image_width: int) -> float:
        """
        Calculate PD using ratio method with iris-size adaptive bias.
        
        Formula: PD_mm = (PD_px / avg_iris_px) × HVID_mm × adaptive_bias
        
        The adaptive bias compensates for the fact that MediaPipe's iris underestimation
        varies with iris size (which correlates with distance and device).
        
        Args:
            pd_px: Inter-pupillary distance in pixels
            avg_iris_px: Average iris diameter in pixels  
            image_height: Image height for resolution-adaptive scaling
            image_width: Image width for orientation detection
            
        Returns:
            PD in millimeters
        """
        # Calculate iris-size adaptive bias
        # Larger iris (close) gets higher bias, smaller iris (far) gets lower bias
        # This matches calibration: close cameras underestimate more
        iris_deviation = avg_iris_px - BIAS_REFERENCE_IRIS_PX  # positive when iris is large
        adaptive_bias = BIAS_BASE + (iris_deviation * BIAS_SCALE_FACTOR)
        
        # Clamp bias to reasonable range
        adaptive_bias = max(1.02, min(1.15, adaptive_bias))
        
        # Base calculation using ratio method
        ratio = pd_px / avg_iris_px
        pd_mm = ratio * HVID_MM * adaptive_bias
        
        # Resolution-adaptive correction (additional)
        if image_height < LOW_RES_THRESHOLD:
            pd_mm *= (1.0 + LOW_RES_ADDITIONAL_BIAS)
        
        # Landscape correction (laptop webcams tend to underestimate more)
        is_landscape = image_width > image_height
        if is_landscape:
            pd_mm *= (1.0 + LANDSCAPE_ADDITIONAL_BIAS)
        
        # Vergence correction (if enabled)
        if APPLY_VERGENCE_CORRECTION:
            pd_mm += VERGENCE_CORRECTION_MM
        
        return pd_mm
    
    def process_frame(self, image: np.ndarray) -> Dict:
        """
        Process single frame through the measurement pipeline.
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            dict with PD measurement and diagnostics
        """
        h, w = image.shape[:2]
        self.frame_count += 1
        
        # Stage 1: Face and iris detection
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
        
        # Stage 2: Extract iris measurements using horizontal width landmarks
        try:
            left_width, right_width, pd_px, left_center, right_center = \
                self._extract_iris_measurements(result.face_landmarks, h)
        except (IndexError, KeyError) as e:
            return {
                'pd_mm': None,
                'confidence': 0.0,
                'is_valid': False,
                'error': f'Iris landmarks not available: {e}'
            }
        
        # Average iris width for stable measurement
        avg_iris_px = (left_width + right_width) / 2
        
        # Stage 3: Quality assessment
        quality_score = self._compute_quality_score(left_width, right_width)
        
        if quality_score < 0.3:
            return {
                'pd_mm': None, 
                'confidence': quality_score,
                'is_valid': False,
                'error': f'Low iris quality: {quality_score:.2f}',
                'iris_diameter_px': round(avg_iris_px, 1)
            }
        
        # Stage 4: Calculate PD using ratio method
        raw_pd_mm = self._calculate_pd(pd_px, avg_iris_px, h, w)
        
        # Stage 5: Temporal smoothing with Kalman filter
        smoothed_pd = self.kalman.update(raw_pd_mm)
        
        # Track history
        self.pd_history.append(raw_pd_mm)
        self.iris_history.append(avg_iris_px)
        
        # Stage 6: Confidence calculation
        if len(self.pd_history) >= 3:
            pd_std = np.std(list(self.pd_history))
            variance_confidence = 1.0 / (1.0 + pd_std / 5.0)  # Lower std = higher confidence
        else:
            variance_confidence = 0.5
        
        # Combine quality and variance confidence
        confidence = 0.6 * quality_score + 0.4 * variance_confidence
        self.confidence_history.append(confidence)
        
        # Stage 7: Validation
        is_valid = (
            PD_MIN_MM <= smoothed_pd <= PD_MAX_MM and
            confidence >= CONFIDENCE_MIN_THRESHOLD
        )
        
        # Calculate adaptive bias for this measurement (for debugging)
        iris_deviation = avg_iris_px - BIAS_REFERENCE_IRIS_PX
        adaptive_bias = max(1.02, min(1.15, BIAS_BASE + (iris_deviation * BIAS_SCALE_FACTOR)))
        
        return {
            'pd_mm': round(smoothed_pd, 2) if is_valid else None,
            'pd_raw': round(raw_pd_mm, 2),
            'confidence': round(confidence, 2),
            'is_valid': is_valid,
            'iris_diameter_px': round(avg_iris_px, 1),
            'left_iris_px': round(left_width, 1),
            'right_iris_px': round(right_width, 1),
            'pd_px': round(pd_px, 1),
            'quality_score': round(quality_score, 2),
            'frame_number': self.frame_count,
            'depth_mm': None,  # Not calculated in ratio method
            # Algorithm info
            'algorithm': 'ratio_adaptive',
            'hvid_mm': HVID_MM,
            'bias_factor': round(adaptive_bias, 3),
            # Landmarks for visualization
            'face_landmarks': result.face_landmarks
        }
    
    def reset(self):
        """Reset all state for new measurement session."""
        self.kalman.reset()
        self.pd_history.clear()
        self.iris_history.clear()
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
        
        # Use Kalman filtered value as final estimate
        if self.kalman.x is not None:
            final_pd = self.kalman.x
        else:
            final_pd = np.median(pd_array)
        
        uncertainty = np.std(pd_array)
        
        return round(final_pd, 2), round(uncertainty, 2)
    
    def visualize(self, image: np.ndarray, result: Dict, 
                  landmarks_px: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create visualization with iris detection overlays.
        
        Args:
            image: Original BGR image
            result: Result dict from process_frame
            landmarks_px: Optional face landmarks array
            
        Returns:
            Annotated image
        """
        viz = image.copy()
        h, w = viz.shape[:2]
        
        # Colors (BGR)
        IRIS_COLOR = (0, 255, 255)    # Yellow - iris circles
        CENTER_COLOR = (0, 255, 0)     # Green - iris centers  
        PD_COLOR = (255, 0, 255)       # Magenta - PD line
        TEXT_COLOR = (255, 255, 255)   # White - text
        BG_COLOR = (0, 0, 0)           # Black - text background
        
        # Extract landmarks if available
        if landmarks_px is not None:
            try:
                # Get iris landmarks
                left_center = landmarks_px[LEFT_IRIS_CENTER][:2]
                right_center = landmarks_px[RIGHT_IRIS_CENTER][:2]
                
                left_iris_l = landmarks_px[LEFT_IRIS_LEFT][:2]
                left_iris_r = landmarks_px[LEFT_IRIS_RIGHT][:2]
                right_iris_l = landmarks_px[RIGHT_IRIS_LEFT][:2]
                right_iris_r = landmarks_px[RIGHT_IRIS_RIGHT][:2]
                
                # Draw iris circles
                left_radius = int(np.linalg.norm(np.array(left_iris_r) - np.array(left_iris_l)) / 2)
                right_radius = int(np.linalg.norm(np.array(right_iris_r) - np.array(right_iris_l)) / 2)
                
                cv2.circle(viz, (int(left_center[0]), int(left_center[1])), left_radius, IRIS_COLOR, 2)
                cv2.circle(viz, (int(right_center[0]), int(right_center[1])), right_radius, IRIS_COLOR, 2)
                
                # Draw iris centers
                cv2.circle(viz, (int(left_center[0]), int(left_center[1])), 5, CENTER_COLOR, -1)
                cv2.circle(viz, (int(right_center[0]), int(right_center[1])), 5, CENTER_COLOR, -1)
                
                # Draw iris width lines
                cv2.line(viz, (int(left_iris_l[0]), int(left_iris_l[1])), 
                        (int(left_iris_r[0]), int(left_iris_r[1])), IRIS_COLOR, 1)
                cv2.line(viz, (int(right_iris_l[0]), int(right_iris_l[1])), 
                        (int(right_iris_r[0]), int(right_iris_r[1])), IRIS_COLOR, 1)
                
                # Draw PD line between centers
                cv2.line(viz, (int(left_center[0]), int(left_center[1])),
                        (int(right_center[0]), int(right_center[1])), PD_COLOR, 2)
                
            except (IndexError, KeyError):
                pass
        
        # Add text overlay with semi-transparent background
        texts = []
        if result.get('pd_mm'):
            texts.append(f"PD: {result['pd_mm']:.1f} mm")
        if result.get('iris_diameter_px'):
            texts.append(f"Iris: {result['iris_diameter_px']:.1f} px")
        if result.get('left_iris_px') and result.get('right_iris_px'):
            texts.append(f"L: {result['left_iris_px']:.1f}  R: {result['right_iris_px']:.1f} px")
        if result.get('confidence'):
            texts.append(f"Conf: {result['confidence']:.2f}")
        if result.get('quality_score'):
            texts.append(f"Quality: {result['quality_score']:.2f}")
        texts.append(f"Algorithm: {result.get('algorithm', 'ratio')}")
        texts.append(f"HVID: {HVID_MM}mm | Bias: {result.get('bias_factor', 'N/A')}")
        
        # Draw text with background
        y_offset = 30
        for text in texts:
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(viz, (10, y_offset - 20), (15 + text_size[0], y_offset + 5), BG_COLOR, -1)
            cv2.putText(viz, text, (12, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)
            y_offset += 30
        
        # Add validity indicator
        validity_text = "VALID" if result.get('is_valid') else "INVALID"
        validity_color = (0, 255, 0) if result.get('is_valid') else (0, 0, 255)
        cv2.putText(viz, validity_text, (w - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, validity_color, 2)
        
        return viz

