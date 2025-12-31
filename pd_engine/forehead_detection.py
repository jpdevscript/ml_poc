"""
Forehead Detection Module - MediaPipe-based Forehead ROI Extraction

Uses MediaPipe FaceLandmarker Tasks API to identify the forehead region
where the calibration card is typically held. This provides a more
accurate ROI for card detection than generic object detection.

Key Landmarks:
- 10: Glabella (between eyebrows)
- 151: Upper forehead center
- 67/297: Left/right temples
- 103/332: Outer eyebrow ends
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple
from dataclasses import dataclass
import os


@dataclass
class ForeheadROI:
    """Result of forehead detection."""
    detected: bool
    roi_box: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    glabella_point: Optional[Tuple[int, int]] = None  # Center reference point
    forehead_landmarks: Optional[np.ndarray] = None  # Key forehead landmarks
    error_message: Optional[str] = None


class ForeheadDetector:
    """
    Detect forehead region using MediaPipe FaceLandmarker Tasks API.
    
    The forehead ROI is computed based on facial landmarks to provide
    a precise region where the calibration card is expected to be held.
    """
    
    # MediaPipe forehead-related landmark indices
    GLABELLA = 10  # Center between eyebrows
    FOREHEAD_CENTER = 151  # Upper forehead
    LEFT_TEMPLE = 67
    RIGHT_TEMPLE = 297
    LEFT_EYEBROW_OUTER = 103
    RIGHT_EYEBROW_OUTER = 332
    NOSE_TIP = 1
    
    # Class-level model cache
    _face_landmarker = None
    _face_landmarker_loaded = False
    
    def __init__(self, debug_dir: Optional[str] = None):
        """
        Initialize ForeheadDetector.
        
        Args:
            debug_dir: Directory to save debug images (None to disable)
        """
        self.debug_dir = debug_dir
        self._step = 0
        self._load_face_landmarker()
    
    def _load_face_landmarker(self):
        """Lazy load MediaPipe FaceLandmarker (Tasks API)."""
        if ForeheadDetector._face_landmarker_loaded:
            return
        
        try:
            # Get model path (same as used in measurement.py)
            model_path = os.path.join(
                os.path.dirname(__file__),  # pd_engine directory
                "..",
                "face_landmarker.task"
            )
            model_path = os.path.abspath(model_path)
            
            if not os.path.exists(model_path):
                print(f"[ForeheadDetector] Model not found at: {model_path}")
                ForeheadDetector._face_landmarker_loaded = True
                return
            
            # Create FaceLandmarker with Tasks API
            base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False
            )
            ForeheadDetector._face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
            ForeheadDetector._face_landmarker_loaded = True
            
        except Exception as e:
            print(f"[ForeheadDetector] Failed to load FaceLandmarker: {e}")
            import traceback
            traceback.print_exc()
            ForeheadDetector._face_landmarker_loaded = True
    
    def _save_debug(self, name: str, image: np.ndarray):
        """Save debug image if debug_dir is set."""
        if self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)
            path = os.path.join(self.debug_dir, f"{self._step:02d}_{name}.jpg")
            cv2.imwrite(path, image)
            print(f"  [Forehead-{self._step:02d}] Saved: {name}")
            self._step += 1
    
    def detect(self, image: np.ndarray) -> ForeheadROI:
        """
        Detect forehead region in image.
        
        Args:
            image: BGR input image
            
        Returns:
            ForeheadROI with detected region
        """
        self._step = 0
        
        if ForeheadDetector._face_landmarker is None:
            return ForeheadROI(
                detected=False,
                error_message="MediaPipe FaceLandmarker not available"
            )
        
        h, w = image.shape[:2]
        
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # Detect landmarks
        results = ForeheadDetector._face_landmarker.detect(mp_image)
        
        if not results.face_landmarks:
            return ForeheadROI(
                detected=False,
                error_message="No face detected"
            )
        
        # Get first face landmarks
        face_landmarks = results.face_landmarks[0]
        
        # Convert to pixel coordinates
        def lm_to_px(idx):
            lm = face_landmarks[idx]
            return (int(lm.x * w), int(lm.y * h))
        
        # Get key forehead landmarks
        glabella = lm_to_px(self.GLABELLA)
        forehead_center = lm_to_px(self.FOREHEAD_CENTER)
        left_temple = lm_to_px(self.LEFT_TEMPLE)
        right_temple = lm_to_px(self.RIGHT_TEMPLE)
        left_eyebrow = lm_to_px(self.LEFT_EYEBROW_OUTER)
        right_eyebrow = lm_to_px(self.RIGHT_EYEBROW_OUTER)
        nose_tip = lm_to_px(self.NOSE_TIP)
        
        # Store key landmarks for debugging
        forehead_landmarks = np.array([
            glabella, forehead_center, left_temple, right_temple,
            left_eyebrow, right_eyebrow
        ])
        
        # Landmarks stored in forehead_landmarks for reference
        
        # Calculate forehead ROI
        # The card on forehead extends:
        # - Horizontally: well beyond temple width (card is ~85.6mm wide, typically wider than face)
        # - Vertically: from below eyebrow level up well beyond top of head
        
        # Face width as reference (between temples)
        face_width = abs(right_temple[0] - left_temple[0])
        face_height = abs(nose_tip[1] - forehead_center[1])
        
        # Estimate inter-pupillary distance (roughly 60-70mm)
        # Card is 85.6mm wide, so card width ~= 1.3x IPD
        # Card should extend beyond temples significantly
        
        # Calculate ROI bounds - MUCH LARGER to capture card
        # X: extend well beyond temples (card is wider than face)
        horizontal_padding = int(face_width * 0.6)  # 60% extra on each side
        roi_x1 = left_temple[0] - horizontal_padding
        roi_x2 = right_temple[0] + horizontal_padding
        
        # Y: Card is held above the eyebrows, resting on forehead/hairline
        # Need to extend well above the forehead center
        card_height_estimate = int(face_width * 0.65)  # Card aspect ~1.586, so height ~63% of width
        
        # Bottom: include glabella area plus some margin below (card may rest on brow ridge)
        roi_y2 = glabella[1] + int(face_height * 0.25)  # 25% below glabella (more generous)
        
        # Top: extend upward for full card + margin (card may extend above head in image)
        roi_y1 = glabella[1] - int(card_height_estimate * 2.5)  # 2.5x card height above glabella
        
        # Clamp to image bounds
        roi_x1 = max(0, roi_x1)
        roi_y1 = max(0, roi_y1)
        roi_x2 = min(w, roi_x2)
        roi_y2 = min(h, roi_y2)
        
        # Validate ROI size
        roi_w = roi_x2 - roi_x1
        roi_h = roi_y2 - roi_y1
        
        if roi_w < 50 or roi_h < 30:
            return ForeheadROI(
                detected=False,
                glabella_point=glabella,
                error_message=f"ROI too small: {roi_w}x{roi_h}"
            )
        
        # ROI computed successfully
        
        return ForeheadROI(
            detected=True,
            roi_box=(roi_x1, roi_y1, roi_x2, roi_y2),
            glabella_point=glabella,
            forehead_landmarks=forehead_landmarks
        )
    
    def extract_roi(self, image: np.ndarray, 
                   roi_result: ForeheadROI) -> Optional[np.ndarray]:
        """
        Extract ROI image from detection result.
        
        Args:
            image: Original BGR image
            roi_result: ForeheadROI detection result
            
        Returns:
            Cropped ROI image or None
        """
        if not roi_result.detected or roi_result.roi_box is None:
            return None
        
        x1, y1, x2, y2 = roi_result.roi_box
        roi_image = image[y1:y2, x1:x2].copy()
        
        # ROI extracted
        
        return roi_image
