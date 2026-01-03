"""
Measurement Module - Iris Detection and Raw PD Calculation

This module uses MediaPipe Face Mesh with refined landmarks to detect
iris centers and calculate raw pupillary distance in pixels.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, List
from dataclasses import dataclass, field

from .utils import (
    LEFT_IRIS_CENTER,
    RIGHT_IRIS_CENTER,
    MAX_YAW_DEGREES,
    euclidean_distance,
    rotation_matrix_to_euler_angles,
    refine_iris_centers_subpixel,
)


@dataclass
class HeadPose:
    """Head pose angles in degrees."""
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    
    @property
    def is_frontal(self) -> bool:
        """Check if head is approximately frontal (yaw < threshold)."""
        return abs(self.yaw) < MAX_YAW_DEGREES


@dataclass
class IrisMeasurement:
    """Result of iris detection and measurement."""
    detected: bool
    left_iris: Optional[Tuple[float, float]] = None  # (x, y) in pixels
    right_iris: Optional[Tuple[float, float]] = None  # (x, y) in pixels
    raw_pd_px: Optional[float] = None  # Raw PD in pixels
    iris_diameter_px: Optional[float] = None  # Average iris diameter in pixels
    head_pose: Optional[HeadPose] = None
    face_landmarks: Optional[np.ndarray] = None
    confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)
    error_message: Optional[str] = None


class IrisMeasurer:
    """
    Iris detection and PD measurement using MediaPipe Face Mesh.
    
    Uses the refined landmarks model for accurate iris tracking.
    Landmarks 468 (left iris) and 473 (right iris) are used as they
    are robust against head pose changes and eyelid occlusion.
    """
    
    def __init__(
        self,
        static_image_mode: bool = True,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize the iris measurer using FaceLandmarker Tasks API.
        
        Args:
            static_image_mode: If True, treats each image independently
            max_num_faces: Maximum number of faces to detect
            refine_landmarks: Enable refined landmarks for iris tracking
            min_detection_confidence: Minimum detection confidence
            min_tracking_confidence: Minimum tracking confidence
        """
        import os
        
        # Get model path
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "face_landmarker.task"
        )
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Face landmarker model not found at {model_path}. "
                "Download from: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
            )
        
        # Create FaceLandmarker with new Tasks API
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )
        self.face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        
        # 3D model points for head pose estimation
        # These are canonical face model coordinates
        self.model_points = np.array([
            [0.0, 0.0, 0.0],          # Nose tip (landmark 1)
            [0.0, -330.0, -65.0],     # Chin (landmark 199)
            [-225.0, 170.0, -135.0],  # Left eye corner (landmark 33)
            [225.0, 170.0, -135.0],   # Right eye corner (landmark 263)
            [-150.0, -150.0, -125.0], # Left mouth corner (landmark 61)
            [150.0, -150.0, -125.0]   # Right mouth corner (landmark 291)
        ], dtype=np.float64)
        
        # Landmark indices for head pose estimation
        self.pose_landmarks = [1, 199, 33, 263, 61, 291]
    
    def measure(self, image: np.ndarray) -> IrisMeasurement:
        """
        Detect iris centers and measure raw PD.
        
        Args:
            image: BGR input image
            
        Returns:
            IrisMeasurement with detection results
        """
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # Process with FaceLandmarker (new Tasks API)
        results = self.face_landmarker.detect(mp_image)
        
        if not results.face_landmarks:
            return IrisMeasurement(
                detected=False,
                error_message="No face detected in image"
            )
        
        # Get first face landmarks
        face_landmarks = results.face_landmarks[0]
        h, w = image.shape[:2]
        
        # Convert landmarks to pixel coordinates
        landmarks_px = np.array([
            [lm.x * w, lm.y * h, lm.z * w]
            for lm in face_landmarks
        ])
        
        # Extract iris centers (landmarks 468 and 473)
        # Note: These are available only with refine_landmarks=True
        try:
            left_iris_approx = (landmarks_px[LEFT_IRIS_CENTER][0], landmarks_px[LEFT_IRIS_CENTER][1])
            right_iris_approx = (landmarks_px[RIGHT_IRIS_CENTER][0], landmarks_px[RIGHT_IRIS_CENTER][1])
        except IndexError:
            return IrisMeasurement(
                detected=False,
                error_message="Iris landmarks not available. Ensure refine_landmarks=True"
            )
        
        # IMPROVEMENT: Refine iris centers to sub-pixel accuracy
        left_iris, right_iris = refine_iris_centers_subpixel(
            image, left_iris_approx, right_iris_approx, roi_size=50
        )
        
        # Calculate raw PD in pixels
        raw_pd_px = euclidean_distance(left_iris, right_iris)
        
        # Calculate iris diameter (using horizontal iris landmarks)
        # Left iris: 469 (right), 471 (left) - horizontal diameter
        # Right iris: 474 (right), 476 (left) - horizontal diameter
        try:
            left_iris_width = euclidean_distance(
                (landmarks_px[469][0], landmarks_px[469][1]),
                (landmarks_px[471][0], landmarks_px[471][1])
            )
            right_iris_width = euclidean_distance(
                (landmarks_px[474][0], landmarks_px[474][1]),
                (landmarks_px[476][0], landmarks_px[476][1])
            )
            iris_diameter_px = (left_iris_width + right_iris_width) / 2
        except IndexError:
            iris_diameter_px = None
        
        # IMPROVEMENT: Estimate head pose with enhanced PnP method
        head_pose = self._estimate_head_pose(landmarks_px, image.shape, use_pnp=True)
        
        # Generate warnings
        warnings = []
        if head_pose and not head_pose.is_frontal:
            warnings.append(
                f"Head yaw angle ({head_pose.yaw:.1f}°) exceeds {MAX_YAW_DEGREES}°. "
                "Please face the camera directly for best accuracy."
            )
        
        # Calculate confidence based on detection quality
        confidence = self._calculate_confidence(face_landmarks, head_pose)
        
        return IrisMeasurement(
            detected=True,
            left_iris=left_iris,
            right_iris=right_iris,
            raw_pd_px=raw_pd_px,
            iris_diameter_px=iris_diameter_px,
            head_pose=head_pose,
            face_landmarks=landmarks_px,
            confidence=confidence,
            warnings=warnings
        )
    
    def _estimate_head_pose(
        self,
        landmarks_px: np.ndarray,
        image_shape: Tuple[int, ...],
        use_pnp: bool = True
    ) -> Optional[HeadPose]:
        """
        Estimate 3D head pose from facial landmarks.
        
        IMPROVEMENT: Uses PnP with 3D face model for more accurate pose estimation.
        Falls back to simple method if PnP fails.
        
        Args:
            landmarks_px: Array of landmark pixel coordinates
            image_shape: Image shape (h, w, c)
            use_pnp: Whether to use PnP method (default: True)
            
        Returns:
            HeadPose with roll, pitch, yaw angles
        """
        import math
        
        try:
            h, w = image_shape[:2] if len(image_shape) >= 2 else (image_shape[0], image_shape[1] if len(image_shape) > 1 else image_shape[0])
            
            # IMPROVEMENT: Try PnP method first for better accuracy
            if use_pnp:
                try:
                    # 3D face model points (in mm, relative to face center)
                    # Based on average human face dimensions
                    model_points_3d = np.array([
                        [0.0, 0.0, 0.0],          # Nose tip (landmark 1)
                        [0.0, -125.0, -20.0],     # Chin (landmark 152)
                        [-50.0, 0.0, -30.0],      # Left eye outer (landmark 33)
                        [50.0, 0.0, -30.0],       # Right eye outer (landmark 263)
                        [-35.0, -50.0, -25.0],    # Left mouth corner (landmark 61)
                        [35.0, -50.0, -25.0],     # Right mouth corner (landmark 291)
                    ], dtype=np.float64)
                    
                    # Corresponding 2D image points
                    image_points_2d = np.array([
                        landmarks_px[1][:2],      # Nose tip
                        landmarks_px[152][:2],    # Chin
                        landmarks_px[33][:2],     # Left eye
                        landmarks_px[263][:2],    # Right eye
                        landmarks_px[61][:2],     # Left mouth
                        landmarks_px[291][:2],    # Right mouth
                    ], dtype=np.float64)
                    
                    # Estimate camera matrix (will be refined if card detected)
                    focal_length = w  # Rough estimate
                    camera_matrix = np.array([
                        [focal_length, 0, w / 2],
                        [0, focal_length, h / 2],
                        [0, 0, 1]
                    ], dtype=np.float64)
                    
                    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
                    
                    # Solve PnP
                    success, rvec, tvec = cv2.solvePnP(
                        model_points_3d, image_points_2d, camera_matrix, dist_coeffs
                    )
                    
                    if success:
                        # Convert rotation vector to matrix
                        R, _ = cv2.Rodrigues(rvec)
                        
                        # Convert to Euler angles
                        roll, pitch, yaw = rotation_matrix_to_euler_angles(R)
                        
                        return HeadPose(roll=roll, pitch=pitch, yaw=yaw)
                except Exception:
                    # Fall through to simple method
                    pass
            
            # Fallback: Simple method (original implementation)
            # Use key landmarks for pose estimation
            # Nose tip: 1, Left eye outer: 33, Right eye outer: 263
            # Left mouth: 61, Right mouth: 291, Forehead: 10
            
            nose_tip = landmarks_px[1][:2]
            left_eye = landmarks_px[33][:2]
            right_eye = landmarks_px[263][:2]
            left_mouth = landmarks_px[61][:2]
            right_mouth = landmarks_px[291][:2]
            chin = landmarks_px[152][:2]
            forehead = landmarks_px[10][:2]
            
            # Calculate eye center
            eye_center = ((left_eye[0] + right_eye[0]) / 2, 
                         (left_eye[1] + right_eye[1]) / 2)
            
            # Calculate mouth center
            mouth_center = ((left_mouth[0] + right_mouth[0]) / 2,
                           (left_mouth[1] + right_mouth[1]) / 2)
            
            # ROLL: Angle of the line connecting the eyes
            eye_dx = right_eye[0] - left_eye[0]
            eye_dy = right_eye[1] - left_eye[1]
            roll = math.degrees(math.atan2(eye_dy, eye_dx))
            
            # YAW: Based on nose position relative to eye center
            # If nose is to the left of eye center, head is turned right (positive yaw)
            eye_width = euclidean_distance(left_eye, right_eye)
            nose_offset = nose_tip[0] - eye_center[0]
            # Normalize by eye width and convert to angle estimate
            # When nose is centered, yaw = 0
            # Max offset would be about 0.5 * eye_width for 45 degree turn
            yaw_ratio = nose_offset / (eye_width * 0.5) if eye_width > 0 else 0
            yaw = math.degrees(math.asin(max(-1, min(1, yaw_ratio * 0.7))))
            
            # PITCH: Based on vertical position of nose relative to eyes and mouth
            # When looking up, nose moves up relative to eyes
            # When looking down, nose moves down
            face_height = euclidean_distance(forehead, chin)
            expected_nose_y = (eye_center[1] + mouth_center[1]) / 2
            nose_offset_y = nose_tip[1] - expected_nose_y
            pitch_ratio = nose_offset_y / (face_height * 0.2) if face_height > 0 else 0
            pitch = math.degrees(math.asin(max(-1, min(1, pitch_ratio * 0.5))))
            
            return HeadPose(roll=roll, pitch=pitch, yaw=yaw)
            
        except (IndexError, ZeroDivisionError):
            return None
    
    def _calculate_confidence(
        self,
        face_landmarks,
        head_pose: Optional[HeadPose]
    ) -> float:
        """
        Calculate measurement confidence based on detection quality.
        
        Args:
            face_landmarks: MediaPipe face landmarks
            head_pose: Estimated head pose
            
        Returns:
            Confidence score between 0 and 1
        """
        confidence = 0.8  # Base confidence for successful detection
        
        # Reduce confidence based on head rotation
        if head_pose:
            yaw_penalty = min(abs(head_pose.yaw) / 45.0, 0.3)
            pitch_penalty = min(abs(head_pose.pitch) / 45.0, 0.2)
            confidence -= (yaw_penalty + pitch_penalty)
        
        return max(0.0, min(1.0, confidence))
    
    def close(self):
        """Release resources."""
        self.face_landmarker.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
