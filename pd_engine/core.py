"""
Core PD Measurement Engine

This module provides the main PDMeasurement class that orchestrates
all modules to produce medical-grade pupillary distance measurements.
"""

import cv2
import numpy as np
from typing import Optional, List, Union
from dataclasses import dataclass, field
from pathlib import Path

from .calibration import CardCalibration, CardDetectionResult
from .measurement import IrisMeasurer, IrisMeasurement, HeadPose
from .corrections import PDCorrector, CorrectionResult
from .photogrammetry import calculate_precise_pd, PhotogrammetryResult
from .utils import (
    calculate_camera_distance,
    draw_landmarks_on_image,
    CARD_WIDTH_MM,
)


@dataclass
class PDResult:
    """
    Complete result of PD measurement.
    
    Contains all intermediate values, corrections applied, and final measurement.
    """
    
    # Detection results
    face_detected: bool = False
    card_detected: bool = False
    
    # Raw measurements
    left_iris_px: Optional[tuple] = None
    right_iris_px: Optional[tuple] = None
    raw_pd_px: Optional[float] = None
    
    # Calibration
    card_corners: Optional[np.ndarray] = None
    card_width_px: Optional[float] = None
    scale_factor_mm_per_px: Optional[float] = None
    camera_distance_mm: Optional[float] = None
    
    # Head pose
    head_pose: Optional[HeadPose] = None
    
    # Corrections applied
    pd_metric_mm: Optional[float] = None
    pd_yaw_corrected_mm: Optional[float] = None
    pd_depth_corrected_mm: Optional[float] = None
    
    # Final result
    pd_final_mm: Optional[float] = None
    
    # Quality metrics
    confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    # Method used
    calibration_method: str = "none"  # "card", "iris", "manual"
    
    @property
    def is_valid(self) -> bool:
        """Check if result contains a valid PD measurement."""
        return self.pd_final_mm is not None and self.face_detected
    
    @property
    def is_medical_grade(self) -> bool:
        """Check if measurement meets medical-grade criteria."""
        return (
            self.is_valid and
            self.card_detected and
            self.confidence >= 0.7 and
            (self.head_pose is None or abs(self.head_pose.yaw) < 5)
        )
    
    def to_dict(self) -> dict:
        """Convert result to dictionary for serialization."""
        return {
            "face_detected": self.face_detected,
            "card_detected": self.card_detected,
            "raw_pd_px": self.raw_pd_px,
            "card_width_px": self.card_width_px,
            "scale_factor_mm_per_px": self.scale_factor_mm_per_px,
            "camera_distance_mm": self.camera_distance_mm,
            "head_pose": {
                "roll": self.head_pose.roll,
                "pitch": self.head_pose.pitch,
                "yaw": self.head_pose.yaw
            } if self.head_pose else None,
            "pd_metric_mm": self.pd_metric_mm,
            "pd_yaw_corrected_mm": self.pd_yaw_corrected_mm,
            "pd_depth_corrected_mm": self.pd_depth_corrected_mm,
            "pd_final_mm": self.pd_final_mm,
            "confidence": self.confidence,
            "calibration_method": self.calibration_method,
            "is_valid": self.is_valid,
            "is_medical_grade": self.is_medical_grade,
            "warnings": self.warnings,
            "errors": self.errors
        }
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        if not self.is_valid:
            return f"PDResult(invalid, errors={self.errors})"
        
        grade = "✓ Medical Grade" if self.is_medical_grade else "⚠ Standard"
        return (
            f"PDResult(PD={self.pd_final_mm:.1f}mm, "
            f"confidence={self.confidence:.0%}, {grade})"
        )


class PDMeasurement:
    """
    Main engine for measuring pupillary distance from images or video.
    
    Combines card calibration, iris detection, and error corrections
    to produce medical-grade PD measurements.
    
    Usage:
        engine = PDMeasurement()
        result = engine.process_image("photo.jpg")
        print(f"PD: {result.pd_final_mm}mm")
    """
    
    def __init__(
        self,
        fallback_to_iris: bool = True,
        default_camera_distance_mm: float = 400.0
    ):
        """
        Initialize the PD measurement engine.
        
        Args:
            fallback_to_iris: Use iris diameter if card not detected
            default_camera_distance_mm: Default camera distance if not estimable
        """
        self.iris_measurer = IrisMeasurer()
        self.corrector = PDCorrector()
        
        self.fallback_to_iris = fallback_to_iris
        self.default_camera_distance = default_camera_distance_mm
    
    def process_image(
        self,
        image_path: Union[str, Path],
        manual_scale_factor: Optional[float] = None,
        debug_dir: Optional[str] = None
    ) -> PDResult:
        """
        Process a single image to measure PD.
        
        Args:
            image_path: Path to the input image
            manual_scale_factor: Optional manual scale factor (mm/px)
            debug_dir: Optional directory to save intermediate detection images
            
        Returns:
            PDResult with measurement details
        """
        # Load image
        image_path = Path(image_path)
        if not image_path.exists():
            return PDResult(errors=[f"Image not found: {image_path}"])
        
        image = cv2.imread(str(image_path))
        if image is None:
            return PDResult(errors=[f"Failed to load image: {image_path}"])
        
        return self.process_frame(image, manual_scale_factor, debug_dir)
    
    def process_frame(
        self,
        frame: np.ndarray,
        manual_scale_factor: Optional[float] = None,
        debug_dir: Optional[str] = None
    ) -> PDResult:
        """
        Process a single frame to measure PD.
        
        Args:
            frame: BGR image as numpy array
            manual_scale_factor: Optional manual scale factor (mm/px)
            debug_dir: Optional directory to save intermediate detection images
            
        Returns:
            PDResult with measurement details
        """
        result = PDResult()
        
        # Step 1: Detect iris and measure raw PD
        iris_result = self.iris_measurer.measure(frame)
        
        if not iris_result.detected:
            result.errors.append(iris_result.error_message or "Face detection failed")
            return result
        
        result.face_detected = True
        result.left_iris_px = iris_result.left_iris
        result.right_iris_px = iris_result.right_iris
        result.raw_pd_px = iris_result.raw_pd_px
        result.head_pose = iris_result.head_pose
        result.warnings.extend(iris_result.warnings)
        
        # Step 2: Get scale factor (from card, manual, or iris fallback)
        scale_factor = None
        camera_distance = self.default_camera_distance
        
        if manual_scale_factor is not None:
            # Use manually provided scale factor
            scale_factor = manual_scale_factor
            result.calibration_method = "manual"
        else:
            # Try card detection using MIDV500
            card_calibration = CardCalibration(debug_dir=debug_dir)
            card_result = card_calibration.detect_card(frame, debug=bool(debug_dir))
            
            if card_result.detected:
                result.card_detected = True
                result.card_corners = card_result.corners
                result.card_width_px = card_result.card_width_px
                result.scale_factor_mm_per_px = card_result.scale_factor
                result.calibration_method = "card_3d"
                
                # Use 3D photogrammetric calculation
                photo_result = calculate_precise_pd(
                    card_corners=card_result.corners,
                    pupil_left_px=result.left_iris_px,
                    pupil_right_px=result.right_iris_px,
                    image_shape=frame.shape[:2],
                    image=frame,
                    head_pose=result.head_pose,
                    debug_dir=debug_dir,
                    debug=bool(debug_dir)
                )
                
                if photo_result.success:
                    # Use 3D photogrammetric result
                    result.camera_distance_mm = photo_result.camera_distance_mm
                    result.pd_metric_mm = photo_result.pd_near_mm
                    result.pd_depth_corrected_mm = photo_result.pd_near_mm  # No far adjustment
                    
                    # Use NEAR PD as the final result (actual measured PD)
                    # The simple_pd_from_scale already uses horizontal pupil distance
                    # which naturally corrects for head rotation, so no yaw correction needed
                    result.pd_final_mm = photo_result.pd_near_mm
                    
                    # High confidence for 3D method
                    result.confidence = 0.9 * iris_result.confidence
                    
                    return result
                else:
                    # Fallback to simple scaling if 3D failed
                    result.warnings.append(f"3D photogrammetry failed: {photo_result.error_message}")
                    scale_factor = card_result.scale_factor
                    camera_distance = calculate_camera_distance(
                        card_result.card_width_px,
                        image_width_px=frame.shape[1]
                    )
            
            if scale_factor is None and self.fallback_to_iris:
                # Fallback: Use iris diameter estimation
                # Average human iris diameter is 11.7mm (range: 10.5-13mm)
                iris_size = iris_result.iris_diameter_px if iris_result.iris_diameter_px else self._estimate_iris_size(iris_result)
                if iris_size and iris_size > 0:
                    # Use 11.7mm as the medical standard iris diameter
                    scale_factor = 11.7 / iris_size
                    result.calibration_method = "iris"
                    
                    # Estimate camera distance from iris diameter for better depth correction
                    # This uses the pinhole camera model: d = (iris_mm * focal_px) / iris_px
                    focal_length = PDCorrector.estimate_focal_length_px(frame.shape[1])
                    camera_distance = PDCorrector.estimate_camera_distance_from_iris(
                        iris_diameter_px=iris_size,
                        focal_length_px=focal_length,
                        avg_iris_diameter_mm=11.7
                    )
                    
                    result.warnings.append(
                        "No card detected. Using iris diameter estimation (less accurate)."
                    )
        
        if scale_factor is None:
            result.errors.append(
                "No calibration available. Please use a card or provide manual scale factor."
            )
            return result
        
        result.scale_factor_mm_per_px = scale_factor
        result.camera_distance_mm = camera_distance
        
        # Step 3: Apply corrections (fallback path - simple scaling)
        # IMPORTANT: Disable depth and vergence corrections as they inflate the PD
        # These corrections assume specific camera/face geometry that may not be accurate
        correction_result = self.corrector.apply_corrections(
            raw_pd_px=result.raw_pd_px,
            scale_factor=scale_factor,
            camera_distance_mm=camera_distance,
            head_pose=result.head_pose,
            apply_yaw_correction=True,  # Keep yaw for significant head turns
            apply_depth_correction=False,  # Disable - adds ~1mm
            apply_vergence_correction=False  # Disable - adds ~1-2mm
        )
        
        # Store intermediate values
        result.pd_metric_mm = correction_result.pd_metric_mm
        result.pd_yaw_corrected_mm = correction_result.pd_yaw_corrected_mm
        result.pd_depth_corrected_mm = correction_result.pd_metric_mm  # No depth correction
        result.pd_final_mm = correction_result.pd_yaw_corrected_mm  # Use yaw-corrected only
        result.warnings.extend(correction_result.warnings)
        
        # Calculate combined confidence
        correction_confidence = correction_result.confidence
        card_confidence = 1.0 if result.card_detected else 0.7
        
        result.confidence = iris_result.confidence * correction_confidence * card_confidence
        
        return result
    
    def process_video(
        self,
        video_path: Union[str, Path],
        sample_rate: int = 5,
        max_frames: int = 30,
        manual_scale_factor: Optional[float] = None
    ) -> List[PDResult]:
        """
        Process a video to measure PD across multiple frames.
        
        Args:
            video_path: Path to the input video
            sample_rate: Sample every Nth frame
            max_frames: Maximum number of frames to process
            manual_scale_factor: Optional manual scale factor (mm/px)
            
        Returns:
            List of PDResult objects, one per processed frame
        """
        video_path = Path(video_path)
        if not video_path.exists():
            return [PDResult(errors=[f"Video not found: {video_path}"])]
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return [PDResult(errors=[f"Failed to open video: {video_path}"])]
        
        results = []
        frame_count = 0
        processed_count = 0
        
        try:
            while processed_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Sample every Nth frame
                if frame_count % sample_rate != 0:
                    continue
                
                result = self.process_frame(frame, manual_scale_factor)
                results.append(result)
                processed_count += 1
        finally:
            cap.release()
        
        return results
    
    def get_aggregated_pd(
        self,
        results: List[PDResult],
        method: str = "median"
    ) -> Optional[float]:
        """
        Aggregate PD measurements from multiple frames.
        
        Args:
            results: List of PDResult objects
            method: Aggregation method ("mean", "median", "best")
            
        Returns:
            Aggregated PD in mm, or None if no valid results
        """
        valid_pds = [r.pd_final_mm for r in results if r.is_valid and r.pd_final_mm]
        
        if not valid_pds:
            return None
        
        if method == "mean":
            return sum(valid_pds) / len(valid_pds)
        elif method == "median":
            sorted_pds = sorted(valid_pds)
            n = len(sorted_pds)
            if n % 2 == 0:
                return (sorted_pds[n//2 - 1] + sorted_pds[n//2]) / 2
            return sorted_pds[n//2]
        elif method == "best":
            # Return measurement with highest confidence
            best = max(results, key=lambda r: r.confidence if r.is_valid else 0)
            return best.pd_final_mm if best.is_valid else None
        
        return None
    
    def visualize(
        self,
        image: np.ndarray,
        result: PDResult
    ) -> np.ndarray:
        """
        Create visualization of the measurement.
        
        Args:
            image: Original BGR image
            result: PDResult from processing
            
        Returns:
            Annotated image
        """
        return draw_landmarks_on_image(
            image,
            iris_left=result.left_iris_px,
            iris_right=result.right_iris_px,
            card_corners=result.card_corners,
            pd_mm=result.pd_final_mm
        )
    
    def _estimate_iris_size(self, iris_result: IrisMeasurement) -> float:
        """
        Estimate iris diameter in pixels from landmarks.
        
        This is a rough estimation used for fallback calibration.
        """
        if iris_result.face_landmarks is None:
            return 0.0
        
        # Use eye landmark distances as a proxy for iris size
        # Landmarks around the iris can be used to estimate diameter
        try:
            # Left iris landmarks (surrounding the center)
            left_landmarks = [469, 470, 471, 472]  # Points around left iris
            
            left_iris_points = iris_result.face_landmarks[left_landmarks][:, :2]
            
            # Calculate average radius
            center = iris_result.face_landmarks[468][:2]
            distances = np.sqrt(np.sum((left_iris_points - center) ** 2, axis=1))
            avg_radius = np.mean(distances)
            
            return avg_radius * 2  # Diameter
        except (IndexError, KeyError):
            return 0.0
    
    def close(self):
        """Release resources."""
        self.iris_measurer.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
