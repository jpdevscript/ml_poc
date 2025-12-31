"""
Error Correction Pipeline - Medical Grade Physics Corrections

This module implements the three critical corrections needed to convert
raw PD measurements to medical-grade accuracy:

1. 3D Head Pose (Cosine Error) - Correct for head rotation
2. Depth Parallax (Forehead Offset) - Correct for card-to-eye depth difference
3. Vergence (Near-to-Far) - Correct for eye convergence during selfie
"""

import math
from typing import Optional, Tuple
from dataclasses import dataclass

from .utils import (
    BROW_TO_CORNEA_OFFSET_MM,
    EYE_CENTER_OF_ROTATION_MM,
    MAX_YAW_DEGREES,
)
from .measurement import HeadPose


@dataclass
class CorrectionResult:
    """Result of applying all corrections."""
    
    # Input values
    raw_pd_px: float
    scale_factor_mm_per_px: float
    camera_distance_mm: float
    head_pose: Optional[HeadPose]
    
    # Intermediate values
    pd_metric_mm: float  # After scale conversion
    pd_yaw_corrected_mm: float  # After head pose correction
    pd_depth_corrected_mm: float  # After depth parallax correction
    pd_final_mm: float  # After vergence correction
    
    # Correction factors applied
    yaw_correction_factor: float
    depth_correction_factor: float
    vergence_correction_factor: float
    
    # Quality metrics
    confidence: float
    warnings: list
    
    @property
    def total_correction_factor(self) -> float:
        """Total multiplicative correction applied."""
        return self.yaw_correction_factor * self.depth_correction_factor * self.vergence_correction_factor


class PDCorrector:
    """
    Error correction pipeline for medical-grade PD measurement.
    
    Applies three corrections based on real-world physics:
    1. Head rotation (cosine error)
    2. Depth parallax (card vs eye depth)
    3. Vergence (eye convergence for near objects)
    """
    
    def __init__(
        self,
        brow_to_cornea_offset_mm: float = BROW_TO_CORNEA_OFFSET_MM,
        eye_cor_distance_mm: float = EYE_CENTER_OF_ROTATION_MM,
        max_yaw_warning_degrees: float = MAX_YAW_DEGREES
    ):
        """
        Initialize the corrector.
        
        Args:
            brow_to_cornea_offset_mm: Distance from brow ridge to cornea apex
            eye_cor_distance_mm: Distance from cornea to eye center of rotation
            max_yaw_warning_degrees: Threshold for head rotation warning
        """
        self.brow_to_cornea_offset = brow_to_cornea_offset_mm
        self.eye_cor_distance = eye_cor_distance_mm
        self.max_yaw_warning = max_yaw_warning_degrees
    
    def apply_corrections(
        self,
        raw_pd_px: float,
        scale_factor: float,
        camera_distance_mm: float,
        head_pose: Optional[HeadPose] = None,
        apply_yaw_correction: bool = True,
        apply_depth_correction: bool = True,
        apply_vergence_correction: bool = True
    ) -> CorrectionResult:
        """
        Apply all corrections to raw PD measurement.
        
        The correction sequence (as per methodology):
        1. Convert to metric: PD_metric = PD_raw * S
        2. Correct for yaw: PD_unrotated = PD_metric / cos(yaw)
        3. Correct for depth: PD_depth = PD_yaw * (D + offset) / D
        4. Correct for vergence: PD_final = PD_depth * (D + 13) / D
        
        Args:
            raw_pd_px: Raw PD measurement in pixels
            scale_factor: Scale factor (mm/px) from calibration
            camera_distance_mm: Estimated distance to camera in mm
            head_pose: Detected head pose (optional)
            apply_yaw_correction: Whether to apply yaw correction
            apply_depth_correction: Whether to apply depth parallax correction
            apply_vergence_correction: Whether to apply vergence correction
            
        Returns:
            CorrectionResult with all intermediate and final values
        """
        warnings = []
        
        # Step 1: Convert to metric
        pd_metric = raw_pd_px * scale_factor
        
        # Step 2: 3D Head Pose Correction (Cosine Error)
        yaw_correction = 1.0
        if apply_yaw_correction and head_pose is not None:
            yaw_correction, yaw_warnings = self._correct_for_yaw(head_pose.yaw)
            warnings.extend(yaw_warnings)
        
        pd_yaw_corrected = pd_metric * yaw_correction
        
        # Step 3: Depth Parallax Correction
        depth_correction = 1.0
        if apply_depth_correction:
            depth_correction = self._correct_for_depth(camera_distance_mm)
        
        pd_depth_corrected = pd_yaw_corrected * depth_correction
        
        # Step 4: Vergence Correction
        vergence_correction = 1.0
        if apply_vergence_correction:
            vergence_correction = self._correct_for_vergence(camera_distance_mm)
        
        pd_final = pd_depth_corrected * vergence_correction
        
        # Calculate overall confidence
        confidence = self._calculate_confidence(head_pose, camera_distance_mm)
        
        return CorrectionResult(
            raw_pd_px=raw_pd_px,
            scale_factor_mm_per_px=scale_factor,
            camera_distance_mm=camera_distance_mm,
            head_pose=head_pose,
            pd_metric_mm=pd_metric,
            pd_yaw_corrected_mm=pd_yaw_corrected,
            pd_depth_corrected_mm=pd_depth_corrected,
            pd_final_mm=pd_final,
            yaw_correction_factor=yaw_correction,
            depth_correction_factor=depth_correction,
            vergence_correction_factor=vergence_correction,
            confidence=confidence,
            warnings=warnings
        )
    
    def _correct_for_yaw(self, yaw_degrees: float) -> Tuple[float, list]:
        """
        Apply 3D head pose correction for yaw rotation.
        
        When the head is rotated (yaw), the distance between eyes appears
        shorter due to foreshortening. We correct by dividing by cos(yaw).
        
        Formula: PD_unrotated = PD_px / cos(yaw)
        
        Args:
            yaw_degrees: Head yaw angle in degrees
            
        Returns:
            Tuple of (correction_factor, warnings)
        """
        warnings = []
        
        # Warn if yaw is too large (correction becomes unreliable)
        if abs(yaw_degrees) > self.max_yaw_warning:
            warnings.append(
                f"Head yaw angle ({yaw_degrees:.1f}°) exceeds recommended maximum "
                f"({self.max_yaw_warning}°). Accuracy may be reduced."
            )
        
        # Clamp yaw to prevent division by very small cosine
        safe_yaw = min(abs(yaw_degrees), 60)  # Don't allow more than 60°
        
        # Calculate correction factor: 1 / cos(yaw)
        yaw_radians = math.radians(safe_yaw)
        correction = 1.0 / math.cos(yaw_radians)
        
        return correction, warnings
    
    def _correct_for_depth(self, camera_distance_mm: float) -> float:
        """
        Apply depth parallax correction for card-to-eye depth difference.
        
        The card rests on the forehead (supraorbital ridge), but the eyes
        (cornea) are set back ~12mm into the skull. This means the card
        is closer to the camera than the eyes, making the eyes appear
        relatively smaller.
        
        Formula: PD_corrected = PD_metric * (D + offset) / D
        
        Args:
            camera_distance_mm: Distance from camera to subject in mm
            
        Returns:
            Correction factor
        """
        if camera_distance_mm <= 0:
            return 1.0
        
        # (D + offset) / D
        correction = (camera_distance_mm + self.brow_to_cornea_offset) / camera_distance_mm
        
        return correction
    
    def _correct_for_vergence(self, camera_distance_mm: float) -> float:
        """
        Apply vergence correction for eye convergence.
        
        When taking a selfie, eyes converge (look inward) to focus on the
        camera. Glasses are made for "Distance PD" (looking at infinity).
        
        The eyes rotate around their Center of Rotation (COR), which is
        ~13mm behind the cornea.
        
        Formula: PD_distance = PD_corrected * (D + 13) / D
        
        Alternative (clinical approximation): Add 3.0mm if D ≈ 400mm
        
        Args:
            camera_distance_mm: Distance from camera to subject in mm
            
        Returns:
            Correction factor
        """
        if camera_distance_mm <= 0:
            return 1.0
        
        # (D + COR_distance) / D
        correction = (camera_distance_mm + self.eye_cor_distance) / camera_distance_mm
        
        return correction
    
    def _calculate_confidence(
        self,
        head_pose: Optional[HeadPose],
        camera_distance_mm: float
    ) -> float:
        """
        Calculate overall confidence in the corrected measurement.
        
        Args:
            head_pose: Detected head pose
            camera_distance_mm: Estimated camera distance
            
        Returns:
            Confidence score between 0 and 1
        """
        confidence = 0.9  # Base confidence
        
        # Reduce confidence for extreme yaw
        if head_pose is not None:
            if abs(head_pose.yaw) > 15:
                confidence -= 0.2
            elif abs(head_pose.yaw) > self.max_yaw_warning:
                confidence -= 0.1
        else:
            # No head pose data means we can't correct
            confidence -= 0.1
        
        # Reduce confidence for unusual camera distances
        if camera_distance_mm < 200:  # Too close
            confidence -= 0.2
        elif camera_distance_mm < 300:
            confidence -= 0.1
        elif camera_distance_mm > 800:  # Too far
            confidence -= 0.1
        
        return max(0.0, min(1.0, confidence))
    
    @staticmethod
    def estimate_pd_without_card(
        raw_pd_px: float,
        iris_diameter_px: float,
        avg_iris_diameter_mm: float = 11.7
    ) -> float:
        """
        Estimate PD using iris diameter as a reference (fallback method).
        
        Average human iris diameter is ~11.7mm, which can be used as a
        reference when no card is available.
        
        Note: This is less accurate than card calibration.
        
        Args:
            raw_pd_px: Raw PD in pixels
            iris_diameter_px: Detected iris diameter in pixels
            avg_iris_diameter_mm: Average iris diameter (default: 11.7mm)
            
        Returns:
            Estimated PD in mm
        """
        if iris_diameter_px <= 0:
            return 0.0
        
        scale_factor = avg_iris_diameter_mm / iris_diameter_px
        return raw_pd_px * scale_factor
