"""
Rimloo Backend Package

This package contains the PD measurement engine and backend services.
"""

from .pd_engine.core import PDMeasurement, PDResult
from .pd_engine.calibration import CardCalibration, CardDetectionResult
from .pd_engine.measurement import IrisMeasurer, IrisMeasurement
from .pd_engine.corrections import PDCorrector, CorrectionResult
from .pd_engine.sam3_segmentation import SAM3Segmenter, SAM3SegmentationResult
from .pd_engine.forehead_detection import ForeheadDetector, ForeheadROI

__all__ = [
    "PDMeasurement",
    "PDResult",
    "CardCalibration",
    "CardDetectionResult",
    "IrisMeasurer",
    "IrisMeasurement",
    "PDCorrector",
    "CorrectionResult",
    "SAM3Segmenter",
    "SAM3SegmentationResult",
    "ForeheadDetector",
    "ForeheadROI",
]
