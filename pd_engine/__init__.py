"""
PD (Pupillary Distance) Measurement Core Engine

A medical-grade library for calculating pupillary distance from images/video
using computer vision and deep learning.
"""

from .core import PDMeasurement, PDResult

__version__ = "0.1.0"
__all__ = ["PDMeasurement", "PDResult"]
