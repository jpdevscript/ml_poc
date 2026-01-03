"""
Temporal Filtering Module - Kalman Filter for Video Sequences

Provides temporal smoothing of PD measurements across video frames
to reduce noise and improve accuracy.
"""

import numpy as np
import cv2
from typing import Optional, List, Tuple


class TemporalPDFilter:
    """
    Kalman filter for temporal smoothing of PD measurements.
    
    Reduces noise in video sequences by predicting and correcting
    PD values based on previous measurements.
    """
    
    def __init__(
        self,
        process_noise: float = 0.1,
        measurement_noise: float = 0.5,
        initial_pd: Optional[float] = None
    ):
        """
        Initialize temporal filter.
        
        Args:
            process_noise: Process noise covariance (how much PD can change between frames)
            measurement_noise: Measurement noise covariance (uncertainty in measurements)
            initial_pd: Initial PD value (None to auto-initialize)
        """
        # 1D Kalman filter (state = PD value)
        self.kf = cv2.KalmanFilter(1, 1)
        
        # State transition matrix (PD stays constant + process noise)
        self.kf.transitionMatrix = np.array([[1.0]], dtype=np.float32)
        
        # Measurement matrix (we directly observe PD)
        self.kf.measurementMatrix = np.array([[1.0]], dtype=np.float32)
        
        # Process noise covariance
        self.kf.processNoiseCov = np.array([[process_noise]], dtype=np.float32)
        
        # Measurement noise covariance
        self.kf.measurementNoiseCov = np.array([[measurement_noise]], dtype=np.float32)
        
        # Error covariance (start with high uncertainty)
        self.kf.errorCovPost = np.array([[1.0]], dtype=np.float32)
        
        # Initialize state
        if initial_pd is not None:
            self.kf.statePre = np.array([[initial_pd]], dtype=np.float32)
            self.kf.statePost = np.array([[initial_pd]], dtype=np.float32)
        else:
            self.kf.statePre = np.array([[65.0]], dtype=np.float32)  # Typical PD
            self.kf.statePost = np.array([[65.0]], dtype=np.float32)
        
        self.initialized = False
        self.measurement_count = 0
    
    def update(self, pd_measurement: float, confidence: float = 1.0) -> float:
        """
        Update filter with new PD measurement.
        
        Args:
            pd_measurement: New PD measurement in mm
            confidence: Confidence in measurement (0-1), used to adjust measurement noise
            
        Returns:
            Filtered PD value
        """
        # Adjust measurement noise based on confidence
        # Lower confidence = higher noise = less weight to measurement
        adjusted_noise = self.kf.measurementNoiseCov[0, 0] / max(confidence, 0.1)
        self.kf.measurementNoiseCov = np.array([[adjusted_noise]], dtype=np.float32)
        
        # Predict
        prediction = self.kf.predict()
        
        # Correct with measurement
        measurement = np.array([[pd_measurement]], dtype=np.float32)
        self.kf.correct(measurement)
        
        self.measurement_count += 1
        if not self.initialized and self.measurement_count >= 2:
            self.initialized = True
        
        return float(self.kf.statePost[0])
    
    def predict(self) -> float:
        """
        Predict next PD value without measurement.
        
        Returns:
            Predicted PD value
        """
        prediction = self.kf.predict()
        return float(prediction[0])
    
    def reset(self, initial_pd: Optional[float] = None):
        """Reset filter to initial state."""
        if initial_pd is not None:
            self.kf.statePre = np.array([[initial_pd]], dtype=np.float32)
            self.kf.statePost = np.array([[initial_pd]], dtype=np.float32)
        else:
            self.kf.statePre = np.array([[65.0]], dtype=np.float32)
            self.kf.statePost = np.array([[65.0]], dtype=np.float32)
        
        self.kf.errorCovPost = np.array([[1.0]], dtype=np.float32)
        self.initialized = False
        self.measurement_count = 0
    
    def get_uncertainty(self) -> float:
        """
        Get current uncertainty (standard deviation) of filtered PD.
        
        Returns:
            Uncertainty in mm
        """
        return float(np.sqrt(self.kf.errorCovPost[0, 0]))


def filter_pd_sequence(
    pd_measurements: List[float],
    confidences: Optional[List[float]] = None,
    process_noise: float = 0.1,
    measurement_noise: float = 0.5
) -> Tuple[List[float], List[float]]:
    """
    Apply temporal filtering to a sequence of PD measurements.
    
    Args:
        pd_measurements: List of PD measurements
        confidences: Optional list of confidence scores
        process_noise: Process noise for Kalman filter
        measurement_noise: Measurement noise for Kalman filter
        
    Returns:
        Tuple of (filtered_pd_values, uncertainties)
    """
    if confidences is None:
        confidences = [1.0] * len(pd_measurements)
    
    filter_obj = TemporalPDFilter(
        process_noise=process_noise,
        measurement_noise=measurement_noise
    )
    
    filtered_values = []
    uncertainties = []
    
    for pd, conf in zip(pd_measurements, confidences):
        filtered_pd = filter_obj.update(pd, conf)
        uncertainty = filter_obj.get_uncertainty()
        filtered_values.append(filtered_pd)
        uncertainties.append(uncertainty)
    
    return filtered_values, uncertainties

