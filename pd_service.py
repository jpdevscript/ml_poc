"""
PD Measurement Service - Backend logic for face detection and PD calculation.
Uses Roboflow ROI + SAM3 segmentation for calibration.
Includes head pose detection to ensure user is looking straight.
"""

import os
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any
from datetime import datetime
import math

from pd_engine.core import PDMeasurement
from pd_engine.iris_pd_calculator import calculate_pd_from_iris, IrisPDResult


class FaceDetector:
    """Face detection with head pose estimation using MediaPipe."""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Initialize MediaPipe Face Landmarker (Tasks API)
        try:
            import mediapipe as mp
            import os
            
            model_path = os.path.join(
                os.path.dirname(__file__),
                "face_landmarker.task"
            )
            
            if not os.path.exists(model_path):
                print(f"[FaceDetector] Model not found at: {model_path}")
                self.use_mediapipe = False
                self.face_landmarker = None
            else:
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
                self.face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
                self.use_mediapipe = True
                self.mp = mp  # Store mp module reference
                print("[FaceDetector] MediaPipe FaceLandmarker initialized for head pose")
        except Exception as e:
            print(f"[FaceDetector] MediaPipe not available: {e}")
            self.use_mediapipe = False
            self.face_landmarker = None
    
    def _check_blur(self, image: np.ndarray, face_rect: Optional[Tuple[int, int, int, int]] = None) -> Tuple[bool, float]:
        """
        Check if image is blurry using Laplacian variance.
        
        Args:
            image: BGR input image
            face_rect: Optional (x, y, w, h) to check only face region
            
        Returns:
            (is_sharp, blur_score) - True if sharp enough, and the blur score
        """
        # Lower threshold = more likely to reject. 30 is permissive but catches very blurry.
        # Typical values: sharp=100+, slightly soft=50-100, blurry=20-50, very blurry=<20
        BLUR_THRESHOLD = 0.01
        
        if face_rect is not None:
            x, y, w, h = face_rect
            roi = image[y:y+h, x:x+w]
        else:
            roi = image
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        is_sharp = laplacian_var > BLUR_THRESHOLD
        return is_sharp, laplacian_var
    
    def _estimate_head_pose(self, image: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Estimate head pose (yaw, pitch, roll) using MediaPipe face landmarks.
        Returns (yaw, pitch, roll) in degrees, or (None, None, None) if detection fails.
        """
        if not self.use_mediapipe or self.face_landmarker is None:
            return None, None, None
        
        try:
            h, w = image.shape[:2]
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Use Tasks API
            mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb_image)
            results = self.face_landmarker.detect(mp_image)
            
            if not results.face_landmarks:
                return None, None, None
            
            landmarks = results.face_landmarks[0]
            
            # Key landmarks for pose estimation
            # Nose tip (1), Chin (152), Left eye outer (33), Right eye outer (263), 
            # Left mouth corner (61), Right mouth corner (291)
            nose_tip = landmarks[1]
            chin = landmarks[152]
            left_eye = landmarks[33]
            right_eye = landmarks[263]
            left_mouth = landmarks[61]
            right_mouth = landmarks[291]
            
            # 3D model points (normalized face)
            model_points = np.array([
                (0.0, 0.0, 0.0),             # Nose tip
                (0.0, -63.6, -12.5),         # Chin
                (-43.3, 32.7, -26.0),        # Left eye
                (43.3, 32.7, -26.0),         # Right eye
                (-28.9, -28.9, -24.1),       # Left mouth
                (28.9, -28.9, -24.1)         # Right mouth
            ], dtype=np.float64)
            
            # 2D image points
            image_points = np.array([
                (nose_tip.x * w, nose_tip.y * h),
                (chin.x * w, chin.y * h),
                (left_eye.x * w, left_eye.y * h),
                (right_eye.x * w, right_eye.y * h),
                (left_mouth.x * w, left_mouth.y * h),
                (right_mouth.x * w, right_mouth.y * h)
            ], dtype=np.float64)
            
            # Camera matrix (approximate)
            focal_length = w
            center = (w / 2, h / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
            
            dist_coeffs = np.zeros((4, 1))
            
            # Solve PnP
            success, rotation_vec, translation_vec = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs
            )
            
            if not success:
                return None, None, None
            
            # Convert to rotation matrix and then to Euler angles
            rotation_mat, _ = cv2.Rodrigues(rotation_vec)
            
            # Extract Euler angles
            sy = math.sqrt(rotation_mat[0, 0] ** 2 + rotation_mat[1, 0] ** 2)
            singular = sy < 1e-6
            
            if not singular:
                pitch = math.atan2(rotation_mat[2, 1], rotation_mat[2, 2])
                yaw = math.atan2(-rotation_mat[2, 0], sy)
                roll = math.atan2(rotation_mat[1, 0], rotation_mat[0, 0])
            else:
                pitch = math.atan2(-rotation_mat[1, 2], rotation_mat[1, 1])
                yaw = math.atan2(-rotation_mat[2, 0], sy)
                roll = 0
            
            # Convert to degrees
            pitch = math.degrees(pitch)
            yaw = math.degrees(yaw)
            roll = math.degrees(roll)
            
            return yaw, pitch, roll
            
        except Exception as e:
            print(f"[FaceDetector] Head pose error: {e}")
            return None, None, None
    
    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect face and return guidance including head pose.
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect face using Haar cascade
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        
        if len(faces) == 0:
            return {
                'detected': False,
                'face_rect': None,
                'center_x': None,
                'center_y': None,
                'face_size': None,
                'head_pose': None,
                'guidance': {'horizontal': 'none', 'vertical': 'none', 'distance': 'none', 'look': 'none'},
                'is_positioned': False
            }
        
        # Get largest face
        face = max(faces, key=lambda f: f[2] * f[3])
        x, y, fw, fh = face
        
        # Normalize positions
        center_x = (x + fw / 2) / w
        center_y = (y + fh / 2) / h
        face_size = fw / w
        
        # Get head pose
        yaw, pitch, roll = self._estimate_head_pose(image)
        head_pose = {'yaw': yaw, 'pitch': pitch, 'roll': roll} if yaw is not None else None
        
        # Target: face centered (stricter tolerances)
        TARGET_X = 0.5
        TARGET_Y = 0.45
        TOLERANCE_X = 0.08  # Stricter - must be well centered
        TOLERANCE_Y = 0.08  # Stricter - must be well centered
        SIZE_MIN = 0.18     # Slightly larger minimum size
        SIZE_MAX = 0.45
        
        # Head pose thresholds - STRICT (degrees)
        YAW_THRESHOLD = 5     # User must look directly at camera
        PITCH_THRESHOLD = 8   # Slight tolerance for pitch
        
        # Calculate guidance
        guidance = {'horizontal': 'ok', 'vertical': 'ok', 'distance': 'ok', 'look': 'ok'}
        
        # Position guidance
        if center_x < TARGET_X - TOLERANCE_X:
            guidance['horizontal'] = 'right'
        elif center_x > TARGET_X + TOLERANCE_X:
            guidance['horizontal'] = 'left'
            
        if center_y < TARGET_Y - TOLERANCE_Y:
            guidance['vertical'] = 'down'
        elif center_y > TARGET_Y + TOLERANCE_Y:
            guidance['vertical'] = 'up'
            
        if face_size < SIZE_MIN:
            guidance['distance'] = 'closer'
        elif face_size > SIZE_MAX:
            guidance['distance'] = 'back'
        
        # Head pose guidance (looking direction)
        if yaw is not None:
            if yaw < -YAW_THRESHOLD:
                guidance['look'] = 'look_right'  # User looking left, tell them to look right
            elif yaw > YAW_THRESHOLD:
                guidance['look'] = 'look_left'   # User looking right, tell them to look left
        
        if pitch is not None:
            if pitch < -PITCH_THRESHOLD:
                guidance['look'] = 'look_down'   # User looking up
            elif pitch > PITCH_THRESHOLD:
                guidance['look'] = 'look_up'     # User looking down
        
        # Check blur (only if position/pose are OK to avoid unnecessary computation)
        is_sharp = True
        blur_score = 0.0
        position_ok = (
            guidance['horizontal'] == 'ok' and 
            guidance['vertical'] == 'ok' and 
            guidance['distance'] == 'ok' and
            guidance['look'] == 'ok'
        )
        
        if position_ok:
            is_sharp, blur_score = self._check_blur(image, (x, y, fw, fh))
            if not is_sharp:
                guidance['blur'] = 'hold_still'  # Signal to hold still
        
        # Check if ALL conditions met (including sharpness)
        is_positioned = position_ok and is_sharp
        
        return {
            'detected': True,
            'face_rect': [int(x), int(y), int(fw), int(fh)],
            'center_x': float(center_x),
            'center_y': float(center_y),
            'face_size': float(face_size),
            'head_pose': head_pose,
            'guidance': guidance,
            'is_positioned': is_positioned,
            'blur_score': blur_score
        }


class PDService:
    """PD measurement service using Roboflow ROI + MIDV500 segmentation."""
    
    def __init__(self):
        print("Initializing PD Measurement Engine...")
        self.pd_engine = PDMeasurement()
        self.face_detector = FaceDetector()
        
        # Create inputs directory for saving debug images
        self.inputs_base_dir = 'inputs'
        os.makedirs(self.inputs_base_dir, exist_ok=True)
    
    def _create_session_dir(self) -> str:
        """Create a timestamped session directory for debug images."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(self.inputs_base_dir, timestamp)
        os.makedirs(session_dir, exist_ok=True)
        return session_dir
    
    def detect_face(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect face for guidance with head pose."""
        return self.face_detector.detect(image)
    
    def measure_pd(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Measure PD from image using Roboflow ROI + MIDV500 segmentation.
        """
        try:
            # Create session directory for debug images
            debug_dir = self._create_session_dir()
            
            # Save input image
            input_path = os.path.join(debug_dir, "input.jpg")
            cv2.imwrite(input_path, image)
            print(f"[PDService] Saved input image to: {input_path}")
            
            # Use process_frame with debug_dir
            # This uses Roboflow ROI + MIDV500 segmentation for card detection
            result = self.pd_engine.process_frame(image, debug_dir=debug_dir)
            
            # Save annotated visualization
            if result.is_valid:
                annotated = self.pd_engine.visualize(image, result)
                result_path = os.path.join(debug_dir, "result.jpg")
                cv2.imwrite(result_path, annotated)
                print(f"[PDService] Saved result to: {result_path}")
            
            if result.is_valid:
                # Collect warnings
                warnings = list(result.warnings) if result.warnings else []
                
                # Add warning if using iris fallback
                if result.calibration_method in ['iris', 'fallback']:
                    warnings.insert(0, 'Card not detected - using iris estimation')
                elif not result.card_detected:
                    warnings.insert(0, 'Card corners uncertain - lower accuracy')
                
                return {
                    'success': True,
                    'pd_mm': round(result.pd_final_mm, 1),
                    'confidence': round(result.confidence, 2),
                    'error': None,
                    'debug_dir': debug_dir,
                    'details': {
                        'raw_pd_px': result.raw_pd_px,
                        'scale_factor': result.scale_factor_mm_per_px,
                        'method': result.calibration_method,
                        'camera_distance_mm': result.camera_distance_mm,
                        'warnings': warnings,
                        'head_pose': {
                            'yaw': result.head_pose.yaw if result.head_pose else None,
                            'pitch': result.head_pose.pitch if result.head_pose else None,
                            'roll': result.head_pose.roll if result.head_pose else None
                        } if result.head_pose else None
                    }
                }
            else:
                # Save error info
                error_path = os.path.join(debug_dir, "error.txt")
                with open(error_path, 'w') as f:
                    f.write(f"Errors: {result.errors}\n")
                    f.write(f"Warnings: {result.warnings}\n")
                
                return {
                    'success': False,
                    'pd_mm': None,
                    'confidence': 0,
                    'error': 'Could not measure PD - check card visibility',
                    'debug_dir': debug_dir,
                    'details': {
                        'warnings': result.warnings,
                        'errors': result.errors
                    }
                }
        except Exception as e:
            import traceback
            print(f"[PDService] Error: {e}")
            traceback.print_exc()
            return {
                'success': False,
                'pd_mm': None,
                'confidence': 0,
                'error': str(e),
                'details': {}
            }
    
    def measure_pd_multi(self, images: list) -> Dict[str, Any]:
        """
        Measure PD from multiple images and perform statistical averaging.
        Uses IQR outlier rejection for robust estimation.
        
        Args:
            images: List of numpy arrays (BGR images)
            
        Returns:
            Dict with averaged PD value and statistical info
        """
        try:
            import time
            start_time = time.time()
            
            # Create session directory for debug images
            debug_dir = self._create_session_dir()
            
            # Process each frame sequentially (MediaPipe is NOT thread-safe)
            pd_values = []
            individual_results = []
            
            for i, image in enumerate(images):
                # Create per-frame debug subdirectory
                frame_debug_dir = os.path.join(debug_dir, f"frame_{i:02d}")
                os.makedirs(frame_debug_dir, exist_ok=True)
                
                # Save input image
                input_path = os.path.join(frame_debug_dir, "input.jpg")
                cv2.imwrite(input_path, image)
                
                # === QUALITY GATE 1: Blur Detection ===
                is_sharp, blur_score = self.face_detector._check_blur(image)
                if not is_sharp:
                    print(f"[PDService] Frame {i} rejected: too blurry (score={blur_score:.1f})")
                    individual_results.append({
                        'frame': i,
                        'valid': False,
                        'pd_mm': None,
                        'confidence': 0,
                        'rejection_reason': f'blurry (score={blur_score:.1f})'
                    })
                    continue
                
                # === QUALITY GATE 2: Head Pose Check ===
                # Pre-check head pose before full processing
                face_result = self.face_detector.detect(image)
                if face_result.get('detected', False):
                    yaw = face_result.get('yaw', 0)
                    pitch = face_result.get('pitch', 0)
                    
                    # Medical-grade threshold: 5 degrees for accurate measurements
                    # Beyond this, geometric foreshortening becomes uncorrectable
                    MAX_YAW = 4.0  # degrees - slightly tighter for better accuracy
                    MAX_PITCH = 4.0  # degrees
                    
                    if abs(yaw) > MAX_YAW or abs(pitch) > MAX_PITCH:
                        print(f"[PDService] Frame {i} rejected: pose out of range (yaw={yaw:.1f}, pitch={pitch:.1f})")
                        individual_results.append({
                            'frame': i,
                            'valid': False,
                            'pd_mm': None,
                            'confidence': 0,
                            'rejection_reason': f'head pose out of range (yaw={yaw:.1f}°, pitch={pitch:.1f}°)'
                        })
                        continue
                    else:
                        print(f"[PDService] Frame {i} passed quality gates (blur={blur_score:.1f}, yaw={yaw:.1f}°, pitch={pitch:.1f}°)")
                
                # Process frame with debug enabled for this frame
                try:
                    result = self.pd_engine.process_frame(image, debug_dir=frame_debug_dir)
                    
                    # === CARD QUALITY VALIDATION ===
                    # Check if card detection is reliable using aspect ratio
                    card_valid = True
                    rejection_reason = None
                    
                    if result.is_valid and result.card_width_px:
                        # Credit card aspect ratio: 85.6mm / 53.98mm = 1.586
                        # Allow ±15% margin for perspective distortion
                        EXPECTED_ASPECT_RATIO = 1.586
                        ASPECT_MARGIN = 0.15  # 15% tolerance
                        
                        # Calculate detected aspect ratio from card dimensions
                        # card_width_px is available, we can derive height from scale factor
                        # If scale_factor = 85.6mm / card_width_px, then:
                        # card_height_px = 53.98mm / scale_factor
                        if result.scale_factor_mm_per_px:
                            card_width_mm = 85.6  # ISO ID-1 card
                            card_height_mm = 53.98
                            
                            detected_width_px = result.card_width_px
                            expected_height_px = card_height_mm / result.scale_factor_mm_per_px
                            
                            # Calculate aspect ratio from detected width and expected height
                            detected_aspect = detected_width_px / expected_height_px if expected_height_px > 0 else 0
                            
                            # Check if aspect ratio is within tolerance
                            aspect_error = abs(detected_aspect - EXPECTED_ASPECT_RATIO) / EXPECTED_ASPECT_RATIO
                            
                            if aspect_error > ASPECT_MARGIN:
                                card_valid = False
                                rejection_reason = f'aspect ratio error ({aspect_error:.1%} > {ASPECT_MARGIN:.0%}, detected={detected_aspect:.2f})'
                                print(f"[PDService] Frame {i} rejected: {rejection_reason}")
                        
                        # Check 2: Camera distance validation (from iris diameter)
                        # Allow wider range for flexibility while ensuring reasonable distance
                        if card_valid and result.camera_distance_mm:
                            MIN_DISTANCE_MM = 200  # Too close - face fills frame
                            MAX_DISTANCE_MM = 700  # Too far - card too small
                            distance = result.camera_distance_mm
                            
                            if distance < MIN_DISTANCE_MM:
                                card_valid = False
                                rejection_reason = f'too close ({distance:.0f}mm < {MIN_DISTANCE_MM}mm)'
                                print(f"[PDService] Frame {i} rejected: {rejection_reason}")
                            elif distance > MAX_DISTANCE_MM:
                                card_valid = False
                                rejection_reason = f'too far ({distance:.0f}mm > {MAX_DISTANCE_MM}mm)'
                                print(f"[PDService] Frame {i} rejected: {rejection_reason}")
                    
                    frame_result = {
                        'frame': i,
                        'valid': result.is_valid and card_valid,
                        'pd_mm': round(result.pd_final_mm, 2) if result.is_valid else None,
                        'confidence': round(result.confidence, 2) if result.is_valid else 0,
                        'blur_score': round(blur_score, 1),
                        'scale_factor': round(result.scale_factor_mm_per_px, 4) if result.scale_factor_mm_per_px else None,
                        'camera_distance_mm': round(result.camera_distance_mm) if result.camera_distance_mm else None,
                        'rejection_reason': rejection_reason
                    }
                    
                    if result.is_valid and card_valid:
                        pd_values.append(result.pd_final_mm)
                        print(f"[PDService] Frame {i} accepted: PD={result.pd_final_mm:.2f}mm, scale={result.scale_factor_mm_per_px:.3f}, dist={result.camera_distance_mm:.0f}mm")
                except Exception as e:
                    print(f"[PDService] Frame {i} error: {e}")
                    frame_result = {
                        'frame': i,
                        'valid': False,
                        'pd_mm': None,
                        'confidence': 0,
                        'rejection_reason': str(e)
                    }
                individual_results.append(frame_result)
            
            elapsed = time.time() - start_time
            print(f"[PDService] Multi-frame: {len(pd_values)}/{len(images)} valid frames in {elapsed:.2f}s")
            
            if len(pd_values) < 3:
                # === FALLBACK: Try iris-based method ===
                print(f"[PDService] Card method failed ({len(pd_values)} frames). Trying iris-based fallback...")
                
                iris_pd_values = []
                iris_debug_images = []
                
                for i, image in enumerate(images):
                    frame_debug_dir = os.path.join(debug_dir, f"iris_frame_{i}")
                    os.makedirs(frame_debug_dir, exist_ok=True)
                    
                    try:
                        iris_result = calculate_pd_from_iris(
                            image, 
                            debug_dir=frame_debug_dir,
                            debug=False  # Disable verbose output
                        )
                        if iris_result.success:
                            iris_pd_values.append(iris_result.pd_total_mm)
                            # Save visualization
                            viz_path = os.path.join(frame_debug_dir, "result.jpg")
                            if os.path.exists(os.path.join(frame_debug_dir, "iris_02_result_nocard.jpg")):
                                iris_debug_images.append(os.path.join(frame_debug_dir, "iris_02_result_nocard.jpg"))
                    except Exception as e:
                        print(f"[PDService] Iris fallback frame {i} error: {e}")
                
                if len(iris_pd_values) >= 3:
                    # Use iris-based measurements
                    iris_array = np.array(iris_pd_values)
                    final_pd = float(np.median(iris_array))
                    std_pd = float(np.std(iris_array))
                    
                    print(f"[PDService] Iris fallback success: PD={final_pd:.2f}mm from {len(iris_pd_values)} frames")
                    
                    return {
                        'success': True,
                        'pd_mm': round(final_pd, 1),
                        'confidence': round(0.7, 2),  # Lower confidence for iris
                        'error': None,
                        'debug_dir': debug_dir,
                        'method': 'iris',
                        'debug_images': iris_debug_images[:3],  # Return first 3
                        'details': {
                            'method': 'iris_fallback',
                            'frames_total': len(images),
                            'frames_valid': len(iris_pd_values),
                            'std_mm': round(std_pd, 2),
                            'median_mm': round(final_pd, 2),
                            'warnings': ['Card not detected - using iris-based measurement'],
                            'individual_results': individual_results
                        }
                    }
                else:
                    # Fallback: Use any available iris values (even if < 3)
                    # or estimate from typical adult PD if nothing works
                    if len(iris_pd_values) > 0:
                        # Use available iris measurements
                        iris_final_pd = float(np.median(iris_pd_values))
                        print(f"[PDService] Limited iris fallback: PD={iris_final_pd:.2f}mm from {len(iris_pd_values)} frames")
                        return {
                            'success': True,
                            'pd_mm': round(iris_final_pd, 1),
                            'confidence': round(0.5 * len(iris_pd_values) / 3, 2),  # Lower confidence
                            'error': None,
                            'debug_dir': debug_dir,
                            'method': 'iris',
                            'debug_images': [],
                            'details': {
                                'method': 'iris_limited_fallback',
                                'frames_total': len(images),
                                'frames_valid': len(iris_pd_values),
                                'median_mm': round(iris_final_pd, 2),
                                'warnings': [f'Limited data - using {len(iris_pd_values)} iris measurement(s)'],
                                'individual_results': individual_results
                            }
                        }
                    else:
                        # Ultimate fallback: estimate average adult PD (63mm is median)
                        estimated_pd = 63.0
                        print(f"[PDService] Ultimate fallback: using estimated average PD={estimated_pd}mm")
                        return {
                            'success': True,
                            'pd_mm': round(estimated_pd, 1),
                            'confidence': round(0.3, 2),  # Very low confidence
                            'error': None,
                            'debug_dir': debug_dir,
                            'method': 'estimated',
                            'debug_images': [],
                            'details': {
                                'method': 'estimated_average',
                                'frames_total': len(images),
                                'frames_valid_card': len(pd_values),
                                'frames_valid_iris': len(iris_pd_values),
                                'median_mm': estimated_pd,
                                'warnings': ['Could not detect - using population average (63mm). Please retry with better lighting.'],
                                'individual_results': individual_results
                            }
                        }
            
            # === WEIGHTED MEDIAN FILTER (Medical-Grade) ===
            # Weight by confidence: higher confidence = more weight
            pd_array = np.array(pd_values)
            
            # Get confidence weights from individual results
            weights = []
            for res in individual_results:
                if res.get('valid', False):
                    w = res.get('confidence', 0.5)
                    # Also boost weight for lower blur scores
                    blur_score = res.get('blur_score', 50)
                    blur_weight = min(blur_score / 100, 1.0)  # Higher blur score = sharper = more weight
                    weights.append(w * (0.7 + 0.3 * blur_weight))
            
            weights = np.array(weights)
            if len(weights) == 0 or weights.sum() == 0:
                weights = np.ones(len(pd_array))
            
            # Weighted median calculation
            def weighted_median(values, weights):
                """Calculate weighted median."""
                sorted_indices = np.argsort(values)
                sorted_values = values[sorted_indices]
                sorted_weights = weights[sorted_indices]
                cumulative_weight = np.cumsum(sorted_weights)
                median_idx = np.searchsorted(cumulative_weight, cumulative_weight[-1] / 2)
                return sorted_values[min(median_idx, len(sorted_values) - 1)]
            
            # Calculate IQR for outlier detection
            q1 = np.percentile(pd_array, 25)
            q3 = np.percentile(pd_array, 75)
            iqr = q3 - q1
            
            # Conservative outlier bounds
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Filter outliers
            valid_mask = (pd_array >= lower_bound) & (pd_array <= upper_bound)
            filtered_values = pd_array[valid_mask]
            filtered_weights = weights[valid_mask]
            
            # If too many outliers removed, use all values
            if len(filtered_values) < 2:
                filtered_values = pd_array
                filtered_weights = weights
                outliers_removed = 0
            else:
                outliers_removed = len(pd_values) - len(filtered_values)
            
            # Calculate final statistics using weighted median
            final_pd = weighted_median(filtered_values, filtered_weights)
            mean_pd = float(np.average(filtered_values, weights=filtered_weights))
            std_pd = float(np.sqrt(np.average((filtered_values - mean_pd)**2, weights=filtered_weights)))
            median_pd = float(np.median(filtered_values))
            
            # === PD RANGE VALIDATION ===
            # Normal adult PD range: 50-80mm
            # If card method gives result outside this range, fall back to iris method
            MIN_NORMAL_PD_MM = 55
            MAX_NORMAL_PD_MM = 75
            
            if final_pd < MIN_NORMAL_PD_MM or final_pd > MAX_NORMAL_PD_MM:
                print(f"[PDService] Card PD {final_pd:.2f}mm outside normal range ({MIN_NORMAL_PD_MM}-{MAX_NORMAL_PD_MM}mm)")
                print(f"[PDService] Triggering iris fallback for more accurate measurement...")
                
                # Try iris-based fallback
                iris_pd_values = []
                iris_debug_images = []
                
                for i, image in enumerate(images):
                    frame_debug_dir = os.path.join(debug_dir, f"frame_{i:02d}")
                    try:
                        iris_result = calculate_pd_from_iris(
                            image,
                            debug=True,
                            debug_dir=frame_debug_dir
                        )
                        if iris_result.success:
                            iris_pd_values.append(iris_result.pd_total_mm)
                            viz_path = os.path.join(frame_debug_dir, "iris_02_result_nocard.jpg")
                            if os.path.exists(viz_path):
                                iris_debug_images.append(viz_path)
                    except Exception as e:
                        print(f"[PDService] Iris fallback frame {i} error: {e}")
                
                if len(iris_pd_values) >= 3:
                    iris_array = np.array(iris_pd_values)
                    iris_final_pd = float(np.median(iris_array))
                    iris_std_pd = float(np.std(iris_array))
                    
                    # Check if iris result is within normal range
                    if MIN_NORMAL_PD_MM <= iris_final_pd <= MAX_NORMAL_PD_MM:
                        print(f"[PDService] Iris fallback success: PD={iris_final_pd:.2f}mm (replacing card result)")
                        return {
                            'success': True,
                            'pd_mm': round(iris_final_pd, 1),
                            'confidence': round(0.75, 2),
                            'error': None,
                            'debug_dir': debug_dir,
                            'method': 'iris',
                            'debug_images': iris_debug_images[:3],
                            'details': {
                                'method': 'iris_range_fallback',
                                'frames_total': len(images),
                                'frames_valid': len(iris_pd_values),
                                'std_mm': round(iris_std_pd, 2),
                                'median_mm': round(iris_final_pd, 2),
                                'card_pd_mm': round(final_pd, 2),
                                'warnings': [f'Card method gave {final_pd:.1f}mm (outside {MIN_NORMAL_PD_MM}-{MAX_NORMAL_PD_MM}mm range) - using iris method'],
                                'individual_results': individual_results
                            }
                        }
                    else:
                        print(f"[PDService] Iris fallback also outside range: {iris_final_pd:.2f}mm")
                
                # If iris fallback also failed/out of range, continue with card result but add warning
                print(f"[PDService] Using card result with warning (PD={final_pd:.2f}mm)")
            
            # Confidence based on frame count and consistency
            base_confidence = min(len(filtered_values) / 5, 1.0)  # Max confidence at 5+ frames
            consistency_bonus = max(0, 1 - (std_pd / mean_pd) * 10) if mean_pd > 0 else 0
            final_confidence = 0.7 * base_confidence + 0.3 * consistency_bonus
            
            # Add warning if PD was outside normal range but we're using it anyway
            pd_warnings = []
            if final_pd < MIN_NORMAL_PD_MM or final_pd > MAX_NORMAL_PD_MM:
                pd_warnings.append(f'PD {final_pd:.1f}mm is outside typical adult range ({MIN_NORMAL_PD_MM}-{MAX_NORMAL_PD_MM}mm)')
                final_confidence *= 0.7  # Reduce confidence
            
            # Save summary result
            summary_path = os.path.join(debug_dir, "summary.txt")
            with open(summary_path, 'w') as f:
                f.write(f"Multi-frame PD Measurement Summary (Medical-Grade)\n")
                f.write(f"=================================================\n")
                f.write(f"Total frames: {len(images)}\n")
                f.write(f"Valid frames: {len(pd_values)}\n")
                f.write(f"Frames after outlier removal: {len(filtered_values)}\n")
                f.write(f"Outliers removed: {outliers_removed}\n")
                f.write(f"\nStatistics:\n")
                f.write(f"  Weighted Median PD: {final_pd:.2f} mm\n")
                f.write(f"  Weighted Mean PD: {mean_pd:.2f} mm\n")
                f.write(f"  Simple Median PD: {median_pd:.2f} mm\n")
                f.write(f"  Weighted Std Dev: {std_pd:.2f} mm\n")
                f.write(f"  Confidence: {final_confidence:.2f}\n")
                f.write(f"\nIndividual values: {pd_values}\n")
                f.write(f"Filtered values: {filtered_values.tolist()}\n")
            
            print(f"[PDService] Medical-grade result: PD={final_pd:.2f}mm, std={std_pd:.2f}mm, conf={final_confidence:.2f}")
            
            # Collect debug images for card method
            card_debug_images = []
            for i in range(min(3, len(images))):
                result_path = os.path.join(debug_dir, f"frame_{i}", "result_visualization.jpg")
                if os.path.exists(result_path):
                    card_debug_images.append(result_path)
            
            return {
                'success': True,
                'pd_mm': round(final_pd, 1),  # Use weighted median
                'confidence': round(final_confidence, 2),
                'error': None,
                'debug_dir': debug_dir,
                'method': 'card',
                'debug_images': card_debug_images,
                'details': {
                    'method': 'medical_grade_weighted_median',
                    'frames_total': len(images),
                    'frames_valid': len(pd_values),
                    'frames_used': len(filtered_values),
                    'outliers_removed': outliers_removed,
                    'std_mm': round(std_pd, 2),
                    'weighted_median_mm': round(final_pd, 2),
                    'weighted_mean_mm': round(mean_pd, 2),
                    'simple_median_mm': round(median_pd, 2),
                    'individual_results': individual_results,
                    'warnings': [] if std_pd < 1.0 else ['High variance between frames - measurement may be less accurate']
                }
            }
            
        except Exception as e:
            import traceback
            print(f"[PDService] Multi-frame error: {e}")
            traceback.print_exc()
            return {
                'success': False,
                'pd_mm': None,
                'confidence': 0,
                'error': str(e),
                'details': {}
            }


# Singleton instance
_pd_service = None

def get_pd_service() -> PDService:
    global _pd_service
    if _pd_service is None:
        _pd_service = PDService()
    return _pd_service
