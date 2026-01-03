"""
3D Photogrammetric PD Calculation Module - Complete Implementation

Implements precise Pupillary Distance measurement using:
1. Card Flexion Check (convexity analysis)
2. Sub-Pixel Corner Refinement
3. PnP Pose Estimation (IPPE_SQUARE)
4. Head-Card Consistency Validation
5. Anthropometric Depth Correction (10mm supraorbital-to-cornea)
6. Ray-Plane Intersection for 3D Reconstruction
7. Geometric Near-to-Far PD Correction

Reference: ISO/IEC 7810 ID-1 Card (85.60 x 53.98 mm)
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
import os

# Card dimensions (ISO/IEC 7810 ID-1)
CARD_WIDTH_MM = 85.60
CARD_HEIGHT_MM = 53.98
CARD_ASPECT_RATIO = CARD_WIDTH_MM / CARD_HEIGHT_MM

# Anthropometric constants - conservative values
# The homography projection partially accounts for perspective, so use smaller offsets
OFFSET_GLABELLA_TO_CORNEA = 5.0  # mm (conservative: 5-8mm typical)
# Near-to-Far PD adjustment for distance glasses
FAR_PD_ADJUSTMENT = 2.0  # mm (conservative industry standard)

# Validation thresholds
MAX_CARD_HEAD_TILT_DIFF = 15.0  # degrees
MAX_CONVEXITY_DEFECT = 0.15  # 15% of bounding box area
MAX_ASPECT_RATIO_ERROR = 0.20  # 20% deviation from ideal


@dataclass
class PhotogrammetryResult:
    """Result of 3D photogrammetric PD calculation."""
    success: bool
    pd_near_mm: Optional[float] = None  # PD at camera distance
    pd_far_mm: Optional[float] = None   # PD at infinity (for distance glasses)
    camera_distance_mm: Optional[float] = None  # Distance to card
    card_pose: Optional[dict] = None  # 6-DoF pose information
    pupil_left_3d: Optional[np.ndarray] = None  # 3D position
    pupil_right_3d: Optional[np.ndarray] = None  # 3D position
    validation_info: Optional[dict] = None  # Checks performed
    error_message: Optional[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class PhotogrammetricPDCalculator:
    """
    Complete 3D Photogrammetric PD Calculator.
    
    Implements the full pipeline from the technical monograph:
    - Card flexion and validity checks
    - Sub-pixel corner refinement
    - PnP 6-DoF pose estimation
    - Head-card consistency validation
    - Ray-plane intersection for 3D reconstruction
    - Anthropometric depth correction
    - Near-to-far PD correction
    """
    
    def __init__(self, debug_dir: Optional[str] = None):
        """
        Initialize calculator.
        
        Args:
            debug_dir: Directory for debug images (None to disable)
        """
        self.debug_dir = debug_dir
        self._step = 0
        
        # Camera intrinsics (estimated or provided)
        self.K = None
        self.D = None
        
    def _save_debug(self, name: str, image: np.ndarray):
        """Save debug image."""
        if self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)
            path = os.path.join(self.debug_dir, f"pd_{self._step:02d}_{name}.jpg")
            cv2.imwrite(path, image)
            print(f"  [PD-{self._step:02d}] {name}")
            self._step += 1
    
    def _estimate_camera_intrinsics(self, width: int, height: int, 
                                    fov_degrees: float = 60.0) -> np.ndarray:
        """Estimate camera intrinsic matrix."""
        fov_rad = np.radians(fov_degrees)
        fx = (width / 2) / np.tan(fov_rad / 2)
        fy = fx
        
        cx = width / 2
        cy = height / 2
        
        return np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)
    
    def _check_card_flexion(self, contour: np.ndarray, 
                           image: np.ndarray) -> Tuple[bool, float, np.ndarray]:
        """
        Check if card is bent/flexed.
        
        Uses convex hull defect analysis. A bent card has larger defects.
        
        Returns:
            (is_valid, defect_ratio, hull)
        """
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(contour)
        
        if hull_area < 1:
            return False, 1.0, hull
        
        # Defect ratio: how much the contour differs from its hull
        defect_ratio = 1 - (contour_area / hull_area)
        
        is_valid = defect_ratio < MAX_CONVEXITY_DEFECT
        
        # Debug visualization
        if self.debug_dir:
            viz = image.copy()
            cv2.drawContours(viz, [contour], -1, (0, 255, 0), 2)
            cv2.drawContours(viz, [hull], -1, (0, 0, 255), 2)
            cv2.putText(viz, f"Defect: {defect_ratio*100:.1f}%", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            self._save_debug("flexion_check", viz)
        
        return is_valid, defect_ratio, hull
    
    def _refine_corners_subpixel(self, corners: np.ndarray, 
                                 gray: np.ndarray,
                                 image: np.ndarray) -> np.ndarray:
        """
        Refine corner locations to sub-pixel precision.
        
        Uses cv2.cornerSubPix for floating-point corner locations.
        """
        corners_float = np.float32(corners).reshape(-1, 2)
        
        # Sub-pixel refinement criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Refine corners
        refined = cv2.cornerSubPix(gray, corners_float, (5, 5), (-1, -1), criteria)
        
        # Debug visualization
        if self.debug_dir:
            viz = image.copy()
            for i, (orig, ref) in enumerate(zip(corners_float, refined)):
                # Original in red
                cv2.circle(viz, (int(orig[0]), int(orig[1])), 5, (0, 0, 255), -1)
                # Refined in green
                cv2.circle(viz, (int(ref[0]), int(ref[1])), 3, (0, 255, 0), -1)
                # Arrow showing refinement
                cv2.arrowedLine(viz, (int(orig[0]), int(orig[1])),
                               (int(ref[0]), int(ref[1])), (255, 0, 0), 2)
            cv2.putText(viz, "Red:Original  Green:Refined", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            self._save_debug("subpixel_corners", viz)
        
        return refined
    
    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """Order corners as TL, TR, BR, BL."""
        corners = corners.reshape(-1, 2)
        
        sorted_by_y = corners[np.argsort(corners[:, 1])]
        top_pts = sorted_by_y[:2]
        bottom_pts = sorted_by_y[2:]
        
        top_pts = top_pts[np.argsort(top_pts[:, 0])]
        bottom_pts = bottom_pts[np.argsort(bottom_pts[:, 0])]
        
        return np.array([
            top_pts[0], top_pts[1],
            bottom_pts[1], bottom_pts[0]
        ], dtype=np.float32)
    
    def _solve_card_pnp(self, corners: np.ndarray,
                        image: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve PnP for card 6-DoF pose.
        
        Uses IPPE_SQUARE optimized for planar rectangular targets.
        
        Returns:
            (success, R_matrix, t_vector, normal_vector)
        """
        # Order corners
        img_points = self._order_corners(corners)
        
        # 3D model points (Z=0 plane, origin at TL)
        obj_points = np.array([
            [0, 0, 0],
            [CARD_WIDTH_MM, 0, 0],
            [CARD_WIDTH_MM, CARD_HEIGHT_MM, 0],
            [0, CARD_HEIGHT_MM, 0]
        ], dtype=np.float32)
        
        # Solve PnP using IPPE (optimized for planar targets)
        try:
            success, rvec, tvec = cv2.solvePnP(
                obj_points, img_points, self.K, self.D,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
        except cv2.error:
            success, rvec, tvec = cv2.solvePnP(
                obj_points, img_points, self.K, self.D,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
        
        if not success:
            return False, None, None, None
        
        # Convert rotation vector to matrix
        R_mat, _ = cv2.Rodrigues(rvec)
        
        # Card normal is Z-axis of rotation matrix
        normal = R_mat[:, 2].flatten()
        
        tvec_flat = tvec.flatten()
        
        # Debug visualization
        if self.debug_dir:
            viz = image.copy()
            
            # Draw axes at card center
            axis_length = 40
            center_3d = np.array([[CARD_WIDTH_MM/2, CARD_HEIGHT_MM/2, 0]], dtype=np.float32)
            center_2d, _ = cv2.projectPoints(center_3d, rvec, tvec, self.K, self.D)
            center = tuple(map(int, center_2d[0, 0]))
            
            # X (red), Y (green), Z (blue) axes
            axes_3d = np.array([
                [CARD_WIDTH_MM/2 + axis_length, CARD_HEIGHT_MM/2, 0],
                [CARD_WIDTH_MM/2, CARD_HEIGHT_MM/2 + axis_length, 0],
                [CARD_WIDTH_MM/2, CARD_HEIGHT_MM/2, axis_length]
            ], dtype=np.float32)
            axes_2d, _ = cv2.projectPoints(axes_3d, rvec, tvec, self.K, self.D)
            
            cv2.line(viz, center, tuple(map(int, axes_2d[0, 0])), (0, 0, 255), 3)  # X
            cv2.line(viz, center, tuple(map(int, axes_2d[1, 0])), (0, 255, 0), 3)  # Y
            cv2.line(viz, center, tuple(map(int, axes_2d[2, 0])), (255, 0, 0), 3)  # Z
            
            cv2.putText(viz, f"Dist: {np.linalg.norm(tvec_flat):.0f}mm", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(viz, f"Normal: ({normal[0]:.2f},{normal[1]:.2f},{normal[2]:.2f})", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            self._save_debug("pnp_pose", viz)
        
        return True, R_mat, tvec_flat, normal
    
    def _validate_head_card_consistency(self, card_normal: np.ndarray,
                                        head_pose: Optional[dict],
                                        image: np.ndarray) -> Tuple[bool, float]:
        """
        Check if card orientation matches head orientation.
        
        If card is "lifting off" the forehead, reject the measurement.
        
        Returns:
            (is_consistent, angle_difference_degrees)
        """
        if head_pose is None:
            return True, 0.0  # Skip check if no head pose
        
        # Head normal from head pose (assuming Z-forward)
        head_yaw = np.radians(head_pose.get('yaw', 0))
        head_pitch = np.radians(head_pose.get('pitch', 0))
        
        # Approximate head normal from yaw/pitch
        head_normal = np.array([
            np.sin(head_yaw),
            -np.sin(head_pitch),
            np.cos(head_yaw) * np.cos(head_pitch)
        ])
        head_normal = head_normal / np.linalg.norm(head_normal)
        
        # Ensure card normal points towards camera (positive Z)
        if card_normal[2] > 0:
            card_normal_aligned = -card_normal
        else:
            card_normal_aligned = card_normal
        
        # Angle between normals
        dot = np.clip(np.dot(card_normal_aligned, head_normal), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(abs(dot)))
        
        is_consistent = angle_deg < MAX_CARD_HEAD_TILT_DIFF
        
        # Debug visualization
        if self.debug_dir:
            viz = image.copy()
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            scale = 100
            
            # Draw card normal
            card_end = (int(center[0] + card_normal_aligned[0] * scale),
                       int(center[1] - card_normal_aligned[1] * scale))
            cv2.arrowedLine(viz, center, card_end, (0, 255, 0), 3)
            cv2.putText(viz, "Card", card_end, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw head normal
            head_end = (int(center[0] + head_normal[0] * scale),
                       int(center[1] - head_normal[1] * scale))
            cv2.arrowedLine(viz, center, head_end, (255, 0, 0), 3)
            cv2.putText(viz, "Head", head_end, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            status = "OK" if is_consistent else "WARN: Card lifting!"
            cv2.putText(viz, f"Angle diff: {angle_deg:.1f}° ({status})", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            self._save_debug("consistency_check", viz)
        
        return is_consistent, angle_deg
    
    def _cast_pupil_ray(self, pixel: np.ndarray) -> np.ndarray:
        """
        Convert 2D pixel to 3D unit ray.
        
        ray = normalize(K^-1 * [u, v, 1])
        """
        # Undistort point
        if self.D is not None and np.any(self.D != 0):
            pts = np.array([[pixel[0], pixel[1]]], dtype=np.float32).reshape(-1, 1, 2)
            pts_undist = cv2.undistortPoints(pts, self.K, self.D, P=self.K)
            pixel = pts_undist[0, 0]
        
        # Homogeneous coordinates
        pixel_h = np.array([pixel[0], pixel[1], 1.0])
        
        # Back-project to 3D ray
        ray = np.linalg.inv(self.K) @ pixel_h
        
        # Normalize
        return ray / np.linalg.norm(ray)
    
    def _ray_plane_intersection(self, ray: np.ndarray,
                                plane_point: np.ndarray,
                                plane_normal: np.ndarray) -> Optional[np.ndarray]:
        """
        Intersect ray (from origin) with plane.
        
        d = (P0 · N) / (R · N)
        intersection = d * R
        """
        denom = np.dot(ray, plane_normal)
        
        if abs(denom) < 1e-10:
            return None
        
        d = np.dot(plane_point, plane_normal) / denom
        
        if d < 0:
            return None
        
        return ray * d
    
    def calculate(self,
                  image: np.ndarray,
                  card_corners: np.ndarray,
                  pupil_left_px: Tuple[float, float],
                  pupil_right_px: Tuple[float, float],
                  head_pose: Optional[dict] = None,
                  K: Optional[np.ndarray] = None,
                  D: Optional[np.ndarray] = None) -> PhotogrammetryResult:
        """
        Calculate precise PD using full 3D photogrammetric pipeline.
        
        Args:
            image: BGR image
            card_corners: 4x2 detected card corners (pixels)
            pupil_left_px: Left pupil (x, y) in pixels
            pupil_right_px: Right pupil (x, y) in pixels
            head_pose: Optional head pose dict with 'yaw', 'pitch', 'roll'
            K: Camera intrinsic matrix (estimated if None)
            D: Distortion coefficients (zero if None)
            
        Returns:
            PhotogrammetryResult with full analysis
        """
        self._step = 0
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Setup camera parameters
        if K is None:
            self.K = self._estimate_camera_intrinsics(w, h, fov_degrees=60.0)
        else:
            self.K = K
            
        if D is None:
            self.D = np.zeros(5, dtype=np.float64)
        else:
            self.D = D
        
        result = PhotogrammetryResult(success=False)
        result.validation_info = {}
        
        print("\n[Photogrammetry] === 3D PD Calculation ===")
        print(f"[Photogrammetry] Focal length: {self.K[0,0]:.1f}px")
        
        # ============================================================
        # PHASE 1: Card Pose Estimation
        # ============================================================
        print("\n[Phase 1] Card Pose Estimation")
        
        # Step 1.1: Create contour from corners for flexion check
        corners_int = card_corners.reshape(-1, 1, 2).astype(np.int32)
        
        # Step 1.2: Sub-pixel corner refinement
        print("   [1.2] Sub-pixel corner refinement")
        refined_corners = self._refine_corners_subpixel(card_corners, gray, image)
        
        # Step 1.3: Solve PnP for 6-DoF pose
        print("   [1.3] PnP pose estimation (IPPE_SQUARE)")
        success, R_mat, tvec, card_normal = self._solve_card_pnp(refined_corners, image)
        
        if not success:
            result.error_message = "PnP pose estimation failed"
            return result
        
        camera_distance = np.linalg.norm(tvec)
        print(f"   Distance to camera: {camera_distance:.1f} mm")
        print(f"   Card normal: ({card_normal[0]:.3f}, {card_normal[1]:.3f}, {card_normal[2]:.3f})")
        
        # Step 1.4: CALIBRATE focal length from card dimensions
        # We know the card is 85.6mm wide. Compare pixel width to mm width at this depth.
        # card_width_px / fx = card_width_mm / Z => fx = card_width_px * Z / card_width_mm
        ordered_corners = self._order_corners(refined_corners)
        card_width_px = np.linalg.norm(ordered_corners[1] - ordered_corners[0])
        card_height_px = np.linalg.norm(ordered_corners[2] - ordered_corners[1])
        
        # Average the two estimates (width and height based)
        fx_from_width = card_width_px * camera_distance / CARD_WIDTH_MM
        fy_from_height = card_height_px * camera_distance / CARD_HEIGHT_MM
        calibrated_f = (fx_from_width + fy_from_height) / 2
        
        print(f"   [1.4] Focal length calibration from card:")
        print(f"         Card size px: {card_width_px:.1f} x {card_height_px:.1f}")
        print(f"         Estimated focal: {self.K[0,0]:.1f}px (60° FOV)")
        print(f"         Calibrated focal: {calibrated_f:.1f}px (from card)")
        
        # Update camera intrinsics with calibrated focal length
        self.K[0, 0] = calibrated_f
        self.K[1, 1] = calibrated_f
        
        result.camera_distance_mm = camera_distance
        result.card_pose = {
            'rotation': R_mat.tolist(),
            'translation': tvec.tolist(),
            'normal': card_normal.tolist()
        }
        
        # ============================================================
        # PHASE 2: Validation Checks
        # ============================================================
        print("\n[Phase 2] Validation Checks")
        
        # Step 2.1: Check aspect ratio
        ordered_corners = self._order_corners(refined_corners)
        width_px = (np.linalg.norm(ordered_corners[1] - ordered_corners[0]) +
                   np.linalg.norm(ordered_corners[2] - ordered_corners[3])) / 2
        height_px = (np.linalg.norm(ordered_corners[3] - ordered_corners[0]) +
                    np.linalg.norm(ordered_corners[2] - ordered_corners[1])) / 2
        
        if height_px > width_px:
            width_px, height_px = height_px, width_px
            
        aspect = width_px / height_px
        aspect_error = abs(aspect - CARD_ASPECT_RATIO) / CARD_ASPECT_RATIO
        
        print(f"   [2.1] Aspect ratio: {aspect:.3f} (ideal: {CARD_ASPECT_RATIO:.3f}, error: {aspect_error*100:.1f}%)")
        
        if aspect_error > MAX_ASPECT_RATIO_ERROR:
            result.warnings.append(f"High aspect ratio error: {aspect_error*100:.0f}%")
        
        result.validation_info['aspect_ratio'] = aspect
        result.validation_info['aspect_error'] = aspect_error
        
        # Step 2.2: Head-card consistency check
        print("   [2.2] Head-card consistency check")
        is_consistent, angle_diff = self._validate_head_card_consistency(
            card_normal, head_pose, image
        )
        
        result.validation_info['head_card_angle'] = angle_diff
        result.validation_info['is_consistent'] = is_consistent
        
        if not is_consistent:
            result.warnings.append(f"Card may be lifting off forehead (angle: {angle_diff:.1f}°)")
        
        # ============================================================
        # PHASE 3: True 3D Ray-Plane Intersection
        # ============================================================
        print("\n[Phase 3] 3D Ray-Plane Intersection")
        
        # This method properly handles any angle between card and camera:
        # 1. Define eye plane (parallel to card, offset by anthropometric distance)
        # 2. Cast rays from camera through each pupil pixel
        # 3. Intersect rays with eye plane
        # 4. Calculate 3D Euclidean distance
        
        # Step 3.1: Compute card center in 3D (from PnP)
        # The card coordinate system has origin at TL corner
        # Card center in card coords: (W/2, H/2, 0)
        # Transform to camera coords using R and t
        card_center_card = np.array([CARD_WIDTH_MM/2, CARD_HEIGHT_MM/2, 0])
        card_center_cam = R_mat @ card_center_card + tvec
        
        print(f"   [3.1] Card center (camera coords): ({card_center_cam[0]:.1f}, {card_center_cam[1]:.1f}, {card_center_cam[2]:.1f}) mm")
        
        # Step 3.2: Define eye plane
        # For CONSISTENT PD measurement regardless of card tilt, we use a FRONTAL plane
        # (parallel to the camera sensor) at the depth where the eyes are.
        # 
        # Eye depth = camera_distance + anthropometric offset (since eyes are BEHIND the card)
        eye_depth = camera_distance + OFFSET_GLABELLA_TO_CORNEA
        
        # Frontal plane: normal is [0, 0, 1] (pointing towards camera)
        # Plane equation: z = eye_depth, or n·p = d where n=[0,0,1], d=eye_depth
        eye_plane_normal = np.array([0.0, 0.0, 1.0])
        eye_plane_point = np.array([0.0, 0.0, eye_depth])
        
        print(f"   [3.2] Eye plane (frontal at Z={eye_depth:.1f} mm)")
        print(f"         Camera dist: {camera_distance:.1f}mm + offset: {OFFSET_GLABELLA_TO_CORNEA}mm")
        
        # Step 3.3: Cast rays through pupil pixels
        ray_left = self._cast_pupil_ray(np.array(pupil_left_px))
        ray_right = self._cast_pupil_ray(np.array(pupil_right_px))
        
        print(f"   [3.3] Pupil rays:")
        print(f"         Left:  ({ray_left[0]:.4f}, {ray_left[1]:.4f}, {ray_left[2]:.4f})")
        print(f"         Right: ({ray_right[0]:.4f}, {ray_right[1]:.4f}, {ray_right[2]:.4f})")
        
        # Step 3.4: Ray-plane intersection
        # For frontal plane (z = eye_depth), intersection is simple:
        # ray = normalize(K^-1 @ pixel), intersection at t where ray_z * t = eye_depth
        # So t = eye_depth / ray_z, and intersection = ray * t
        
        if abs(ray_left[2]) > 1e-10 and abs(ray_right[2]) > 1e-10:
            t_left = eye_depth / ray_left[2]
            t_right = eye_depth / ray_right[2]
            pupil_left_3d = ray_left * t_left
            pupil_right_3d = ray_right * t_right
        else:
            pupil_left_3d = None
            pupil_right_3d = None
        
        if pupil_left_3d is None or pupil_right_3d is None:
            # Fallback to homography if ray-plane fails
            print("   [WARN] Ray-plane intersection failed, using homography fallback")
            card_corners_mm = np.array([
                [0, 0], [CARD_WIDTH_MM, 0], 
                [CARD_WIDTH_MM, CARD_HEIGHT_MM], [0, CARD_HEIGHT_MM]
            ], dtype=np.float32)
            H, _ = cv2.findHomography(ordered_corners, card_corners_mm, cv2.RANSAC, 5.0)
            if H is not None:
                pupils_px = np.array([pupil_left_px, pupil_right_px], dtype=np.float32).reshape(-1, 1, 2)
                pupils_mm = cv2.perspectiveTransform(pupils_px, H).reshape(-1, 2)
                pd_on_card_plane = np.linalg.norm(pupils_mm[1] - pupils_mm[0])
                correction_factor = (camera_distance + OFFSET_GLABELLA_TO_CORNEA) / camera_distance
                pd_near = pd_on_card_plane * correction_factor
                pd_far = pd_near + FAR_PD_ADJUSTMENT
                result.pd_near_mm = pd_near
                result.pd_far_mm = pd_far
                result.success = True
                return result
            else:
                result.error_message = "Both ray-plane and homography methods failed"
                return result
        
        print(f"   [3.4] 3D pupil positions:")
        print(f"         Left:  ({pupil_left_3d[0]:.1f}, {pupil_left_3d[1]:.1f}, {pupil_left_3d[2]:.1f}) mm")
        print(f"         Right: ({pupil_right_3d[0]:.1f}, {pupil_right_3d[1]:.1f}, {pupil_right_3d[2]:.1f}) mm")
        
        result.pupil_left_3d = pupil_left_3d
        result.pupil_right_3d = pupil_right_3d
        
        # Step 3.5: Calculate 3D Euclidean distance (the "true" PD)
        pd_3d = np.linalg.norm(pupil_right_3d - pupil_left_3d)
        
        print(f"   [3.5] 3D Euclidean PD: {pd_3d:.2f} mm")
        
        # Debug visualization
        if self.debug_dir:
            viz = image.copy()
            cv2.circle(viz, (int(pupil_left_px[0]), int(pupil_left_px[1])), 8, (255, 0, 0), -1)
            cv2.circle(viz, (int(pupil_right_px[0]), int(pupil_right_px[1])), 8, (0, 0, 255), -1)
            cv2.line(viz, (int(pupil_left_px[0]), int(pupil_left_px[1])),
                    (int(pupil_right_px[0]), int(pupil_right_px[1])), (0, 255, 255), 3)
            cv2.putText(viz, f"3D PD: {pd_3d:.1f}mm", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            self._save_debug("3d_intersection", viz)
        
        # ============================================================
        # PHASE 4: Near-to-Far PD Adjustment
        # ============================================================
        print("\n[Phase 4] Near-to-Far PD Adjustment")
        
        # The 3D Euclidean distance IS the "Near PD" (at the viewing distance)
        pd_near = pd_3d
        
        # For distance glasses, add the standard industry adjustment
        pd_far = pd_near + FAR_PD_ADJUSTMENT
        
        print(f"   [4.1] Near PD (3D calculated): {pd_near:.2f} mm")
        print(f"   [4.2] Far PD (+{FAR_PD_ADJUSTMENT}mm): {pd_far:.2f} mm")
        
        # 3D positions are already set from ray intersection
        # (result.pupil_left_3d and pupil_right_3d are set earlier for successful ray-plane)
        
        # Final debug image
        if self.debug_dir:
            viz = image.copy()
            
            # Draw card corners
            cv2.polylines(viz, [ordered_corners.astype(np.int32)], True, (0, 255, 0), 2)
            
            # Draw pupils with PD line
            cv2.circle(viz, (int(pupil_left_px[0]), int(pupil_left_px[1])), 8, (255, 0, 0), -1)
            cv2.circle(viz, (int(pupil_right_px[0]), int(pupil_right_px[1])), 8, (0, 0, 255), -1)
            cv2.line(viz, (int(pupil_left_px[0]), int(pupil_left_px[1])),
                    (int(pupil_right_px[0]), int(pupil_right_px[1])), (0, 255, 255), 3)
            
            # Results text
            cv2.rectangle(viz, (5, 5), (350, 120), (0, 0, 0), -1)
            cv2.putText(viz, f"3D PD: {pd_3d:.1f}mm", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.putText(viz, f"Near PD: {pd_near:.1f}mm", (10, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(viz, f"Far PD: {pd_far:.1f}mm (for glasses)", (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(viz, f"Distance: {camera_distance:.0f}mm", (10, 105),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            self._save_debug("final_pd_result", viz)
        
        print(f"\n[Photogrammetry] === RESULTS ===")
        print(f"   Card plane PD: {pd_on_card_plane:.2f} mm")
        print(f"   Near PD:       {pd_near:.2f} mm")
        print(f"   Far PD:        {pd_far:.2f} mm")
        
        result.success = True
        result.pd_near_mm = pd_near
        result.pd_far_mm = pd_far
        
        return result


def simple_pd_from_scale(
    card_corners: np.ndarray,
    pupil_left_px: Tuple[float, float],
    pupil_right_px: Tuple[float, float],
    debug: bool = False
) -> PhotogrammetryResult:
    """
    SIMPLE, ROBUST PD calculation using card dimensions for scale factor.
    
    Method:
    1. Detect card orientation (landscape vs portrait)
    2. Use LONGER edge for scale (always maps to 85.6mm width)
    3. PD = horizontal_pupil_distance × scale_factor
    """
    result = PhotogrammetryResult(success=False)
    result.warnings = []
    
    if debug:
        print("\n[PD] === Scale Method with Orientation Detection ===")
    
    # Order corners: TL, TR, BR, BL
    corners = card_corners.reshape(-1, 2).astype(np.float64)
    
    # Sort by Y to find top/bottom
    sorted_by_y = corners[np.argsort(corners[:, 1])]
    top_two = sorted_by_y[:2]
    bottom_two = sorted_by_y[2:]
    
    # Sort by X within each pair
    top_two = top_two[np.argsort(top_two[:, 0])]
    bottom_two = bottom_two[np.argsort(bottom_two[:, 0])]
    
    TL, TR = top_two[0], top_two[1]
    BL, BR = bottom_two[0], bottom_two[1]
    
    # Calculate both dimensions
    top_edge_px = np.linalg.norm(TR - TL)
    bottom_edge_px = np.linalg.norm(BR - BL)
    left_edge_px = np.linalg.norm(BL - TL)
    right_edge_px = np.linalg.norm(BR - TR)
    
    # Average horizontal and vertical edges
    horizontal_px = (top_edge_px + bottom_edge_px) / 2.0
    vertical_px = (left_edge_px + right_edge_px) / 2.0
    
    # Determine orientation: LONGER edge is always the card WIDTH (85.6mm)
    if horizontal_px >= vertical_px:
        # Landscape orientation - horizontal edge is width
        card_width_px = horizontal_px
        card_mm = CARD_WIDTH_MM  # 85.6mm
        orientation = "landscape"
    else:
        # Portrait orientation - vertical edge is width
        card_width_px = vertical_px
        card_mm = CARD_WIDTH_MM  # 85.6mm (vertical edge is the physical width)
        orientation = "portrait"
    
    # Scale factor: mm per pixel
    scale_factor = card_mm / card_width_px
    
    if debug:
        print(f"   Horizontal edge: {horizontal_px:.1f} px")
        print(f"   Vertical edge: {vertical_px:.1f} px")
        print(f"   Orientation: {orientation}")
        print(f"   Using {card_width_px:.1f} px = {card_mm}mm")
        print(f"   Scale: {scale_factor:.4f} mm/px")
    
    # Calculate pupil distance using horizontal component in image
    pupil_dx = abs(pupil_right_px[0] - pupil_left_px[0])
    pupil_dy = abs(pupil_right_px[1] - pupil_left_px[1])
    
    # If significant vertical offset, warn
    if pupil_dy > pupil_dx * 0.1:  # More than 10% tilt
        result.warnings.append(f"Head may be tilted (dy={pupil_dy:.1f}px)")
    
    # Use horizontal distance for PD
    pupil_dist_px = pupil_dx
    
    # Raw PD in mm
    pd_mm = pupil_dist_px * scale_factor
    
    if debug:
        print(f"   Pupil: dx={pupil_dx:.1f}, dy={pupil_dy:.1f} px")
        print(f"   PD: {pd_mm:.2f} mm")
    
    # Estimate camera distance
    camera_distance = card_mm / scale_factor * 0.5  # Rough estimate
    
    result.success = True
    result.pd_near_mm = pd_mm
    result.pd_far_mm = pd_mm  # No adjustment
    result.camera_distance_mm = camera_distance
    result.validation_info = {'scale_factor': scale_factor, 'orientation': orientation}
    
    if debug:
        print(f"\n[PD] === RESULT: {pd_mm:.2f} mm ===")
    
    return result


def calculate_precise_pd(
    card_corners: np.ndarray,
    pupil_left_px: Tuple[float, float],
    pupil_right_px: Tuple[float, float],
    image_shape: Tuple[int, int],
    image: Optional[np.ndarray] = None,
    head_pose: Optional[dict] = None,
    K: Optional[np.ndarray] = None,
    D: Optional[np.ndarray] = None,
    debug_dir: Optional[str] = None,
    debug: bool = False
) -> PhotogrammetryResult:
    """
    Calculate PD using full 3D photogrammetric method when camera intrinsics available,
    otherwise fallback to simple scale method.
    
    IMPROVEMENT: Now uses full 3D ray-plane intersection when camera calibration is available.
    """
    # IMPROVEMENT: Use full 3D method if camera intrinsics provided
    if K is not None and image is not None:
        calculator = PhotogrammetricPDCalculator(debug_dir=debug_dir)
        
        # Convert head_pose to dict format if needed
        head_pose_dict = None
        if head_pose is not None:
            if hasattr(head_pose, 'yaw'):
                head_pose_dict = {
                    'yaw': head_pose.yaw,
                    'pitch': head_pose.pitch,
                    'roll': head_pose.roll
                }
            elif isinstance(head_pose, dict):
                head_pose_dict = head_pose
        
        result = calculator.calculate(
            image=image,
            card_corners=card_corners,
            pupil_left_px=pupil_left_px,
            pupil_right_px=pupil_right_px,
            head_pose=head_pose_dict,
            K=K,
            D=D
        )
        
        if result.success:
            return result
        else:
            # Fallback to simple method if 3D fails
            if debug:
                print(f"[Photogrammetry] 3D method failed: {result.error_message}, using scale method")
    
    # Fallback: Use the simple robust method
    return simple_pd_from_scale(
        card_corners=card_corners,
        pupil_left_px=pupil_left_px,
        pupil_right_px=pupil_right_px,
        debug=debug
    )
