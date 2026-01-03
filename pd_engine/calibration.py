"""
Calibration Module - Hybrid Card Detection Pipeline

Pipeline:
1. Roboflow YOLO for fast card ROI detection
2. MIDV500 segmentation on ROI patch
3. Virtual Corner Regression (edge fitting + intersection)
4. Homography-based metric rectification
5. Recalibrate to global coordinates for PD calculation

ISO/IEC 7810 ID-1 Card Dimensions:
- Width: 85.60 mm
- Height: 53.98 mm
- Corner Radius: 3.18 mm (handled via virtual corners)
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
import os
from dotenv import load_dotenv

from .forehead_detection import ForeheadDetector, ForeheadROI
from .sam3_segmentation import SAM3Segmenter, SAM3SegmentationResult
from .utils import validate_card_flatness

load_dotenv()


# Card dimensions (ISO/IEC 7810 ID-1)
CARD_WIDTH_MM = 85.60
CARD_HEIGHT_MM = 53.98
CARD_ASPECT_RATIO = CARD_WIDTH_MM / CARD_HEIGHT_MM

# Configuration
API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_WORKSPACE = "ori-einstein-i703x"
ROBOFLOW_PROJECT = "credit-card-for-glasses"
ROBOFLOW_VERSION = 2


@dataclass
class CardDetectionResult:
    """Result of card detection with virtual corners."""
    detected: bool
    corners: Optional[np.ndarray] = None  # Virtual corners in GLOBAL coords: TL, TR, BR, BL
    card_width_px: Optional[float] = None
    card_height_px: Optional[float] = None
    scale_factor: Optional[float] = None  # mm per pixel
    homography: Optional[np.ndarray] = None  # Pixel to mm transformation
    confidence: float = 0.0
    error_message: Optional[str] = None


class CardCalibration:
    """
    Hybrid card detection: Roboflow ROI + MIDV500 Segmentation.
    
    Uses YOLO-based detection for fast card localization,
    then deep learning segmentation for precise corner detection.
    """
    
    # Class-level model caches
    _midv500_model = None
    _midv500_loaded = False
    _roboflow_model = None
    _roboflow_loaded = False
    
    def __init__(self, debug_dir: Optional[str] = None):
        """
        Initialize CardCalibration.
        
        Args:
            debug_dir: Directory to save debug images (None to disable)
        """
        self.debug_dir = debug_dir
        self._step = 0
        
    def _save_debug(self, name: str, image: np.ndarray):
        """Save debug image if debug_dir is set."""
        if self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)
            path = os.path.join(self.debug_dir, f"{self._step:02d}_{name}.jpg")
            cv2.imwrite(path, image)
            print(f"  ✓ [{self._step:02d}] Saved: {name}.jpg")
            self._step += 1
    
    def _load_roboflow_model(self):
        """Lazy load Roboflow model."""
        if CardCalibration._roboflow_loaded:
            return CardCalibration._roboflow_model is not None
        
        try:
            from roboflow import Roboflow
            
            rf = Roboflow(api_key=API_KEY)
            
            project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
            CardCalibration._roboflow_model = project.version(ROBOFLOW_VERSION).model
            CardCalibration._roboflow_loaded = True
            return True
        except Exception as e:
            print(f"  ⚠️ Roboflow not available: {e}")
            CardCalibration._roboflow_loaded = True
            return False
    
    def _load_midv500_model(self):
        """Lazy load MIDV500 model."""
        if CardCalibration._midv500_loaded:
            return CardCalibration._midv500_model is not None
            
        try:
            import torch
            from midv500models.pre_trained_models import create_model
            
            CardCalibration._midv500_model = create_model("Unet_resnet34_2020-05-19")
            CardCalibration._midv500_model.eval()
            CardCalibration._midv500_loaded = True
            return True
        except ImportError as e:
            print(f"  ⚠️ MIDV500 dependencies not available: {e}")
            CardCalibration._midv500_loaded = True
            return False
    
    def _detect_roi_roboflow(self, image: np.ndarray, 
                             image_path: Optional[str] = None) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect card ROI using Roboflow YOLO model.
        
        Returns:
            (x1, y1, x2, y2) bounding box or None
        """
        if not self._load_roboflow_model():
            return None
        
        try:
            import supervision as sv
            
            # Roboflow needs file path or URL
            if image_path:
                result = CardCalibration._roboflow_model.predict(
                    image_path, confidence=0.4, overlap=0.3
                ).json()
            else:
                # Save temp file
                temp_path = "/tmp/card_detect_temp.jpg"
                cv2.imwrite(temp_path, image)
                result = CardCalibration._roboflow_model.predict(
                    temp_path, confidence=0.4, overlap=0.3
                ).json()
            
            detections = sv.Detections.from_inference(result)
            
            if len(detections) == 0:
                return None
            
            bbox = detections.xyxy[0]
            return tuple(map(int, bbox))
            
        except Exception as e:
            print(f"  ⚠️ Roboflow detection failed: {e}")
            return None
    
    def _detect_roi_forehead(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect card ROI using MediaPipe forehead detection.
        
        This method uses facial landmarks to identify the forehead region
        where the calibration card is expected to be held.
        
        Returns:
            (x1, y1, x2, y2) bounding box or None
        """
        try:
            # Create forehead detector with same debug dir
            forehead_detector = ForeheadDetector(debug_dir=self.debug_dir)
            
            # Detect forehead region
            forehead_result = forehead_detector.detect(image)
            
            if not forehead_result.detected:
                print(f"  ⚠️ Forehead detection failed: {forehead_result.error_message}")
                return None
            
            return forehead_result.roi_box
            
        except Exception as e:
            print(f"  ⚠️ Forehead detection error: {e}")
            return None
    
    def _segment_card_midv500(self, roi: np.ndarray) -> Optional[np.ndarray]:
        """
        Segment card using MIDV500 model on ROI.
        
        Args:
            roi: BGR image of card region
            
        Returns:
            Binary mask (0/1) or None if failed
        """
        if not self._load_midv500_model():
            return None
            
        import torch
        import albumentations as albu
        from iglovikov_helper_functions.utils.image_utils import pad, unpad
        from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
        
        # Convert BGR to RGB
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        # Preprocessing
        transform = albu.Compose([albu.Normalize(p=1)], p=1)
        padded_image, pads = pad(roi_rgb, factor=32, border=cv2.BORDER_CONSTANT)
        
        x = transform(image=padded_image)["image"]
        x = torch.unsqueeze(tensor_from_rgb_image(x), 0)
        
        # Inference
        with torch.no_grad():
            prediction = CardCalibration._midv500_model(x)[0][0]
        
        # Create binary mask
        mask = (prediction > 0).cpu().numpy().astype(np.uint8)
        mask = unpad(mask, pads)
        
        return mask
    
    def _segment_card_sam3(self, roi: np.ndarray) -> Optional[np.ndarray]:
        """
        Segment card using SAM3 model via Roboflow workflow.
        
        Args:
            roi: BGR image of card region
            
        Returns:
            Binary mask (0/255) or None if failed
        """
        try:
            segmenter = SAM3Segmenter(debug_dir=self.debug_dir)
            result = segmenter.segment(roi)
            
            if not result.success:
                print(f"  ⚠️ SAM3 segmentation failed: {result.error_message}")
                return None
            
            return result.mask
            
        except Exception as e:
            print(f"  ⚠️ SAM3 segmentation error: {e}")
            return None
    
    def _preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Preprocess mask: fill holes, morphological cleanup.
        
        Handles both 0/1 and 0/255 input formats.
        """
        # Normalize to 0/255 range if needed
        if mask.max() <= 1:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)
        
        # Step 1: Morphological closing to fill small holes
        kernel_close = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        
        # Step 2: Fill interior holes using flood fill
        h, w = mask.shape
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        mask_copy = mask.copy()
        cv2.floodFill(mask_copy, flood_mask, (0, 0), 255)
        
        # Invert to get holes
        holes = cv2.bitwise_not(mask_copy)
        
        # Combine original mask with filled holes
        mask = cv2.bitwise_or(mask, holes)
        
        # Threshold to ensure binary
        mask = (mask > 127).astype(np.uint8)
        
        # Step 3: Morphological opening to remove noise
        kernel_open = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        
        return mask
    
    def _detect_card_corners(self, mask: np.ndarray, roi: np.ndarray, debug: bool = False) -> Optional[np.ndarray]:
        """
        Detect card corners from segmentation mask using multiple methods with voting.
        
        Enhanced approach:
        1. Find the largest contour
        2. Try multiple corner detection methods
        3. Validate and vote for best corners
        4. Refine to sub-pixel accuracy
        
        Args:
            mask: Binary mask (0/1 or 0/255)
            roi: Original ROI image for debug visualization
            debug: Whether to save debug images
            
        Returns:
            4 corners as np.ndarray of shape (4, 2) in order TL, TR, BR, BL, or None
        """
        # Ensure mask is 0/255
        if mask.max() <= 1:
            mask_vis = (mask * 255).astype(np.uint8)
        else:
            mask_vis = mask.astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(mask_vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get largest contour
        largest = max(contours, key=cv2.contourArea)
        
        # Save contour debug
        if self.debug_dir:
            contour_debug = roi.copy()
            cv2.drawContours(contour_debug, [largest], -1, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(self.debug_dir, "contour.jpg"), contour_debug)
        
        # Multiple detection methods
        methods = []
        
        # Method 1: approxPolyDP
        epsilon = 0.02 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        if len(approx) == 4:
            corners1 = approx.reshape(4, 2).astype(np.float32)
            methods.append(('approxPolyDP', corners1))
        
        # Method 2: minAreaRect
        rect = cv2.minAreaRect(largest)
        corners2 = cv2.boxPoints(rect).astype(np.float32)
        methods.append(('minAreaRect', corners2))
        
        # Method 3: Harris corner detection on edges
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        edges = cv2.Canny(gray_roi, 50, 150)
        corners_harris = cv2.cornerHarris(edges, 2, 3, 0.04)
        corners_harris = cv2.dilate(corners_harris, None)
        
        # Extract top 4 Harris corners
        if np.max(corners_harris) > 0.01:
            corner_coords = np.argwhere(corners_harris > 0.01 * corners_harris.max())
            if len(corner_coords) >= 4:
                # Get 4 corners closest to expected positions
                corners3 = self._extract_harris_corners(corner_coords, roi.shape[:2])
                if corners3 is not None:
                    methods.append(('harris', corners3))
        
        # Method 4: Shi-Tomasi corner detection
        corners_shitomasi = cv2.goodFeaturesToTrack(
            edges, maxCorners=4, qualityLevel=0.01, minDistance=20
        )
        if corners_shitomasi is not None and len(corners_shitomasi) == 4:
            corners4 = corners_shitomasi.reshape(4, 2).astype(np.float32)
            methods.append(('shi-tomasi', corners4))
        
        # Validate and select best method
        valid_corners = []
        for method_name, corners in methods:
            corners_ordered = self._order_corners(corners)
            if self._validate_corners(corners_ordered):
                valid_corners.append((method_name, corners_ordered))
        
        if not valid_corners:
            # Fallback to first method
            if methods:
                corners = self._order_corners(methods[0][1])
                method = methods[0][0]
            else:
                return None
        elif len(valid_corners) == 1:
            corners = valid_corners[0][1]
            method = valid_corners[0][0]
        else:
            # Average valid corners
            corners = np.mean([c for _, c in valid_corners], axis=0)
            method = f"voted({len(valid_corners)} methods)"
        
        # Refine to sub-pixel accuracy
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        from .utils import refine_corners_subpixel
        corners = refine_corners_subpixel(gray_roi, corners, window_size=5)
        
        # Save corners debug
        if self.debug_dir:
            corners_debug = roi.copy()
            cv2.polylines(corners_debug, [corners.astype(np.int32)], True, (0, 255, 0), 3)
            labels = ['TL', 'TR', 'BR', 'BL']
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
            for i, pt in enumerate(corners):
                cv2.circle(corners_debug, (int(pt[0]), int(pt[1])), 10, colors[i], -1)
                cv2.putText(corners_debug, labels[i], (int(pt[0])+15, int(pt[1])+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(corners_debug, f"Method: {method}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(self.debug_dir, "corners.jpg"), corners_debug)
        
        if debug:
            print(f"  Corner detection method: {method}")
        
        return corners
    
    def _extract_harris_corners(self, corner_coords: np.ndarray, roi_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Extract 4 corners from Harris corner detection results."""
        if len(corner_coords) < 4:
            return None
        
        # Convert to (x, y) format
        points = corner_coords[:, [1, 0]].astype(np.float32)
        
        # Find corners closest to expected positions (TL, TR, BR, BL)
        h, w = roi_shape
        expected_positions = [
            (w * 0.25, h * 0.25),  # TL
            (w * 0.75, h * 0.25),  # TR
            (w * 0.75, h * 0.75),  # BR
            (w * 0.25, h * 0.75),  # BL
        ]
        
        selected = []
        used = set()
        for exp_pos in expected_positions:
            distances = [np.linalg.norm(p - exp_pos) for p in points]
            best_idx = np.argmin(distances)
            if best_idx not in used:
                selected.append(points[best_idx])
                used.add(best_idx)
        
        if len(selected) == 4:
            return np.array(selected, dtype=np.float32)
        return None
    
    def _validate_corners(self, corners: np.ndarray) -> bool:
        """Validate that corners form a reasonable rectangle."""
        if corners.shape != (4, 2):
            return False
        
        # Check aspect ratio
        width = np.linalg.norm(corners[1] - corners[0])
        height = np.linalg.norm(corners[3] - corners[0])
        
        if height == 0:
            return False
        
        aspect = width / height
        # Card aspect ratio is ~1.586, allow 0.5-3.0 range
        if aspect < 0.5 or aspect > 3.0:
            return False
        
        # Check minimum size
        if width < 50 or height < 30:
            return False
        
        return True
    
    def _extract_convex_hull(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """Extract convex hull from mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        largest = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest)
        
        return hull
    
    def _fit_edge_lines(self, hull: np.ndarray, roi_shape: Tuple[int, int]) -> dict:
        """
        Fit lines to the four edges using direct point classification.
        
        Simplified approach:
        1. Use minAreaRect for orientation
        2. Classify points by Y (top/bottom) and X (left/right) in original frame
        3. Prune corners and fit lines
        """
        h, w = roi_shape[:2]
        
        # Get rotated bounding box
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)  # 4 corner points of the rotated rect
        
        # Extract hull points
        points = hull.reshape(-1, 2).astype(np.float32)
        
        # Sort box points to get corners: TL, TR, BR, BL
        # First find top-most points
        sorted_by_y = box[np.argsort(box[:, 1])]
        top_two = sorted_by_y[:2]
        bottom_two = sorted_by_y[2:]
        
        # Sort each pair by X
        top_two = top_two[np.argsort(top_two[:, 0])]
        bottom_two = bottom_two[np.argsort(bottom_two[:, 0])]
        
        # Box corners: TL, TR, BR, BL
        box_tl = top_two[0]
        box_tr = top_two[1]
        box_br = bottom_two[1]
        box_bl = bottom_two[0]
        
        # Define line equations for each edge of the box
        def pts_to_line(p1, p2):
            """Convert two points to line equation Ax + By + C = 0."""
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            # Normal: (-dy, dx)
            A, B = -dy, dx
            C = -(A * p1[0] + B * p1[1])
            # Normalize
            norm = np.sqrt(A*A + B*B)
            if norm > 1e-6:
                A, B, C = A/norm, B/norm, C/norm
            return (A, B, C)
        
        # Define edge lines from box corners
        top_line = pts_to_line(box_tl, box_tr)
        bottom_line = pts_to_line(box_bl, box_br)
        left_line = pts_to_line(box_tl, box_bl)
        right_line = pts_to_line(box_tr, box_br)
        
        # Classify hull points to each edge based on distance
        def point_to_line_dist(pt, line):
            A, B, C = line
            return abs(A * pt[0] + B * pt[1] + C)
        
        top_pts, bottom_pts, left_pts, right_pts = [], [], [], []
        
        for pt in points:
            dists = {
                'top': point_to_line_dist(pt, top_line),
                'bottom': point_to_line_dist(pt, bottom_line),
                'left': point_to_line_dist(pt, left_line),
                'right': point_to_line_dist(pt, right_line)
            }
            closest = min(dists, key=dists.get)
            
            # Only add if close enough (within 20% of box size)
            threshold = max(h, w) * 0.15
            if dists[closest] < threshold:
                if closest == 'top':
                    top_pts.append(pt)
                elif closest == 'bottom':
                    bottom_pts.append(pt)
                elif closest == 'left':
                    left_pts.append(pt)
                else:
                    right_pts.append(pt)
        
        # Prune corner points (15% from each end)
        def prune_corners(pts, axis):
            if len(pts) < 5:
                return pts
            pts = sorted(pts, key=lambda p: p[axis])
            n = len(pts)
            start = max(1, int(n * 0.15))
            end = min(n - 1, int(n * 0.85))
            return pts[start:end]
        
        # For horizontal edges, sort by X; for vertical edges, sort by Y
        top_pts = prune_corners(top_pts, axis=0)
        bottom_pts = prune_corners(bottom_pts, axis=0)
        left_pts = prune_corners(left_pts, axis=1)
        right_pts = prune_corners(right_pts, axis=1)
        
        # Fit lines using robust regression
        def fit_line(pts, fallback_line) -> Tuple[float, float, float]:
            if len(pts) < 2:
                return fallback_line  # Use box line as fallback
            pts_arr = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
            vx, vy, x0, y0 = cv2.fitLine(pts_arr, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
            A, B = -vy, vx
            C = -(A * x0 + B * y0)
            norm = np.sqrt(A*A + B*B)
            if norm > 1e-6:
                A, B, C = A/norm, B/norm, C/norm
            return (A, B, C)
        
        return {
            'top': fit_line(top_pts, top_line),
            'bottom': fit_line(bottom_pts, bottom_line),
            'left': fit_line(left_pts, left_line),
            'right': fit_line(right_pts, right_line)
        }
    
    def _enforce_parallel_edges(self, edge_lines: dict) -> dict:
        """
        Enforce that opposite edges are parallel.
        Cards have parallel top/bottom and left/right edges.
        
        The approach: average the directions of opposite edge pairs
        and adjust both lines to use the averaged direction.
        """
        def get_direction(line):
            """Get direction vector from line equation Ax + By + C = 0."""
            A, B, C = line
            # Direction is perpendicular to normal (A, B)
            return (B, -A)  # Normalized since A, B are normalized
        
        def make_parallel_to(line, direction, point_on_line=None):
            """Create new line parallel to given direction, passing through a point on original line."""
            A, B, C = line
            dx, dy = direction
            # New normal is perpendicular to direction: (-dy, dx)
            new_A, new_B = -dy, dx
            # Normalize
            norm = np.sqrt(new_A*new_A + new_B*new_B)
            if norm > 1e-6:
                new_A, new_B = new_A/norm, new_B/norm
            
            # Find a point on the original line for the new C
            if point_on_line is None:
                # Pick a point on original line
                if abs(B) > 1e-6:
                    px, py = 0, -C/B
                elif abs(A) > 1e-6:
                    px, py = -C/A, 0
                else:
                    return line
            else:
                px, py = point_on_line
            
            new_C = -(new_A * px + new_B * py)
            return (new_A, new_B, new_C)
        
        def average_direction(dir1, dir2):
            """Average two direction vectors (handling opposite directions)."""
            d1 = np.array(dir1)
            d2 = np.array(dir2)
            # Check if directions are opposite (dot product < 0)
            if np.dot(d1, d2) < 0:
                d2 = -d2
            avg = (d1 + d2) / 2
            norm = np.linalg.norm(avg)
            if norm > 1e-6:
                avg = avg / norm
            return tuple(avg)
        
        # Enforce parallel horizontal edges (top and bottom)
        top_dir = get_direction(edge_lines['top'])
        bottom_dir = get_direction(edge_lines['bottom'])
        avg_horizontal = average_direction(top_dir, bottom_dir)
        
        # Enforce parallel vertical edges (left and right)
        left_dir = get_direction(edge_lines['left'])
        right_dir = get_direction(edge_lines['right'])
        avg_vertical = average_direction(left_dir, right_dir)
        
        return {
            'top': make_parallel_to(edge_lines['top'], avg_horizontal),
            'bottom': make_parallel_to(edge_lines['bottom'], avg_horizontal),
            'left': make_parallel_to(edge_lines['left'], avg_vertical),
            'right': make_parallel_to(edge_lines['right'], avg_vertical)
        }
    
    def _compute_virtual_corners(self, edge_lines: dict) -> Optional[np.ndarray]:
        """Compute virtual corners as line intersections."""
        def line_intersection(l1, l2):
            A1, B1, C1 = l1
            A2, B2, C2 = l2
            D = A1 * B2 - A2 * B1
            if abs(D) < 1e-10:
                return None
            x = (B1 * C2 - B2 * C1) / D
            y = (A2 * C1 - A1 * C2) / D
            return (x, y)
        
        for name in ['top', 'bottom', 'left', 'right']:
            if edge_lines.get(name) is None:
                return None
        
        tl = line_intersection(edge_lines['top'], edge_lines['left'])
        tr = line_intersection(edge_lines['top'], edge_lines['right'])
        br = line_intersection(edge_lines['bottom'], edge_lines['right'])
        bl = line_intersection(edge_lines['bottom'], edge_lines['left'])
        
        if None in [tl, tr, br, bl]:
            return None
        
        return np.array([tl, tr, br, bl], dtype=np.float32)
    
    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """Order corners as TL, TR, BR, BL."""
        sorted_by_y = corners[np.argsort(corners[:, 1])]
        top_pts = sorted_by_y[:2][np.argsort(sorted_by_y[:2, 0])]
        bottom_pts = sorted_by_y[2:][np.argsort(sorted_by_y[2:, 0])]
        
        return np.array([
            top_pts[0], top_pts[1],
            bottom_pts[1], bottom_pts[0]
        ], dtype=np.float32)
    
    def _check_orientation(self, corners: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
        """Check orientation and return correct dimensions."""
        width_px = (np.linalg.norm(corners[1] - corners[0]) + 
                   np.linalg.norm(corners[2] - corners[3])) / 2
        height_px = (np.linalg.norm(corners[3] - corners[0]) + 
                    np.linalg.norm(corners[2] - corners[1])) / 2
        
        if width_px > height_px:
            return corners, (CARD_WIDTH_MM, CARD_HEIGHT_MM)
        else:
            return corners, (CARD_HEIGHT_MM, CARD_WIDTH_MM)
    
    def _compute_homography(self, corners: np.ndarray, 
                           card_dims: Tuple[float, float]) -> Optional[np.ndarray]:
        """Compute homography from pixel corners to metric space."""
        width_mm, height_mm = card_dims
        
        pts_dst = np.array([
            [0, 0],
            [width_mm, 0],
            [width_mm, height_mm],
            [0, height_mm]
        ], dtype=np.float32)
        
        H, _ = cv2.findHomography(corners, pts_dst, cv2.RANSAC, 5.0)
        return H
    
    def detect_card(self, image: np.ndarray, 
                   image_path: Optional[str] = None,
                   debug: bool = False,
                   use_forehead_roi: bool = True) -> CardDetectionResult:
        """
        Detect credit card using forehead ROI + MIDV500 pipeline.
        
        Pipeline:
        1. Try forehead detection (MediaPipe) for ROI
        2. Fallback to Roboflow YOLO if forehead fails
        3. Apply MIDV500 segmentation on ROI
        4. Extract virtual corners via edge fitting
        
        Args:
            image: BGR input image
            image_path: Optional path to image (for Roboflow fallback)
            debug: Enable debug output
            use_forehead_roi: Try forehead detection first (default: True)
            
        Returns:
            CardDetectionResult with corners in GLOBAL coordinates
        """
        self._step = 0
        h, w = image.shape[:2]
        roi_method = "none"
        
        if debug:
            print("\n" + "=" * 60)
            print("  FOREHEAD-BASED CARD DETECTION")
            print("  Method: Forehead ROI + MIDV500 Segmentation")
            print("=" * 60)
            print(f"  Image size: {w}x{h}px")
            if self.debug_dir:
                print(f"  Debug dir: {self.debug_dir}")
        
        if self.debug_dir:
            # Save input image
            os.makedirs(self.debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(self.debug_dir, "input.jpg"), image)
        
        bbox = None
        
        # ============================================================
        # PHASE 1A: Forehead ROI Detection (Primary)
        # ============================================================
        if use_forehead_roi:
            if debug:
                print("\n[Phase 1A] Forehead ROI Detection (MediaPipe)")
            
            bbox = self._detect_roi_forehead(image)
            
            if bbox is not None:
                roi_method = "forehead"
                if debug:
                    x1, y1, x2, y2 = bbox
                    print(f"  ✓ Forehead ROI: ({x1},{y1})-({x2},{y2}) = {x2-x1}x{y2-y1}px")
            else:
                if debug:
                    print("  ⚠️ Forehead detection failed, trying Roboflow fallback")
        
        # ============================================================
        # PHASE 1B: Roboflow ROI Detection (Fallback)
        # ============================================================
        if bbox is None:
            if debug:
                print("\n[Phase 1B] Roboflow ROI Detection (Fallback)")
            
            bbox = self._detect_roi_roboflow(image, image_path)
            
            if bbox is not None:
                roi_method = "roboflow"
            else:
                print("  ✗ Both forehead and Roboflow detection failed")
                return CardDetectionResult(
                    detected=False,
                    error_message="Could not detect card ROI (forehead and Roboflow failed)"
                )
        
        x1, y1, x2, y2 = bbox
        ml_w, ml_h = x2 - x1, y2 - y1
        
        if debug:
            print(f"  ✓ Card detected at ({x1},{y1})-({x2},{y2}) = {ml_w}x{ml_h}px")
        
        # ROI detected and extracted
        
        # Add padding (15%)
        padding_x = int(ml_w * 0.15)
        padding_y = int(ml_h * 0.15)
        
        roi_x1 = max(0, x1 - padding_x)
        roi_y1 = max(0, y1 - padding_y)
        roi_x2 = min(w, x2 + padding_x)
        roi_y2 = min(h, y2 + padding_y)
        
        roi = image[roi_y1:roi_y2, roi_x1:roi_x2].copy()
        roi_h, roi_w = roi.shape[:2]
        
        if debug:
            print(f"  ✓ ROI extracted: ({roi_x1},{roi_y1})-({roi_x2},{roi_y2}) = {roi_w}x{roi_h}px")
        
        # ROI extracted for segmentation
        
        # ============================================================
        # PHASE 2: SAM3 Segmentation on ROI
        # ============================================================
        if debug:
            print("\n[Phase 2] SAM3 Segmentation (Roboflow Workflow)")
        
        mask = self._segment_card_sam3(roi)
        
        if mask is None:
            print("  ✗ SAM3 segmentation failed")
            return CardDetectionResult(
                detected=False,
                error_message="SAM3 segmentation failed"
            )
        
        if debug:
            print(f"  ✓ SAM3 mask generated")
        
        # Save mask for debugging
        if self.debug_dir:
            cv2.imwrite(os.path.join(self.debug_dir, "mask.jpg"), mask)
        
        # ============================================================
        # PHASE 3: Mask Preprocessing (hole filling)
        # ============================================================
        if debug:
            print("\n[Phase 3] Mask Preprocessing (hole filling)")
        
        mask = self._preprocess_mask(mask)
        
        # Mask preprocessed
        
        mask_area = np.sum(mask > 0)
        roi_area = roi_h * roi_w
        mask_percent = (mask_area / roi_area) * 100
        
        if debug:
            print(f"  ✓ Mask area: {mask_area} pixels ({mask_percent:.1f}% of ROI)")
        
        # Save ROI for debugging
        if self.debug_dir:
            cv2.imwrite(os.path.join(self.debug_dir, "roi.jpg"), roi)
        
        # Save processed mask for debugging (0/1 -> 0/255)
        if self.debug_dir:
            mask_vis = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask
            cv2.imwrite(os.path.join(self.debug_dir, "mask_processed.jpg"), mask_vis)
        
        # ============================================================
        # PHASE 4: Corner Detection from Segmentation
        # ============================================================
        if debug:
            print("\n[Phase 4] Corner Detection")
        
        corners_roi = self._detect_card_corners(mask, roi, debug=debug)
        
        if corners_roi is None:
            print("  ✗ Could not detect card corners")
            return CardDetectionResult(
                detected=False,
                error_message="Could not detect card corners"
            )
        
        if debug:
            print("  ✓ Card corners detected")
        
        # ============================================================
        # PHASE 7: Convert corners to GLOBAL coordinates
        # ============================================================
        if debug:
            print("\n[Phase 7] Global Coordinate Recalibration")
        
        # Add ROI offset to convert to global image coordinates
        roi_offset = np.array([roi_x1, roi_y1], dtype=np.float32)
        corners_global = corners_roi + roi_offset
        
        if debug:
            print(f"  ✓ ROI offset: ({roi_x1}, {roi_y1})")
            for i, (name, pt) in enumerate(zip(['TL', 'TR', 'BR', 'BL'], corners_global)):
                print(f"    {name}: ({pt[0]:.1f}, {pt[1]:.1f})")
        
        # Check orientation
        corners_global, card_dims = self._check_orientation(corners_global)
        
        # Calculate dimensions
        width_px = (np.linalg.norm(corners_global[1] - corners_global[0]) + 
                   np.linalg.norm(corners_global[2] - corners_global[3])) / 2
        height_px = (np.linalg.norm(corners_global[3] - corners_global[0]) + 
                    np.linalg.norm(corners_global[2] - corners_global[1])) / 2
        
        if height_px > width_px:
            width_px, height_px = height_px, width_px
        
        scale_factor = CARD_WIDTH_MM / width_px
        
        # Check aspect ratio
        aspect = width_px / height_px
        aspect_error = abs(aspect - CARD_ASPECT_RATIO) / CARD_ASPECT_RATIO
        
        # IMPROVEMENT: Validate card flatness
        is_flat, convexity_defect, flatness_info = validate_card_flatness(corners_global, image)
        if not is_flat:
            if debug:
                print(f"   ⚠️ Card may be bent (convexity: {convexity_defect*100:.1f}%)")
        
        if aspect_error > 0.20:
            if debug:
                print(f"   ⚠️ Aspect error {aspect_error*100:.0f}% > 20%, using ML bbox fallback")
            
            # Use ML bbox as fallback
            height_px = float(ml_h)
            width_px = height_px * CARD_ASPECT_RATIO
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            
            corners_global = np.array([
                [cx - width_px/2, cy - height_px/2],
                [cx + width_px/2, cy - height_px/2],
                [cx + width_px/2, cy + height_px/2],
                [cx - width_px/2, cy + height_px/2]
            ], dtype=np.float32)
            
            scale_factor = CARD_WIDTH_MM / width_px
            aspect = CARD_ASPECT_RATIO
        
        # ============================================================
        # PHASE 8: Compute Homography
        # ============================================================
        if debug:
            print("\n[Phase 8] Homography Computation")
        
        H = self._compute_homography(corners_global, card_dims)
        
        confidence = max(0, 1.0 - aspect_error)
        
        if debug:
            print("  ✓ Homography matrix computed")
            print("\n" + "=" * 60)
            print("  ✅ CARD DETECTION COMPLETE")
            print("=" * 60)
            print(f"  Dimensions: {width_px:.1f} x {height_px:.1f} px")
            print(f"  Aspect ratio: {aspect:.3f} (ideal: {CARD_ASPECT_RATIO:.3f})")
            print(f"  Scale factor: {scale_factor:.4f} mm/px")
            print(f"  Confidence: {confidence:.1%}")
        
        # Save final result visualization
        if self.debug_dir:
            final_viz = image.copy()
            cv2.polylines(final_viz, [corners_global.astype(np.int32)], True, (0, 255, 0), 3)
            for pt in corners_global:
                cv2.circle(final_viz, (int(pt[0]), int(pt[1])), 8, (0, 0, 255), -1)
            info = f"Card: {width_px:.0f}x{height_px:.0f}px | Scale: {scale_factor:.4f}mm/px"
            cv2.putText(final_viz, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(self.debug_dir, "result.jpg"), final_viz)
        
        return CardDetectionResult(
            detected=True,
            corners=corners_global,
            card_width_px=width_px,
            card_height_px=height_px,
            scale_factor=scale_factor,
            homography=H,
            confidence=confidence
        )
    
    def transform_points_to_mm(self, points: np.ndarray, 
                               result: CardDetectionResult) -> Optional[np.ndarray]:
        """
        Transform pixel coordinates to millimeters using homography.
        
        Args:
            points: Nx2 array of pixel coordinates
            result: CardDetectionResult with homography
            
        Returns:
            Nx2 array of coordinates in mm
        """
        if result.homography is None:
            return None
        
        pts = points.reshape(-1, 1, 2).astype(np.float32)
        pts_mm = cv2.perspectiveTransform(pts, result.homography)
        
        return pts_mm.reshape(-1, 2)
