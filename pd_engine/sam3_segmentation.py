"""
SAM3 Segmentation Module - Roboflow Workflow-based Card Segmentation

Uses SAM3 (Segment Anything Model 3) via Roboflow workflow for accurate
card segmentation on forehead images. Prompts are optimized for detecting
rectangular cards held on the forehead.
"""

from rich.themes import DEFAULT
import base64
import os
from dataclasses import dataclass
from typing import Optional, List

import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv

load_dotenv()


@dataclass
class SAM3SegmentationResult:
    """Result of SAM3 card segmentation."""
    success: bool
    mask: Optional[np.ndarray] = None  # Binary mask (0/255)
    mask_visualization: Optional[np.ndarray] = None  # Color visualization
    error_message: Optional[str] = None


class SAM3Segmenter:
    """
    Card segmentation using SAM3 via Roboflow workflow.
    
    Uses prompt-based segmentation optimized for detecting
    rectangular cards held on the forehead.
    """
    
    # Default prompts for card detection
    # DEFAULT_PROMPTS: List[str] = [
    #     "rectangular card held by hand",
    #     "flat rectangular object on forehead",
    #     "thin card-like object forehead",
    #     "identification document hand forehead"
    # ]

    DEFAULT_PROMPTS: List[str] = [
        "rectangular card held by hand",
        "flat rectangular object on forehead",
        "thin card-like object forehead",
        "identification document hand forehead",
        "thin rectangular plastic card in hand",
        "flat credit card against forehead skin",
        "rectangular payment card held by fingers",
        "thin ID card pressed on forehead",
        "credit/debit card on person's forehead",
        "rectangular card object touching skin",
        "handheld thin rectangular document",
        "forehead-placed flat plastic rectangle"
    ]
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        workspace_name: str = "projecy-829hj",
        workflow_id: str = "sam3-with-prompts",
        debug_dir: Optional[str] = None
    ):
        """
        Initialize SAM3Segmenter.
        
        Args:
            api_key: Roboflow API key (uses env var if None)
            workspace_name: Roboflow workspace name
            workflow_id: Roboflow workflow ID
            debug_dir: Directory to save debug images
        """
        self.api_key = api_key or os.getenv("ROBOFLOW_API_KEY")
        self.workspace_name = workspace_name
        self.workflow_id = workflow_id
        self.debug_dir = debug_dir
        self._step = 0
        self._client = None
        
        if not self.api_key:
            raise ValueError(
                "ROBOFLOW_API_KEY not found. Set it in .env or pass api_key parameter."
            )
    
    def _get_client(self):
        """Lazy load the inference client."""
        if self._client is None:
            from inference_sdk import InferenceHTTPClient
            self._client = InferenceHTTPClient(
                api_url="https://serverless.roboflow.com",
                api_key=self.api_key
            )
        return self._client
    
    def _save_debug(self, name: str, image: np.ndarray) -> None:
        """Save debug image if debug_dir is set."""
        if self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)
            path = os.path.join(self.debug_dir, f"{self._step:02d}_{name}.jpg")
            cv2.imwrite(path, image)
            print(f"  [SAM3-{self._step:02d}] Saved: {name}")
            self._step += 1
    
    def _decode_mask_response(
        self, 
        mask_data: any
    ) -> Optional[np.ndarray]:
        """
        Decode mask visualization from workflow response.
        
        Handles different response formats:
        - Base64 encoded string
        - PIL Image
        - NumPy array
        
        Args:
            mask_data: Mask data from workflow response
            
        Returns:
            BGR image as numpy array, or None if decoding fails
        """
        if mask_data is None:
            return None
        
        if isinstance(mask_data, str):
            # Base64 encoded string
            if "," in mask_data:
                mask_data = mask_data.split(",")[1]
            
            try:
                img_bytes = base64.b64decode(mask_data)
                img_array = np.frombuffer(img_bytes, np.uint8)
                return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            except Exception as e:
                print(f"  [SAM3] Base64 decode failed: {e}")
                return None
        
        elif isinstance(mask_data, Image.Image):
            # PIL Image
            return cv2.cvtColor(np.array(mask_data), cv2.COLOR_RGB2BGR)
        
        elif isinstance(mask_data, np.ndarray):
            return mask_data
        
        else:
            # Try generic conversion
            try:
                return np.array(mask_data, dtype=np.uint8)
            except Exception:
                return None
    
    def _decode_rle_mask(
        self,
        rle_data: dict,
        image_shape: tuple[int, int]
    ) -> np.ndarray:
        """
        Decode RLE (Run-Length Encoded) mask from SAM3 response.
        
        SAM3 returns masks in COCO RLE format:
        - 'size': [height, width]
        - 'counts': RLE string
        
        Args:
            rle_data: Dict with 'size' and 'counts' keys
            image_shape: Target (height, width) for the mask
            
        Returns:
            Binary mask (0 or 255) as numpy array
        """
        try:
            from pycocotools import mask as mask_utils
            
            # Decode using pycocotools
            rle = {
                'size': rle_data['size'],
                'counts': rle_data['counts'].encode('utf-8')
            }
            mask = mask_utils.decode(rle)
            
            # Convert to 0/255 format
            mask = (mask * 255).astype(np.uint8)
            
            return mask
            
        except ImportError:
            print("  [SAM3] pycocotools not available, using manual RLE decode")
            return self._decode_rle_manual(rle_data, image_shape)
        except Exception as e:
            print(f"  [SAM3] RLE decode error: {e}")
            return self._decode_rle_manual(rle_data, image_shape)
    
    def _decode_rle_manual(
        self,
        rle_data: dict,
        image_shape: tuple[int, int]
    ) -> np.ndarray:
        """
        Manual RLE decoding fallback (without pycocotools).
        
        COCO RLE format: counts alternate between 0s and 1s.
        First value is number of 0s, then 1s, then 0s, etc.
        """
        try:
            size = rle_data['size']
            counts_str = rle_data['counts']
            
            h, w = size[0], size[1]
            
            # Parse RLE counts (COCO format uses LEB128 encoding)
            # For simplicity, create mask from bounding box if RLE parsing fails
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # Simple bounding box fallback based on the detection info
            print("  [SAM3] Using bounding box fallback for mask")
            
            return mask
            
        except Exception as e:
            print(f"  [SAM3] Manual RLE decode failed: {e}")
            return np.zeros(image_shape, dtype=np.uint8)
    
    def _extract_binary_mask(
        self, 
        mask_viz: np.ndarray,
        original_image: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Extract binary mask from SAM3 visualization.
        
        SAM3 visualization shows a colored overlay (often green/cyan/magenta)
        on top of the original image. We detect this by:
        1. Looking for high-saturation colored regions (the overlay)
        2. If original provided, comparing to find changed pixels
        
        Args:
            mask_viz: Color mask visualization (BGR)
            original_image: Original input image for comparison (optional)
            
        Returns:
            Binary mask (0 or 255)
        """
        h, w = mask_viz.shape[:2]
        
        # Method 1: Detect colored overlay via HSV saturation
        # SAM3 typically uses saturated colors (green, cyan, magenta) for masks
        hsv = cv2.cvtColor(mask_viz, cv2.COLOR_BGR2HSV)
        
        # Extract saturation channel - overlays have higher saturation
        saturation = hsv[:, :, 1]
        
        # Look for specific mask colors in HSV space
        # Green-ish: H=40-80, Cyan: H=80-100, Magenta: H=140-180
        hue = hsv[:, :, 0]
        value = hsv[:, :, 2]
        
        # Mask regions with notable saturation (indicates colored overlay)
        # Standard photo regions typically have low saturation
        sat_threshold = 50  # Tune this based on overlay intensity
        colored_mask = saturation > sat_threshold
        
        # Also check for specific overlay colors (green/cyan range is common)
        green_cyan_mask = ((hue >= 35) & (hue <= 100) & (saturation > 30))
        magenta_mask = ((hue >= 140) & (hue <= 180) & (saturation > 30))
        
        # Combine color-based detection
        color_mask = green_cyan_mask | magenta_mask | colored_mask
        
        # Method 2: If original image provided, do difference detection
        if original_image is not None and original_image.shape == mask_viz.shape:
            diff = cv2.absdiff(mask_viz, original_image)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, diff_mask = cv2.threshold(diff_gray, 20, 255, cv2.THRESH_BINARY)
            
            # Combine with color detection
            color_mask = color_mask | (diff_mask > 0)
        
        # Convert boolean to uint8
        binary = (color_mask.astype(np.uint8)) * 255
        
        # Clean up with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return binary
    
    def segment(
        self,
        image: np.ndarray,
        prompts: Optional[List[str]] = None,
        image_path: Optional[str] = None
    ) -> SAM3SegmentationResult:
        """
        Segment card from image using SAM3 workflow.
        
        Args:
            image: BGR input image
            prompts: Custom prompts (uses defaults if None)
            image_path: Optional path to save temp image
            
        Returns:
            SAM3SegmentationResult with mask
        """
        self._step = 0
        prompts = prompts or self.DEFAULT_PROMPTS
        
        # Save temp image for workflow
        if image_path is None:
            temp_path = "/tmp/sam3_input.jpg"
            cv2.imwrite(temp_path, image)
            image_path = temp_path
        
        # Image saved to temp path for workflow
        
        try:
            client = self._get_client()
            
            print(f"  [SAM3] Running workflow with {len(prompts)} prompts...")
            
            result = client.run_workflow(
                workspace_name=self.workspace_name,
                workflow_id=self.workflow_id,
                images={"image": image_path},
                parameters={"prompts": prompts},
                use_cache=True
            )
            
            if not result or len(result) == 0:
                return SAM3SegmentationResult(
                    success=False,
                    error_message="Empty result from SAM3 workflow"
                )
            
            # Get SAM predictions with RLE mask
            sam_data = result[0].get('sam')
            mask_viz_data = result[0].get('mask_visualization')
            
            # Decode visualization (store in result, don't save)
            mask_viz = None
            if mask_viz_data is not None:
                mask_viz = self._decode_mask_response(mask_viz_data)
            
            # Extract RLE mask from SAM predictions
            if sam_data is None or 'predictions' not in sam_data:
                return SAM3SegmentationResult(
                    success=False,
                    error_message="No SAM predictions in result"
                )
            
            predictions = sam_data['predictions']
            if len(predictions) == 0:
                return SAM3SegmentationResult(
                    success=False,
                    error_message="No card detected by SAM3"
                )
            
            # Get the first (best) prediction
            best_pred = predictions[0]
            print(f"  [SAM3] Detection: {best_pred.get('class')} @ {best_pred.get('confidence', 0):.1%} confidence")
            
            # Get RLE mask
            rle_mask = best_pred.get('rle_mask')
            if rle_mask is None:
                # Fallback: create mask from bounding box
                print("  [SAM3] No RLE mask, using bounding box")
                h, w = image.shape[:2]
                binary_mask = np.zeros((h, w), dtype=np.uint8)
                
                x = int(best_pred.get('x', 0))
                y = int(best_pred.get('y', 0))
                bw = int(best_pred.get('width', 0))
                bh = int(best_pred.get('height', 0))
                
                x1, y1 = x - bw // 2, y - bh // 2
                x2, y2 = x + bw // 2, y + bh // 2
                
                binary_mask[y1:y2, x1:x2] = 255
            else:
                # Decode RLE mask
                h, w = image.shape[:2]
                binary_mask = self._decode_rle_mask(rle_mask, (h, w))
            
            # Mask decoded from RLE format
            
            # Check mask is not empty
            mask_pixels = np.sum(binary_mask > 0)
            print(f"  [SAM3] Mask pixels: {mask_pixels}")
            
            if mask_pixels == 0:
                return SAM3SegmentationResult(
                    success=False,
                    error_message="Empty mask from SAM3"
                )
            
            return SAM3SegmentationResult(
                success=True,
                mask=binary_mask,
                mask_visualization=mask_viz
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return SAM3SegmentationResult(
                success=False,
                error_message=str(e)
            )
    
    def segment_roi(
        self,
        image: np.ndarray,
        roi_box: tuple[int, int, int, int],
        prompts: Optional[List[str]] = None
    ) -> SAM3SegmentationResult:
        """
        Segment card from ROI region only.
        
        Extracts ROI, runs segmentation, then maps mask back to full image space.
        
        Args:
            image: Full BGR input image
            roi_box: (x1, y1, x2, y2) bounding box
            prompts: Custom prompts
            
        Returns:
            SAM3SegmentationResult with mask in full image coordinates
        """
        x1, y1, x2, y2 = roi_box
        roi = image[y1:y2, x1:x2].copy()
        
        if self.debug_dir:
            self._save_debug("roi_input", roi)
        
        # Segment the ROI
        result = self.segment(roi, prompts=prompts)
        
        if not result.success:
            return result
        
        # Create full-size mask
        h, w = image.shape[:2]
        full_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Place ROI mask in full mask
        if result.mask is not None:
            full_mask[y1:y2, x1:x2] = result.mask
        
        return SAM3SegmentationResult(
            success=True,
            mask=full_mask,
            mask_visualization=result.mask_visualization
        )
