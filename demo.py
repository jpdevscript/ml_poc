#!/usr/bin/env python3
"""
PD Measurement Demo Script

Test the PD measurement pipeline on custom images with full debug output.
Uses SAM3 via Roboflow workflow for card segmentation.

Usage:
    python demo.py <image_path> [options]
    
Examples:
    python demo.py photo.jpg
    python demo.py photo.jpg --output-dir my_debug
    python demo.py photo.jpg --card-only
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pd_engine.core import PDMeasurement, PDResult
from pd_engine.calibration import CardCalibration, CardDetectionResult


def create_debug_dir(base_dir: str = "debug") -> str:
    """Create timestamped debug directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_dir = os.path.join(base_dir, timestamp)
    os.makedirs(debug_dir, exist_ok=True)
    return debug_dir


def print_header(title: str) -> None:
    """Print formatted header."""
    line = "=" * 70
    print(f"\n{line}")
    print(f"  {title}")
    print(line)


def print_section(title: str) -> None:
    """Print formatted section header."""
    print(f"\n{'-' * 40}")
    print(f"  {title}")
    print(f"{'-' * 40}")


def test_card_detection(
    image_path: str, 
    debug_dir: str, 
    use_forehead: bool = True
) -> Optional[CardDetectionResult]:
    """
    Test card detection pipeline only.
    
    Args:
        image_path: Path to input image
        debug_dir: Directory to save debug images
        use_forehead: Whether to use forehead ROI detection
        
    Returns:
        CardDetectionResult or None if failed
    """
    print_header("CARD DETECTION TEST")
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image: {image_path}")
        return None
    
    h, w = image.shape[:2]
    print(f"Image: {image_path}")
    print(f"Size: {w}x{h}")
    print(f"Debug: {debug_dir}")
    print(f"Forehead ROI: {'enabled' if use_forehead else 'disabled'}")
    
    # Save input
    cv2.imwrite(os.path.join(debug_dir, "00_input.jpg"), image)
    
    # Run card detection
    calibration = CardCalibration(debug_dir=debug_dir)
    result = calibration.detect_card(
        image, 
        debug=True, 
        use_forehead_roi=use_forehead
    )
    
    print_section("RESULT")
    
    if result.detected:
        print("  ✓ Card detected!")
        print(f"    Width: {result.card_width_px:.1f} px")
        print(f"    Height: {result.card_height_px:.1f} px")
        print(f"    Scale: {result.scale_factor:.4f} mm/px")
        print(f"    Confidence: {result.confidence:.1%}")
        
        if result.corners is not None:
            print("    Corners:")
            for lbl, corner in zip(['TL', 'TR', 'BR', 'BL'], result.corners):
                print(f"      {lbl}: ({corner[0]:.1f}, {corner[1]:.1f})")
    else:
        print("  ✗ Card NOT detected")
        print(f"    Error: {result.error_message}")
    
    return result


def test_full_pd(
    image_path: str, 
    debug_dir: str, 
    use_forehead: bool = True
) -> Optional[PDResult]:
    """
    Test full PD measurement pipeline with quality gates.
    
    Args:
        image_path: Path to input image
        debug_dir: Directory to save debug images
        use_forehead: Whether to use forehead ROI detection
        
    Returns:
        PDResult or None if failed
    """
    from pd_engine.corrections import PDCorrector
    from pd_engine.measurement import IrisMeasurer
    
    print_header("FULL PD MEASUREMENT TEST")
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image: {image_path}")
        return None
    
    h, w = image.shape[:2]
    print(f"Image: {image_path}")
    print(f"Size: {w}x{h}")
    print(f"Debug: {debug_dir}")
    
    # Save input
    cv2.imwrite(os.path.join(debug_dir, "input.jpg"), image)
    
    # === QUALITY GATE 1: Blur Detection ===
    print_section("QUALITY GATES")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_sharp = blur_score > 30.0
    print(f"  Blur Score: {blur_score:.1f} (threshold=30)")
    print(f"  Sharpness: {'✓ PASS' if is_sharp else '✗ FAIL - too blurry'}")
    
    # === QUALITY GATE 2: Iris Detection for camera distance ===
    iris_measurer = IrisMeasurer()
    iris_result = iris_measurer.measure(image)
    iris_measurer.close()
    
    if iris_result.detected:
        print(f"  Face Detection: ✓ PASS")
        if iris_result.head_pose:
            yaw = iris_result.head_pose.yaw
            pitch = iris_result.head_pose.pitch
            print(f"  Head Pose: yaw={yaw:.1f}°, pitch={pitch:.1f}°")
            if abs(yaw) > 15 or abs(pitch) > 15:
                print(f"    ⚠️ Pose outside recommended range (±15°)")
            else:
                print(f"    ✓ Pose OK")
        
        # Iris-based camera distance estimation
        if iris_result.iris_diameter_px:
            focal_length = PDCorrector.estimate_focal_length_px(w)
            camera_dist = PDCorrector.estimate_camera_distance_from_iris(
                iris_result.iris_diameter_px,
                focal_length
            )
            print(f"  Iris Diameter: {iris_result.iris_diameter_px:.1f} px")
            print(f"  Est. Camera Distance (from iris): {camera_dist:.0f} mm")
    else:
        print(f"  Face Detection: ✗ FAIL - {iris_result.error_message}")
    
    # Create PD engine
    print_section("PD MEASUREMENT")
    engine = PDMeasurement()
    
    try:
        result = engine.process_frame(image, debug_dir=debug_dir)
        
        if result.is_valid:
            print(f"  ✓ PD: {result.pd_final_mm:.2f} mm")
            print(f"    Confidence: {result.confidence:.1%}")
            print(f"    Method: {result.calibration_method}")
            print(f"    Medical Grade: {'Yes' if result.is_medical_grade else 'No'}")
            
            if result.raw_pd_px:
                print(f"    Raw PD: {result.raw_pd_px:.1f} px")
            if result.scale_factor_mm_per_px:
                print(f"    Scale: {result.scale_factor_mm_per_px:.4f} mm/px")
            if result.camera_distance_mm:
                print(f"    Camera Distance: {result.camera_distance_mm:.0f} mm")
            
            if result.head_pose:
                print(f"    Head Pose: yaw={result.head_pose.yaw:.1f}°, "
                      f"pitch={result.head_pose.pitch:.1f}°, "
                      f"roll={result.head_pose.roll:.1f}°")
            
            if result.warnings:
                print("    Warnings:")
                for w in result.warnings:
                    print(f"      ⚠️ {w}")
            
            # Save visualization
            viz = engine.visualize(image, result)
            viz_path = os.path.join(debug_dir, "result_visualization.jpg")
            cv2.imwrite(viz_path, viz)
            print(f"\n    → Visualization: {viz_path}")
            
        else:
            print("  ✗ PD Measurement FAILED")
            if result.errors:
                for e in result.errors:
                    print(f"    Error: {e}")
            if result.warnings:
                for w in result.warnings:
                    print(f"    Warning: {w}")
        
        return result
        
    finally:
        engine.close()


def print_summary(debug_dir: str) -> None:
    """Print summary of generated files."""
    print_header("SUMMARY")
    print(f"  Debug output: {debug_dir}")
    
    files = sorted(os.listdir(debug_dir))
    print(f"  Files ({len(files)}):")
    for f in files:
        size = os.path.getsize(os.path.join(debug_dir, f)) // 1024
        print(f"    - {f} ({size}KB)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test PD measurement on images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("-o", "--output-dir", default="debug",
                        help="Base directory for debug output")
    parser.add_argument("--card-only", action="store_true",
                        help="Only test card detection")
    parser.add_argument("--no-forehead", action="store_true",
                        help="Disable forehead ROI detection")
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found: {args.image_path}")
        sys.exit(1)
    
    # Create debug directory
    debug_dir = create_debug_dir(args.output_dir)
    
    print_header("PD MEASUREMENT DEMO")
    print(f"  Input: {args.image_path}")
    print(f"  Output: {debug_dir}")
    print(f"  Mode: {'Card Detection Only' if args.card_only else 'Full PD Measurement'}")
    
    use_forehead = not args.no_forehead
    
    if args.card_only:
        test_card_detection(args.image_path, debug_dir, use_forehead)
    else:
        test_full_pd(args.image_path, debug_dir, use_forehead)
    
    print_summary(debug_dir)
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
