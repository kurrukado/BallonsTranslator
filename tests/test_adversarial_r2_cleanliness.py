"""
Adversarial Verification Suite for Requirement R2: Visual Cleanliness & Zero Seams
Empirical Challenger: challenger_2
Date: 2026-09-04

Scope:
1. Mathematical verification of Unmasked MAE == 0.000000 across all 5 real manga samples.
   - Pixel-by-pixel exact identity verification (uint8 array_equal)
   - Analysis of mask mutations during inpainting (before vs after inpaint)
   - RGBA channel integrity & alpha preservation
2. Stress-testing gradient background detection (_is_gradient_bg):
   - Vertical gradient slope response vs threshold 8.0
   - Flat background with text strokes (false-positive gradient detection analysis)
   - Horizontal & diagonal gradients (directional blindness analysis)
   - Screentone & halftone dot patterns with varying noise
3. Stress-testing Laplacian pyramid multi-band blending:
   - Seam continuity and boundary step discontinuity analysis
   - Interaction between Laplacian pyramid blending and hard binary mask clipping (np.where)
   - Edge cases: non-power-of-2 dimensions, small crops (<16px)
"""

import os
import sys
import time
import math
import numpy as np
import cv2
import torch
from typing import Dict, List, Tuple

# Set UTF-8 encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.textblock_mask import _is_gradient_bg
from modules.inpaint.base import _laplacian_pyramid_blend, LamaLarge
from modules.textdetector.detector_ctd import ComicTextDetector


def test_section(name: str):
    print("\n" + "=" * 80)
    print(f"🔬 RUNNING TEST SUITE: {name}")
    print("=" * 80)


# =========================================================================
# SUITE 1: STRESS-TESTING GRADIENT BACKGROUND DETECTION (_is_gradient_bg)
# =========================================================================
def run_gradient_detection_stress_tests() -> Dict:
    test_section("SUITE 1: Gradient Detection (_is_gradient_bg) Stress-Testing")
    results = {}

    # Test 1.1: Linear Vertical Gradients (Varying slope S from 0.1 to 3.0 gray/px)
    print("\n--- Test 1.1: Vertical Linear Gradients (Slope Response) ---")
    h, w = 100, 100
    y_coords = np.arange(h, dtype=np.float32)[:, None]
    slopes = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.5, 2.0, 2.5]
    slope_results = []
    for s in slopes:
        # Create gradient starting at 20 + s * y
        grad = np.clip(20.0 + s * y_coords, 0, 255).astype(np.uint8)
        grad_img = np.repeat(grad, w, axis=1)
        is_grad = _is_gradient_bg(grad_img, threshold=8.0)
        gy = cv2.Sobel(grad_img, cv2.CV_32F, 0, 1, ksize=3)
        mean_gy = float(np.mean(np.abs(gy)))
        slope_results.append({
            "slope": s,
            "mean_sobel_y": round(mean_gy, 4),
            "is_gradient_detected": is_grad,
            "expected_detected": (s * 8.0 > 8.0)
        })
        print(f"  Slope {s:4.1f} gray/px -> Mean |Sobel Y|: {mean_gy:6.2f} | Detected: {is_grad}")
    results["vertical_slopes"] = slope_results

    # Test 1.2: Flat Background with Text (The Text Edge Gradient Paradox)
    print("\n--- Test 1.2: Flat Background with Text (Speech Balloon Text Edge Paradox) ---")
    # Synthetic speech balloon: flat white (255) with black text characters (0)
    flat_balloon = np.full((120, 150), 255, dtype=np.uint8)
    cv2.putText(flat_balloon, "HELLO", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2, cv2.LINE_AA)
    cv2.putText(flat_balloon, "WORLD", (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2, cv2.LINE_AA)
    
    gy_balloon = cv2.Sobel(flat_balloon, cv2.CV_32F, 0, 1, ksize=3)
    mean_gy_balloon = float(np.mean(np.abs(gy_balloon)))
    is_grad_balloon = _is_gradient_bg(flat_balloon, threshold=8.0)
    print(f"  Pure Flat Background with Text -> Mean |Sobel Y|: {mean_gy_balloon:.2f} | Detected as Gradient: {is_grad_balloon}")
    results["flat_balloon_with_text"] = {
        "mean_sobel_y": round(mean_gy_balloon, 4),
        "falsely_detected_as_gradient": is_grad_balloon
    }

    # Test 1.3: Directional Gradients (Horizontal, Diagonal, Radial)
    print("\n--- Test 1.3: Directional Gradient Sensitivity (Directional Blindness) ---")
    x_coords = np.arange(w, dtype=np.float32)[None, :]
    # Pure horizontal gradient (slope = 1.5 gray/px)
    horiz_grad = np.clip(20.0 + 1.5 * x_coords, 0, 255).astype(np.uint8)
    horiz_img = np.repeat(horiz_grad, h, axis=0)
    is_grad_horiz = _is_gradient_bg(horiz_img, threshold=8.0)
    gy_horiz = float(np.mean(np.abs(cv2.Sobel(horiz_img, cv2.CV_32F, 0, 1, ksize=3))))
    print(f"  Horizontal Gradient (Slope 1.5) -> Mean |Sobel Y|: {gy_horiz:.2f} | Detected: {is_grad_horiz}")

    # Diagonal 45 deg gradient (slope = 1.5 along diagonal)
    diag_grid = (x_coords + y_coords) * 0.75
    diag_img = np.clip(diag_grid, 0, 255).astype(np.uint8)
    is_grad_diag = _is_gradient_bg(diag_img, threshold=8.0)
    gy_diag = float(np.mean(np.abs(cv2.Sobel(diag_img, cv2.CV_32F, 0, 1, ksize=3))))
    print(f"  Diagonal Gradient 45° (Slope 1.5) -> Mean |Sobel Y|: {gy_diag:.2f} | Detected: {is_grad_diag}")

    # Radial gradient from center
    xx, yy = np.meshgrid(np.arange(w) - w/2, np.arange(h) - h/2)
    radius = np.sqrt(xx**2 + yy**2)
    radial_img = np.clip(radius * 2.5, 0, 255).astype(np.uint8)
    is_grad_radial = _is_gradient_bg(radial_img, threshold=8.0)
    gy_radial = float(np.mean(np.abs(cv2.Sobel(radial_img, cv2.CV_32F, 0, 1, ksize=3))))
    print(f"  Radial Gradient -> Mean |Sobel Y|: {gy_radial:.2f} | Detected: {is_grad_radial}")

    results["directional_gradients"] = {
        "horizontal": {"mean_sobel_y": round(gy_horiz, 4), "detected": is_grad_horiz},
        "diagonal": {"mean_sobel_y": round(gy_diag, 4), "detected": is_grad_diag},
        "radial": {"mean_sobel_y": round(gy_radial, 4), "detected": is_grad_radial}
    }

    # Test 1.4: Manga Screentone Patterns (Halftone Dot Grids)
    print("\n--- Test 1.4: Halftone Screentone Patterns (Dot Frequency Response) ---")
    halftone_results = []
    for dot_spacing in [4, 6, 8, 12]:
        # Flat screentone (uniform dot grid on gray background)
        tone_img = np.full((120, 120), 200, dtype=np.uint8)
        tone_img[::dot_spacing, ::dot_spacing] = 50
        gy_tone = float(np.mean(np.abs(cv2.Sobel(tone_img, cv2.CV_32F, 0, 1, ksize=3))))
        is_grad_tone = _is_gradient_bg(tone_img, threshold=8.0)
        halftone_results.append({
            "spacing_px": dot_spacing,
            "mean_sobel_y": round(gy_tone, 4),
            "detected_as_gradient": is_grad_tone
        })
        print(f"  Uniform Halftone (Grid {dot_spacing}px) -> Mean |Sobel Y|: {gy_tone:6.2f} | Flagged as Gradient: {is_grad_tone}")
    results["halftone_screentones"] = halftone_results

    # Test 1.5: Noise Immunity Test
    print("\n--- Test 1.5: Noise Immunity Test (Gaussian Noise on Flat Background) ---")
    noise_results = []
    for sigma in [1.0, 2.0, 3.0, 5.0, 8.0, 10.0]:
        noise = np.random.normal(0, sigma, (100, 100))
        noisy_flat = np.clip(128.0 + noise, 0, 255).astype(np.uint8)
        gy_noise = float(np.mean(np.abs(cv2.Sobel(noisy_flat, cv2.CV_32F, 0, 1, ksize=3))))
        is_grad_noise = _is_gradient_bg(noisy_flat, threshold=8.0)
        noise_results.append({
            "sigma": sigma,
            "mean_sobel_y": round(gy_noise, 4),
            "flagged_as_gradient": is_grad_noise
        })
        print(f"  Flat Background + Gaussian Noise (σ={sigma:4.1f}) -> Mean |Sobel Y|: {gy_noise:6.2f} | Flagged: {is_grad_noise}")
    results["noise_immunity"] = noise_results

    return results


# =========================================================================
# SUITE 2: LAPLACIAN PYRAMID BLENDING & BOUNDARY SEAM ANALYSIS
# =========================================================================
def run_laplacian_blending_stress_tests() -> Dict:
    test_section("SUITE 2: Laplacian Pyramid Blending Boundary Transition Analysis")
    results = {}

    # Test 2.1: Boundary Seam Evaluation on Sloped Gradient
    print("\n--- Test 2.1: Sloped Gradient Inpaint Seam Test ---")
    h, w = 128, 128
    # Background: Linear vertical gradient 50 -> 200
    y = np.linspace(50, 200, h, dtype=np.float32)[:, None]
    orig_img = np.repeat(y, w, axis=1).astype(np.uint8)

    # Inpainted synthetic reconstruction: slightly shifted gradient (simulating AI inpaint tone delta)
    inpaint_img = np.repeat(y + 15.0, w, axis=1).astype(np.uint8)

    # Circular mask in center
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (64, 64), 30, 255, -1)

    # Apply _laplacian_pyramid_blend
    blended = _laplacian_pyramid_blend(inpaint_img, orig_img, mask, levels=4)

    # Evaluate boundary gradient: difference across mask perimeter
    elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    boundary = cv2.dilate(mask, elem) - cv2.erode(mask, elem)
    
    # Check max gradient magnitude along boundary
    grad_x = cv2.Sobel(blended.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blended.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(grad_x**2 + grad_y**2)
    boundary_mag_mean = float(np.mean(mag[boundary > 0]))
    boundary_mag_max = float(np.max(mag[boundary > 0]))
    print(f"  Laplacian Blended Boundary Gradient -> Mean: {boundary_mag_mean:.2f}, Max: {boundary_mag_max:.2f}")

    # Compare against naive cut-and-paste
    naive = np.where(mask > 0, inpaint_img, orig_img)
    grad_x_naive = cv2.Sobel(naive.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    grad_y_naive = cv2.Sobel(naive.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    mag_naive = np.sqrt(grad_x_naive**2 + grad_y_naive**2)
    boundary_naive_mean = float(np.mean(mag_naive[boundary > 0]))
    boundary_naive_max = float(np.max(mag_naive[boundary > 0]))
    print(f"  Naive Cut-and-Paste Boundary Gradient -> Mean: {boundary_naive_mean:.2f}, Max: {boundary_naive_max:.2f}")
    
    seam_reduction = (1.0 - (boundary_mag_max / max(1.0, boundary_naive_max))) * 100.0
    print(f"  Seam Peak Gradient Reduction: {seam_reduction:.1f}%")

    results["gradient_seam"] = {
        "laplacian_mean_grad": round(boundary_mag_mean, 2),
        "laplacian_max_grad": round(boundary_mag_max, 2),
        "naive_mean_grad": round(boundary_naive_mean, 2),
        "naive_max_grad": round(boundary_naive_max, 2),
        "seam_reduction_pct": round(seam_reduction, 1)
    }

    # Test 2.2: Hard Binary Clipping vs Soft Laplacian Blend Interaction
    print("\n--- Test 2.2: Hard Binary Mask np.where Interaction ---")
    # In base.py:
    # line 258: blended = _laplacian_pyramid_blend(roi_inpainted, roi_orig, roi_mask, levels=4)
    # line 270: result_rgb = np.where(mask_bin, result_rgb, img_rgb)
    # If mask_bin is binary (roi_mask > 0), what happens to blended pixels outside roi_mask?
    clipped_result = np.where(mask > 0, blended, orig_img)
    mag_clipped = np.sqrt(cv2.Sobel(clipped_result.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)**2 +
                          cv2.Sobel(clipped_result.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)**2)
    boundary_clipped_max = float(np.max(mag_clipped[boundary > 0]))
    boundary_clipped_mean = float(np.mean(mag_clipped[boundary > 0]))
    print(f"  Boundary Gradient when clipped to (mask > 0): Mean: {boundary_clipped_mean:.2f}, Max: {boundary_clipped_max:.2f}")
    results["clipped_blend_interaction"] = {
        "boundary_clipped_mean": round(boundary_clipped_mean, 2),
        "boundary_clipped_max": round(boundary_clipped_max, 2)
    }

    # Test 2.3: Dimension Edge Cases for _laplacian_pyramid_blend
    print("\n--- Test 2.3: Dimension Edge Cases (Odd sizes, Small Crops) ---")
    odd_sizes = [(63, 63), (65, 47), (31, 33), (17, 19), (15, 15), (8, 8)]
    dimension_results = []
    for sh, sw in odd_sizes:
        try:
            a = np.full((sh, sw), 100, dtype=np.uint8)
            b = np.full((sh, sw), 200, dtype=np.uint8)
            m = np.zeros((sh, sw), dtype=np.uint8)
            m[sh//4:3*sh//4, sw//4:3*sw//4] = 255
            res = _laplacian_pyramid_blend(a, b, m, levels=4)
            shape_match = (res.shape == (sh, sw))
            dimension_results.append({"size": (sh, sw), "success": True, "shape_match": shape_match})
            print(f"  Crop Size ({sh:2d}, {sw:2d}) with 4 levels -> Success: True | Shape match: {shape_match}")
        except Exception as e:
            dimension_results.append({"size": (sh, sw), "success": False, "error": str(e)})
            print(f"  Crop Size ({sh:2d}, {sw:2d}) with 4 levels -> FAILED: {e}")
    results["dimension_edge_cases"] = dimension_results

    return results


# =========================================================================
# SUITE 3: MATHEMATICAL INVARIANT EMPIRICAL VERIFICATION (5 REAL MANGA SAMPLES)
# =========================================================================
def run_empirical_real_manga_invariant_tests() -> Dict:
    test_section("SUITE 3: Mathematical Invariant Unmasked MAE == 0.000000 on 5 Real Manga Samples")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Initializing Detector & Inpainter on {device.upper()}...")
    detector = ComicTextDetector(device=device, detect_size=1536)
    detector._load_model()
    inpainter = LamaLarge(device=device, inpaint_size=2048, inpaint_passes=2)
    inpainter._load_model()

    samples = [
        ("Page_1_Action", os.path.join(ROOT_DIR, "data", "real_manga_samples", "manga_page1_action.png")),
        ("Page_2_SliceOfLife", os.path.join(ROOT_DIR, "data", "real_manga_samples", "manga_page2_sliceoflife.png")),
        ("Page_3_Mystery_Dark", os.path.join(ROOT_DIR, "data", "real_manga_samples", "manga_page3_mystery.png")),
        ("Page_4_User_Mystery", os.path.join(ROOT_DIR, "data", "real_manga_samples", "user_sample_mystery.png")),
        ("Page_5_Scanlation_Tone", os.path.join(ROOT_DIR, "data", "real_manga_samples", "manga_page5_scanlation.png"))
    ]

    empirical_pages = []

    for tag, path in samples:
        print(f"\n--- Testing Invariant on {tag} ---")
        img_bgr = cv2.imread(path)
        assert img_bgr is not None, f"Image not found: {path}"
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        total_pixels = h * w

        # 1. Detect text and generate initial mask
        mask, blk_list = detector._detect(img_rgb, None)
        initial_mask_copy = mask.copy()
        initial_masked_px = int(np.count_nonzero(initial_mask_copy > 0))

        # 2. Run Inpainting
        t0 = time.time()
        inpainted_rgb = inpainter.inpaint(img_rgb, mask, blk_list)
        t_inpaint = time.time() - t0

        final_mask_copy = mask.copy()
        final_masked_px = int(np.count_nonzero(final_mask_copy > 0))

        # 3. Check mask mutation (did secondary pass or inpainting expand the mask?)
        mask_diff = np.count_nonzero(final_mask_copy != initial_mask_copy)
        mask_expanded_px = final_masked_px - initial_masked_px

        # 4. Strict Pixel-by-Pixel Verification outside final_mask
        unmasked_final = (final_mask_copy == 0)
        unmasked_final_px = int(np.count_nonzero(unmasked_final))

        diff_final = np.abs(inpainted_rgb[unmasked_final].astype(np.int32) - img_rgb[unmasked_final].astype(np.int32))
        max_diff_final = int(np.max(diff_final)) if diff_final.size > 0 else 0
        mae_final = float(np.mean(diff_final)) if diff_final.size > 0 else 0.0
        altered_final_px = int(np.count_nonzero(diff_final > 0))

        # 5. Check Pixel Identity outside INITIAL mask (prior to any inpainting mutation)
        unmasked_initial = (initial_mask_copy == 0)
        diff_initial = np.abs(inpainted_rgb[unmasked_initial].astype(np.int32) - img_rgb[unmasked_initial].astype(np.int32))
        max_diff_initial = int(np.max(diff_initial)) if diff_initial.size > 0 else 0
        mae_initial = float(np.mean(diff_initial)) if diff_initial.size > 0 else 0.0
        altered_initial_px = int(np.count_nonzero(diff_initial > 0))

        # 6. Check exact uint8 array identity
        exact_equal_final = np.array_equal(inpainted_rgb[unmasked_final], img_rgb[unmasked_final])

        # 7. Check Inpainted Region Texture
        masked_region = (final_mask_copy > 0)
        inpainted_std = float(np.std(inpainted_rgb[masked_region])) if np.any(masked_region) else 0.0
        
        # 8. Check Mean Box Laplacian Variance
        box_vars = []
        for blk in blk_list:
            bx1, by1, bx2, by2 = [int(v) for v in blk.xyxy]
            bx1, by1 = max(0, bx1), max(0, by1)
            bx2, by2 = min(w, bx2), min(h, by2)
            if bx2 > bx1 and by2 > by1:
                crop = inpainted_rgb[by1:by2, bx1:bx2]
                g_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                b_var = float(cv2.Laplacian(g_crop, cv2.CV_64F).var())
                box_vars.append(b_var)
        mean_lap = float(np.mean(box_vars)) if box_vars else 0.0

        page_record = {
            "page": tag,
            "dimensions": f"{w}x{h}",
            "total_pixels": total_pixels,
            "initial_mask_px": initial_masked_px,
            "final_mask_px": final_masked_px,
            "mask_expanded_px": mask_expanded_px,
            "unmasked_final_px": unmasked_final_px,
            "unmasked_mae_final": mae_final,
            "max_abs_diff_final": max_diff_final,
            "altered_unmasked_px_final": altered_final_px,
            "exact_uint8_equal_final": exact_equal_final,
            "unmasked_mae_initial": mae_initial,
            "altered_unmasked_px_initial": altered_initial_px,
            "inpainted_std": round(inpainted_std, 2),
            "mean_box_laplacian": round(mean_lap, 1),
            "inpaint_time_s": round(t_inpaint, 2)
        }
        empirical_pages.append(page_record)

        print(f"  Total Pixels: {total_pixels:,} | Mask Pixels: {final_masked_px:,} ({final_masked_px/total_pixels*100:.2f}%)")
        print(f"  Mask Mutation: initial={initial_masked_px:,} -> final={final_masked_px:,} (Δ = +{mask_expanded_px:,} px)")
        print(f"  Unmasked MAE (vs Final Mask): {mae_final:.6f} | Max AE: {max_diff_final} | Altered Px: {altered_final_px}")
        print(f"  Exact Bit-Level Uint8 Preservation: {'✓ PERFECT' if exact_equal_final else '✗ FAILED'}")
        if mask_expanded_px > 0:
            print(f"  ⚠️ Note: Mask was expanded by 2nd-pass by {mask_expanded_px} px. Unmasked MAE vs Initial Mask: {mae_initial:.6f} (Altered px: {altered_initial_px})")

    return {"pages": empirical_pages}


# =========================================================================
# SUITE 4: RGBA ALPHA CHANNEL PRESERVATION STRESS-TEST
# =========================================================================
def run_rgba_alpha_channel_stress_tests() -> Dict:
    test_section("SUITE 4: RGBA Alpha Channel Preservation & Leakage Stress-Test")
    # InpainterBase.inpaint:
    # Test handling of 4-channel RGBA images with transparency
    h, w = 120, 120
    img_rgba = np.full((h, w, 4), 200, dtype=np.uint8)
    # Transparent border
    img_rgba[:20, :, 3] = 0
    img_rgba[-20:, :, 3] = 0
    img_rgba[:, :20, 3] = 0
    img_rgba[:, -20:, 3] = 0
    # Semi-transparent center
    img_rgba[20:-20, 20:-20, 3] = 255

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[40:80, 40:80] = 255

    # Run OpenCV tela inpainter as dummy to check pipeline channel flow
    from modules.inpaint.base import OpenCVInpainter
    tela = OpenCVInpainter()
    out_rgba = tela.inpaint(img_rgba, mask.copy(), None)

    # Check channels
    is_rgba = (out_rgba.ndim == 3 and out_rgba.shape[2] == 4)
    alpha_diff = np.abs(out_rgba[:, :, 3].astype(np.int32) - img_rgba[:, :, 3].astype(np.int32))
    unmasked = (mask == 0)
    unmasked_alpha_diff = int(np.max(alpha_diff[unmasked]))
    unmasked_rgb_diff = int(np.max(np.abs(out_rgba[unmasked, :3].astype(np.int32) - img_rgba[unmasked, :3].astype(np.int32))))

    print(f"  Output is 4-channel RGBA: {is_rgba}")
    print(f"  Max Alpha Diff on Unmasked Pixels: {unmasked_alpha_diff}")
    print(f"  Max RGB Diff on Unmasked Pixels: {unmasked_rgb_diff}")

    return {
        "is_rgba": is_rgba,
        "unmasked_alpha_diff": unmasked_alpha_diff,
        "unmasked_rgb_diff": unmasked_rgb_diff
    }


if __name__ == "__main__":
    print("🚀 STARTING ADVERSARIAL CHALLENGER VERIFICATION SUITE")
    t_start = time.time()

    suite1 = run_gradient_detection_stress_tests()
    suite2 = run_laplacian_blending_stress_tests()
    suite4 = run_rgba_alpha_channel_stress_tests()
    suite3 = run_empirical_real_manga_invariant_tests()

    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"🏁 ALL ADVERSARIAL TESTS COMPLETED IN {total_time:.2f}s")
    print("=" * 80)
