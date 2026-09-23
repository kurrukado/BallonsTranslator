import os
import sys
import time
import argparse
import re
from pathlib import Path
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

# Force UTF-8 stdout & stderr
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

from utils.gemini_proxy_launcher import ensure_gemini_proxy_running
ensure_gemini_proxy_running()

from utils.textblock import TextBlock
from utils.io_utils import imread, imwrite
from modules import init_module_registries, TEXTDETECTORS, OCR, INPAINTERS, TRANSLATORS
init_module_registries()


def wrap_text_to_lines(text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    """Wraps text into lines that fit within max_width."""
    words = text.split()
    if not words:
        return [text]
    lines = []
    current_line = []
    
    for word in words:
        test_line = " ".join(current_line + [word])
        # Use font.getbbox() or font.getlength()
        bbox = font.getbbox(test_line)
        w = bbox[2] - bbox[0]
        if w <= max_width or not current_line:
            current_line.append(word)
        else:
            lines.append(" ".join(current_line))
            current_line = [word]
            
    if current_line:
        lines.append(" ".join(current_line))
    return lines


def fit_text_in_box(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_path: str,
    box_w: int,
    box_h: int,
    min_size: int = 12,
    max_size: int = 40
) -> Tuple[ImageFont.ImageFont, List[str], int]:
    """Uses binary search to find the optimal font size that fits inside box_w x box_h."""
    best_font = ImageFont.truetype(font_path, size=min_size)
    best_lines = [text]
    best_size = min_size
    
    low = min_size
    high = max_size
    
    while low <= high:
        mid = (low + high) // 2
        try:
            test_font = ImageFont.truetype(font_path, size=mid)
        except Exception:
            break
            
        lines = wrap_text_to_lines(text, test_font, box_w)
        line_heights = []
        total_h = 0
        fits_width = True
        
        for line in lines:
            bbox = test_font.getbbox(line)
            lw = bbox[2] - bbox[0]
            lh = bbox[3] - bbox[1] + 4
            line_heights.append(lh)
            total_h += lh
            if lw > box_w:
                fits_width = False
                break
                
        if fits_width and total_h <= box_h:
            best_font = test_font
            best_lines = lines
            best_size = mid
            low = mid + 1  # Try larger size
        else:
            high = mid - 1 # Reduce size
            
    return best_font, best_lines, best_size


def run_manga_pipeline_on_image(
    img_path: Path,
    detector,
    ocr_engine,
    inpainter,
    translator,
    output_dir: Path
):
    print("\n" + "=" * 80)
    print(f"  PROCESSING REAL MANGA IMAGE: {img_path.name}")
    print("=" * 80)
    
    # 1. Load Image
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"[ERROR] Could not read image: {img_path}")
        return
    im_h, im_w = img_bgr.shape[:2]
    print(f"• Image Dimensions: {im_w} x {im_h} px | Channels: {img_bgr.shape[2]}")
    
    metrics = {}
    
    # 2. Step 1: Text Detection (CTD on CUDA)
    t0 = time.perf_counter()
    mask, blk_list = detector._detect(img_bgr, None)
    metrics["detect_ms"] = (time.perf_counter() - t0) * 1000
    print(f"• [Step 1] Detection (CTD CUDA): Found {len(blk_list)} text blocks in {metrics['detect_ms']:.1f} ms")
    
    if not blk_list:
        print("  [WARN] No text blocks detected in this image.")
        return

    # Refine is_balloon for each block
    for blk in blk_list:
        # Check aspect ratio & label
        bx1, by1, bx2, by2 = blk.xyxy
        bw, bh = bx2 - bx1, by2 - by1
        aspect = max(bw / max(bh, 1), bh / max(bw, 1))
        
        # In comictextdetector, single vertical tall lines without bubble often have label 'other' or high aspect
        if blk.label in ['other', 'sfx', 'narration']:
            blk.is_balloon = False
        elif aspect > 4.5 and len(blk.lines) <= 1:
            blk.is_balloon = False
        else:
            blk.is_balloon = True
            
    # 3. Step 2: Japanese OCR (MangaOCR on CUDA with fallback)
    t0 = time.perf_counter()
    ocr_engine._ocr_blk_list(img_bgr, blk_list)
    metrics["ocr_ms"] = (time.perf_counter() - t0) * 1000
    print(f"• [Step 2] OCR (MangaOCR CUDA): Recognized {len(blk_list)} blocks in {metrics['ocr_ms']:.1f} ms")
    
    # 4. Step 3: LaMa Inpainting (Remove Japanese text on CUDA)
    t0 = time.perf_counter()
    inpainted_bgr = inpainter.inpaint(img_bgr, mask)
    metrics["inpaint_ms"] = (time.perf_counter() - t0) * 1000
    print(f"• [Step 3] Inpainting (LaMa 512 CUDA): Inpainted mask in {metrics['inpaint_ms']:.1f} ms")
    
    # 5. Step 4: Context-Aware Translation (Gemini 3.7 Flash)
    t0 = time.perf_counter()
    translator.translate_textblk_lst(blk_list, page_context={"img_shape": (im_h, im_w)})
    metrics["trans_ms"] = (time.perf_counter() - t0) * 1000
    print(f"• [Step 4] Translation (Gemini 3.7 Flash): Translated {len(blk_list)} blocks in {metrics['trans_ms']:.1f} ms")
    
    # 6. Step 5: Typesetting with Auto 2-Font System
    t0 = time.perf_counter()
    font_baloo_path = str(PROJECT_ROOT / "fonts" / "Yuki-CCMarianChurchlandJournal.ttf") if (PROJECT_ROOT / "fonts" / "Yuki-CCMarianChurchlandJournal.ttf").exists() else "C:/Windows/Fonts/arial.ttf"
    font_sriracha_path = str(PROJECT_ROOT / "fonts" / "Yuki-Ripsnort BB.ttf") if (PROJECT_ROOT / "fonts" / "Yuki-Ripsnort BB.ttf").exists() else "C:/Windows/Fonts/arial.ttf"
    
    # Convert inpainted image to PIL (RGB)
    inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
    out_pil = Image.fromarray(inpainted_rgb)
    draw = ImageDraw.Draw(out_pil)
    
    cjk_pattern = re.compile(r'[\u3040-\u30ff\u4e00-\u9fff]')
    block_records = []
    
    print("\n" + "-" * 80)
    print(f"  BLOCK DETAILS & TYPESETTING LOG ({len(blk_list)} blocks)")
    print("-" * 80)
    
    for idx, blk in enumerate(blk_list):
        x1, y1, x2, y2 = blk.xyxy
        bw, bh = max(10, x2 - x1), max(10, y2 - y1)
        raw_jp = blk.get_text().strip()
        trans_vi = blk.translation.strip()
        is_dialogue = blk.is_in_balloon()
        
        # Check CJK residual
        has_cjk = bool(cjk_pattern.search(trans_vi))
        
        if is_dialogue:
            font_path = font_baloo_path
            font_name = "Baloo 2 Bold"
            style_name = "Dialogue"
            text_color = (0, 0, 0)
        else:
            font_path = font_sriracha_path
            font_name = "Sriracha Regular"
            style_name = "Narration/SFX"
            text_color = (180, 20, 20) if blk.label == 'sfx' else (30, 30, 130)
            
        # Fit text into box
        padding_x = int(bw * 0.08)
        padding_y = int(bh * 0.08)
        inner_w = max(10, bw - padding_x * 2)
        inner_h = max(10, bh - padding_y * 2)
        
        opt_font, lines, opt_size = fit_text_in_box(
            draw, trans_vi, font_path, inner_w, inner_h, min_size=11, max_size=36
        )
        
        # Draw lines centered in box
        line_heights = [opt_font.getbbox(l)[3] - opt_font.getbbox(l)[1] + 3 for l in lines]
        total_text_h = sum(line_heights)
        curr_y = y1 + padding_y + max(0, (inner_h - total_text_h) // 2)
        
        for line, lh in zip(lines, line_heights):
            bbox = opt_font.getbbox(line)
            lw = bbox[2] - bbox[0]
            curr_x = x1 + padding_x + max(0, (inner_w - lw) // 2)
            draw.text((curr_x, curr_y), line, fill=text_color, font=opt_font)
            curr_y += lh
            
        block_records.append({
            "id": idx + 1,
            "xyxy": [x1, y1, x2, y2],
            "is_balloon": is_dialogue,
            "style": style_name,
            "font": font_name,
            "font_size": opt_size,
            "raw_jp": raw_jp,
            "trans_vi": trans_vi,
            "has_cjk": has_cjk
        })
        
        tag = "BALLOON [Dialogue]" if is_dialogue else "OUTSIDE [Narration/SFX]"
        print(f"[{idx + 1:02d}] {tag:22s} | Font: {font_name} (sz={opt_size}) | Box: {x1},{y1} -> {x2},{y2}")
        print(f"     JP: \"{raw_jp}\"")
        print(f"     VI: \"{trans_vi}\"")
        if has_cjk:
            print(f"     [WARNING] Leftover CJK detected!")
            
    metrics["render_ms"] = (time.perf_counter() - t0) * 1000
    metrics["total_ms"] = sum(metrics.values())
    
    # 7. Create Side-by-Side Comparison Image
    orig_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_pil = Image.fromarray(orig_rgb)
    
    side_by_side = Image.new('RGB', (im_w * 2 + 30, im_h + 60), color=(240, 240, 240))
    s_draw = ImageDraw.Draw(side_by_side)
    
    # Header titles
    title_font = ImageFont.truetype(font_baloo_path, size=24)
    s_draw.text((40, 15), f"ORIGINAL SCAN: {img_path.name}", fill=(40, 40, 40), font=title_font)
    s_draw.text((im_w + 50, 15), f"TRANSLATED (MangaOCR + Gemini 3.7 + 2 Fonts)", fill=(180, 20, 20), font=title_font)
    
    side_by_side.paste(orig_pil, (20, 50))
    side_by_side.paste(out_pil, (im_w + 30, 50))
    
    # Save outputs
    out_img_path = output_dir / f"translated_{img_path.stem}.png"
    out_compare_path = output_dir / f"compare_{img_path.stem}.png"
    
    out_pil.save(str(out_img_path))
    side_by_side.save(str(out_compare_path))
    
    print("\n" + "-" * 80)
    print(f"  PERFORMANCE BENCHMARK FOR {img_path.name}")
    print("-" * 80)
    print(f"  • Detection (CTD)    : {metrics['detect_ms']:7.1f} ms")
    print(f"  • OCR (MangaOCR)     : {metrics['ocr_ms']:7.1f} ms")
    print(f"  • Inpainting (LaMa)  : {metrics['inpaint_ms']:7.1f} ms")
    print(f"  • Translation (LLM)  : {metrics['trans_ms']:7.1f} ms")
    print(f"  • Typesetting/Render : {metrics['render_ms']:7.1f} ms")
    print(f"  -------------------------------------------")
    print(f"  • TOTAL PIPELINE     : {metrics['total_ms']:7.1f} ms ({metrics['total_ms']/1000:.2f} s)")
    print(f"  ✓ Output Saved: {out_img_path}")
    print(f"  ✓ Comparison Saved: {out_compare_path}")
    
    return metrics, block_records


def main():
    parser = argparse.ArgumentParser(description="Test full translation pipeline on real manga scans.")
    parser.add_argument("--input_dir", type=str, default="data/real_manga_samples", help="Path to real manga images folder")
    parser.add_argument("--output_dir", type=str, default="tmp/live_output/real_test", help="Output directory")
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Gather image files
    exts = [".png", ".jpg", ".jpeg", ".webp"]
    if input_path.is_file():
        img_files = [input_path]
    elif input_path.is_dir():
        img_files = [p for p in input_path.iterdir() if p.suffix.lower() in exts]
    else:
        print(f"[ERROR] Input path does not exist: {input_path}")
        return
        
    print("=" * 80)
    print("  REAL MANGA TRANSLATION VALIDATION SUITE (JAPANESE -> VIETNAMESE)")
    print(f"  • Found {len(img_files)} real manga scan image(s) to process.")
    print("=" * 80)
    
    # Initialize Core Modules
    print("\n[Init] Initializing AI Models on GPU (CUDA)...")
    t0 = time.perf_counter()
    
    CTDClass = TEXTDETECTORS.module_dict["ctd"]
    detector = CTDClass(device='cuda', detect_size=1024)
    detector._load_model()
    
    OCRClass = OCR.module_dict.get("paddle_ocr") or OCR.module_dict.get("windows_ocr")
    ocr_engine = OCRClass()
    ocr_engine._load_model()
    
    LaMaClass = INPAINTERS.module_dict["lama_large_512px"]
    inpainter = LaMaClass(device='cuda', inpaint_size=1024)
    inpainter._load_model()
    
    from utils.gemini_proxy_launcher import get_saved_api_key
    saved_key = get_saved_api_key()
    TranslatorClass = TRANSLATORS.module_dict["LLM_API_Translator"]
    translator = TranslatorClass(
        lang_source='English',
        lang_target='Tiếng Việt',
        model='gemini-3.5-flash-lite',
        provider='Google',
        apikey=saved_key,
        raise_unsupported_lang=False
    )
    translator.story_genre = "Manga Drama / Mystery"
    
    init_time = (time.perf_counter() - t0) * 1000
    print(f"✓ All core models initialized in {init_time:.1f} ms.")
    
    all_results = []
    
    for img_file in img_files:
        try:
            res = run_manga_pipeline_on_image(
                img_file, detector, ocr_engine, inpainter, translator, output_path
            )
            if res:
                all_results.append((img_file.name, res[0], res[1]))
        except Exception as e:
            print(f"[ERROR] Processing failed on {img_file.name}: {e}")
            import traceback
            traceback.print_exc()
            
    print("\n" + "=" * 80)
    print(f"  REAL MANGA VALIDATION SUMMARY ({len(all_results)}/{len(img_files)} SUCCESSFUL)")
    print("=" * 80)
    for name, metrics, records in all_results:
        cjk_fails = sum(1 for r in records if r["has_cjk"])
        print(f"  • {name:30s} | Blocks: {len(records):2d} | CJK Residuals: {cjk_fails} | Total Time: {metrics['total_ms']:.0f} ms")
    print("=" * 80)


if __name__ == '__main__':
    main()
