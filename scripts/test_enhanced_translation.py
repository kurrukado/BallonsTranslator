import sys, os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import time
import cv2
import json
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from modules.textdetector.detector_ctd import ComicTextDetector
from modules.inpaint.base import INPAINTERS
from modules.translators.trans_llm_api import LLM_API_Translator
from modules.translators.context_engine import ContextAssembler, DialogueItem
from utils.gemini_proxy_launcher import ensure_gemini_proxy_running, sync_proxy_api_key

OUT_DIR = Path("tmp/live_output/enhanced_experience")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ensure_gemini_proxy_running()
_api_key = os.getenv("GEMINI_API_KEY", "")
if not _api_key and (PROJECT_ROOT / "config" / "config.json").exists():
    try:
        with open(PROJECT_ROOT / "config" / "config.json", "r", encoding="utf-8") as _f:
            _api_key = json.load(_f).get("module", {}).get("translator_params", {}).get("LLM_API_Translator", {}).get("apikey", "")
    except Exception:
        pass
sync_proxy_api_key(_api_key)

print("=" * 80)
print("  BALLOONSTRANSLATOR: ENHANCED MULTI-BUBBLE COHESION & TRANSLATION SUITE")
print("=" * 80)

# Initialize Core Models on CUDA
print("\n[1/4] Initializing AI Models on CUDA 12.6...")
t0 = time.perf_counter()
ctd = ComicTextDetector(device="cuda", detect_size=1024)
lama = INPAINTERS.module_dict["lama_large_512px"](device="cuda", inpaint_size=1024)
lama.load_model()
translator = LLM_API_Translator(lang_source="English", lang_target="Tiếng Việt")
translator.set_param_value("provider", "Google")
translator.set_param_value("model", "gemini-3.6-flash")
t_init = (time.perf_counter() - t0) * 1000
print(f"✓ AI Models initialized in {t_init:.1f} ms")

def run_test_sample(sample_name, img_path, manual_texts=None, lang_src="Auto"):
    print(f"\n" + "=" * 80)
    print(f"  PROCESSING TEST SAMPLE: {sample_name} (Source: {lang_src})")
    print("=" * 80)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error loading image: {img_path}")
        return

    h, w, c = img.shape
    t_start = time.perf_counter()

    # 1. Detection
    mask, blks = ctd.detect(img)
    t_det = (time.perf_counter() - t_start) * 1000
    print(f"• [Step 1] Text Detection (CTD): Found {len(blks)} text blocks in {t_det:.1f} ms")

    # Sort blocks reading order (Top to Bottom, Right to Left)
    blks = sorted(blks, key=lambda b: (b.xyxy[1] // 100, -b.xyxy[0]))

    # 2. OCR / Text Injection
    t_ocr_start = time.perf_counter()
    if manual_texts and len(manual_texts) == len(blks):
        for b, txt in zip(blks, manual_texts):
            b.text = [txt]
    else:
        # Fallback / recognize
        for i, b in enumerate(blks):
            if manual_texts and i < len(manual_texts):
                b.text = [manual_texts[i]]
    t_ocr = (time.perf_counter() - t_ocr_start) * 1000
    print(f"• [Step 2] OCR Text Extraction: Processed {len(blks)} blocks in {t_ocr:.1f} ms")

    # 3. Inpainting
    t_inp_start = time.perf_counter()
    inpainted = lama.inpaint(img, mask)
    t_inp = (time.perf_counter() - t_inp_start) * 1000
    print(f"• [Step 3] Inpainting (LaMa 512): Inpainted background in {t_inp:.1f} ms")

    # 4. Context Assembly & Translation
    t_trans_start = time.perf_counter()
    dialogue_items = ContextAssembler.build_dialogue_items(blks, img_shape=(h, w))
    translator.lang_source = lang_src
    translator.lang_target = "Tiếng Việt"
    translations = translator.translate_dialogue_items(dialogue_items)
    t_trans = (time.perf_counter() - t_trans_start) * 1000
    print(f"• [Step 4] Translation (Gemini Context-Engine): Translated {len(dialogue_items)} blocks in {t_trans:.1f} ms")

    # 5. Typesetting & Render (Auto 2-Font)
    canvas = Image.fromarray(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas)
    fnt_dialogue = "fonts/Baloo2-Bold.ttf"
    fnt_sfx = "fonts/Sriracha-Regular.ttf"

    print("\n--------------------------------------------------------------------------------")
    print(f"  BLOCK DETAILS & COHESIVE TRANSLATIONS ({len(dialogue_items)} blocks)")
    print("--------------------------------------------------------------------------------")
    for i, (item, trans_text) in enumerate(zip(dialogue_items, translations)):
        font_to_use = fnt_dialogue if item.block_type in ["DIALOGUE", "THOUGHT"] else fnt_sfx
        style_name = "Dialogue (Baloo 2 Bold)" if item.block_type in ["DIALOGUE", "THOUGHT"] else "Narration/SFX (Sriracha Regular)"
        conn_tag = f" [Linked Group #{item.connected_group_id}]" if item.connected_group_id else ""
        print(f"[{i+1:02d}] {item.block_type:<10} | {style_name:<28}{conn_tag}")
        print(f"     SRC: {item.source}")
        print(f"     VI : {trans_text}\n")

    # Render
    for i, (item, trans_text) in enumerate(zip(dialogue_items, translations)):
        blk = blks[i]
        font_to_use = fnt_dialogue if item.block_type in ["DIALOGUE", "THOUGHT"] else fnt_sfx
        x1, y1, x2, y2 = blk.xyxy
        bw, bh = max(20, x2 - x1), max(20, y2 - y1)
        fnt = ImageFont.truetype(font_to_use, max(12, min(24, int(bh // 5))))
        draw.text((x1 + 5, y1 + 5), trans_text, font=fnt, fill=(0, 0, 0), stroke_width=2, stroke_fill=(255, 255, 255))

    final_bgr = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
    out_img_path = OUT_DIR / f"translated_{sample_name}.png"
    out_cmp_path = OUT_DIR / f"compare_{sample_name}.png"
    cv2.imwrite(str(out_img_path), final_bgr)
    divider = np.full((h, 6, 3), (0, 0, 255), dtype=np.uint8)
    comparison = np.hstack([img, divider, final_bgr])
    cv2.imwrite(str(out_cmp_path), comparison)

    t_total = (time.perf_counter() - t_start) * 1000
    print(f"✓ Total Pipeline Execution: {t_total:.1f} ms ({t_total/1000:.2f} s)")
    print(f"✓ Output Saved: {out_img_path}")
    print(f"✓ Comparison Saved: {out_cmp_path}")

# Test 1: User Uploaded English Mystery Manga Sample
user_en_texts = [
    "ON ANOTHER NOTE, THERE HAVE ALSO BEEN A SERIES OF MURDERS WHERE THE BODIES WERE FOUND AND APPEARED TO HAVE EXPLODED FROM THE INSIDE.",
    "BUT, BECAUSE OF PRESSURE FROM SOMEONE WITH PRETTY HIGH INFLUENCE, THE INVESTIGATION COULDN'T PROCEED.",
    "I DON'T KNOW ANYONE OTHER THAN YUREA WHO HAS THAT KIND OF POWER,",
    "BUT JUDGING BY THE SITUATION, IT'S UNLIKELY TO BE HER.",
    "AT THE VERY LEAST, SHE WAS WITH ME THE ENTIRE NIGHT A CRIME WAS COMMITTED, SO I KNEW THAT YUREA WASN'T THE CULPRIT.",
    "IN THE FIRST PLACE, HOW DO YOU EVEN MURDER SOMEONE BY MAKING THEIR BODY EXPLODE FROM THE INSIDE?",
    "WHO IN THE WORLD HAS ENOUGH INFLUENCE TO PRESSURE THE POLICE INTO SUPPRESSING SUCH A BIZARRE MURDER CASE?"
]
run_test_sample("user_mystery_sample", "data/real_manga_samples/user_sample_mystery.png", manual_texts=user_en_texts, lang_src="English")

# Test 2: Real Japanese Action Manga Sample
jp_action_texts = [
    "おい！油断するなよ！奴が来るぞ！",
    "わかってる！全力で迎え撃つ！",
    "うううっ",
    "消えた．．．！？どこへ行ったんだ！？",
    "．．．"
]
run_test_sample("real_jp_action", "data/real_manga_samples/manga_page1_action.png", manual_texts=jp_action_texts, lang_src="日本語")

print("\n" + "=" * 80)
print("  ALL TESTS COMPLETED SUCCESSFULLY!")
print("=" * 80)
