import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import json
import torch
from PIL import Image, ImageDraw, ImageFont

from modules.textdetector.detector_ctd import ComicTextDetector
from modules.inpaint.base import INPAINTERS
from utils.gemini_proxy_launcher import ensure_gemini_proxy_running, sync_proxy_api_key
import openai

# 1. Paths & Setup
IMG_PATH = "data/real_manga_samples/user_sample_mystery.png"
OUT_DIR = Path("tmp/live_output/user_sample")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ensure_gemini_proxy_running()
_api_key = os.getenv("GEMINI_API_KEY", "")
if not _api_key:
    try:
        with open("config/config.json", "r", encoding="utf-8") as _f:
            _api_key = json.load(_f).get("module", {}).get("translator_params", {}).get("LLM_API_Translator", {}).get("apikey", "")
    except Exception:
        pass
sync_proxy_api_key(_api_key)

client = openai.OpenAI(
    api_key=_api_key or "dummy_key_for_proxy",
    base_url="http://127.0.0.1:8080/v1"
)

# 2. Load Image
img = cv2.imread(IMG_PATH)
h, w, c = img.shape
print(f"Loaded image: {w}x{h} px")

# 3. Detection
ctd = ComicTextDetector(device="cuda", detect_size=1024)
mask, textblocks = ctd.detect(img)
print(f"Detected {len(textblocks)} text blocks")

# Sort blocks reading order (Top to Bottom, Right to Left)
textblocks = sorted(textblocks, key=lambda b: (b.xyxy[1] // 100, -b.xyxy[0]))

# 4. Inpainting
lama = INPAINTERS.module_dict["lama_large_512px"](device="cuda", inpaint_size=1024)
lama.load_model()
inpainted_img = lama.inpaint(img, mask)

# 5. English Text from User Sample
dialogues = [
    {
        "id": 1,
        "panel": "Panel 1 (Top-Right)",
        "source": "ON ANOTHER NOTE, THERE HAVE ALSO BEEN A SERIES OF MURDERS WHERE THE BODIES WERE FOUND AND APPEARED TO HAVE EXPLODED FROM THE INSIDE."
    },
    {
        "id": 2,
        "panel": "Panel 1 (Top-Left)",
        "source": "BUT, BECAUSE OF PRESSURE FROM SOMEONE WITH PRETTY HIGH INFLUENCE, THE INVESTIGATION COULDN'T PROCEED."
    },
    {
        "id": 3,
        "panel": "Panel 2 (Middle-Right)",
        "source": "I DON'T KNOW ANYONE OTHER THAN YUREA WHO HAS THAT KIND OF POWER,"
    },
    {
        "id": 4,
        "panel": "Panel 2 (Middle-Right Connected)",
        "source": "BUT JUDGING BY THE SITUATION, IT'S UNLIKELY TO BE HER."
    },
    {
        "id": 5,
        "panel": "Panel 2 (Middle-Left)",
        "source": "AT THE VERY LEAST, SHE WAS WITH ME THE ENTIRE NIGHT A CRIME WAS COMMITTED, SO I KNEW THAT YUREA WASN'T THE CULPRIT."
    },
    {
        "id": 6,
        "panel": "Panel 3 (Bottom-Right)",
        "source": "IN THE FIRST PLACE, HOW DO YOU EVEN MURDER SOMEONE BY MAKING THEIR BODY EXPLODE FROM THE INSIDE?"
    },
    {
        "id": 7,
        "panel": "Panel 3 (Bottom-Left)",
        "source": "WHO IN THE WORLD HAS ENOUGH INFLUENCE TO PRESSURE THE POLICE INTO SUPPRESSING SUCH A BIZARRE MURDER CASE?"
    }
]

system_prompt = (
    "You are an elite Manga localization translator specializing in Vietnamese.\n"
    "Deliver a natural, gripping, and cohesive translation for this detective mystery manga.\n"
    "Ensure multi-bubble cohesion for 3->4->5 and 6->7. Return strictly JSON: {\"translations\": [{\"id\": 1, \"translation\": \"...\"}]}"
)

resp = client.chat.completions.create(
    model="gemini-3.5-flash-lite",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Translate into natural Vietnamese:\n{json.dumps(dialogues, ensure_ascii=False, indent=2)}"}
    ],
    response_format={"type": "json_object"},
    timeout=30.0
)

res_json = json.loads(resp.choices[0].message.content)
trans_map = {item["id"]: item["translation"] for item in res_json.get("translations", [])}

# 6. Typesetting & Render (Auto 2-Font)
canvas = Image.fromarray(cv2.cvtColor(inpainted_img, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(canvas)

font_dialogue_path = "fonts/Baloo2-Bold.ttf"
font_narration_path = "fonts/Sriracha-Regular.ttf"

def render_balanced_text(draw, text, xyxy, font_path, is_balloon=True):
    x1, y1, x2, y2 = xyxy
    box_w = max(20, x2 - x1)
    box_h = max(20, y2 - y1)
    
    # Margin padding (10% padding for oval balloons)
    pad_x = int(box_w * 0.12)
    pad_y = int(box_h * 0.12)
    target_w = box_w - 2 * pad_x
    target_h = box_h - 2 * pad_y
    
    words = text.split()
    if not words:
        return
    
    # Binary search optimal font size
    best_font_sz = 12
    best_lines = [text]
    
    low, high = 10, 32
    while low <= high:
        mid = (low + high) // 2
        try:
            fnt = ImageFont.truetype(font_path, mid)
        except Exception:
            fnt = ImageFont.load_default()
        
        # Word wrap with balanced line widths
        lines = []
        cur_line = []
        for w_word in words:
            test_line = " ".join(cur_line + [w_word])
            bbox = draw.textbbox((0, 0), test_line, font=fnt)
            w_px = bbox[2] - bbox[0]
            if w_px <= target_w or not cur_line:
                cur_line.append(w_word)
            else:
                lines.append(" ".join(cur_line))
                cur_line = [w_word]
        if cur_line:
            lines.append(" ".join(cur_line))
        
        # Calculate total text height
        line_heights = [draw.textbbox((0, 0), l, font=fnt)[3] - draw.textbbox((0, 0), l, font=fnt)[1] for l in lines]
        total_h = sum(line_heights) + int(len(lines) * mid * 0.3)
        max_line_w = max([draw.textbbox((0, 0), l, font=fnt)[2] - draw.textbbox((0, 0), l, font=fnt)[0] for l in lines]) if lines else 0
        
        if total_h <= target_h and max_line_w <= target_w:
            best_font_sz = mid
            best_lines = lines
            low = mid + 1
        else:
            high = mid - 1
            
    fnt = ImageFont.truetype(font_path, best_font_sz)
    line_bboxes = [draw.textbbox((0, 0), l, font=fnt) for l in best_lines]
    line_heights = [b[3] - b[1] for b in line_bboxes]
    line_spacing = int(best_font_sz * 0.25)
    total_text_h = sum(line_heights) + (len(best_lines) - 1) * line_spacing
    
    # Centered start y
    start_y = y1 + (box_h - total_text_h) // 2
    cur_y = start_y
    
    for i, line in enumerate(best_lines):
        bbox = draw.textbbox((0, 0), line, font=fnt)
        lw = bbox[2] - bbox[0]
        cur_x = x1 + (box_w - lw) // 2
        
        # Stroke / Outline
        stroke_w = max(1, best_font_sz // 8)
        draw.text((cur_x, cur_y), line, font=fnt, fill=(0, 0, 0), stroke_width=stroke_w, stroke_fill=(255, 255, 255))
        cur_y += line_heights[i] + line_spacing

print("\nRendering Vietnamese translation with Auto 2-Font:")
for i, blk in enumerate(textblocks):
    trans_text = trans_map.get(i + 1, "")
    if not trans_text:
        continue
    # Blocks 1, 2, 3, 4, 5, 6, 7: Dialogue / Thought in bubbles
    font_p = font_dialogue_path
    render_balanced_text(draw, trans_text, blk.xyxy, font_p, is_balloon=True)
    print(f"[{i+1}] Rendered -> \"{trans_text}\" (Font: Baloo 2 Bold)")

final_img = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)

# Save Outputs
out_trans_path = OUT_DIR / "translated_user_sample.png"
out_compare_path = OUT_DIR / "compare_user_sample.png"

cv2.imwrite(str(out_trans_path), final_img)

# Side-by-side comparison
divider = np.full((h, 8, 3), (0, 0, 255), dtype=np.uint8)
comparison = np.hstack([img, divider, final_img])
cv2.imwrite(str(out_compare_path), comparison)

print(f"\n[SUCCESS] Output saved to:")
print(f"  • Translated: {out_trans_path}")
print(f"  • Compare:    {out_compare_path}")
