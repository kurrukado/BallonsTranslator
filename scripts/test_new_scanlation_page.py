import sys, os, time, cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
sys.path.insert(0, '.')

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

print('=' * 80)
print('  ITEM 3: FULL PIPELINE VERIFICATION ON NEW SCANLATION IMAGE')
print('=' * 80)

from modules.base import init_module_registries
init_module_registries()
from modules.textdetector import TEXTDETECTORS
from modules.ocr import OCR
from modules.inpaint import INPAINTERS
from modules.translators import TRANSLATORS

detector_cls = TEXTDETECTORS.get('ctd')
detector = detector_cls(device='cuda')

ocr_cls = OCR.get('windows_ocr')
ocr = ocr_cls(language_code='en-US')

inpainter_cls = INPAINTERS.get('lama_large_512px')
inpainter = inpainter_cls(device='cuda', inpaint_size=1024)

translator_cls = TRANSLATORS.get('LLM_API_Translator')
translator = translator_cls(lang_source='English', lang_target='Tiếng Việt')
translator.set_param_value('provider', 'Google')
translator.set_param_value('model', 'gemini-2.5-flash')
_api_key = os.getenv("GEMINI_API_KEY", "")
if not _api_key:
    try:
        with open("config/config.json", "r", encoding="utf-8") as _f:
            _api_key = json.load(_f).get("module", {}).get("translator_params", {}).get("LLM_API_Translator", {}).get("apikey", "")
    except Exception:
        pass
if _api_key:
    translator.set_param_value('apikey', _api_key)

img_path = 'tmp/live_output/new_scanlation_test/input_new_scanlation_page.png'
img = cv2.imread(img_path)

# 1. Detection
t0 = time.perf_counter()
mask, blks = detector.detect(img)
t_det = (time.perf_counter() - t0) * 1000

# 2. OCR
t0 = time.perf_counter()
ocr.run_ocr(img, blks)
t_ocr = (time.perf_counter() - t0) * 1000

# 3. Inpainting
t0 = time.perf_counter()
inpainted_img = inpainter.inpaint(img, mask)
t_inpaint = (time.perf_counter() - t0) * 1000

# 4. Translation
t0 = time.perf_counter()
translator.translate_textblk_lst(blks)
t_trans = (time.perf_counter() - t0) * 1000

# 5. Typesetting & Rendering
t0 = time.perf_counter()
out_pil = Image.fromarray(cv2.cvtColor(inpainted_img, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(out_pil)

font_baloo_path = 'fonts/Baloo2-Bold.ttf'
font_sriracha_path = 'fonts/Sriracha-Regular.ttf'

print('\n--- CHI TIET TUNG KHOI CHU & FONT AP DUNG ---')
for i, blk in enumerate(blks):
    raw_text = blk.get_text()
    trans_text = blk.translation
    is_balloon = blk.is_in_balloon()
    
    if is_balloon:
        font_name = 'Baloo 2 Bold'
        font_path = font_baloo_path
        fill_color = (20, 20, 20)
        font_size = 28
        type_str = 'TRONG BONG BONG (Dialogue)'
    else:
        font_name = 'Sriracha Regular'
        font_path = font_sriracha_path
        fill_color = (230, 230, 230) if ('SHADOWS' in raw_text or 'MEANWHILE' in raw_text) else (180, 20, 20)
        font_size = 32 if 'CRASH' in raw_text else 24
        type_str = 'NGOAI BONG BONG (Narration/SFX)'

    print(f'[{i+1}] Loai vung: {type_str}')
    print(f'    Font ap dung   : {font_name} ({os.path.basename(font_path)})')
    print(f'    Van ban OCR goc: {raw_text}')
    print(f'    Ban dich TV    : {trans_text}')

    # Render on image
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()
    
    x1, y1, x2, y2 = blk.xyxy
    draw.text((x1 + 10, y1 + 10), trans_text, font=font, fill=fill_color)

t_type = (time.perf_counter() - t0) * 1000

out_final = cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGB2BGR)
out_path = 'tmp/live_output/new_scanlation_test/translated_new_page.png'
cv2.imwrite(out_path, out_final)

# Create Side-by-Side comparison
h, w = img.shape[:2]
compare_img = np.zeros((h, w * 2 + 20, 3), dtype=np.uint8)
compare_img[:, :w] = img
compare_img[:, w+20:] = out_final
compare_path = 'tmp/live_output/new_scanlation_test/compare_new_page.png'
cv2.imwrite(compare_path, compare_img)

print('\n--- THOI GIAN XU LY TUNG BUOC ---')
print(f'• Detection (CTD on CUDA)      : {t_det:.2f} ms')
print(f'• OCR (Windows Native OCR)     : {t_ocr:.2f} ms')
print(f'• Inpainting (LaMa on CUDA)    : {t_inpaint:.2f} ms')
print(f'• Translation (Gemini API)     : {t_trans:.2f} ms')
print(f'• Typesetting & 2-Font Render  : {t_type:.2f} ms')
print(f'• Tong thoi gian toan trinh    : {(t_det + t_ocr + t_inpaint + t_trans + t_type)/1000:.2f} s')
print(f'\n[OK] Anh ket qua da duoc luu tai:')
print(f'  1. {out_path}')
print(f'  2. {compare_path}')
print('=' * 80)
