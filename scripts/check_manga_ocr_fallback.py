"""Self-check for the manga-ocr -> Gemini Vision fallback wiring.

Uses a stubbed local model and a fake LLM, so it needs no model download and no
network. Fails if: the fallback stops triggering on empty/garbled blocks, the
fallback language is not forwarded, or an API error destroys the local reading.

Run:  .\\venv\\Scripts\\python.exe scripts\\check_manga_ocr_fallback.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np

from utils.textblock import TextBlock
from modules.ocr.ocr_manga import MangaOCR
import modules.ocr.ocr_llm_api as llm_api

img = np.full((200, 200, 3), 255, np.uint8)
cv2.putText(img, "DOOM", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 6)


def make_blk():
    b = TextBlock(xyxy=[20, 60, 180, 150], is_balloon=False)
    b.text = []
    return b


captured = {}


class FakeLLMError:
    def __init__(self, **kw):
        captured.update(kw)

    def run_ocr(self, img, blks, *a, **k):
        for b in blks:
            b.text = ["[ERROR: Boom]"]


class FakeLLMGood:
    def __init__(self, **kw):
        captured.update(kw)

    def run_ocr(self, img, blks, *a, **k):
        for b in blks:
            b.text = ["ゴゴゴゴ"]


# --- 1. Empty local reading triggers fallback; API error must NOT lose data. ---
llm_api.LLM_OCR = FakeLLMError
ocr = MangaOCR(device="cpu", fallback_llm=True, fallback_language="Auto Detect")
ocr.model = lambda pil: ""  # force an empty local reading
blk = make_blk()
ocr._ocr_blk_list(img, [blk])
assert captured.get("language") == "Auto Detect", f"language not forwarded: {captured}"
assert blk.text == [""], f"error path lost/kept bad text: {blk.text}"

# --- 2. Successful fallback replaces the empty local reading. ---
llm_api.LLM_OCR = FakeLLMGood
ocr._llm_ocr_instance = None  # rebuild with the good fake
blk2 = make_blk()
ocr._ocr_blk_list(img, [blk2])
assert blk2.text == ["ゴゴゴゴ"], f"fallback text not applied: {blk2.text}"

# --- 3. Fallback disabled -> local-only, no LLM instance created. ---
llm_api.LLM_OCR = FakeLLMError
ocr_off = MangaOCR(device="cpu", fallback_llm=False)
ocr_off.model = lambda pil: ""
blk3 = make_blk()
ocr_off._ocr_blk_list(img, [blk3])
assert ocr_off._llm_ocr_instance is None, "LLM instantiated while fallback disabled"
assert blk3.text == [""]

# --- 4. Clean Latin reading from the Japanese engine -> routed to fallback. ---
llm_api.LLM_OCR = FakeLLMGood
ocr_en = MangaOCR(device="cpu", fallback_llm=True, fallback_language="Auto Detect")
ocr_en.model = lambda pil: "HELLO WORLD"
blk4 = make_blk()
ocr_en._ocr_blk_list(img, [blk4])
assert ocr_en._llm_ocr_instance is not None, "Latin-script block was not routed to fallback"

# --- 5. Genuine Japanese reading stays local (no wasted API call). ---
ocr_en._llm_ocr_instance = None
ocr_en.model = lambda pil: "おい！逃げろ！"
blk5 = TextBlock(xyxy=[0, 0, 120, 40], is_balloon=True)  # blank crop: not stylized
blk5.text = []
ocr_en._ocr_blk_list(img, [blk5])
assert ocr_en._llm_ocr_instance is None, "Japanese block wrongly routed to fallback"
assert blk5.text == ["おい！逃げろ！"]

# --- 6. Validator must accept fullwidth CJK punctuation as legitimate text. ---
from utils.ocr_validator import score_ocr_quality
_, susp, why = score_ocr_quality("「まさか…！？」", None, lang="ja")
assert not susp, f"valid Japanese punctuation flagged as {why}"

print("OK: fallback triggers on empty/garbled/Latin text, forwards language, restores on API error, respects toggle")
