import re
import cv2
import numpy as np
from PIL import Image
from typing import List, Optional

from .base import OCRBase, register_OCR, TextBlock, DEVICE_SELECTOR, DEFAULT_DEVICE, LOGGER
from utils.ocr_validator import score_ocr_quality, is_high_stylization_crop

_CJK_RE = re.compile(r'[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uff66-\uff9f]')
_LATIN_WORD_RE = re.compile(r'[A-Za-z]{3,}')


@register_OCR("manga_ocr")
class MangaOCR(OCRBase):
    """
    Offline Japanese Manga OCR (kha-white/manga-ocr).
    Specifically trained for Japanese manga vertical text, furigana removal, and speech bubbles.
    100% Local GPU (CUDA) execution, zero token cost.

    Stylized / brush-lettered / SFX blocks that the local model reads as garbage are
    optionally re-routed to Gemini Vision OCR (batched, 1 request per page).
    """
    params = {
        "device": DEVICE_SELECTOR(),
        "fallback_llm": {
            "type": "checkbox",
            "value": True,
            "description": "Route stylized/garbled blocks to Gemini Vision OCR (Consumes Gemini API quota).",
        },
        "fallback_language": {
            "type": "selector",
            "options": ["Auto Detect", "Japanese", "English"],
            "value": "Auto Detect",
            "description": "Language hint for the Gemini Vision fallback.",
        },
        "description": "manga-ocr (kha-white) - Offline Japanese Manga OCR"
    }

    _load_model_keys = {"model"}

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.model = None
        self._llm_ocr_instance = None

    def _load_model(self):
        if self.model is None:
            device = self.get_param_value("device") if hasattr(self, "get_param_value") else getattr(self, "device", DEFAULT_DEVICE)
            LOGGER.info(f"[manga-ocr] Initializing MangaOcr on {device}...")
            from manga_ocr import MangaOcr
            force_cpu = "cpu" in str(device).lower()
            self.model = MangaOcr(force_cpu=force_cpu)
            LOGGER.info("[manga-ocr] Model loaded successfully.")

    @staticmethod
    def _to_pil_rgb(img: np.ndarray) -> Optional[Image.Image]:
        if img is None or img.size == 0:
            return None
        if img.ndim == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[-1] == 4:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.shape[-1] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img
        return Image.fromarray(img_rgb)

    def ocr_img(self, img: np.ndarray) -> str:
        if self.model is None:
            self._load_model()
        pil_img = self._to_pil_rgb(img)
        if pil_img is None:
            return ""
        return self.model(pil_img)

    def _build_llm_ocr(self):
        """Instantiate LLM_OCR from the user's configured `llm_ocr` params (provider, model,
        prompt, api key...) so the fallback honours their setup instead of hardcoding one."""
        from .ocr_llm_api import LLM_OCR
        params = {}
        try:
            from utils.config import pcfg
            cfg = (pcfg.module.ocr_params or {}).get("llm_ocr") or {}
            for k, v in cfg.items():
                if k.startswith("__"):
                    continue
                params[k] = v["value"] if isinstance(v, dict) and "value" in v else v
        except Exception as e:
            LOGGER.warning(f"[manga-ocr] Could not read llm_ocr config ({e}); using defaults.")
        # The fallback language wins over whatever the standalone llm_ocr module is set to.
        params["language"] = self.get_param_value("fallback_language")
        return LLM_OCR(**params)

    def _run_llm_fallback(self, img: np.ndarray, fallback_blks: List[TextBlock]) -> None:
        """Re-OCR stylized/garbled blocks with Gemini Vision (batched: 1 request per page)."""
        try:
            if self._llm_ocr_instance is None:
                self._llm_ocr_instance = self._build_llm_ocr()
            orig_texts = {id(blk): blk.get_text() for blk in fallback_blks}
            self._llm_ocr_instance.run_ocr(img, fallback_blks)
            for blk in fallback_blks:
                res_text = blk.get_text()
                if res_text.startswith("[ERROR:"):
                    # API/network failure: never lose the local model's reading.
                    orig = orig_texts.get(id(blk), "") or ""
                    blk.text = [orig]
                    LOGGER.warning(f"[manga-ocr] Fallback failed ({res_text}), keeping local text: \"{orig}\"")
                else:
                    LOGGER.info(f"[Gemini Vision OCR] Block: Successfully transcribed -> \"{res_text}\"")
        except Exception as e:
            LOGGER.error(f"[manga-ocr] Fallback OCR error: {e}")

    def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs) -> None:
        if self.model is None:
            self._load_model()

        im_h, im_w = img.shape[:2]
        fallback_llm = self.get_param_value("fallback_llm")
        validator_lang = "en" if self.get_param_value("fallback_language") == "English" else "ja"
        fallback_blks: List[TextBlock] = []

        for i, blk in enumerate(blk_list):
            x1, y1, x2, y2 = [int(v) for v in blk.xyxy]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(im_w, x2), min(im_h, y2)

            if x2 <= x1 or y2 <= y1:
                blk.text = [""]
                continue

            crop = img[y1:y2, x1:x2]
            pil_img = self._to_pil_rgb(crop)
            if pil_img is None:
                blk.text = [""]
                continue

            try:
                recognized_text = self.model(pil_img).strip()
                blk.text = [recognized_text] if recognized_text else [""]
                LOGGER.info(f"[manga-ocr] Block #{i+1}: \"{recognized_text}\"")
            except Exception as e:
                LOGGER.warning(f"[manga-ocr] Block #{i+1} failed: {e}")
                blk.text = [""]
                recognized_text = ""

            if not fallback_llm:
                continue

            # Route empty, garbled, stylized, or Latin-script blocks to Gemini Vision.
            # manga-ocr only speaks Japanese, so a pure-Latin reading means the crop is
            # most likely an English page that this engine silently misread.
            _, is_suspicious, reason = score_ocr_quality(recognized_text, crop, lang=validator_lang)
            is_stylized, style_reason = is_high_stylization_crop(crop)
            latin_mismatch = (
                validator_lang != "en"
                and bool(_LATIN_WORD_RE.search(recognized_text))
                and not _CJK_RE.search(recognized_text)
            )
            if not recognized_text or is_suspicious or is_stylized or latin_mismatch:
                fallback_blks.append(blk)
                if is_stylized:
                    why = style_reason
                elif latin_mismatch:
                    why = "latin_script_mismatch"
                else:
                    why = reason if recognized_text else "empty_text"
                LOGGER.info(f"[manga-ocr] Block #{i+1}: Fallback to Gemini Vision OCR ({why})")

        if fallback_llm and fallback_blks:
            LOGGER.info(f"[manga-ocr] Gemini Vision Fallback for {len(fallback_blks)} stylized/unreadable blocks.")
            self._run_llm_fallback(img, fallback_blks)
