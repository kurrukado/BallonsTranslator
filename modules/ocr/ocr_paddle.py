import cv2
import numpy as np
import os
import os.path as osp
import subprocess
import sys
from typing import List, Optional

from .base import OCRBase, register_OCR, TextBlock, LOGGER

# Try RapidOCR (Official PaddleOCR 3.0 ONNX Engine) first, then native paddleocr
HAS_RAPID_PADDLE = False
HAS_NATIVE_PADDLE = False

try:
    from rapidocr_onnxruntime import RapidOCR
    HAS_RAPID_PADDLE = True
except ImportError:
    HAS_RAPID_PADDLE = False

try:
    from paddleocr import PaddleOCR as PaddleOCRInstance
    HAS_NATIVE_PADDLE = True
except ImportError:
    HAS_NATIVE_PADDLE = False


def _auto_install_paddleocr():
    python = sys.executable
    LOGGER.info("[PaddleOCR] paddleocr / rapidocr not found. Auto-installing from PyPI/GitHub...")
    try:
        subprocess.check_call(
            [python, "-m", "pip", "install", "rapidocr_onnxruntime", "paddleocr"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        LOGGER.info("[PaddleOCR] Auto-install succeeded.")
        global HAS_RAPID_PADDLE, HAS_NATIVE_PADDLE
        try:
            from rapidocr_onnxruntime import RapidOCR
            HAS_RAPID_PADDLE = True
        except ImportError:
            pass
        try:
            from paddleocr import PaddleOCR as PaddleOCRInstance
            HAS_NATIVE_PADDLE = True
        except ImportError:
            pass
        return HAS_RAPID_PADDLE or HAS_NATIVE_PADDLE
    except Exception as e:
        LOGGER.error(f"[PaddleOCR] Auto-install failed: {e}")
        return False


def _preprocess_crop_for_paddle(crop: np.ndarray) -> np.ndarray:
    """Enhance manga speech bubble text contrast and add padding for PaddleOCR recognition."""
    if crop is None or crop.size == 0:
        return crop
    
    # 1. Convert to RGB if needed
    if crop.ndim == 3 and crop.shape[-1] == 4:
        crop = cv2.cvtColor(crop, cv2.COLOR_RGBA2RGB)
    elif crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)

    h, w = crop.shape[:2]
    
    # 2. Add border padding (10-15px) so letters at edges are recognized clearly
    pad = 12
    crop_padded = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    # 3. Upscale if text region is very small (< 48px)
    if min(h, w) < 48:
        scale = max(2, int(48 / max(1, min(h, w))))
        crop_padded = cv2.resize(crop_padded, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    return crop_padded


@register_OCR("paddleocr3")
@register_OCR("paddleocr")
@register_OCR("paddle_ocr")
class PaddleOCREngine(OCRBase):
    """
    PaddleOCR 3.0 Engine for Comic/Manga Text Recognition.
    Supports English, Japanese, Chinese, and Multilingual recognition.
    Powered by PaddlePaddle official models (PP-OCRv4 / PP-OCR 3.0).

    GitHub: https://github.com/PaddlePaddle/PaddleOCR
    """
    params = {
        "language": {
            "type": "selector",
            "options": ["en", "japan", "ch", "korean", "chinese_cht"],
            "value": "en",
            "description": "Language for OCR recognition.",
        },
        "use_angle_cls": {
            "type": "checkbox",
            "value": True,
            "description": "Use text orientation classification for vertical/rotated text.",
        },
        "drop_score": {
            "type": "line_editor",
            "value": 0.4,
            "description": "Minimum confidence score for recognition results (0.1-0.9).",
        },
        "fallback_llm": {
            "type": "checkbox",
            "value": False,
            "description": "Fallback to Gemini Vision OCR when local text quality is low or suspicious.",
        },
    }

    download_file_on_load = False
    download_file_list = []

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self._rapid_instance = None
        self._native_instance = None
        self._llm_ocr_instance = None
        self.lang = self.params.get("language", {}).get("value", "en")
        self.use_angle_cls = self.params.get("use_angle_cls", {}).get("value", True)
        self.drop_score = float(self.params.get("drop_score", {}).get("value", 0.4))
        self.fallback_llm = self.params.get("fallback_llm", {}).get("value", False)

    def _load_model(self):
        global HAS_RAPID_PADDLE, HAS_NATIVE_PADDLE
        if not HAS_RAPID_PADDLE and not HAS_NATIVE_PADDLE:
            if not _auto_install_paddleocr():
                LOGGER.error("[PaddleOCR] Neither rapidocr_onnxruntime nor paddleocr is available.")
                return

        # 1. Prefer RapidOCR (PaddleOCR 3.0 ONNX Engine)
        if HAS_RAPID_PADDLE:
            try:
                from rapidocr_onnxruntime import RapidOCR
                self._rapid_instance = RapidOCR()
                LOGGER.info("[PaddleOCR] Initialized PaddleOCR 3.0 (RapidOCR ONNX Engine) successfully.")
                return
            except Exception as e:
                LOGGER.warning(f"[PaddleOCR] RapidOCR init failed: {e}. Trying native PaddleOCR...")

        # 2. Fallback to native PaddleOCR
        if HAS_NATIVE_PADDLE:
            try:
                from paddleocr import PaddleOCR as PaddleOCRInstance
                self._native_instance = PaddleOCRInstance(
                    lang=self.lang,
                    use_angle_cls=self.use_angle_cls,
                    show_log=False,
                )
                LOGGER.info(f"[PaddleOCR] Initialized native PaddleOCR 3.0 (lang={self.lang}) successfully.")
            except Exception as e:
                LOGGER.error(f"[PaddleOCR] Native PaddleOCR init failed: {e}")

    def _recognize_single_crop(self, crop: np.ndarray) -> str:
        """Run OCR on a single preprocessed crop image."""
        if crop is None or crop.size == 0:
            return ""

        # RapidOCR Path (PaddleOCR ONNX)
        if self._rapid_instance is not None:
            try:
                result, elapse = self._rapid_instance(crop)
                if result:
                    lines = []
                    for item in result:
                        if len(item) >= 3:
                            box, text, score = item[0], item[1], float(item[2])
                            if score >= self.drop_score and text and text.strip():
                                lines.append(text.strip())
                    if lines:
                        return " ".join(lines)
            except Exception as e:
                LOGGER.warning(f"[PaddleOCR] RapidOCR recognition warning: {e}")

        # Native PaddleOCR Path
        if self._native_instance is not None:
            try:
                results = self._native_instance.ocr(crop, det=False, cls=self.use_angle_cls)
                if results and results[0]:
                    lines = []
                    for line in results[0]:
                        if len(line) >= 2:
                            text, confidence = line[0] if isinstance(line[0], str) else line[1]
                            conf_val = float(confidence) if not isinstance(text, float) else 1.0
                            if conf_val >= self.drop_score and text and text.strip():
                                lines.append(text.strip())
                    if lines:
                        return " ".join(lines)
            except Exception as e:
                LOGGER.warning(f"[PaddleOCR] Native PaddleOCR recognition warning: {e}")

        return ""

    def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs) -> None:
        if self._rapid_instance is None and self._native_instance is None:
            self._load_model()
        if self._rapid_instance is None and self._native_instance is None:
            LOGGER.error("[PaddleOCR] Engine not available, skipping OCR.")
            return

        from utils.ocr_validator import generate_multiview_crops, score_ocr_quality, is_high_stylization_crop

        fallback_blks: List[TextBlock] = []
        fallback_reasons: dict = {}

        for i, blk in enumerate(blk_list):
            x1, y1, x2, y2 = blk.xyxy
            crop = img[max(0, y1):min(img.shape[0], y2), max(0, x1):min(img.shape[1], x2)]
            if crop.size == 0:
                blk.text = [""]
                continue

            # Check for heavy brush lettering / distressed SFX
            is_stylized, reason_stylized = is_high_stylization_crop(crop)
            if is_stylized and self.fallback_llm:
                fallback_blks.append(blk)
                fallback_reasons[id(blk)] = reason_stylized
                LOGGER.info(f"[PaddleOCR] Block #{i+1}: Stylized lettering detected ({reason_stylized}) -> Queueing Gemini Vision")
                continue

            views = generate_multiview_crops(crop)
            best_text = ""
            best_score = -1.0

            for vname, vcrop in views:
                t = self._recognize_single_crop(vcrop)
                if t:
                    score, is_susp, r = score_ocr_quality(t, crop, lang=self.lang)
                    if score > best_score:
                        best_score = score
                        best_text = t
                    if not is_susp and score >= 0.85:
                        break

            score, is_suspicious, reason = score_ocr_quality(best_text, crop, lang=self.lang)

            if not is_suspicious and best_text:
                blk.text = [best_text]
                LOGGER.info(f"[PaddleOCR 3.0] Block #{i+1}: Validated ({score:.2f}) -> \"{best_text}\"")
            else:
                if self.fallback_llm:
                    fallback_blks.append(blk)
                    fallback_reasons[id(blk)] = reason
                    LOGGER.info(f"[PaddleOCR] Block #{i+1}: Suspicious text ({reason}) -> Queueing Gemini Vision Fallback")
                else:
                    blk.text = [best_text] if best_text else [""]

        # Gemini Vision Fallback execution
        if self.fallback_llm and fallback_blks:
            LOGGER.info(f"[PaddleOCR] Triggering Gemini Vision Fallback for {len(fallback_blks)} text blocks.")
            try:
                from .ocr_llm_api import LLM_OCR
                if self._llm_ocr_instance is None:
                    ocr_lang = "English" if self.lang.startswith("en") else ("Japanese" if "ja" in self.lang else "English")
                    self._llm_ocr_instance = LLM_OCR(
                        provider="Google",
                        model="gemini-2.5-flash-lite",
                        language=ocr_lang
                    )
                orig_texts = {id(blk): blk.get_text() for blk in fallback_blks}
                self._llm_ocr_instance.run_ocr(img, fallback_blks)
                for blk in fallback_blks:
                    res_text = blk.get_text()
                    if res_text.startswith("[ERROR:") and orig_texts.get(id(blk)):
                        blk.text = [orig_texts[id(blk)]]
                    else:
                        LOGGER.info(f"[Gemini Vision OCR] Block: Successfully transcribed -> \"{res_text}\"")
            except Exception as e:
                LOGGER.error(f"[PaddleOCR] Gemini Vision Fallback error: {e}")

    def ocr_img(self, img: np.ndarray) -> str:
        if self._rapid_instance is None and self._native_instance is None:
            self._load_model()
        if self._rapid_instance is None and self._native_instance is None:
            return ""

        from utils.ocr_validator import generate_multiview_crops
        views = generate_multiview_crops(img)
        for _, vcrop in views:
            txt = self._recognize_single_crop(vcrop)
            if txt:
                return txt
        return ""