import cv2
import numpy as np
from PIL import Image
from typing import List

from .base import OCRBase, register_OCR, TextBlock, DEVICE_SELECTOR, DEFAULT_DEVICE, LOGGER


@register_OCR("manga_ocr")
class MangaOCR(OCRBase):
    """
    Offline Japanese Manga OCR (kha-white/manga-ocr).
    Specifically trained for Japanese manga vertical text, furigana removal, and speech bubbles.
    100% Local GPU (CUDA) execution, zero token cost.
    """
    params = {
        "device": DEVICE_SELECTOR(),
        "description": "manga-ocr (kha-white) - Offline Japanese Manga OCR"
    }

    _load_model_keys = {"model"}

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.model = None

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

    def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs) -> None:
        if self.model is None:
            self._load_model()

        im_h, im_w = img.shape[:2]
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
