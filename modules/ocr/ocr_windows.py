import torch
import asyncio
import cv2
import numpy as np
from typing import List, Optional

from .base import OCRBase, register_OCR, TextBlock, LOGGER

try:
    import winrt.windows.media.ocr as winrt_ocr
    import winrt.windows.globalization as winrt_glob
    import winrt.windows.graphics.imaging as winrt_imaging
    import winrt.windows.storage.streams as winrt_streams
    HAS_WINRT = True
except Exception:
    HAS_WINRT = False


@register_OCR("windows_ocr")
@register_OCR("ocr_windows")
class WindowsOCR(OCRBase):
    """
    Ultra-fast native Windows OCR engine (0.01s, offline, zero token cost)
    Specialized for English, European, and Latin-script manga/comic scanlations.
    Includes automatic fallback to LLM Vision OCR when needed.
    """
    params = {
        "language_code": {
            "type": "selector",
            "options": ["en-US", "ja-JP", "zh-Hans-CN", "ko-KR"],
            "value": "en-US",
            "description": "Windows OCR language profile.",
        },
        "fallback_llm": {
            "type": "checkbox",
            "value": True,
            "description": "Automatically fallback to Gemini Vision OCR when result is empty or unreadable (Consumes Gemini API quota).",
        },
        "min_text_len": {
            "type": "selector",
            "options": [1, 2, 3, 5, 10],
            "value": 1,
            "description": "Minimum characters required before triggering fallback.",
        },
    }

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.language_code = self.params["language_code"]["value"]
        self.fallback_llm = self.params["fallback_llm"]["value"]
        self.min_text_len = self.params["min_text_len"]["value"]
        self._engine = None
        self._llm_ocr_instance = None

    def _load_model(self):
        if not HAS_WINRT:
            LOGGER.warning("[WindowsOCR] WinRT OCR library is not available.")
            return

        lang_str = self.params["language_code"]["value"] or "en-US"
        try:
            lang = winrt_glob.Language(lang_str)
            self._engine = winrt_ocr.OcrEngine.try_create_from_language(lang)
            if self._engine is None:
                LOGGER.warning(f"[WindowsOCR] Language {lang_str} not installed in Windows. Falling back to default language.")
                self._engine = winrt_ocr.OcrEngine.try_create_from_user_profile_languages()
        except Exception as e:
            LOGGER.error(f"[WindowsOCR] Failed to initialize Windows OcrEngine: {e}")
            self._engine = None

    async def _recognize_single_crop(self, crop: np.ndarray) -> str:
        if self._engine is None or crop is None or crop.size == 0:
            return ""

        if crop.ndim == 2:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        elif crop.shape[-1] == 4:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_RGBA2RGB)
        elif crop.shape[-1] == 3:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        else:
            crop_rgb = crop

        success, buf = cv2.imencode(".png", cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
        if not success:
            return ""

        stream = winrt_streams.InMemoryRandomAccessStream()
        writer = winrt_streams.DataWriter(stream)
        writer.write_bytes(buf.tobytes())
        await writer.store_async()
        await writer.flush_async()
        writer.detach_stream()
        stream.seek(0)

        decoder = await winrt_imaging.BitmapDecoder.create_async(stream)
        software_bitmap = await decoder.get_software_bitmap_async()
        result = await self._engine.recognize_async(software_bitmap)

        lines = [line.text.strip() for line in result.lines if line.text.strip()]
        return " ".join(lines)

    async def _recognize_crop_async(self, crop: np.ndarray) -> str:
        if self._engine is None or crop is None or crop.size == 0:
            return ""

        # Preprocessing: auto-upscale small crops & add white padding so WinRT OCR detects small text
        h, w = crop.shape[:2]
        scale = max(2.0, min(4.0, 100.0 / max(1, min(h, w)))) if (h < 80 or w < 80) else 1.5
        scaled_crop = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        padded_crop = cv2.copyMakeBorder(scaled_crop, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=[255, 255, 255])

        text = await self._recognize_single_crop(padded_crop)
        if text.strip():
            return text.strip()

        # Local Pass 2 (Offline Adaptive Contrast Enhancement for faint/dark background text)
        gray = cv2.cvtColor(padded_crop, cv2.COLOR_BGR2GRAY) if padded_crop.ndim == 3 else padded_crop
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        # If dark background, invert
        if np.mean(gray) < 120:
            enhanced = cv2.bitwise_not(enhanced)
        
        text_pass2 = await self._recognize_single_crop(enhanced)
        return text_pass2.strip()

    def _is_high_stylization_crop(self, crop: np.ndarray) -> tuple:
        """
        Analyze image crop gradients and edge statistics to detect heavy brush lettering,
        distressed fonts, or hand-drawn comic SFX before attempting standard WinRT OCR.
        """
        if crop is None or crop.size == 0:
            return False, ""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        h, w = gray.shape
        if h < 15 or w < 15:
            return False, ""

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        mag_active = mag[mag > 25]
        mag_std = float(np.std(mag_active)) if mag_active.size > 20 else 0.0

        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.count_nonzero(edges) / max(1, h * w))

        if mag_std > 70.0 or (edge_density > 0.12 and mag_std > 50.0):
            return True, f"high_stylization (mag_std={mag_std:.1f}, edge_density={edge_density:.2f})"
        return False, ""

    def _is_suspicious_text(self, text: str, blk: TextBlock, crop: np.ndarray) -> tuple:
        """
        Analyze WinRT OCR output for garbage characters or missing text.
        Allows all valid 1-word exclamations ('YUREA!', 'WHAT?', 'NO!') with alphanumeric characters.
        """
        if not text or not text.strip():
            return True, "empty_text"

        cleaned = text.strip()
        has_alnum = any(c.isalnum() for c in cleaned)
        if not has_alnum:
            return True, "no_alphanumeric_letters"

        standard_punct = set(" .,!?'\"-:;()[]/…–—~")
        garbage_chars = [c for c in cleaned if not (c.isalnum() or c in standard_punct)]
        if len(garbage_chars) / max(1, len(cleaned)) > 0.40:
            return True, f"high_garbage_ratio ({len(garbage_chars)}/{len(cleaned)})"

        return False, ""

    def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs) -> None:
        if self._engine is None:
            self._load_model()

        fallback_blks: List[TextBlock] = []
        fallback_reasons: dict = {}

        async def _process_all():
            for i, blk in enumerate(blk_list):
                x1, y1, x2, y2 = blk.xyxy
                crop = img[max(0, y1):min(img.shape[0], y2), max(0, x1):min(img.shape[1], x2)]
                if crop.size == 0:
                    blk.text = [""]
                    continue

                from utils.ocr_validator import generate_multiview_crops, score_ocr_quality, is_high_stylization_crop
                is_stylized, reason_stylized = is_high_stylization_crop(crop)
                if is_stylized and self.fallback_llm:
                    fallback_blks.append(blk)
                    fallback_reasons[id(blk)] = reason_stylized
                    LOGGER.info(f"[WindowsOCR] Block #{i+1}: Pre-routing to Gemini Vision OCR ({reason_stylized})")
                    continue

                views = generate_multiview_crops(crop)
                best_text = ""
                best_score = -1.0

                for vname, vcrop in views:
                    try:
                        t = await self._recognize_crop_async(vcrop)
                        t = t.strip() if t else ""
                    except Exception as e:
                        LOGGER.warning(f"[WindowsOCR] Block #{i+1} view '{vname}' error: {e}")
                        t = ""
                    if t:
                        s, is_s, r = score_ocr_quality(t, crop, lang=self.language_code)
                        if s > best_score:
                            best_score = s
                            best_text = t
                        if not is_s and s >= 0.85:
                            break

                score, is_suspicious, reason_suspicious = score_ocr_quality(best_text, crop, lang=self.language_code)
                is_outside_balloon = not blk.is_in_balloon() if hasattr(blk, 'is_in_balloon') else not getattr(blk, 'is_balloon', True)
                should_fallback = is_suspicious or not best_text or (is_outside_balloon and score < 0.85)
                if not should_fallback and best_text:
                    blk.text = [best_text]
                    LOGGER.info(f"[WindowsOCR] Block #{i+1}: Validated ({score:.2f}) -> \"{best_text}\"")
                else:
                    if self.fallback_llm:
                        fallback_blks.append(blk)
                        reason = reason_suspicious or ("outside_balloon_borderline" if is_outside_balloon else "low_quality")
                        fallback_reasons[id(blk)] = reason
                        LOGGER.info(f"[WindowsOCR] Block #{i+1}: Fallback to Gemini Vision OCR ({reason})")
                    else:
                        blk.text = [best_text] if best_text else [""]

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                loop.run_until_complete(_process_all())
            else:
                asyncio.run(_process_all())
        except Exception:
            asyncio.run(_process_all())

        if self.fallback_llm and fallback_blks:
            LOGGER.info(f"[WindowsOCR] Triggering Gemini Vision Fallback for {len(fallback_blks)} unreadable/stylized text blocks.")
            try:
                from .ocr_llm_api import LLM_OCR
                if self._llm_ocr_instance is None:
                    ocr_lang = "English" if self.language_code.startswith("en") else ("Japanese" if "ja" in self.language_code else "English")
                    self._llm_ocr_instance = LLM_OCR(
                        provider="Google",
                        model="gemini-3.5-flash-lite",
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
                LOGGER.error(f"[WindowsOCR] Fallback OCR error: {e}")

    def ocr_img(self, img: np.ndarray) -> str:
        if self._engine is None:
            self._load_model()

        async def _run():
            return await self._recognize_crop_async(img)

        try:
            return asyncio.run(_run())
        except Exception as e:
            LOGGER.error(f"[WindowsOCR] ocr_img failed: {e}")
            return ""
