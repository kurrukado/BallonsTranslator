import re
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

STANDARD_PUNCTUATION = set(" .,!?'\"-;:()[]/…-—~_?!")

def generate_multiview_crops(crop: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """
    Generate multi-representation views of an image crop for robust OCR recognition:
    1. Standard upscaled & padded view.
    2. CLA E contrast-enhanced view (for faint/shaded manga bubbles).
    3. Inverted view (for white/light text on dark/black backgrounds).
    """
    if crop is None or crop.size == 0:
        return []

    # Ensure 3-channel RGB
    if crop.ndim == 2:
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    elif crop.shape[-1] == 4:
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_RGBA2RGB)
    else:
        crop_rgb = crop.copy()

    h, w = crop_rgb.shape[:2]
    
    # 1. Standard View with Upscaling & White Border
    scale = max(2.0, min(4.0, 140.0 / max(1, min(h, w)))) if (h < 80 or w < 80) else 1.5
    scaled = cv2.resize(crop_rgb, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_CUBIC)
    pad = 16
    standard_view = cv2.copyMakeBorder(scaled, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    views = [('standard', standard_view)]

    # 2. CLAHE Contrast Enhanced View
    gray = cv2.cvtColor(standard_view, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    enhanced_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
    views.append(('clahe', enhanced_rgb))

    # 3. Inverted View (if dark background or dark art)
    mean_lum = float(np.mean(gray))
    if mean_lum < 140:
        inverted_rgb = cv2.bitwise_not(standard_view)
        views.append(('inverted', inverted_rgb))

    # 4. Super-Resolution Sharpened View for Small Text & Furigana
    if h < 35 or w < 50:
        kernel = np.array([[0, -0.5, 0], [-0.5, 3.0, -0.5], [0, -0.5, 0]], dtype=np.float32)
        sharpened = cv2.filter2D(standard_view, -1, kernel)
        views.append(('sharpened', sharpened))

    return views

def score_ocr_quality(text: str, crop: Optional[np.ndarray] = None, lang: str = 'en') -> Tuple[float, bool, str]:
    """
    Comprehensive OCR Text Quality & Anomaly Scoring.
    Returns:
        quality_score: float (0.0 to 1.0)
        is_suspicious: bool (True if output warrants secondary OCR Line / Gemini Vision fallback)
        reason: str
    """
    if not text or not text.strip():
        return 0.0, True, 'empty_text'

    cleaned = text.strip()
    total_chars = len(cleaned)

    # 1. Alphanumeric Check
    alnum_chars = [c for c in cleaned if c.isalnum()]
    if not alnum_chars:
        return 0.0, True, 'no_alphanumeric_characters'

    valid_chars = [c for c in cleaned if (c.isalnum() or c in STANDARD_PUNCTUATION)]
    valid_ratio = len(valid_chars) / max(1, total_chars)
    garbage_ratio = 1.0 - valid_ratio
    
    # 2. Garbage Characters Check
    if garbage_ratio > 0.20:
        return max(0.1, 1.0 - garbage_ratio), True, f'high_garbage_ratio ({garbage_ratio:.2f})'

    # 3. Broken / Impossible Punctuation Sequences (e.g. repeated commas or caret artifacts)
    if re.search(r'[,;:]{2,}|[~]{3,}|\^|`', cleaned):
        return 0.3, True, 'abnormal_punctuation_symbols'

    # 4. English / Latin Language-Specific Checks
    if lang.lower().startswith(('en', 'eng')):
        words = cleaned.split()
        if words:
            vowels = set('aeiouyAEIOUY')
            no_vowel_words = 0
            for w in words:
                clean_w = ''.join(c for c in w if c.isalpha())
                if len(clean_w) >= 4 and not any(c in vowels for c in clean_w):
                    no_vowel_words += 1
            if no_vowel_words >= max(1, len(words) // 2):
                return 0.35, True, f'unpronounceable_words ({no_vowel_words}/{len(words)})'

    # 5. Aspect Ratio & Box Dimension vs Character Count
    if crop is not None and crop.size > 0:
        ch, cw = crop.shape[:2]
        crop_area = ch * cw
        # If crop is very large (e.g. 200x100) but only 1-2 chars returned
        if crop_area > 15000 and total_chars <= 2:
            return 0.4, True, f'area_length_mismatch (area={crop_area}, chars={total_chars})'

    # Score calculation
    score = min(1.0, max(0.0, valid_ratio * 0.95))
    is_suspicious = (score < 0.65)
    reason = 'low_quality_score' if is_suspicious else 'valid'

    return score, is_suspicious, reason

def is_high_stylization_crop(crop: np.ndarray) -> Tuple[bool, str]:
    """
    Analyze gradient variance and edge density to detect heavy brush fonts, distressed SFX.
    """
    if crop is None or crop.size == 0:
        return False, ''
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
    h, w = gray.shape
    if h < 15 or w < 15:
        return False, ''

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag_active = mag[mag > 25]
    mag_std = float(np.std(mag_active)) if mag_active.size > 20 else 0.0

    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges) / max(1, h * w))

    if mag_std > 70.0 or (edge_density > 0.12 and mag_std > 50.0):
        return True, f'high_stylization (mag_std={mag_std:.1f}, edge_density={edge_density:.2f})'
    return False, ''