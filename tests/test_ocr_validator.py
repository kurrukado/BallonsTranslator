import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, r"d:\BallonsTranslator")

import cv2
import numpy as np
from utils.ocr_validator import generate_multiview_crops, score_ocr_quality, is_high_stylization_crop

def run_tests():
    dark_crop = np.full((60, 150, 3), 30, dtype=np.uint8)
    cv2.putText(dark_crop, "WHISPER...", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)
    views = generate_multiview_crops(dark_crop)
    view_names = [v[0] for v in views]
    print(f"Dark crop generated views: {view_names}")
    assert "standard" in view_names and "clahe" in view_names and "inverted" in view_names

    valid_texts = [
        "I CANNOT BELIEVE THIS HAPPENED!!",
        "WHAT...?!",
        "DON'T TOUCH THAT!",
        "KYAAAHHH!",
        "He said: Let us go right now."
    ]
    for vt in valid_texts:
        score, is_susp, r = score_ocr_quality(vt, dark_crop, lang="en")
        print(f"Valid text '{vt}': score={score:.2f}, susp={is_susp}, reason={r}")
        assert not is_susp, f"Failed on valid text: {vt}"

    garbage_texts = [
        "",
        "    ",
        "!!!,,,;;;^^^",
        "qwrtyp sdfghjkl",
        "@@##$$%%^^"
    ]
    for gt in garbage_texts:
        score, is_susp, r = score_ocr_quality(gt, dark_crop, lang="en")
        print(f"Garbage text '{gt}': score={score:.2f}, susp={is_susp}, reason={r}")
        assert is_susp, f"Failed on garbage text: {gt}"

    print("✅ TEST PASSED: OCR Multi-View & Quality Scoring validation verified!")

if __name__ == "__main__":
    run_tests()
