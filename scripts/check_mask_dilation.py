"""Self-check for the CTD mask-dilation fix.

Fails if the per-block dilation kernel grows unbounded again (the regression that
turned text blocks into solid slabs and made LaMa paint flat "white arena" fills).

Run:  .\\venv\\Scripts\\python.exe scripts\\check_mask_dilation.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from modules.textdetector.detector_ctd import _block_dilate_kernel

# 1. Kernel must stay small, odd, and capped regardless of the user's slider value.
for base_ksize in range(0, 40):
    for sfx in (False, True):
        k = _block_dilate_kernel(base_ksize, sfx)
        assert k % 2 == 1, f"kernel must be odd, got {k} (base={base_ksize})"
        assert 5 <= k <= 9, f"kernel {k} out of [5, 9] bounds (base={base_ksize})"

# 2. SFX/stroke blocks may dilate a touch more, never less.
for base_ksize in range(0, 40):
    assert _block_dilate_kernel(base_ksize, True) >= _block_dilate_kernel(base_ksize, False)

print("OK: mask dilation kernel bounded to [5, 9], odd, sfx >= normal")

# 3. End-to-end (optional): real model must not inflate the raw mask past 1.6x.
_SAMPLE = os.path.join(ROOT, "data", "real_manga_samples", "user_sample_mystery.png")
_MODEL = os.path.join(ROOT, "data", "models", "comictextdetector.pt")
if os.path.exists(_SAMPLE) and os.path.exists(_MODEL):
    import cv2
    import numpy as np
    from modules.textdetector.detector_ctd import ComicTextDetector

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    img = cv2.cvtColor(cv2.imread(_SAMPLE), cv2.COLOR_BGR2RGB)
    det = ComicTextDetector(device=dev, detect_size=1536)
    det.load_model()
    _, raw_mask, blk_list = det.model(img)
    final_mask, _ = det._detect(img, None)
    raw = int(np.count_nonzero(raw_mask > 0))
    fin = int(np.count_nonzero(final_mask > 0))
    ratio = fin / max(1, raw)
    print(f"mask coverage raw={raw} final={fin} inflation=x{ratio:.2f}")
    assert ratio < 1.6, f"mask inflated x{ratio:.2f} (>1.6) -> solid slabs / white-arena risk"
    print("OK: end-to-end mask inflation within bounds")
else:
    print("SKIP: model/sample not present, ran kernel bound check only")
