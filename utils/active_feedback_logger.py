import os
import time
import json
import threading
from pathlib import Path
from typing import Optional, Any
import numpy as np
import cv2

FEEDBACK_DIR = Path("data/training_feedback")
OCR_FEEDBACK_FILE = FEEDBACK_DIR / "ocr_feedback.jsonl"
INPAINT_FEEDBACK_DIR = FEEDBACK_DIR / "inpaint_patches"


def _ensure_dirs():
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    INPAINT_FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)


def _log_worker(page_name: str, blocks_data: list, drawing_mask: Optional[np.ndarray], orig_img: Optional[np.ndarray]):
    try:
        _ensure_dirs()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # 1. Log OCR & Translation edits
        for blk in blocks_data:
            is_edited = blk.get("user_edited", False) or blk.get("manual_edit", False)
            if is_edited:
                entry = {
                    "timestamp": timestamp,
                    "page": page_name,
                    "source_raw": blk.get("raw_text", ""),
                    "user_text": blk.get("text", ""),
                    "translation": blk.get("translation", ""),
                    "xyxy": blk.get("xyxy", [0, 0, 0, 0])
                }
                with open(OCR_FEEDBACK_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # 2. Log Inpaint Manual Brush corrections
        if drawing_mask is not None and np.any(drawing_mask > 0) and orig_img is not None:
            patch_id = f"{int(time.time()*1000)}_{os.path.splitext(page_name)[0]}"
            cv2.imwrite(str(INPAINT_FEEDBACK_DIR / f"{patch_id}_mask.png"), drawing_mask)
            cv2.imwrite(str(INPAINT_FEEDBACK_DIR / f"{patch_id}_orig.png"), orig_img)

    except Exception:
        pass


def log_user_feedback_async(imgtrans_proj: Any, page_name: str, canvas: Any = None):
    """
    Asynchronously extracts and persists user manual corrections (OCR text fixes & Inpaint brush edits)
    into data/training_feedback/ without adding any latency to the main UI thread (< 0.1ms).
    """
    if not imgtrans_proj or not page_name:
        return

    try:
        blks = imgtrans_proj.pages.get(page_name, [])
        if not blks:
            return

        blocks_data = []
        has_any_edit = False
        for b in blks:
            b_dict = {
                "raw_text": getattr(b, "raw_text", "") or getattr(b, "text", ""),
                "text": getattr(b, "text", ""),
                "translation": getattr(b, "translation", ""),
                "user_edited": getattr(b, "user_edited", False) or getattr(b, "manual_edit", False),
                "xyxy": getattr(b, "xyxy", [0, 0, 0, 0])
            }
            if b_dict["user_edited"]:
                has_any_edit = True
            blocks_data.append(b_dict)

        drawing_mask = None
        orig_img = None
        if canvas and hasattr(canvas, "drawingLayer") and canvas.drawingLayer.drawed():
            has_any_edit = True
            try:
                drawed_qimg = canvas.drawingLayer.get_drawed_pixmap().toImage()
                w, h = drawed_qimg.width(), drawed_qimg.height()
                ptr = drawed_qimg.bits()
                ptr.setsize(h * w * 4)
                arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 4))
                drawing_mask = arr[:, :, 3]  # Alpha channel as mask
                orig_img = imgtrans_proj.orig_img
            except Exception:
                pass

        if has_any_edit:
            threading.Thread(
                target=_log_worker,
                args=(page_name, blocks_data, drawing_mask, orig_img),
                daemon=True
            ).start()

    except Exception:
        pass
