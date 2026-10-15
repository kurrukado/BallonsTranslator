import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import unittest
import numpy as np
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QImage, QPainter, QColor

# Headless Qt App
app = QApplication.instance() or QApplication(sys.argv)

from utils.active_feedback_logger import log_user_feedback_async
from ui.io_thread import ImgSaveThread


class DummyDrawingLayer:
    def __init__(self, w=100, h=100):
        self.w, self.h = w, h

    def drawed(self):
        return True

    def get_drawed_pixmap(self):
        from PyQt6.QtGui import QPixmap
        img = QImage(self.w, self.h, QImage.Format.Format_ARGB32)
        img.fill(QColor(255, 0, 0, 128))
        return QPixmap.fromImage(img)


class DummyCanvas:
    def __init__(self):
        self.drawingLayer = DummyDrawingLayer()


class DummyProj:
    def __init__(self):
        self.pages = {"test_page.png": [{"manual_edit": True, "raw_text": "A", "text": "B"}]}
        self.img_array = np.zeros((100, 100, 3), dtype=np.uint8)


class TestPageSwitchResilience(unittest.TestCase):
    def test_feedback_logger_no_use_after_free(self):
        proj = DummyProj()
        canvas = DummyCanvas()
        # Call multiple times in rapid succession
        for _ in range(5):
            log_user_feedback_async(proj, "test_page.png", canvas)
        # Should complete without segfault or memory access violation

    def test_io_thread_array_isolation(self):
        import tempfile
        saver = ImgSaveThread()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            original_arr = np.zeros((50, 50, 3), dtype=np.uint8)
            saver.saveImg(tmp_path, original_arr)

            with saver._lock:
                queued_path, queued_arr, _, _ = saver.im_save_list[0]
                # Mutate original
                original_arr[:] = 255
                # Queued array must remain 0 (independent copy)
                self.assertEqual(np.sum(queued_arr), 0)
                self.assertFalse(np.shares_memory(original_arr, queued_arr))
            
            saver.wait(500)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass


if __name__ == "__main__":
    unittest.main()
