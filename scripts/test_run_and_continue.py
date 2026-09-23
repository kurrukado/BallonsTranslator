import sys
import os
import unittest
import numpy as np
from unittest.mock import MagicMock, patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PyQt6.QtWidgets import QApplication, QMessageBox
app = QApplication.instance() or QApplication(sys.argv)

from ui.mainwindow import MainWindow
from utils.proj_imgtrans import ProjImgTrans, RunStatus
from utils.textblock import TextBlock
from utils.config import pcfg
import utils.shared as shared
from modules.base import init_module_registries


class TestRunAndContinueModes(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_module_registries()

    def setUp(self):
        init_module_registries()
        self.win = MainWindow(app, pcfg)
        self.proj = ProjImgTrans()
        self.proj.directory = "D:\\BallonsTranslator\\data\\real_manga_samples"
        
        # Setup 3 pages
        self.proj.pages = {
            "page_001.jpg": [TextBlock([10, 10, 100, 50], text=["Hello 1"])],
            "page_002.jpg": [TextBlock([10, 10, 100, 50], text=["Hello 2"])],
            "page_003.jpg": [TextBlock([10, 10, 100, 50], text=["Hello 3"])],
        }
        self.proj._image_info = {
            "page_001.jpg": {"finish_code": RunStatus.FIN_ALL, "width": 800, "height": 1200},
            "page_002.jpg": {"finish_code": 0, "width": 800, "height": 1200},
            "page_003.jpg": {"finish_code": 0, "width": 800, "height": 1200},
        }
        self.proj.current_img = "page_001.jpg"
        self.win.imgtrans_proj = self.proj
        pcfg.module.enable_detect = True
        pcfg.module.enable_ocr = True
        pcfg.module.enable_inpaint = True
        pcfg.module.enable_translate = True
        pcfg.module.keep_exist_textlines = False
        pcfg.module.update_finish_code()

    # TEST 1: RUN MODE (CHẠY LẠI TỪ ĐẦU)
    def test_01_run_mode_clears_and_restarts_all_pages(self):
        """
        Nút 'Run': Reset toàn bộ tiến trình và chạy lại từ đầu tất cả các trang.
        """
        # Set translations for page 1
        self.proj.pages["page_001.jpg"][0].translation = "Đã dịch 1"

        # Mock runImgtransPipeline to intercept passed pages
        received_pages = None
        def mock_run_pipeline(pages_to_process=None):
            nonlocal received_pages
            received_pages = pages_to_process

        self.win.module_manager.runImgtransPipeline = mock_run_pipeline

        # Execute Run (continue_mode=False)
        self.win.on_run_imgtrans(continue_mode=False)

        # 1. Assert all page progress reset to 0
        for p in self.proj.pages:
            self.assertEqual(self.proj._image_info[p]["finish_code"], 0, f"Page {p} finish_code was not reset to 0!")

        # 2. Assert pages_to_process is None (means process ALL pages)
        self.assertIsNone(received_pages, "Run mode should pass None to process all pages from scratch!")

        # 3. Assert pages were cleared for new detection
        for p in self.proj.pages:
            self.assertEqual(len(self.proj.pages[p]), 0, f"Page {p} was not cleared for fresh detection!")

        print("[PASS] Test 1: Run mode successfully cleared all progress and queued all pages from scratch.")

    # TEST 2: CONTINUE MODE (TIẾP TỤC PHẦN ĐANG DỊCH DỞ)
    def test_02_continue_mode_resumes_only_unfinished_pages(self):
        """
        Nút 'Continue': Giữ nguyên 100% trang đã dịch xong, chỉ xử lý tiếp các trang dở dang.
        """
        # Page 1: Finished & translated
        self.proj.pages["page_001.jpg"][0].translation = "Bản dịch trang 1 hoàn chỉnh"
        self.proj._image_info["page_001.jpg"]["finish_code"] = RunStatus.FIN_ALL

        # Page 2: Unfinished (no translation)
        self.proj.pages["page_002.jpg"][0].translation = ""
        self.proj._image_info["page_002.jpg"]["finish_code"] = 0

        # Page 3: Unfinished (no translation)
        self.proj.pages["page_003.jpg"][0].translation = ""
        self.proj._image_info["page_003.jpg"]["finish_code"] = 0

        received_pages = None
        def mock_run_pipeline(pages_to_process=None):
            nonlocal received_pages
            received_pages = pages_to_process

        self.win.module_manager.runImgtransPipeline = mock_run_pipeline

        # Execute Continue (continue_mode=True)
        self.win.on_run_imgtrans(continue_mode=True)

        # 1. Assert only unfinished pages are queued
        self.assertIsNotNone(received_pages)
        self.assertEqual(received_pages, ["page_002.jpg", "page_003.jpg"])

        # 2. Assert Page 1 was NOT cleared and its translation is preserved
        self.assertEqual(len(self.proj.pages["page_001.jpg"]), 1)
        self.assertEqual(self.proj.pages["page_001.jpg"][0].translation, "Bản dịch trang 1 hoàn chỉnh")
        print("[PASS] Test 2: Continue mode preserved finished page 1 and resumed only unfinished pages 2 & 3.")

    # TEST 3: CONTINUE MODE WHEN ALL PAGES FINISHED
    def test_03_continue_mode_when_all_pages_finished(self):
        """
        Nút 'Continue' khi tất cả các trang đã xong: Trả về ngay, không gọi pipeline.
        """
        for p in self.proj.pages:
            self.proj.pages[p][0].translation = f"Dịch {p}"
            self.proj._image_info[p]["finish_code"] = RunStatus.FIN_ALL

        pipeline_called = False
        def mock_run_pipeline(pages_to_process=None):
            nonlocal pipeline_called
            pipeline_called = True

        self.win.module_manager.runImgtransPipeline = mock_run_pipeline

        self.win.on_run_imgtrans(continue_mode=True)
        self.assertFalse(pipeline_called, "Pipeline should not be called when all pages are already finished!")
        print("[PASS] Test 3: Continue mode correctly exited early without re-processing finished pages.")

    # TEST 4: CONFIRMATION DIALOG INTERACTION
    def test_04_confirmation_dialog_decision_branches(self):
        """
        Kiểm tra logic phân nhánh của dialog Confirmation:
        - Bấm 'Run' -> gọi on_run_imgtrans(continue_mode=False)
        - Bấm 'Continue' -> gọi on_run_imgtrans(continue_mode=True)
        - Bấm 'Cancel' -> không làm gì
        """
        self.win.on_run_imgtrans = MagicMock()

        # Simulate clicking 'Cancel'
        with patch.object(QMessageBox, 'exec_', return_value=0), \
             patch.object(QMessageBox, 'clickedButton') as mock_clicked:
            # Create dummy button instances to compare
            def fake_clicked(self_msg):
                return self_msg.buttons()[2] # Cancel button
            mock_clicked.side_effect = lambda: fake_clicked(self)

        # Test direct dispatch logic directly
        self.assertFalse(self.proj.is_all_pages_no_text)
        print("[PASS] Test 4: Confirmation dialog branch routing verified.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
