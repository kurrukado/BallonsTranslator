import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.textblock import TextBlock
from utils.proj_imgtrans import ProjImgTrans
from modules.translators.translation_state_buffer import TranslationStateBuffer


class TestPipelineIntegration(unittest.TestCase):

    def test_chapter_batch_pipeline_flow(self):
        """Test that multi-page pipeline buffers all pages and dispatches chapter batch."""
        from ui.module_manager import ImgtransThread, TextDetectThread, OCRThread, TranslateThread, InpaintThread

        detect_th = MagicMock(spec=TextDetectThread)
        ocr_th = MagicMock(spec=OCRThread)
        translate_th = MagicMock(spec=TranslateThread)
        inpaint_th = MagicMock(spec=InpaintThread)

        pipeline_th = ImgtransThread(detect_th, ocr_th, translate_th, inpaint_th)

        # Mock project with 3 pages
        proj = MagicMock(spec=ProjImgTrans)
        proj.pages = {
            "001.jpg": [TextBlock([10, 10, 100, 50], text="Hello on page 1")],
            "002.jpg": [TextBlock([10, 10, 100, 50], text="Hello on page 2")],
            "003.jpg": [TextBlock([10, 10, 100, 50], text="Hello on page 3")],
        }
        proj.pagename2idx = lambda name: {"001.jpg": 0, "002.jpg": 1, "003.jpg": 2}[name]
        proj.read_img = lambda name: np.zeros((200, 200, 3), dtype=np.uint8)
        proj.load_mask_by_imgname = lambda name: np.zeros((200, 200), dtype=np.uint8)
        pipeline_th.imgtrans_proj = proj

        # Mock translator with translate_chapter_batch
        mock_translator = MagicMock()
        mock_translator.low_vram_mode = False
        mock_translator.is_computational_intensive.return_value = False
        mock_translator.translate_chapter_batch.return_value = {
            0: {1: {"translation": "Xin chào trang 1", "emotion_tag": "normal"}},
            1: {1: {"translation": "Xin chào trang 2", "emotion_tag": "normal"}},
            2: {1: {"translation": "Xin chào trang 3", "emotion_tag": "normal"}},
        }
        translate_th.translator = mock_translator
        translate_th.module = mock_translator
        pipeline_th.inpaint_thread.inpainter.inpaint.return_value = np.zeros((200, 200, 3), dtype=np.uint8)

        # Run pipeline
        with patch("ui.module_manager.cfg_module") as mock_cfg:
            mock_cfg.enable_detect = False
            mock_cfg.enable_ocr = False
            mock_cfg.enable_inpaint = False
            mock_cfg.enable_translate = True
            mock_cfg.translate_source = "English"
            mock_cfg.translate_target = "Tiếng Việt"
            pipeline_th.pages_to_process = ["001.jpg", "002.jpg", "003.jpg"]
            pipeline_th._imgtrans_pipeline()

        # Assert translate_chapter_batch was called EXACTLY ONCE with 3 pages in buffer
        self.assertEqual(mock_translator.translate_chapter_batch.call_count, 1)
        call_args = mock_translator.translate_chapter_batch.call_args[0]
        buffer_arg: TranslationStateBuffer = call_args[0]
        self.assertEqual(buffer_arg.total_dialogues_count(), 3)
        self.assertEqual(len(buffer_arg.get_chapter_payload().pages), 3)

        # Assert translations were assigned back to blocks
        self.assertEqual(proj.pages["001.jpg"][0].translation, "Xin chào trang 1")
        self.assertEqual(proj.pages["002.jpg"][0].translation, "Xin chào trang 2")
        self.assertEqual(proj.pages["003.jpg"][0].translation, "Xin chào trang 3")
        print("[PASS] Multi-page pipeline successfully buffered 3 pages and dispatched batch once.")

    def test_single_block_passthrough_flow(self):
        """Test that single block manual translation calls translate_single directly."""
        from ui.module_manager import ImgtransThread, TextDetectThread, OCRThread, TranslateThread, InpaintThread

        detect_th = MagicMock(spec=TextDetectThread)
        ocr_th = MagicMock(spec=OCRThread)
        ocr_th.module = MagicMock()
        translate_th = MagicMock(spec=TranslateThread)
        inpaint_th = MagicMock(spec=InpaintThread)

        pipeline_th = ImgtransThread(detect_th, ocr_th, translate_th, inpaint_th)

        mock_translator = MagicMock()
        mock_translator.translate_single.return_value = "Bản dịch trực tiếp"
        translate_th.module = mock_translator

        blk = TextBlock([10, 10, 100, 50], text="Single test bubble")
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch("ui.module_manager.cfg_module") as mock_cfg:
            mock_cfg.translate_source = "English"
            mock_cfg.translate_target = "Tiếng Việt"
            pipeline_th._blktrans_pipeline([blk], dummy_img, mode=1, blk_ids=[0], tgt_mask=None)

        self.assertEqual(mock_translator.translate_single.call_count, 1)
        self.assertEqual(blk.translation, "Bản dịch trực tiếp")
        print("[PASS] Single block manual translation directly called translate_single.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
