import sys
import os
import unittest
import numpy as np
import cv2
from unittest.mock import MagicMock, patch

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PyQt6.QtWidgets import QApplication
app = QApplication.instance() or QApplication(sys.argv)

from utils.textblock import TextBlock, sort_regions
from utils.fontformat import TextAlignment
from utils.proj_imgtrans import ProjImgTrans, RunStatus
from utils.config import pcfg
from modules.translators.translation_proxy import (
    TranslationProxy, PageState, TranslationMode,
    clean_and_repair_json, DialogueBlock
)
from modules.translators.trans_llm_api import LLM_API_Translator
from modules.translators.context_engine import (
    ContextAssembler, DialogueItem, CharacterMemory, CharacterProfile, GlossaryManager, SFX_EN_VN
)
from modules.inpaint.base import InpainterBase, LamaInpainterMPE, LamaLarge
from ui.mainwindow import MainWindow


class TestHardenedEdgeCases(unittest.TestCase):

    # =========================================================================
    # SUBSYSTEM 1: Empty, Whitespace, and Special Symbol Text Blocks from OCR
    # =========================================================================

    def test_01_empty_and_whitespace_blocks_handling(self):
        """
        Verify that empty, whitespace-only, and zero-width text blocks:
        - Do not trigger empty API calls.
        - Are safely initialized or cleared.
        - Do not corrupt 1:1 mapping of remaining non-empty blocks.
        """
        translator = LLM_API_Translator("English", "Tiếng Việt")
        
        # Test 1A: Page with ONLY empty and whitespace blocks
        blks_empty = [
            TextBlock([10, 10, 50, 50], text=[""]),
            TextBlock([60, 10, 100, 50], text=["   \n\t  "]),
            TextBlock([110, 10, 150, 50], text=["\u200b\ufeff"])
        ]
        # Initially set some stale translation to simulate previous run
        blks_empty[0].translation = "Stale text 1"
        blks_empty[1].translation = "Stale text 2"
        
        translator.translate_textblk_lst(blks_empty)
        # All non-user-edited empty blocks must have translation reset to empty
        self.assertEqual(blks_empty[0].translation, "")
        self.assertEqual(blks_empty[1].translation, "")
        self.assertEqual(blks_empty[2].translation, "")
        print("   ✓ Case 1A (All-empty page handled cleanly): PASSED")

        # Test 1B: Mixed page: valid blocks + whitespace blocks
        blks_mixed = [
            TextBlock([10, 10, 100, 50], text=["Hello world"]),
            TextBlock([10, 60, 100, 100], text=["   "]),
            TextBlock([10, 110, 100, 150], text=["Goodbye"])
        ]
        blks_mixed[0].translation = ""
        blks_mixed[1].translation = "Old text"
        blks_mixed[2].translation = ""

        # Mock translate_dialogue_items
        with patch.object(translator, 'translate_dialogue_items', return_value=["Xin chào thế giới", "Tạm biệt"]):
            translator.translate_textblk_lst(blks_mixed)
            self.assertEqual(blks_mixed[0].translation, "Xin chào thế giới")
            self.assertEqual(blks_mixed[2].translation, "Tạm biệt")
        print("   ✓ Case 1B (Mixed valid & empty blocks 1:1 alignment): PASSED")

    def test_02_special_symbols_and_punctuation_bypass(self):
        """
        Verify that special symbols (???, ..., ♪, ★, 「」, …?!?) bypass the LLM API locally,
        cost 0 tokens, and assign correct emotion tags.
        """
        translator = LLM_API_Translator("English", "Tiếng Việt")
        items = [
            DialogueItem(id=1, source="...?!"),
            DialogueItem(id=2, source="???"),
            DialogueItem(id=3, source="♪ ~ ★"),
            DialogueItem(id=4, source="「...」"),
            DialogueItem(id=5, source="!")
        ]

        # Call translate_dialogue_items - should complete locally without API calls
        results = translator.translate_dialogue_items(items)
        self.assertEqual(results[0], "...?!")
        self.assertEqual(results[1], "???")
        self.assertEqual(results[2], "♪ ~ ★")
        self.assertEqual(results[3], "「...」")
        self.assertEqual(results[4], "!")

        # Verify emotion tagging
        self.assertEqual(items[0].emotion_tag, "surprise")
        self.assertEqual(items[1].emotion_tag, "normal")
        self.assertEqual(items[4].emotion_tag, "shout")
        print("   ✓ Case 1C (Special symbols zero-token bypass & emotion tagging): PASSED")

    # =========================================================================
    # SUBSYSTEM 2: Vertical Text vs Horizontal Text Parsing & Sound Effects (SFX)
    # =========================================================================

    def test_03_vertical_horizontal_and_single_line_alignment(self):
        """
        Verify that:
        - Single-line or empty polygon blocks recalculate alignment without RuntimeWarning or NaN.
        - TextBlock min_rect safely handles degenerate or missing lines.
        - Japanese manga reading order (RTL) vs Western/Webtoon reading order (LTR) is respected.
        """
        # Test single-line block
        blk_single = TextBlock([10, 10, 50, 100])
        blk_single.lines = [[[10, 10], [50, 10], [50, 100], [10, 100]]]
        blk_single.recalulate_alignment()
        self.assertEqual(blk_single.alignment, TextAlignment.Center)

        # Test empty lines block
        blk_empty = TextBlock([20, 20, 80, 80])
        blk_empty.lines = []
        blk_empty.recalulate_alignment()
        self.assertEqual(blk_empty.alignment, TextAlignment.Center)
        rect = blk_empty.min_rect()
        self.assertIsNotNone(rect)

        # Test reading order: Vertical Manga (RTL)
        b1_right = TextBlock([800, 100, 900, 300])
        b1_right.vertical = True
        b2_left = TextBlock([100, 100, 200, 300])
        b2_left.vertical = True
        sorted_manga = sort_regions([b2_left, b1_right], right_to_left=True)
        self.assertEqual(sorted_manga[0], b1_right)
        self.assertEqual(sorted_manga[1], b2_left)

        # Test reading order: Horizontal Webtoon (LTR)
        b1_h_left = TextBlock([100, 100, 200, 150])
        b1_h_left.vertical = False
        b2_h_right = TextBlock([800, 100, 900, 150])
        b2_h_right.vertical = False
        sorted_webtoon = sort_regions([b2_h_right, b1_h_left], right_to_left=False)
        self.assertEqual(sorted_webtoon[0], b1_h_left)
        self.assertEqual(sorted_webtoon[1], b2_h_right)
        print("   ✓ Case 2A (Alignment calculation & Direction reading order): PASSED")

    def test_04_sfx_detection_and_vietnamese_localization(self):
        """
        Verify that Japanese and English sound effects (SFX) are accurately classified
        and mapped into natural Vietnamese onomatopoeia.
        """
        sfx_samples = [
            ("ドン", "SFX"),
            ("ドキドキ", "SFX"),
            ("ゴゴゴ", "SFX"),
            ("POW!", "SFX"),
            ("BOOM", "SFX"),
            ("CRASH", "SFX"),
            ("GASP", "SFX"),
            ("CLICK", "SFX"),
            ("（I wonder if she knows...）", "THOUGHT"),
            ("[THREE DAYS LATER]", "NARRATION")
        ]

        for text, expected_type in sfx_samples:
            classified = ContextAssembler.classify_block_type(text, vertical=True)
            self.assertEqual(classified, expected_type, f"Failed for '{text}': got {classified}, expected {expected_type}")

        # Check SFX dictionary mapping
        self.assertIn("POW", SFX_EN_VN)
        self.assertEqual(SFX_EN_VN["POW"], "BỐP!")
        self.assertEqual(SFX_EN_VN["BOOM"], "ĐÙNG!")
        self.assertEqual(SFX_EN_VN["ドキドキ"], "THỊCH THỊCH")
        self.assertEqual(SFX_EN_VN["ドン"], "RẦM!")
        print("   ✓ Case 2B (SFX classification & Vietnamese onomatopoeia dictionary): PASSED")

    # =========================================================================
    # SUBSYSTEM 3: Large Image Resolutions & Non-Standard Aspect Ratios (LaMa)
    # =========================================================================

    def test_05_lama_large_resolution_and_aspect_ratio_safety(self):
        """
        Verify that:
        - Image resolutions > 2048px (e.g. 2400x3200) are resized and modulo-8 padded safely.
        - Extreme aspect ratios (e.g. 1:8 tall webtoon strip or 8:1 wide banner) are handled without dimension mismatch.
        - Degenerate zero-size crops are caught before invoking neural network.
        - Memory safe inpainting catches OOM and falls back cleanly.
        """
        inpainter = LamaLarge()
        inpainter.model = MagicMock()
        inpainter.model.mpe = None
        
        # Mock generator returning same shape tensor
        def mock_generator(img, mask, rel_pos=None, direct=None):
            return img.clone()
        inpainter.model.side_effect = mock_generator

        # 3A: Large resolution image (2400 x 3200)
        large_img = np.full((3200, 2400, 3), 200, dtype=np.uint8)
        large_mask = np.zeros((3200, 2400), dtype=np.uint8)
        large_mask[1000:1200, 800:1400] = 255

        result_large = inpainter._inpaint(large_img, large_mask)
        self.assertEqual(result_large.shape, (3200, 2400, 3), "Large image inpainting output resolution altered!")
        print("   ✓ Case 3A (Large resolution 2400x3200 inpainting resolution integrity): PASSED")

        # 3B: Extreme aspect ratio webtoon strip (250 x 2500)
        webtoon_img = np.full((2500, 250, 3), 150, dtype=np.uint8)
        webtoon_mask = np.zeros((2500, 250), dtype=np.uint8)
        webtoon_mask[500:600, 50:200] = 255

        result_webtoon = inpainter._inpaint(webtoon_img, webtoon_mask)
        self.assertEqual(result_webtoon.shape, (2500, 250, 3), "Webtoon strip inpainting output resolution altered!")
        print("   ✓ Case 3B (Extreme aspect ratio 250x2500 inpainting resolution integrity): PASSED")

        # 3C: Degenerate zero-size inputs
        empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
        empty_mask = np.zeros((0, 0), dtype=np.uint8)
        res_empty = inpainter.inpaint(empty_img, empty_mask)
        self.assertEqual(res_empty.size, 0)
        print("   ✓ Case 3C (Degenerate zero-size crop guard): PASSED")

        # 3D: Out Of Memory (OOM) handling
        fail_inpainter = LamaLarge()
        call_count = [0]
        def oom_inpaint(img, mask, textblock_list=None):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("CUDA error: out of memory while allocating tensor")
            return img.copy()
        
        fail_inpainter._inpaint = oom_inpaint
        fail_inpainter.moveToDevice = MagicMock()
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        test_mask = np.zeros((100, 100), dtype=np.uint8)
        
        res_oom = fail_inpainter.memory_safe_inpaint(test_img, test_mask)
        self.assertEqual(res_oom.shape, (100, 100, 3))
        self.assertEqual(call_count[0], 2, "Failed to retry on CPU after CUDA OOM!")
        print("   ✓ Case 3D (CUDA OOM detection and CPU fallback): PASSED")

    # =========================================================================
    # SUBSYSTEM 4: Network Failure, Timeout & Rate-Limit Fast Fallback
    # =========================================================================

    def test_06_fast_429_rate_limit_and_timeout_recovery(self):
        """
        Verify that:
        - HTTP 429 RateLimitError immediately advances to next candidate model without wasting 3 retries.
        - Network timeouts retry with backoff and succeed on backup model.
        - Malformed or truncated JSON from smaller models is repaired or caught.
        """
        proxy = TranslationProxy()
        proxy.start_session([0])
        proxy.mark_page_completed(0, [{"id": 1, "text": "Rate Limit Test"}])

        translator = LLM_API_Translator("English", "Tiếng Việt")
        translator._initialize_client = MagicMock(return_value=True)
        translator.client = MagicMock()

        call_sequence = []
        def mock_chat_create(model, **kwargs):
            call_sequence.append(model)
            if model == "gemini-3.5-flash-lite":
                raise Exception("429 Too Many Requests: Resource exhausted, quota exceeded")
            mock_choice = MagicMock()
            mock_choice.message.content = '{"pages": [{"page_index": 0, "dialogues": [{"id": 1, "translation": "Thành công qua fallback", "emotion_tag": "normal"}]}]}'
            mock_resp = MagicMock()
            mock_resp.choices = [mock_choice]
            return mock_resp

        translator.client.chat.completions.create.side_effect = mock_chat_create
        res = translator.translate_chapter_batch(proxy)

        # Verify gemini-3.5-flash-lite was called ONLY ONCE before fast-fallback to gemini-3.5-flash
        lite_calls = [m for m in call_sequence if m == "gemini-3.5-flash-lite"]
        self.assertEqual(len(lite_calls), 1, f"Expected 1 call before fast fallback, got {len(lite_calls)}")
        self.assertEqual(res[0][1]["translation"], "Thành công qua fallback")
        print("   ✓ Case 4A (Fast 429 Rate Limit zero-delay model transition): PASSED")

        # 4B: JSON Repair for missing brackets / trailing commas
        malformed_json_1 = '```json\n{"pages": [{"page_index": 0, "dialogues": [{"id": 1, "translation": "Bản dịch bị thiếu ngoặc"}]'
        repaired_1 = clean_and_repair_json(malformed_json_1)
        self.assertIn("pages", repaired_1)
        self.assertEqual(repaired_1["pages"][0]["dialogues"][0]["translation"], "Bản dịch bị thiếu ngoặc")

        malformed_json_2 = '{"pages": [{"page_index": 0, "dialogues": [{"id": 1, "translation": "Dấu phẩy thừa",}],}],}'
        repaired_2 = clean_and_repair_json(malformed_json_2)
        self.assertIn("pages", repaired_2)
        print("   ✓ Case 4B (Malformed & truncated JSON cleaner and repair engine): PASSED")

    # =========================================================================
    # SUBSYSTEM 5: Project Saving & Canvas Text Layer Synchronization
    # =========================================================================

    def test_07_project_save_and_mismatched_canvas_items(self):
        """
        Verify that:
        - Project save automatically initializes proj_path if unset.
        - updateTextBlkList handles mismatched lengths between textblk_item_list and pairwidget_list without dropping items.
        - get_mask_path, get_inpainted_path, get_result_path return empty string safely if current_img is None.
        - Manual save (Ctrl+S) preserves text layers and translations.
        """
        import tempfile
        import shutil
        tmp_dir = tempfile.mkdtemp()
        try:
            proj = ProjImgTrans()
            proj.directory = tmp_dir
            # proj.proj_path is None initially
            blk = TextBlock([10, 10, 100, 50], text=["Save edge test"])
            blk.translation = "Đã lưu an toàn"
            proj.pages = {"test.jpg": [blk]}
            proj._image_info = {"test.jpg": {"finish_code": RunStatus.FIN_ALL}}

            # Save should auto-set proj_path and succeed
            proj.save()
            self.assertIsNotNone(proj.proj_path)
            self.assertTrue(os.path.exists(proj.proj_path))

            # Test safe path getters when current_img is None
            proj.current_img = None
            self.assertEqual(proj.get_mask_path(), "")
            self.assertEqual(proj.get_inpainted_path(), "")
            self.assertEqual(proj.get_result_path(), "")
            print("   ✓ Case 5A (Project save auto-path and safe path getters when None): PASSED")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        # Test Canvas SceneTextManager with mismatched list lengths
        win = MainWindow(app, pcfg)
        test_proj = ProjImgTrans()
        test_proj.directory = "D:\\BallonsTranslator\\data\\real_manga_samples"
        test_proj.pages = {"p1.png": [TextBlock([10, 10, 50, 50]), TextBlock([60, 60, 100, 100])]}
        test_proj.current_img = "p1.png"
        win.imgtrans_proj = test_proj
        win.canvas.imgtrans_proj = test_proj
        win.st_manager.imgtrans_proj = test_proj

        win.st_manager.updateSceneTextitems()
        self.assertEqual(len(win.st_manager.textblk_item_list), 2)
        
        # Simulate desync: simulate pairwidget_list having 1 item while textblk_item_list has 2
        win.st_manager.pairwidget_list = win.st_manager.pairwidget_list[:1]
        
        # updateTextBlkList must NOT drop the second text block!
        win.st_manager.updateTextBlkList()
        current_blks = test_proj.current_block_list()
        self.assertEqual(len(current_blks), 2, "Second text block was dropped due to zip() mismatch!")
        print("   ✓ Case 5B (updateTextBlkList zero-drop guarantee on list length mismatch): PASSED")


if __name__ == "__main__":
    unittest.main(verbosity=2)
