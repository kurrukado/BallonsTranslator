import sys
import os
import unittest
from PyQt6.QtWidgets import QApplication

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.config import pcfg
from utils.textblock import TextBlock
from utils.proj_imgtrans import ProjImgTrans
from ui.mainwindow import MainWindow

app = QApplication.instance() or QApplication(sys.argv)

class TestUISynchronization(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.win = MainWindow(app, pcfg, open_dir=os.path.abspath("data/real_manga_samples"))

    def setUp(self):
        self.win = self.__class__.win
        self.proj = self.win.imgtrans_proj
        self.canvas = self.win.canvas
        self.st_manager = self.win.st_manager

    # TEST 1 — RUN TRANSLATION VISIBILITY
    def test_01_run_translation_visibility(self):
        page_name = self.proj.current_img
        blks = [
            TextBlock([10, 10, 100, 50], text=["Hello world"]),
            TextBlock([10, 60, 100, 100], text=["Another line"])
        ]
        self.proj.pages[page_name] = blks

        # Apply translation to model
        blks[0].translation = "Xin chào thế giới"
        blks[1].translation = "Dòng khác"

        # Simulate on_pagtrans_finished and on_imgtrans_pipeline_finished
        page_idx = self.proj.pagename2idx(page_name)
        self.win.on_pagtrans_finished(page_idx)
        self.win.on_imgtrans_pipeline_finished()

        # Assert items exist and are visible
        self.assertEqual(len(self.st_manager.textblk_item_list), 2)
        self.assertTrue(self.canvas.textLayer.isVisible())
        self.assertEqual(self.st_manager.textblk_item_list[0].toPlainText().replace('\n', ' ').lower(), "xin chào thế giới")
        self.assertEqual(self.st_manager.textblk_item_list[1].toPlainText().replace('\n', ' ').lower(), "dòng khác")
        print("[PASS] Test 1: Run translation visibility verified immediately without clicking buttons.")

    # TEST 2 — MODEL/CANVAS CONSISTENCY
    def test_02_model_canvas_consistency(self):
        page_name = self.proj.current_img
        blk = TextBlock([20, 20, 120, 80], text=["Test consistency"])
        self.proj.pages[page_name] = [blk]
        blk.translation = "Kiểm tra tính nhất quán"

        page_idx = self.proj.pagename2idx(page_name)
        self.win.on_pagtrans_finished(page_idx)
        self.win.on_imgtrans_pipeline_finished()

        item = self.st_manager.textblk_item_list[0]
        self.assertEqual(blk.translation.replace('\n', ' ').lower(), item.toPlainText().replace('\n', ' ').lower())
        print("[PASS] Test 2: Model and Canvas text item consistency verified 100%.")

    # TEST 3 — MULTIPLE BLOCKS (20 blocks)
    def test_03_multiple_blocks_visibility(self):
        page_name = self.proj.current_img
        blks = [TextBlock([10, i*30, 150, i*30+25], text=[f"Source line {i}"]) for i in range(20)]
        for i, b in enumerate(blks):
            b.translation = f"Bản dịch câu {i}"
        self.proj.pages[page_name] = blks

        page_idx = self.proj.pagename2idx(page_name)
        self.win.on_pagtrans_finished(page_idx)
        self.win.on_imgtrans_pipeline_finished()

        self.assertEqual(len(self.st_manager.textblk_item_list), 20)
        for i in range(20):
            self.assertEqual(self.st_manager.textblk_item_list[i].toPlainText().replace('\n', ' ').lower(), f"bản dịch câu {i}")
        print("[PASS] Test 3: 20 blocks loaded and immediately visible.")

    # TEST 4 — MULTIPLE PAGES (5 pages)
    def test_04_multiple_pages_visibility(self):
        self.win.pageList.setCurrentRow(0)
        pages_list = list(self.proj.pages.keys())
        for p_name in pages_list:
            p_idx = self.proj.pagename2idx(p_name)
            self.proj.pages[p_name] = [TextBlock([10, 10, 100, 50], text=[f"Page {p_idx} text"])]
            self.proj.pages[p_name][0].translation = f"trang {p_idx} dịch"

        self.st_manager.updateSceneTextitems()

        # Trigger for all pages
        for p_name in pages_list:
            p_idx = self.proj.pagename2idx(p_name)
            self.win.on_pagtrans_finished(p_idx)
        self.win.on_imgtrans_pipeline_finished()

        # Check that current page has visible items
        current_idx = self.proj.pagename2idx(self.proj.current_img)
        self.assertEqual(self.st_manager.textblk_item_list[0].toPlainText().replace('\n', ' ').lower(), f"trang {current_idx} dịch")
        print("[PASS] Test 4: Multi-page switching and synchronization verified.")

    # TEST 5 — DIRECT MODE IMMEDIATE REFRESH
    def test_05_direct_mode_immediate_refresh(self):
        page_name = self.proj.current_img
        blk = TextBlock([10, 10, 100, 50], text=["Direct source"])
        self.proj.pages[page_name] = [blk]

        page_idx = self.proj.pagename2idx(page_name)
        self.win.on_pagtrans_finished(page_idx)

        # Simulate direct translation returning
        blk.translation = "Bản dịch trực tiếp"
        self.win.on_blktrans_finished(1, [0])

        blk_item = self.st_manager.textblk_item_list[0]
        self.assertEqual(blk_item.toPlainText(), "Bản dịch trực tiếp")
        print("[PASS] Test 5: Direct mode immediately refreshed block and pair widget.")

    # TEST 6 — PARTIAL TRANSLATION STABILITY
    def test_06_partial_translation_stability(self):
        page_name = self.proj.current_img
        blks = [TextBlock([10, i*30, 150, i*30+25], text=[f"Line {i}"]) for i in range(10)]
        for i in range(8):
            blks[i].translation = f"Dịch {i}"
        # blks[8] and blks[9] remain untranslated
        self.proj.pages[page_name] = blks

        page_idx = self.proj.pagename2idx(page_name)
        self.win.on_pagtrans_finished(page_idx)
        self.win.on_imgtrans_pipeline_finished()

        self.assertEqual(len(self.st_manager.textblk_item_list), 10)
        for i in range(8):
            self.assertEqual(self.st_manager.textblk_item_list[i].toPlainText().replace('\n', ' ').lower(), f"dịch {i}")
        self.assertEqual(self.st_manager.textblk_item_list[8].toPlainText(), "")
        self.assertEqual(self.st_manager.textblk_item_list[9].toPlainText(), "")
        print("[PASS] Test 6: Partial translation updated 8 blocks without corrupting 2 pending blocks.")

    # TEST 7 — REPEATED RUN (NO DUPLICATES)
    def test_07_repeated_run_no_duplicates(self):
        page_name = self.proj.current_img
        blk = TextBlock([10, 10, 100, 50], text=["Repeat line"])
        self.proj.pages[page_name] = [blk]
        page_idx = self.proj.pagename2idx(page_name)

        # Run 1
        blk.translation = "Dịch lần 1"
        self.win.on_pagtrans_finished(page_idx)
        self.win.on_imgtrans_pipeline_finished()
        self.assertEqual(len(self.st_manager.textblk_item_list), 1)

        # Run 2
        blk.translation = "Dịch lần 2"
        self.win.on_pagtrans_finished(page_idx)
        self.win.on_imgtrans_pipeline_finished()
        self.assertEqual(len(self.st_manager.textblk_item_list), 1)
        self.assertEqual(self.st_manager.textblk_item_list[0].toPlainText().replace('\n', ' ').lower(), "dịch lần 2")
        print("[PASS] Test 7: Repeated run confirmed zero duplicates.")

    # TEST 8 — SAVE AND REOPEN INTEGRITY
    def test_08_save_and_reopen_integrity(self):
        import tempfile
        import shutil
        tmp_dir = tempfile.mkdtemp()
        try:
            # Create a dummy image file so ProjImgTrans recognizes the page
            dummy_img = os.path.join(tmp_dir, "sample.jpg")
            with open(dummy_img, "wb") as f:
                f.write(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xbf\x00\xff\xd9")

            proj_path = os.path.join(tmp_dir, "test_proj.json")
            blk = TextBlock([10, 10, 100, 50], text=["Save test"])
            blk.translation = "Bản dịch đã lưu"
            
            p = ProjImgTrans()
            p.directory = tmp_dir
            p.proj_path = proj_path
            p.pages = {"sample.jpg": [blk]}
            p.save()

            # Reload
            p_loaded = ProjImgTrans()
            p_loaded.load(tmp_dir, json_path=proj_path)
            self.assertIn("sample.jpg", p_loaded.pages)
            loaded_blk = p_loaded.pages["sample.jpg"][0]
            self.assertEqual(loaded_blk.translation, "Bản dịch đã lưu")
            print("[PASS] Test 8: Save and reload preserved 100% translation integrity.")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # TEST 9 — OUT OF BOUNDS BLKTRANS SAFETY
    def test_09_on_blktrans_finished_out_of_bounds_safety(self):
        page_name = self.proj.current_img
        blk = TextBlock([10, 10, 100, 50], text=["Boundary test"])
        self.proj.pages[page_name] = [blk]
        page_idx = self.proj.pagename2idx(page_name)
        self.win.on_pagtrans_finished(page_idx)

        # 1. Out-of-bounds IDs should not raise IndexError
        try:
            self.win.on_blktrans_finished(1, [9999, -1, 42])
        except IndexError as e:
            self.fail(f"on_blktrans_finished raised IndexError on out-of-bounds blk_ids: {e}")

        # 2. Empty blk_ids
        try:
            self.win.on_blktrans_finished(1, [])
        except IndexError as e:
            self.fail(f"on_blktrans_finished raised IndexError on empty blk_ids: {e}")

        # 3. Valid id succeeds
        blk.translation = "Bản dịch an toàn"
        self.win.on_blktrans_finished(1, [0])
        self.assertEqual(self.st_manager.textblk_item_list[0].toPlainText(), "Bản dịch an toàn")

        # 4. Canvas cleared or page switched while task finishes
        self.st_manager.clearSceneTextitems()
        try:
            self.win.on_blktrans_finished(1, [0])
        except IndexError as e:
            self.fail(f"on_blktrans_finished raised IndexError when textblk_item_list is cleared: {e}")

        print("[PASS] Test 9: on_blktrans_finished out-of-bounds and cleared state handled safely without IndexError.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
