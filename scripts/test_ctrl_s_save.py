import sys
import os
import unittest
import numpy as np

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PyQt6.QtWidgets import QApplication
app = QApplication.instance() or QApplication(sys.argv)

from ui.mainwindow import MainWindow
from utils.proj_imgtrans import ProjImgTrans, RunStatus
from utils.textblock import TextBlock
from utils.config import pcfg
from modules.base import init_module_registries


class TestCtrlSSave(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_module_registries()

    def setUp(self):
        init_module_registries()
        self.win = MainWindow(app, pcfg)
        self.proj = ProjImgTrans()
        self.proj.directory = "D:\\BallonsTranslator\\data\\real_manga_samples"
        
        # Add sample page with block
        blk = TextBlock([10, 10, 100, 50], text=["Hello manga"])
        blk.translation = "Xin chào manga"
        self.proj.pages = {"sample_01.png": [blk]}
        self.proj._image_info = {"sample_01.png": {"finish_code": RunStatus.FIN_ALL, "width": 800, "height": 1200}}
        self.proj.current_img = "sample_01.png"
        self.proj.img_array = np.zeros((1200, 800, 3), dtype=np.uint8)
        self.proj.inpainted_array = np.zeros((1200, 800, 3), dtype=np.uint8)
        
        pcfg.let_uppercase_flag = False
        self.win.imgtrans_proj = self.proj
        self.win.canvas.imgtrans_proj = self.proj
        self.win.st_manager.imgtrans_proj = self.proj
        
        # Load text blocks into scene
        self.win.st_manager.updateSceneTextitems()

    def test_01_manual_save_preserves_text_layer_and_translations(self):
        """
        Kiểm tra khi bấm Ctrl + S (manual_save):
        - Text layer trên canvas vẫn hiển thị (isVisible == True)
        - Nội dung dịch của TextBlkItem không bị biến mất
        - Khối điều khiển viền ô chữ (txtblkShapeControl) được phục hồi
        """
        # Ensure block item loaded
        self.assertEqual(len(self.win.st_manager.textblk_item_list), 1)
        blk_item = self.win.st_manager.textblk_item_list[0]
        self.assertEqual(blk_item.toPlainText(), "Xin chào manga")

        # Select block item
        blk_item.setSelected(True)
        self.win.st_manager.txtblkShapeControl.setBlkItem(blk_item)

        # Trigger manual save (Ctrl + S)
        self.win.manual_save()

        # Assert text layer is visible
        self.assertTrue(self.win.canvas.textLayer.isVisible(), "TextLayer must remain visible after Ctrl+S save!")
        
        # Assert text translation was not wiped
        self.assertEqual(blk_item.toPlainText(), "Xin chào manga", "Text item translation was wiped out!")
        self.assertEqual(self.proj.pages["sample_01.png"][0].translation, "Xin chào manga", "Project block translation was wiped out!")

        # Assert shape control is still attached
        self.assertIsNotNone(self.win.st_manager.txtblkShapeControl.blk_item, "txtblkShapeControl lost its attached block item!")
        self.assertEqual(self.win.st_manager.txtblkShapeControl.blk_item, blk_item)

        print("[PASS] Test 1: Manual save (Ctrl+S) preserved text layer, translations, and active text box frame.")

    def test_02_manual_save_syncs_from_transpair_if_needed(self):
        """
        Kiểm tra nếu người dùng gõ bản dịch ở ô bên phải (TransPairWidget) rồi bấm Ctrl + S:
        - Bản dịch được đồng bộ sang TextBlkItem và lưu vào project
        """
        blk_item = self.win.st_manager.textblk_item_list[0]
        blk_item.setPlainText("") # simulate empty canvas text item
        pair_widget = self.win.st_manager.pairwidget_list[0]
        pair_widget.e_trans.setPlainText("Bản dịch mới từ ô nhập liệu")

        self.win.manual_save()

        self.assertEqual(blk_item.toPlainText(), "Bản dịch mới từ ô nhập liệu")
        self.assertEqual(self.proj.pages["sample_01.png"][0].translation, "Bản dịch mới từ ô nhập liệu")

        print("[PASS] Test 2: Manual save synced right panel translation to canvas and project.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
