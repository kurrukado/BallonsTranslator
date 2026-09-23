import sys
import os
import unittest
from PyQt6.QtWidgets import QApplication

# Set headless platform offscreen for testing in CLI
os.environ["QT_QPA_PLATFORM"] = "offscreen"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.config import pcfg
from ui.scenetext_manager import SceneTextManager
from ui.canvas import Canvas
from utils.proj_imgtrans import ProjImgTrans
from utils.textblock import TextBlock

app = QApplication.instance() or QApplication(sys.argv)

print("[INFO] Testing Canvas and SceneTextManager synchronization...")

# Create mock objects
proj = ProjImgTrans()
proj.pages = {"001.jpg": [TextBlock([10, 10, 100, 100], text=["Hello"])]}
proj.set_current_img("001.jpg")

canvas = Canvas(proj)
st_manager = SceneTextManager(canvas)

print("Before Translation:")
print("  canvas.textLayer.isVisible():", canvas.textLayer.isVisible())
print("  st_manager text items count:", len(st_manager.textblk_item_list))

# Simulate Translation returning
blk = proj.pages["001.jpg"][0]
blk.translation = "Xin chào"

# Check if canvas automatically knows about translation
print("After Translation without update:")
print("  blk.translation in model:", blk.translation)
if st_manager.textblk_item_list:
    print("  st_manager item 0 text:", st_manager.textblk_item_list[0].toPlainText())
else:
    print("  st_manager has NO text items loaded on Canvas yet!")

# Now simulate what on_pagtrans_finished does:
st_manager.updateSceneTextitems()
canvas.textLayer.show()
canvas.update()

print("After st_manager.updateSceneTextitems() and textLayer.show():")
print("  canvas.textLayer.isVisible():", canvas.textLayer.isVisible())
print("  st_manager text items count:", len(st_manager.textblk_item_list))
print("  st_manager item 0 text:", st_manager.textblk_item_list[0].toPlainText())
