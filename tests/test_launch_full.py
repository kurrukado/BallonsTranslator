import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

# Setup Qt
from qtpy.QtWidgets import QApplication
from qtpy.QtCore import QLocale
import utils.shared as shared
from utils.config import load_config, pcfg
from modules.base import init_module_registries

app = QApplication(sys.argv)
shared.DEFAULT_DISPLAY_LANG = 'en_US'
shared.PROGRAM_PATH = str(PROJECT_ROOT)
shared.load_cache()
load_config('config/config.json')

init_module_registries()

from ui.mainwindow import MainWindow
mw = MainWindow(app, pcfg)
print("[TEST SUCCESS] MainWindow instantiated and loaded all modules successfully!")
mw.close()
sys.exit(0)
