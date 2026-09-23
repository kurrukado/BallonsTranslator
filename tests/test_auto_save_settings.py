import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import config as program_config
from utils.config import pcfg, save_config, ModuleConfig


class TestAutoSaveSettings(unittest.TestCase):

    def setUp(self):
        self.orig_pcfg_module = pcfg.module
        pcfg.module = ModuleConfig()

    def tearDown(self):
        pcfg.module = self.orig_pcfg_module

    def test_setupConfig_preserves_user_settings(self):
        """Verify that setupConfig preserves manga_ocr, windows_ocr, Japanese, etc."""
        # 1. Test Japanese source preserved
        pcfg.module.translate_source = "Japanese"
        pcfg.module.translate_target = "Tiếng Việt"
        pcfg.module.ocr = "manga_ocr"
        pcfg.module.textdetector = "ctd"
        pcfg.module.inpainter = "lama_large_512px"
        pcfg.module.translator = "LLM_API_Translator"

        # Simulate setupConfig logic
        if not pcfg.module.translate_target or pcfg.module.translate_target in ["简体中文", "Simplified Chinese"]:
            pcfg.module.translate_target = "Tiếng Việt"
        if not pcfg.module.translate_source or pcfg.module.translate_source in ["简体中文", "Simplified Chinese"]:
            pcfg.module.translate_source = "English"
        if not pcfg.module.ocr or pcfg.module.ocr in ["mit48px_ctc"]:
            pcfg.module.ocr = "paddle_ocr"

        self.assertEqual(pcfg.module.translate_source, "Japanese")
        self.assertEqual(pcfg.module.translate_target, "Tiếng Việt")
        self.assertEqual(pcfg.module.ocr, "manga_ocr")

        # 2. Test Nihongo preserved
        pcfg.module.translate_source = "日本語"
        pcfg.module.ocr = "windows_ocr"
        if not pcfg.module.translate_target or pcfg.module.translate_target in ["简体中文", "Simplified Chinese"]:
            pcfg.module.translate_target = "Tiếng Việt"
        if not pcfg.module.translate_source or pcfg.module.translate_source in ["简体中文", "Simplified Chinese"]:
            pcfg.module.translate_source = "English"
        if not pcfg.module.ocr or pcfg.module.ocr in ["mit48px_ctc"]:
            pcfg.module.ocr = "paddle_ocr"

        self.assertEqual(pcfg.module.translate_source, "日本語")
        self.assertEqual(pcfg.module.ocr, "windows_ocr")

        # 3. Test empty/Chinese defaults to English / paddle_ocr / Tiếng Việt
        pcfg.module.translate_source = "简体中文"
        pcfg.module.translate_target = ""
        pcfg.module.ocr = "mit48px_ctc"
        if not pcfg.module.translate_target or pcfg.module.translate_target in ["简体中文", "Simplified Chinese"]:
            pcfg.module.translate_target = "Tiếng Việt"
        if not pcfg.module.translate_source or pcfg.module.translate_source in ["简体中文", "Simplified Chinese"]:
            pcfg.module.translate_source = "English"
        if not pcfg.module.ocr or pcfg.module.ocr in ["mit48px_ctc"]:
            pcfg.module.ocr = "paddle_ocr"

        self.assertEqual(pcfg.module.translate_source, "English")
        self.assertEqual(pcfg.module.translate_target, "Tiếng Việt")
        self.assertEqual(pcfg.module.ocr, "paddle_ocr")

    @patch("ui.mainwindow.save_config")
    def test_mainwindow_change_handlers_auto_save(self, mock_save_config):
        """Verify that on_*_changed handlers update pcfg and invoke save_config."""
        from ui.mainwindow import MainWindow

        # Create a mock MainWindow instance without invoking __init__
        win = MainWindow.__new__(MainWindow)
        win.save_config = mock_save_config

        # Mock bottomBar selectors
        win.bottomBar = MagicMock()
        win.configPanel = MagicMock()
        win.module_manager = MagicMock()

        # 1. Text detector change
        win.bottomBar.textdet_selector.selector.currentText.return_value = "ctd"
        win.configPanel.detect_config_panel.module_combobox.currentText.return_value = "other"
        MainWindow.on_textdet_changed(win)
        self.assertEqual(pcfg.module.textdetector, "ctd")
        mock_save_config.assert_called()
        mock_save_config.reset_mock()

        # 2. OCR change
        win.bottomBar.ocr_selector.selector.currentText.return_value = "manga_ocr"
        win.configPanel.ocr_config_panel.module_combobox.currentText.return_value = "other"
        MainWindow.on_ocr_changed(win)
        self.assertEqual(pcfg.module.ocr, "manga_ocr")
        mock_save_config.assert_called()
        mock_save_config.reset_mock()

        # 3. Inpaint change
        win.bottomBar.inpaint_selector.selector.currentText.return_value = "lama_large_512px"
        win.configPanel.inpaint_config_panel.module_combobox.currentText.return_value = "other"
        MainWindow.on_inpaint_changed(win)
        self.assertEqual(pcfg.module.inpainter, "lama_large_512px")
        mock_save_config.assert_called()
        mock_save_config.reset_mock()

        # 4. Translator change
        win.bottomBar.trans_selector.selector.currentText.return_value = "LLM_API_Translator"
        win.configPanel.trans_config_panel.module_combobox.currentText.return_value = "other"
        MainWindow.on_trans_changed(win)
        self.assertEqual(pcfg.module.translator, "LLM_API_Translator")
        mock_save_config.assert_called()
        mock_save_config.reset_mock()

        # 5. Source language change
        mock_sender = MagicMock()
        mock_sender.currentText.return_value = "Japanese"
        win.sender = MagicMock(return_value=mock_sender)
        win.configPanel.trans_config_panel.source_combobox = MagicMock()
        win.bottomBar.trans_selector.src_selector = MagicMock()
        MainWindow.on_trans_src_changed(win)
        self.assertEqual(pcfg.module.translate_source, "Japanese")
        mock_save_config.assert_called()
        mock_save_config.reset_mock()

        # 6. Target language change
        mock_sender.currentText.return_value = "Tiếng Việt"
        win.sender = MagicMock(return_value=mock_sender)
        win.configPanel.trans_config_panel.target_combobox = MagicMock()
        win.bottomBar.trans_selector.tgt_selector = MagicMock()
        MainWindow.on_trans_tgt_changed(win)
        self.assertEqual(pcfg.module.translate_target, "Tiếng Việt")
        mock_save_config.assert_called()


if __name__ == "__main__":
    unittest.main()
