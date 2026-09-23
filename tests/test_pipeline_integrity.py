import sys
import time
import uuid
import numpy as np

sys.path.insert(0, r"d:\BallonsTranslator")

from utils import shared
from utils.proj_imgtrans import ProjImgTrans
from utils.textblock import TextBlock
from ui.module_manager import ImgtransThread, ModuleManager
from PyQt6.QtWidgets import QApplication

app = QApplication.instance() or QApplication(sys.argv)

def test_pipeline_signals_and_race_conditions():
    print("🚀 Starting Pipeline Coordinator & Signal Synchronization Test...")

    # Create project with 2 pages
    proj = ProjImgTrans()
    proj.directory = "."
    h, w = 400, 300
    dummy_img = np.full((h, w, 3), 255, dtype=np.uint8)
    
    blk1 = TextBlock(xyxy=[50, 50, 200, 100], lines=[[50, 50, 200, 50, 200, 100, 50, 100]])
    blk1.text = ["HELLO WORLD"]
    blk2 = TextBlock(xyxy=[30, 150, 180, 220], lines=[[30, 150, 180, 150, 180, 220, 30, 220]])
    blk2.text = ["TEST DIALOGUE"]

    proj.pages = {
        "page_001.png": [blk1],
        "page_002.png": [blk2]
    }
    proj.img_dict = {
        "page_001.png": dummy_img,
        "page_002.png": dummy_img
    }
    proj._pagename2idx = {
        "page_001.png": 0,
        "page_002.png": 1
    }
    proj._image_info = {
        "page_001.png": {"finish_code": 0},
        "page_002.png": {"finish_code": 0}
    }
    proj.read_img = lambda name: dummy_img
    proj.save_mask = lambda name, m: None
    proj.save_inpainted = lambda name, inp: None

    # Track signals emitted
    page_trans_finished_emits = []
    pipeline_finished_emits = []
    pipeline_stopped_emits = []

    # Mock components
    class MockDetector:
        def detect(self, img, proj):
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            return mask, [blk1]

    class MockOCR:
        def run_ocr(self, img, blk_list):
            for blk in blk_list:
                blk.text = ["MOCK OCR TEXT"]

    class MockInpainter:
        def inpaint(self, img, mask, blk_list=None):
            return img.copy()

    class MockTranslator:
        def __init__(self):
            self.low_vram_mode = False
        def is_computational_intensive(self):
            return False
        def translate_textblk_lst(self, blk_list, page_context=None):
            for blk in blk_list:
                blk.translation = "BẢN DỊCH THỬ NGHIỆM"

    class MockThread:
        def __init__(self):
            self.num_process_pages = 2
            self.finished_counter = 2
        def isRunning(self):
            return False
        def quit(self):
            pass
        def requestStop(self):
            pass

    class MockDetectorThread(MockThread):
        def __init__(self):
            super().__init__()
            self.textdetector = MockDetector()

    class MockOCRThread(MockThread):
        def __init__(self):
            super().__init__()
            self.ocr = MockOCR()

    class MockInpaintThread(MockThread):
        def __init__(self):
            super().__init__()
            self.inpainter = MockInpainter()

    class MockTranslateThread(MockThread):
        def __init__(self):
            super().__init__()
            self.translator = MockTranslator()
            self.module = self.translator
            class MockSig:
                def connect(self, fn):
                    pass
            self.module_thread_stopped = MockSig()
            self.progress_changed = MockSig()
            self.finish_translate_page = MockSig()
        def runTranslatePipeline(self, proj):
            pass
        def push_pagekey_queue(self, pagekey):
            pass
        def _extract_page_context(self, pages, name):
            return None

    td_th = MockDetectorThread()
    ocr_th = MockOCRThread()
    trans_th = MockTranslateThread()
    inp_th = MockInpaintThread()

    imgtrans_th = ImgtransThread(td_th, ocr_th, trans_th, inp_th)

    def on_page_fin(idx):
        page_trans_finished_emits.append(idx)

    def on_pipe_fin():
        pipeline_finished_emits.append(True)

    def on_pipe_stop():
        pipeline_stopped_emits.append(True)

    imgtrans_th.page_trans_finished.connect(on_page_fin)
    imgtrans_th.pipeline_finished.connect(on_pipe_fin)
    imgtrans_th.pipeline_stopped.connect(on_pipe_stop)

    # 1. Test standard runImgtransPipeline execution
    job_id = uuid.uuid4().hex
    imgtrans_th.imgtrans_proj = proj
    imgtrans_th.pages_to_process = None
    imgtrans_th.num_pages = len(proj.pages)
    imgtrans_th.active_job_id = job_id
    imgtrans_th.process_idx_to_page_idx = {0: 0, 1: 1}

    # Run direct pipeline job
    imgtrans_th._imgtrans_pipeline()

    print(f"Page trans finished emissions: {page_trans_finished_emits}")
    print(f"Pipeline finished emissions: {len(pipeline_finished_emits)}")

    # Verify: Exactly 2 page finishes (one for each page in order) and exactly 1 pipeline finish
    assert page_trans_finished_emits == [0, 1], f"Expected [0, 1], got {page_trans_finished_emits}"
    assert len(pipeline_finished_emits) == 1, f"Expected 1 completion, got {len(pipeline_finished_emits)}"
    assert len(pipeline_stopped_emits) == 0, "No stop expected"

    # 2. Test cancellation behavior
    page_trans_finished_emits.clear()
    pipeline_finished_emits.clear()
    pipeline_stopped_emits.clear()

    imgtrans_th.stop_requested = True
    imgtrans_th._imgtrans_pipeline()

    # When stopped, no completion should be emitted, only pipeline_stopped
    assert len(page_trans_finished_emits) == 0, "No page finishes should be emitted when cancelled"
    assert len(pipeline_finished_emits) == 0, "No pipeline finish should be emitted when cancelled"
    assert len(pipeline_stopped_emits) == 1, "Pipeline stopped should be emitted"

    print("✅ TEST PASSED: Pipeline Coordinator & Signal Synchronization verified!")

if __name__ == "__main__":
    test_pipeline_signals_and_race_conditions()
