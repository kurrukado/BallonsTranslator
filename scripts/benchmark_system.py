import sys, os, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

import torch

t0 = time.perf_counter()
import modules.textdetector.detector_ctd
import modules.ocr.ocr_manga
import modules.inpaint.base
import modules.translators.trans_llm_api
t_import = (time.perf_counter() - t0) * 1000

from modules.textdetector.base import TEXTDETECTORS
from modules.ocr.base import OCR
from modules.inpaint.base import INPAINTERS

vram_idle = 0
vram_peak = 0
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    vram_idle = torch.cuda.memory_allocated() / (1024 ** 2)
    
    ctd = TEXTDETECTORS.module_dict['ctd'](device='cuda', detect_size=1024)
    ctd.load_model()
    ocr = OCR.module_dict['manga_ocr'](device='cuda')
    ocr.load_model()
    lama = INPAINTERS.module_dict['lama_large_512px'](device='cuda', inpaint_size=1024)
    lama.load_model()
    
    vram_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

print("=" * 60)
print("  SYSTEM PERFORMANCE & MEMORY BENCHMARK REPORT")
print("=" * 60)
print(f"• Module Import Time (Cold Start) : {t_import:.1f} ms")
print(f"• GPU Idle Allocation             : {vram_idle:.1f} MB")
print(f"• Peak GPU Memory (3 Core Models) : {vram_peak:.1f} MB ({vram_peak/1024:.2f} GB / 4.00 GB VRAM)")
print("=" * 60)
