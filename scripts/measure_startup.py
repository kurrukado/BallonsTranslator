import time
import sys
import os
sys.path.insert(0, '.')

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

print('=' * 80)
print('  BALLOONSTRANSLATOR: STARTUP LATENCY BENCHMARK')
print('=' * 80)

t_start_total = time.perf_counter()

# Phase 1: Python stdlib
t0 = time.perf_counter()
import json, urllib.request, subprocess, os.path as osp
t_stdlib = time.perf_counter() - t0
print(f'• [Phase 1] Stdlib & Core Modules        : {t_stdlib*1000:.2f} ms')

# Phase 2: PyQt6 / GUI Framework
t0 = time.perf_counter()
import qtpy
from qtpy.QtWidgets import QApplication
from qtpy.QtCore import QTranslator, QLocale, Qt
from qtpy.QtGui import QIcon, QFontDatabase, QGuiApplication, QFont
t_qt = time.perf_counter() - t0
print(f'• [Phase 2] PyQt6 & Qt Framework         : {t_qt*1000:.2f} ms')

# Phase 3: PyTorch & GPU Runtime
t0 = time.perf_counter()
import torch
cuda_avail = torch.cuda.is_available()
t_torch = time.perf_counter() - t0
print(f'• [Phase 3] PyTorch Runtime (CUDA={cuda_avail}) : {t_torch*1000:.2f} ms')

# Phase 4: Gemini Proxy Check & API Key Sync
t0 = time.perf_counter()
from utils.gemini_proxy_launcher import ensure_gemini_proxy_running
proxy_status = ensure_gemini_proxy_running()
t_proxy = time.perf_counter() - t0
print(f'• [Phase 4] Gemini Proxy Check & Sync    : {t_proxy*1000:.2f} ms (Status: {proxy_status})')

# Phase 5: Module Registries & Cache
t0 = time.perf_counter()
import utils.shared as shared
shared.load_cache()
from modules.base import init_module_registries
from modules.prepare_local_files import prepare_local_files_forall
init_module_registries()
prepare_local_files_forall()
t_registries = time.perf_counter() - t0
print(f'• [Phase 5] Module Registries & Files    : {t_registries*1000:.2f} ms')

# Phase 6: MainWindow UI & Widget Construction
t0 = time.perf_counter()
from utils.config import pcfg, load_config
from ui.mainwindow import MainWindow

app = QApplication.instance() or QApplication(['', '-platform', 'offscreen'])
load_config()
win = MainWindow(app, pcfg)
t_ui = time.perf_counter() - t0
print(f'• [Phase 6] MainWindow UI & Components   : {t_ui*1000:.2f} ms')

t_total = time.perf_counter() - t_start_total
print('=' * 80)
print(f'✓ TOTAL STARTUP TIME: {t_total:.2f} s ({t_total*1000:.2f} ms)')
print('=' * 80)
