import subprocess
import sys
import time
import os
from pathlib import Path

# Force UTF-8 stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

PYTHON_EXE = sys.executable

TEST_SUITES = [
    ("1. Run vs Continue Branching Suite", "scripts/test_run_and_continue.py"),
    ("2. Ctrl+S Save Text Box Preservation", "scripts/test_ctrl_s_save.py"),
    ("3. Context-Aware Manga Translation", "scripts/test_context_aware_translation.py"),
    ("4. UI State Synchronization Suite", "scripts/test_ui_sync.py"),
    ("5. Translation Proxy Full Test Suite", "scripts/test_translation_proxy_full.py"),
    ("6. Deep Context & Pronoun Locking", "tests/test_deep_context_translation.py"),
    ("7. Vietnamese Font Rendering Suite", "scripts/test_fonts_vietnamese.py"),
    ("8. OCR Multi-View & Anomaly Scoring Suite", "tests/test_ocr_validator.py"),
    ("9. Pipeline Coordinator & Atomic Commit Suite", "tests/test_pipeline_integrity.py"),
    ("10. Real Manga Inpainting & Pipeline", "scripts/test_real_manga.py"),
]

def main():
    print("=" * 80)
    print("   BALLOONSTRANSLATOR COMPREHENSIVE REGRESSION & PIPELINE TEST SUITE")
    print("=" * 80)
    
    results = []
    total_start = time.perf_counter()
    
    for name, script_path in TEST_SUITES:
        full_path = PROJECT_ROOT / script_path
        if not full_path.exists():
            print(f"\n[SKIP] {name}: File not found ({script_path})")
            results.append((name, "SKIPPED", 0.0, "File not found"))
            continue
            
        print(f"\n>> RUNNING: {name} ({script_path})...")
        t0 = time.perf_counter()
        
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        
        proc = subprocess.run(
            [PYTHON_EXE, str(full_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env
        )
        duration = (time.perf_counter() - t0)
        
        if proc.returncode == 0:
            print(f"   ✓ PASSED in {duration:.2f}s")
            results.append((name, "PASSED", duration, ""))
        else:
            print(f"   ✗ FAILED with code {proc.returncode} in {duration:.2f}s")
            err_msg = proc.stderr.strip() or proc.stdout.strip()
            lines = err_msg.splitlines()
            snippet = "\n".join(lines[-8:]) if len(lines) > 8 else err_msg
            print(f"     Error details:\n{snippet}")
            results.append((name, "FAILED", duration, snippet))
            
    total_duration = time.perf_counter() - total_start
    
    print("\n" + "=" * 80)
    print(f"   FINAL TEST SUMMARY ({len(results)} SUITES)")
    print("=" * 80)
    print(f"{'Test Suite':<45} | {'Status':<10} | {'Duration':<10}")
    print("-" * 80)
    
    passed_count = 0
    for name, status, dur, _ in results:
        status_str = f"✓ {status}" if status == "PASSED" else f"✗ {status}"
        print(f"{name:<45} | {status_str:<10} | {dur:6.2f}s")
        if status == "PASSED":
            passed_count += 1
            
    print("-" * 80)
    print(f"Overall Result: {passed_count}/{len(results)} Suites Passed ({(passed_count/len(results))*100:.1f}%) in {total_duration:.2f}s")
    print("=" * 80)
    
    if passed_count == len(results):
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == '__main__':
    main()
