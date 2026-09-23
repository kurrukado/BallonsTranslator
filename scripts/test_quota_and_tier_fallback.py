import os
import sys
import time
import json

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.quota_tracker import QUOTA_TRACKER, QuotaTracker, CACHE_DIR, QUOTA_STATE_FILE, RESUME_STATE_FILE
from modules.translators.translation_proxy import TranslationProxy
from modules.translators.trans_llm_api import LLM_API_Translator

def test_quota_tracker_unit():
    print("\n" + "=" * 80)
    print("🧪 1. TESTING QUOTA TRACKER UNIT & TIER ORDER")
    print("=" * 80)

    # Clear spurious entries
    QUOTA_TRACKER.exhausted_today.clear()
    QUOTA_TRACKER._save_state()

    # Verify priority models
    candidates = QUOTA_TRACKER.get_candidate_models(preferred_model="gemini-3.8-flash")
    print(f"Candidate models for standard batch: {candidates}")
    assert candidates[0] == "gemini-3.8-flash", f"Expected gemini-3.8-flash first, got {candidates[0]}"
    assert "gemini-3.7-flash" in candidates, "Expected gemini-3.7-flash in candidates"
    assert "gemini-3.6-flash" in candidates, "Expected gemini-3.6-flash in candidates"
    assert "gemini-3.5-flash" in candidates, "Expected gemini-3.5-flash in candidates"
    assert "gemini-3.5-flash-lite" in candidates, "Expected gemini-3.5-flash-lite in candidates"
    
    # Verify blacklisted models are not included
    assert "gemini-2.5-pro" not in candidates, "gemini-2.5-pro must be excluded"
    assert "gemini-3.1-flash-lite" not in candidates, "gemini-3.1-flash-lite must be excluded"
    assert "gemini-2.5-flash-lite" not in candidates, "gemini-2.5-flash-lite must be excluded"
    assert "gemini-3-flash" not in candidates, "gemini-3-flash must be excluded"

    print("✓ Model tier ordering and blacklist filtering verified.")

def test_long_chapter_multi_subbatch():
    print("\n" + "=" * 80)
    print("🚀 2. TESTING LONG CHAPTER (>150 DIALOGUES) DISPATCH & THROTTLE")
    print("=" * 80)

    proxy = TranslationProxy()
    # Create 6 pages, 30 dialogues each = 180 dialogues total (>150 dialogues)
    page_indices = list(range(1, 7))
    proxy.start_session(page_indices, timeout_seconds=120.0)

    for p_idx in page_indices:
        dialogues = []
        for d_id in range(1, 31):
            dialogues.append({
                "id": d_id,
                "text": f"Page {p_idx} dialogue sentence #{d_id}: This is a long conversational test line for chapter translation.",
                "reading_order": d_id,
                "block_type": "DIALOGUE" if d_id % 3 != 0 else "NARRATION"
            })
        proxy.mark_page_completed(p_idx, dialogues, page_name=f"page_{p_idx:03d}.png")

    total_count = proxy.total_dialogues_count()
    print(f"Total buffered dialogues: {total_count} across {len(page_indices)} pages.")
    assert total_count == 180, f"Expected 180 dialogues, got {total_count}"

    sub_payloads = proxy.build_sub_batch_payloads(src_lang="English", tgt_lang="Tiếng Việt", max_dialogues_per_batch=35)
    print(f"Partitioned into {len(sub_payloads)} sub-batches (max 35 dialogues/batch).")
    assert len(sub_payloads) >= 5, f"Expected at least 5 sub-batches, got {len(sub_payloads)}"

    translator = LLM_API_Translator(
        lang_source="English",
        lang_target="Tiếng Việt",
        provider="Gemini Proxy",
        model="gemini-3.5-flash-lite",
        apikey=os.getenv("GEMINI_API_KEY", "")
    )

    print("\nExecuting chapter batch translation with Tiered Model & Proactive RPM Throttle...")
    t0 = time.time()
    result_map = translator.translate_chapter_batch(proxy, src_lang="English", tgt_lang="Tiếng Việt")
    elapsed = time.time() - t0

    print(f"\n✓ Chapter Batch Translation Completed in {elapsed:.2f}s!")
    total_translated = sum(len(p_res) for p_res in result_map.values())
    print(f"Total translated dialogues unpacked: {total_translated}/{total_count}")
    assert total_translated == total_count, f"Mismatch: translated {total_translated} vs {total_count}"

    # Print quota state
    print(f"\nQuota State Cache File: {QUOTA_STATE_FILE}")
    if os.path.exists(QUOTA_STATE_FILE):
        with open(QUOTA_STATE_FILE, "r", encoding="utf-8") as f:
            state_data = json.load(f)
        print("Usage Today Summary:")
        for m, count in state_data.get("usage_today", {}).items():
            print(f" - {m}: {count} requests used (Remaining RPD: {QUOTA_TRACKER.get_remaining_rpd(m)})")

def test_quota_exhaustion_simulation():
    print("\n" + "=" * 80)
    print("🛑 3. TESTING ALL-QUOTA EXHAUSTION SIMULATION & SAFE RESUME STATE")
    print("=" * 80)

    # Temporarily mark all models as exhausted
    original_exhausted = dict(QUOTA_TRACKER.exhausted_today)
    try:
        for m in QUOTA_TRACKER.profiles.keys():
            QUOTA_TRACKER.record_429_exhaustion(m)

        assert QUOTA_TRACKER.are_all_quotas_exhausted() is True, "Expected all quotas to be exhausted"
        print("✓ All models simulated as exhausted (RPD == 0).")

        proxy = TranslationProxy()
        proxy.start_session([1], timeout_seconds=60.0)
        proxy.mark_page_completed(1, [{"id": 1, "text": "Hello world.", "reading_order": 1, "block_type": "DIALOGUE"}], page_name="test_page.png")

        translator = LLM_API_Translator(
            lang_source="English",
            lang_target="Tiếng Việt",
            provider="Gemini Proxy",
            model="gemini-3.5-flash-lite",
            apikey=os.getenv("GEMINI_API_KEY", "")
        )

        caught_expected_error = False
        try:
            translator.translate_chapter_batch(proxy, src_lang="English", tgt_lang="Tiếng Việt")
        except RuntimeError as e:
            caught_expected_error = True
            print(f"\n[Verified Exception Message]:\n{e}\n")
            assert "Đã dùng hết quota ngày hôm nay" in str(e)
            assert os.path.exists(RESUME_STATE_FILE), "Resume state file must exist!"
            with open(RESUME_STATE_FILE, "r", encoding="utf-8") as f:
                resume_data = json.load(f)
            print("Saved Resume Checkpoint:")
            print(json.dumps(resume_data, ensure_ascii=False, indent=2))

        assert caught_expected_error, "Pipeline must catch quota exhaustion and raise RuntimeError with resume state!"
        print("✓ Quota exhaustion clean exit and resume state persistence verified.")

    finally:
        # Restore real quota state
        QUOTA_TRACKER.exhausted_today = original_exhausted
        QUOTA_TRACKER._save_state()

if __name__ == "__main__":
    test_quota_tracker_unit()
    test_long_chapter_multi_subbatch()
    test_quota_exhaustion_simulation()
    print("\n" + "=" * 80)
    print("🎉 ALL QUOTA TRACKER & TIER FALLBACK TESTS PASSED 100%!")
    print("=" * 80)
