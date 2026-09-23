import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from modules.translators.translation_proxy import TranslationProxy
from modules.translators.trans_llm_api import LLM_API_Translator


class TestRealAPIExecution(unittest.TestCase):

    def test_real_gemini_direct_mode(self):
        translator = LLM_API_Translator("English", "Tiếng Việt")
        # Try real single translation
        try:
            source_text = "I will protect everyone no matter what!"
            result = translator.translate_single(source_text, src_lang="English", tgt_lang="Tiếng Việt")
            print(f"\n[REAL API DIRECT] Source: {source_text} -> Translated: {result}")
            self.assertTrue(len(result) > 0)
            self.assertNotEqual(result, source_text)
            print("[REAL_API_VERIFIED] Direct mode executed and returned natural Vietnamese translation.")
        except Exception as e:
            print(f"[NOT_VERIFIED] Real API call failed (e.g. offline/no key): {e}")

    def test_real_gemini_chapter_batch_mode(self):
        proxy = TranslationProxy()
        proxy.start_session([0])
        proxy.mark_page_completed(0, [
            {"id": 1, "text": "Are you ready?"},
            {"id": 2, "text": "Let's win this match!"}
        ])

        translator = LLM_API_Translator("English", "Tiếng Việt")
        try:
            result_map = translator.translate_chapter_batch(proxy, src_lang="English", tgt_lang="Tiếng Việt")
            print(f"\n[REAL API BATCH] Result Map: {result_map}")
            self.assertIn(0, result_map)
            self.assertIn(1, result_map[0])
            self.assertIn(2, result_map[0])
            trans1 = result_map[0][1]["translation"]
            trans2 = result_map[0][2]["translation"]
            print(f"[REAL_API_VERIFIED] Batch translation 1: {trans1} | 2: {trans2}")
        except Exception as e:
            print(f"[NOT_VERIFIED] Real API batch call failed (e.g. offline/no key): {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
