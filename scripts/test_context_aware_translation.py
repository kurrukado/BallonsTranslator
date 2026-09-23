import sys
import os
import unittest
import json

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.translators.translation_proxy import (
    TranslationProxy,
    ChapterTranslationPayload,
    PageBatch,
    DialogueBlock,
    clean_and_repair_json
)
from modules.translators.trans_llm_api import LLM_API_Translator


class TestContextAwareTranslation(unittest.TestCase):

    def setUp(self):
        self.proxy = TranslationProxy()
        self.translator = LLM_API_Translator("English", "Tiếng Việt")

    # TEST 1: COMPLETE CLAUSE NOT TRUNCATED
    def test_01_complete_clause_not_truncated(self):
        """
        Critical Test: 'their cohabitation life starts a new phase'
        Must NOT be truncated into only 'Cuộc sống chung'.
        Must preserve full proposition: subject + predicate.
        """
        source_text = "their cohabitation life starts a new phase"
        
        # Test Direct Mode
        translated = self.translator.translate_single(source_text, src_lang="English", tgt_lang="Tiếng Việt")
        print(f"[DEBUG Test 1] Direct translated: '{translated}'")
        
        # Validations
        translated_lower = translated.lower()
        
        # Must contain subject meaning
        has_subject = any(w in translated_lower for w in ["cuộc sống", "sống chung", "ở chung"])
        # Must contain possessive/pronoun
        has_pronoun = any(w in translated_lower for w in ["họ", "hai người", "của họ"])
        # Must contain predicate (verb + phase)
        has_predicate = any(w in translated_lower for w in ["bước sang", "bắt đầu", "giai đoạn", "chặng", "thời kỳ"])
        
        self.assertTrue(has_subject, f"Missing subject in '{translated}'")
        self.assertTrue(has_predicate, f"Missing predicate in '{translated}' (truncated error!)")
        
        # Explicit rejection of truncated bad results
        self.assertNotEqual(translated.strip().upper(), "CUỘC SỐNG CHUNG", "Truncated to bare noun phrase!")
        self.assertNotEqual(translated.strip(), "Cuộc sống chung", "Truncated to bare noun phrase!")
        print("[PASS] Test 1: Full proposition preserved, predicate not truncated.")

    # TEST 2: FRAGMENT SPLIT ACROSS TWO BLOCKS
    def test_02_fragment_split_across_two_blocks(self):
        """
        Multi-bubble sentence stitching:
        Block 1: 'their cohabitation life'
        Block 2: 'starts a new phase'
        """
        self.proxy.start_session([0])
        self.proxy.mark_page_completed(
            page_index=0,
            dialogues=[
                {"id": 1, "text": "their cohabitation life", "reading_order": 1},
                {"id": 2, "text": "starts a new phase", "reading_order": 2}
            ],
            page_name="page_001.jpg"
        )
        
        payload = self.proxy.build_chapter_payload()
        self.assertEqual(len(payload.pages[0].blocks), 2)
        
        # Simulate realistic LLM response following context-aware stitching guidelines
        mock_response = {
            "pages": [
                {
                    "page_index": 0,
                    "dialogues": [
                        {"id": 1, "translation": "Cuộc sống chung của họ"},
                        {"id": 2, "translation": "bước sang một giai đoạn mới."}
                    ]
                }
            ]
        }
        
        valid, res_map, msg = self.proxy.validate_and_unpack(payload, json.dumps(mock_response))
        self.assertTrue(valid, f"Validation failed: {msg}")
        self.assertEqual(res_map[0][1]["translation"], "Cuộc sống chung của họ")
        self.assertEqual(res_map[0][2]["translation"], "bước sang một giai đoạn mới")
        print("[PASS] Test 2: Multi-bubble fragments stitched and partitioned with 1:1 ID integrity.")

    # TEST 3: FRAGMENT THOUGHT CONTINUATION
    def test_03_fragment_thought_continuation(self):
        """
        Block 1: 'I never thought'
        Block 2: 'you would come back.'
        """
        self.proxy.start_session([0])
        self.proxy.mark_page_completed(
            page_index=0,
            dialogues=[
                {"id": 10, "text": "I never thought", "block_type": "THOUGHT"},
                {"id": 11, "text": "you would come back.", "block_type": "THOUGHT"}
            ],
            page_name="page_001.jpg"
        )
        payload = self.proxy.build_chapter_payload()
        
        mock_response = {
            "pages": [
                {
                    "page_index": 0,
                    "dialogues": [
                        {"id": 10, "translation": "Mình chưa bao giờ nghĩ rằng..."},
                        {"id": 11, "translation": "cậu sẽ quay trở lại."}
                    ]
                }
            ]
        }
        valid, res_map, msg = self.proxy.validate_and_unpack(payload, mock_response)
        self.assertTrue(valid, f"Validation failed: {msg}")
        self.assertIn("nghĩ", res_map[0][10]["translation"].lower())
        self.assertIn("quay", res_map[0][11]["translation"].lower())
        print("[PASS] Test 3: Inner thought fragment continuity verified.")

    # TEST 4: FALSE-MERGE PREVENTION
    def test_04_false_merge_prevention(self):
        """
        Block 1: 'Good morning.'
        Block 2: 'What are you doing?'
        Separate utterances must NOT be merged or blurred together.
        """
        self.proxy.start_session([0])
        self.proxy.mark_page_completed(
            page_index=0,
            dialogues=[
                {"id": 1, "text": "Good morning."},
                {"id": 2, "text": "What are you doing?"}
            ],
            page_name="page_001.jpg"
        )
        payload = self.proxy.build_chapter_payload()
        
        mock_response = {
            "pages": [
                {
                    "page_index": 0,
                    "dialogues": [
                        {"id": 1, "translation": "Chào buổi sáng."},
                        {"id": 2, "translation": "Cậu đang làm gì đấy?"}
                    ]
                }
            ]
        }
        valid, res_map, msg = self.proxy.validate_and_unpack(payload, mock_response)
        self.assertTrue(valid, f"Validation failed: {msg}")
        self.assertEqual(res_map[0][1]["translation"], "Chào buổi sáng")
        self.assertEqual(res_map[0][2]["translation"], "Cậu đang làm gì đấy?")
        print("[PASS] Test 4: Standalone utterances correctly kept separate without false merging.")

    # TEST 5: TITLE VS CLAUSE (NO HALLUCINATION)
    def test_05_title_vs_clause_no_hallucination(self):
        """
        Standalone Title: 'Their Cohabitation Life'
        Must translate as noun phrase, must NOT hallucinate unmentioned predicate.
        """
        source_title = "Their Cohabitation Life"
        trans_title = self.translator.translate_single(
            source_title,
            src_lang="English",
            tgt_lang="Tiếng Việt",
            context_hints={"block_type": "TITLE"}
        )
        print(f"[DEBUG Test 5] Title translated: '{trans_title}'")
        
        trans_lower = trans_title.lower()
        self.assertTrue(any(w in trans_lower for w in ["cuộc sống", "sống chung", "ở chung"]))
        
        # Must NOT hallucinate 'starts a new phase' when source has no verb
        self.assertNotIn("bước sang", trans_lower)
        self.assertNotIn("giai đoạn mới", trans_lower)
        print("[PASS] Test 5: Standalone title faithfully translated without hallucinating predicates.")

    # TEST 6: DIRECT MODE WITH CONTEXT HINTS
    def test_06_direct_mode_with_context_hints(self):
        """
        Direct translation of fragment with surrounding context hints.
        """
        target_fragment = "starts a new phase."
        context_hints = {
            "previous": "their cohabitation life",
            "next": "",
            "block_type": "DIALOGUE"
        }
        res = self.translator.translate_single(
            target_fragment,
            src_lang="English",
            tgt_lang="Tiếng Việt",
            context_hints=context_hints
        )
        print(f"[DEBUG Test 6] Fragment with context translated: '{res}'")
        res_lower = res.lower()
        self.assertTrue(any(w in res_lower for w in ["bước sang", "bắt đầu", "giai đoạn", "chặng"]), f"Invalid fragment translation '{res}'")
        print("[PASS] Test 6: Direct Mode accurately translated fragment using context hints.")

    # TEST 7: CROSS-PAGE CONTINUITY
    def test_07_cross_page_sentence_continuity(self):
        """
        Sentence spans across Page 0 and Page 1:
        Page 0: 'Even if the world ends tomorrow'
        Page 1: 'I will still stand by your side.'
        """
        self.proxy.start_session([0, 1])
        self.proxy.mark_page_completed(0, [{"id": 1, "text": "Even if the world ends tomorrow"}], "p0.jpg")
        self.proxy.mark_page_completed(1, [{"id": 1, "text": "I will still stand by your side."}], "p1.jpg")
        
        payloads = self.proxy.build_sub_batch_payloads()
        self.assertEqual(len(payloads[0].pages), 2)
        
        mock_response = {
            "pages": [
                {
                    "page_index": 0,
                    "dialogues": [{"id": 1, "translation": "Dù cho ngày mai thế giới có tận thế..."}]
                },
                {
                    "page_index": 1,
                    "dialogues": [{"id": 1, "translation": "tớ vẫn sẽ luôn sát cánh bên cậu."}]
                }
            ]
        }
        valid, res_map, msg = self.proxy.validate_and_unpack(payloads[0], mock_response)
        self.assertTrue(valid, f"Validation failed: {msg}")
        self.assertIn("tận thế", res_map[0][1]["translation"].lower())
        self.assertIn("sát cánh", res_map[1][1]["translation"].lower())
        print("[PASS] Test 7: Cross-page sentence continuity and 1:1 ID mapping verified.")

    # TEST 8: ID INTEGRITY AND REJECTIONS
    def test_08_id_integrity_and_rejections(self):
        """
        Validate strict rejection of missing IDs, unknown IDs, and empty strings.
        """
        self.proxy.start_session([0])
        self.proxy.mark_page_completed(0, [{"id": "b1", "text": "Line 1"}, {"id": "b2", "text": "Line 2"}], "p0.jpg")
        payload = self.proxy.build_chapter_payload()
        
        # Case A: Missing b2
        bad_resp_missing = {"pages": [{"page_index": 0, "dialogues": [{"id": "b1", "translation": "Dòng 1"}]}]}
        valid, _, msg = self.proxy.validate_and_unpack(payload, bad_resp_missing)
        self.assertFalse(valid)
        self.assertIn("Omitted IDs", msg)
        
        # Case B: Unknown ID b99
        bad_resp_unknown = {"pages": [{"page_index": 0, "dialogues": [{"id": "b1", "translation": "Dòng 1"}, {"id": "b99", "translation": "Lạ"}]}]}
        valid, _, msg = self.proxy.validate_and_unpack(payload, bad_resp_unknown)
        self.assertFalse(valid)
        self.assertIn("Unknown/Hallucinated ID", msg)
        
        # Case C: Empty translation
        bad_resp_empty = {"pages": [{"page_index": 0, "dialogues": [{"id": "b1", "translation": "Dòng 1"}, {"id": "b2", "translation": "  "}]}]}
        valid, _, msg = self.proxy.validate_and_unpack(payload, bad_resp_empty)
        self.assertFalse(valid)
        self.assertIn("Empty translation", msg)
        print("[PASS] Test 8: All ID integrity violations strictly caught and rejected.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
