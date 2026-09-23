import sys
import os
import time
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.translators.translation_proxy import (
    TranslationProxy, PageState, TranslationMode, RequestStatus,
    ChapterTranslationPayload, ChapterTranslationResponse,
    PageResponseBatch, DialogueResponseBlock
)
from modules.translators.trans_llm_api import LLM_API_Translator


class TestTranslationProxyFullSuite(unittest.TestCase):

    def setUp(self):
        self.proxy = TranslationProxy()

    # Test 1: 5 pages arrive asynchronously -> all buffered in correct order
    def test_01_async_page_arrival_and_batch_queuing(self):
        page_indices = [0, 1, 2, 3, 4]
        self.proxy.start_session(page_indices, timeout_seconds=10.0)

        # Pages arrive out-of-order asynchronously
        self.proxy.mark_page_completed(2, [{"id": 1, "text": "Page 2 Text"}])
        self.proxy.mark_page_completed(0, [{"id": 1, "text": "Page 0 Text"}])
        self.proxy.mark_page_completed(4, [{"id": 1, "text": "Page 4 Text"}])
        self.assertFalse(self.proxy.is_batch_ready())

        self.proxy.mark_page_completed(1, [{"id": 1, "text": "Page 1 Text"}])
        self.proxy.mark_page_completed(3, [{"id": 1, "text": "Page 3 Text"}])
        self.assertTrue(self.proxy.is_batch_ready())

        payload = self.proxy.build_chapter_payload()
        self.assertEqual(len(payload.pages), 5)
        # Verify strict sequential ordering is maintained
        for idx, p in enumerate(payload.pages):
            self.assertEqual(p.page_index, idx)
        print("[PASS] Test 1: 5 pages arrived asynchronously and buffered in exact order.")

    # Test 2: Structured JSON response -> all IDs preserved 1:1
    def test_02_structured_json_id_preservation(self):
        self.proxy.start_session([0])
        self.proxy.mark_page_completed(0, [
            {"id": "b_101", "text": "Wait!"},
            {"id": "b_102", "text": "Don't go there."}
        ])
        payload = self.proxy.build_chapter_payload()

        mock_json = {
            "pages": [{
                "page_index": 0,
                "dialogues": [
                    {"id": "b_101", "translation": "Chờ đã!", "emotion_tag": "shout"},
                    {"id": "b_102", "translation": "Đừng đến đó.", "emotion_tag": "fear"}
                ]
            }]
        }
        valid, res_map, err = self.proxy.validate_and_unpack(payload, mock_json)
        self.assertTrue(valid, err)
        self.assertEqual(res_map[0]["b_101"]["translation"], "Chờ đã!")
        self.assertEqual(res_map[0]["b_102"]["translation"], "Đừng đến đó")
        print("[PASS] Test 2: Structured JSON response unpacked with 1:1 ID preservation.")

    # Test 3: Gemini returns IDs in shuffled order -> proxy restores original ordering
    def test_03_shuffled_id_restoration(self):
        self.proxy.start_session([0])
        self.proxy.mark_page_completed(0, [
            {"id": 1, "text": "First"},
            {"id": 2, "text": "Second"},
            {"id": 3, "text": "Third"}
        ])
        payload = self.proxy.build_chapter_payload()

        # Shuffled output from Gemini (3, 1, 2)
        shuffled_json = {
            "pages": [{
                "page_index": 0,
                "dialogues": [
                    {"id": 3, "translation": "Thứ ba"},
                    {"id": 1, "translation": "Thứ nhất"},
                    {"id": 2, "translation": "Thứ hai"}
                ]
            }]
        }
        valid, res_map, err = self.proxy.validate_and_unpack(payload, shuffled_json)
        self.assertTrue(valid, err)
        # Verify the unpacked keys match original sequence
        ordered_keys = list(res_map[0].keys())
        self.assertEqual(ordered_keys, [1, 2, 3])
        print("[PASS] Test 3: Shuffled ID response properly restored to original sequential ordering.")

    # Test 4: Gemini omits one ID -> validation failure triggered
    def test_04_omitted_id_validation_failure(self):
        self.proxy.start_session([0])
        self.proxy.mark_page_completed(0, [
            {"id": 1, "text": "Hello"},
            {"id": 2, "text": "World"}
        ])
        payload = self.proxy.build_chapter_payload()

        # Gemini omits ID 2
        omitted_json = {
            "pages": [{
                "page_index": 0,
                "dialogues": [
                    {"id": 1, "translation": "Xin chào"}
                ]
            }]
        }
        valid, res_map, err = self.proxy.validate_and_unpack(payload, omitted_json)
        self.assertFalse(valid)
        self.assertIn("Omitted IDs", err)
        print("[PASS] Test 4: Omitted ID detected and rejected by validator.")

    # Test 5: Gemini adds unknown/hallucinated ID -> validation failure
    def test_05_unknown_id_rejection(self):
        self.proxy.start_session([0])
        self.proxy.mark_page_completed(0, [{"id": 1, "text": "Hello"}])
        payload = self.proxy.build_chapter_payload()

        unknown_id_json = {
            "pages": [{
                "page_index": 0,
                "dialogues": [
                    {"id": 1, "translation": "Xin chào"},
                    {"id": 999, "translation": "Hallucinated dialogue"}
                ]
            }]
        }
        valid, res_map, err = self.proxy.validate_and_unpack(payload, unknown_id_json)
        self.assertFalse(valid)
        self.assertIn("Unknown/Hallucinated ID", err)
        print("[PASS] Test 5: Hallucinated/unknown ID detected and rejected by validator.")

    # Test 6: HTTP 429 Rate Limit -> fallback chain activated
    def test_06_http_429_rate_limit_fallback(self):
        from utils.quota_tracker import QUOTA_TRACKER
        QUOTA_TRACKER.exhausted_today.clear()

        self.proxy.start_session([0])
        self.proxy.mark_page_completed(0, [{"id": 1, "text": "Test"}])

        translator = LLM_API_Translator("English", "Tiếng Việt")
        translator.params["override model"]["value"] = "gemini-3.5-flash-lite"
        translator._initialize_client = MagicMock(return_value=True)
        translator.client = MagicMock()

        call_records = []
        def mock_chat_create(model, **kwargs):
            call_records.append(model)
            if model == "gemini-3.5-flash-lite":
                raise Exception("429 Too Many Requests: Quota exceeded")
            mock_choice = MagicMock()
            mock_choice.message.content = '{"pages": [{"page_index": 0, "dialogues": [{"id": 1, "translation": "Kiểm tra", "emotion_tag": "normal"}]}]}'
            mock_resp = MagicMock()
            mock_resp.choices = [mock_choice]
            return mock_resp

        translator.client.chat.completions.create.side_effect = mock_chat_create
        res_map = translator.translate_chapter_batch(self.proxy)

        self.assertIn("gemini-3.5-flash-lite", call_records)
        self.assertIn("gemini-3.1-flash-lite", call_records)
        self.assertEqual(res_map[0][1]["translation"], "Kiểm tra")
        print("[PASS] Test 6: HTTP 429 seamlessly triggered fallback chain to next Tier 1 model.")

    # Test 7: Primary model timeout -> fallback
    def test_07_primary_model_timeout_fallback(self):
        from utils.quota_tracker import QUOTA_TRACKER
        QUOTA_TRACKER.exhausted_today.clear()

        self.proxy.start_session([0])
        self.proxy.mark_page_completed(0, [{"id": 1, "text": "Timeout Test"}])

        translator = LLM_API_Translator("English", "Tiếng Việt")
        translator.params["override model"]["value"] = "gemini-3.5-flash-lite"
        translator._initialize_client = MagicMock(return_value=True)
        translator.client = MagicMock()

        call_records = []
        def mock_chat_create(model, **kwargs):
            call_records.append(model)
            if model == "gemini-3.5-flash-lite":
                raise TimeoutError("Request timed out after 30s")
            mock_choice = MagicMock()
            mock_choice.message.content = '{"pages": [{"page_index": 0, "dialogues": [{"id": 1, "translation": "Thành công sau timeout", "emotion_tag": "normal"}]}]}'
            mock_resp = MagicMock()
            mock_resp.choices = [mock_choice]
            return mock_resp

        translator.client.chat.completions.create.side_effect = mock_chat_create
        res_map = translator.translate_chapter_batch(self.proxy)

        self.assertEqual(res_map[0][1]["translation"], "Thành công sau timeout")
        print("[PASS] Test 7: Timeout recovered via fallback chain.")

    # Test 8: Direct mode -> bypass buffer
    def test_08_direct_mode_buffer_bypass(self):
        translator = LLM_API_Translator("English", "Tiếng Việt")
        translator._initialize_client = MagicMock(return_value=True)
        translator.client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Bản dịch trực tiếp ngay lập tức"
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        translator.client.chat.completions.create.return_value = mock_resp

        # Buffer is empty, direct request must succeed immediately
        res = translator.translate_single("Direct test without queuing.")
        self.assertEqual(res, "Bản dịch trực tiếp ngay lập tức")
        self.assertTrue(self.proxy.is_empty())
        print("[PASS] Test 8: Direct mode bypassed buffer and returned plain string.")

    # Test 9: Direct mode while RUN session is active -> independent execution
    def test_09_concurrent_direct_during_run_mode(self):
        self.proxy.start_session([0, 1])
        self.proxy.mark_page_completed(0, [{"id": 1, "text": "Page 0 in Run Session"}])

        translator = LLM_API_Translator("English", "Tiếng Việt")
        translator._initialize_client = MagicMock(return_value=True)
        translator.client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Chỉnh sửa thủ công độc lập"
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        translator.client.chat.completions.create.return_value = mock_resp

        direct_res = translator.translate_single("Manual bubble edit")
        self.assertEqual(direct_res, "Chỉnh sửa thủ công độc lập")
        # Run session buffer must remain untouched
        self.assertEqual(self.proxy.total_dialogues_count(), 1)
        print("[PASS] Test 9: Direct request executed independently without disrupting active RUN session.")

    # Test 10: Duplicate page enqueue prevention
    def test_10_duplicate_page_enqueue_prevention(self):
        self.proxy.start_session([0])
        self.proxy.mark_page_completed(0, [{"id": 1, "text": "Original Text"}])
        # Attempt duplicate enqueue
        self.proxy.mark_page_completed(0, [{"id": 1, "text": "Duplicate Text"}])

        payload = self.proxy.build_chapter_payload()
        self.assertEqual(len(payload.pages), 1)
        self.assertEqual(payload.pages[0].blocks[0].text, "Original Text")
        print("[PASS] Test 10: Duplicate page enqueue successfully prevented.")

    # Test 11: Stop Run cancellation and cleanup
    def test_11_stop_run_cancellation_and_cleanup(self):
        self.proxy.start_session([0, 1, 2])
        self.proxy.mark_page_completed(0, [{"id": 1, "text": "Page 0"}])
        self.assertFalse(self.proxy.is_cancelled())

        self.proxy.cancel_session()
        self.assertTrue(self.proxy.is_cancelled())
        self.assertTrue(self.proxy.is_empty())
        self.assertFalse(self.proxy.is_batch_ready())
        print("[PASS] Test 11: Cancel session cleanly purged buffer and marked state as cancelled.")

    # Test 12: 20-block page batch stress test with 100% ID integrity
    def test_12_twenty_block_page_stress(self):
        self.proxy.start_session([0])
        blocks_input = [{"id": i, "text": f"Manga dialogue line #{i}"} for i in range(1, 21)]
        self.proxy.mark_page_completed(0, blocks_input)

        payload = self.proxy.build_chapter_payload()
        self.assertEqual(len(payload.pages[0].blocks), 20)

        # Mock full 20-dialogue response
        dialogues_resp = [{"id": i, "translation": f"Lời thoại tiếng Việt #{i}", "emotion_tag": "normal"} for i in range(1, 21)]
        mock_json = {"pages": [{"page_index": 0, "dialogues": dialogues_resp}]}

        valid, res_map, err = self.proxy.validate_and_unpack(payload, mock_json)
        self.assertTrue(valid, err)
        self.assertEqual(len(res_map[0]), 20)
        for i in range(1, 21):
            self.assertEqual(res_map[0][i]["translation"], f"Lời thoại tiếng Việt #{i}")
        print("[PASS] Test 12: 20-block stress batch processed with 100% ID integrity.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
