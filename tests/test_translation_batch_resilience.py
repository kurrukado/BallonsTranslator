import os
import sys
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.translators.translation_proxy import (
    TranslationProxy, PageBatch, DialogueBlock,
    ChapterTranslationPayload, clean_and_repair_json
)
from utils.quota_tracker import QuotaTracker


class TestTranslationBatchResilience(unittest.TestCase):

    def setUp(self):
        self.proxy = TranslationProxy()
        # Build dummy payload: 2 pages, page 0 (ids: '1', '2'), page 1 (ids: '3', '4')
        self.payload = ChapterTranslationPayload(
            pages=[
                PageBatch(
                    page_id="page_000.png",
                    page_index=0,
                    blocks=[
                        DialogueBlock(id="1", text="Hello", page_index=0),
                        DialogueBlock(id="2", text="World", page_index=0),
                    ]
                ),
                PageBatch(
                    page_id="page_001.png",
                    page_index=1,
                    blocks=[
                        DialogueBlock(id="3", text="Good", page_index=1),
                        DialogueBlock(id="4", text="Morning", page_index=1),
                    ]
                ),
            ]
        )

    def test_standard_nested_pages_response(self):
        """Standard {"pages": [...]} response from Gemini."""
        raw_json = '''{
            "pages": [
                {
                    "page_index": 0,
                    "dialogues": [
                        {"id": "1", "translation": "Xin chào"},
                        {"id": "2", "translation": "Thế giới"}
                    ]
                },
                {
                    "page_index": 1,
                    "dialogues": [
                        {"id": "3", "translation": "Chào"},
                        {"id": "4", "translation": "Buổi sáng"}
                    ]
                }
            ]
        }'''
        valid, res_map, err = self.proxy.validate_and_unpack(self.payload, raw_json)
        self.assertTrue(valid, f"Validation should succeed: {err}")
        self.assertEqual(res_map[0]["1"]["translation"], "Xin chào")
        self.assertEqual(res_map[1]["4"]["translation"], "Buổi sáng")

    def test_flat_translations_adaptation(self):
        """
        User Custom Prompt instructs: {"translations": [{"id": ..., "translation": ...}]}.
        Proxy must adapt flat list across payload pages without omitting IDs.
        """
        raw_json = '''{
            "translations": [
                {"id": "1", "translation": "Xin chào", "emotion_tag": "normal"},
                {"id": "2", "translation": "Thế giới", "emotion_tag": "normal"},
                {"id": "3", "translation": "Chào", "emotion_tag": "normal"},
                {"id": "4", "translation": "Buổi sáng", "emotion_tag": "normal"}
            ]
        }'''
        valid, res_map, err = self.proxy.validate_and_unpack(self.payload, raw_json)
        self.assertTrue(valid, f"Flat translations should unpack cleanly: {err}")
        self.assertIn(0, res_map)
        self.assertIn(1, res_map)
        self.assertEqual(res_map[0]["1"]["translation"], "Xin chào")
        self.assertEqual(res_map[0]["2"]["translation"], "Thế giới")
        self.assertEqual(res_map[1]["3"]["translation"], "Chào")
        self.assertEqual(res_map[1]["4"]["translation"], "Buổi sáng")

    def test_1_based_page_index_normalization(self):
        """Model returns 1-based indexing [1, 2] instead of zero-based [0, 1]."""
        raw_json = '''{
            "pages": [
                {
                    "page_index": 1,
                    "dialogues": [
                        {"id": "1", "translation": "Xin chào"},
                        {"id": "2", "translation": "Thế giới"}
                    ]
                },
                {
                    "page_index": 2,
                    "dialogues": [
                        {"id": "3", "translation": "Chào"},
                        {"id": "4", "translation": "Buổi sáng"}
                    ]
                }
            ]
        }'''
        valid, res_map, err = self.proxy.validate_and_unpack(self.payload, raw_json)
        self.assertTrue(valid, f"1-based index should be normalized to 0-based: {err}")
        self.assertIn(0, res_map)
        self.assertIn(1, res_map)
        self.assertEqual(res_map[0]["1"]["translation"], "Xin chào")

    def test_regex_repair_on_truncated_json(self):
        """Truncated stream containing valid block objects."""
        truncated_raw = '''```json
        {
          "translations": [
            {"id": "1", "translation": "Xin chào"},
            {"id": "2", "translation": "Thế giới"},
            {"id": "3", "translation": "Chào"},
            {"id": "4", "translation": "Buổi sáng"}
        '''
        repaired = clean_and_repair_json(truncated_raw)
        self.assertIsInstance(repaired, dict)
        valid, res_map, err = self.proxy.validate_and_unpack(self.payload, repaired)
        self.assertTrue(valid, f"Truncated repair should succeed: {err}")
        self.assertEqual(res_map[0]["1"]["translation"], "Xin chào")

    def test_rpm_slot_reservation(self):
        """Ensure wait_for_rpm_slot reserves timestamp and throttles immediate repeat calls."""
        tracker = QuotaTracker()
        model = "gemini-3.5-flash-lite"
        # Reset last request time
        tracker.last_request_time[model] = 0.0

        # Slot 1: should return immediately and reserve slot
        d1 = tracker.wait_for_rpm_slot(model)
        self.assertLess(d1, 0.1)
        self.assertGreater(tracker.last_request_time.get(model, 0.0), 0.0)

        # Slot 2 immediately after: should compute throttle delay ~ 4.2s (15 RPM)
        throttle_needed = tracker.get_required_rpm_throttle(model)
        self.assertGreater(throttle_needed, 3.5)


if __name__ == "__main__":
    unittest.main()
