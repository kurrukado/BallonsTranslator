import sys
import os
import json
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.translators.translation_state_buffer import (
    TranslationStateBuffer,
    DialogueEntry,
    PageBatchEntry,
    ChapterBatchRequest,
    ChapterBatchResponse,
    PageBatchResponse,
    DialogueResponseEntry,
)
from modules.translators.trans_llm_api import LLM_API_Translator


class TestTranslationProxyWorkflow(unittest.TestCase):

    def setUp(self):
        self.buffer = TranslationStateBuffer()
        self.mock_chapter_data = {
            0: [
                {"id": 1, "text": "Are you ready to begin the mission?"},
                {"id": 2, "text": "Yes, Captain! We're all set."},
                {"id": 3, "text": "Don't let your guard down for even a second..."},
            ],
            1: [
                {"id": 1, "text": "Look at those monsters over there!"},
                {"id": 2, "text": "FIRE AT WILL!!"},
            ],
            2: [
                {"id": 1, "text": "They're faster than we expected!"},
                {"id": 2, "text": "Fall back to sector four!"},
                {"id": 3, "text": "I can't leave you behind..."},
                {"id": 4, "text": "Just go, that's an order!"},
            ],
            3: [
                {"id": 1, "text": "Is everyone safe?"},
            ],
            4: [
                {"id": 1, "text": "We made it through somehow."},
                {"id": 2, "text": "Good work, team. Rest up for tomorrow."},
            ],
        }

    def test_01_buffer_queuing_and_serialization(self):
        """Test queuing multi-page dialogues and serializing to ChapterBatchRequest."""
        self.buffer.clear()
        self.assertTrue(self.buffer.is_empty())

        for p_idx, dialogues in self.mock_chapter_data.items():
            self.buffer.add_page_dialogues(p_idx, dialogues)

        self.assertFalse(self.buffer.is_empty())
        self.assertEqual(self.buffer.total_dialogues_count(), 12)

        payload = self.buffer.get_chapter_payload()
        self.assertEqual(len(payload.pages), 5)
        self.assertEqual(payload.pages[0].page_index, 0)
        self.assertEqual(len(payload.pages[0].dialogues), 3)
        self.assertEqual(payload.pages[2].dialogues[1].text, "Fall back to sector four!")

        # Verify JSON serialization
        json_output = payload.model_dump_json()
        parsed = json.loads(json_output)
        self.assertIn("pages", parsed)
        self.assertEqual(len(parsed["pages"]), 5)
        print("[PASS] Test 1: Buffer queuing and chapter serialization successful.")

    def test_02_structured_json_unpacking_and_id_integrity(self):
        """Test unpacking structured Gemini JSON response with 1:1 ID and emotion integrity."""
        mock_response_json = {
            "pages": [
                {
                    "page_index": 0,
                    "dialogues": [
                        {"id": 1, "text": "Cậu đã sẵn sàng bắt đầu nhiệm vụ chưa?", "emotion_tag": "normal"},
                        {"id": 2, "text": "Rõ, đội trưởng! Bọn em đã sẵn sàng.", "emotion_tag": "normal"},
                        {"id": 3, "text": "Đừng lơ là cảnh giác dù chỉ một giây...", "emotion_tag": "whisper"}
                    ]
                },
                {
                    "page_index": 1,
                    "dialogues": [
                        {"id": 1, "text": "Nhìn lũ quái vật đằng kia kìa!", "emotion_tag": "surprise"},
                        {"id": 2, "text": "BẮN TỰ DO!!", "emotion_tag": "shout"}
                    ]
                },
                {
                    "page_index": 2,
                    "dialogues": [
                        {"id": 1, "text": "Chúng nhanh hơn chúng ta tưởng tượng!", "emotion_tag": "fear"},
                        {"id": 2, "text": "Rút lui về khu vực bốn!", "emotion_tag": "shout"},
                        {"id": 3, "text": "Tớ không thể bỏ cậu lại được...", "emotion_tag": "whisper"},
                        {"id": 4, "text": "Đi mau, đây là mệnh lệnh!", "emotion_tag": "shout"}
                    ]
                },
                {
                    "page_index": 3,
                    "dialogues": [
                        {"id": 1, "text": "Mọi người đều an toàn cả chứ?", "emotion_tag": "normal"}
                    ]
                },
                {
                    "page_index": 4,
                    "dialogues": [
                        {"id": 1, "text": "Bằng cách nào đó chúng ta đã vượt qua.", "emotion_tag": "normal"},
                        {"id": 2, "text": "Làm tốt lắm cả đội. Hãy nghỉ ngơi cho ngày mai.", "emotion_tag": "normal"}
                    ]
                }
            ]
        }

        unpacked = self.buffer.unpack_response(mock_response_json)
        self.assertEqual(len(unpacked), 5)
        self.assertEqual(unpacked[0][1]["translation"], "Cậu đã sẵn sàng bắt đầu nhiệm vụ chưa?")
        self.assertEqual(unpacked[1][2]["emotion_tag"], "shout")
        self.assertEqual(unpacked[2][3]["emotion_tag"], "whisper")
        self.assertEqual(unpacked[4][2]["translation"], "Làm tốt lắm cả đội. Hãy nghỉ ngơi cho ngày mai.")
        print("[PASS] Test 2: Structured response unpacked with 1:1 ID and emotion tags preserved.")

    def test_03_fallback_routing_on_rate_limit(self):
        """Test automatic fallback from Flash-Lite to Flash upon rate limit (429) or failure."""
        translator = LLM_API_Translator("English", "Tiếng Việt")
        translator.params['model']['value'] = "gemini-2.0-flash-lite"
        
        # Populate buffer
        self.buffer.clear()
        self.buffer.add_page_dialogues(0, [{"id": 1, "text": "Stand back!"}])

        # Mock OpenAI / Proxy client responses
        mock_client = MagicMock()
        
        # First call with primary model fails (429), second call with fallback succeeds
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps({
            "pages": [{
                "page_index": 0,
                "dialogues": [{"id": 1, "text": "Lùi lại mau!", "emotion_tag": "shout"}]
            }]
        })
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        def mock_create(*args, **kwargs):
            model = kwargs.get("model", "")
            if model == "gemini-2.0-flash-lite":
                raise Exception("Rate limit exceeded 429 Quota Exceeded")
            return mock_completion

        mock_client.chat.completions.create.side_effect = mock_create
        translator.client = mock_client
        translator._select_api_key = MagicMock(return_value="test-key")
        translator._initialize_client = MagicMock(return_value=True)

        result_map = translator.translate_chapter_batch(self.buffer)
        self.assertIn(0, result_map)
        self.assertEqual(result_map[0][1]["translation"], "Lùi lại mau!")
        self.assertEqual(result_map[0][1]["emotion_tag"], "shout")
        print("[PASS] Test 3: Fallback sequence seamlessly recovered and completed chapter translation.")

    def test_04_direct_single_passthrough_bypass(self):
        """Test that single manual UI edits bypass chapter buffering and execute directly."""
        translator = LLM_API_Translator("English", "Tiếng Việt")
        translator.params['model']['value'] = "gemini-2.0-flash-lite"

        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Đúng như dự đoán của tôi."
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_completion

        translator.client = mock_client
        translator._select_api_key = MagicMock(return_value="test-key")
        translator._initialize_client = MagicMock(return_value=True)

        single_result = translator.translate_single("Just as I expected.")
        self.assertEqual(single_result, "Đúng như dự đoán của tôi.")
        print("[PASS] Test 4: Direct single passthrough returned string with zero batching overhead.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
