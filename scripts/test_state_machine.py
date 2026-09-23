import sys
import os
import unittest
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.translators.translation_state_buffer import TranslationStateBuffer, PageState


class TestStateMachine(unittest.TestCase):

    def test_session_flow_normal(self):
        buf = TranslationStateBuffer()
        buf.start_session([0, 1, 2], timeout_seconds=10.0)

        self.assertFalse(buf.is_batch_ready())
        self.assertEqual(buf._page_states[0], PageState.PENDING)

        buf.mark_page_ready(0, [{"id": 1, "text": "Hello"}])
        self.assertFalse(buf.is_batch_ready())

        buf.mark_page_ready(1, [{"id": 1, "text": "World"}])
        self.assertFalse(buf.is_batch_ready())

        buf.mark_page_ready(2, [{"id": 1, "text": "Done"}])
        self.assertTrue(buf.is_batch_ready())
        self.assertEqual(len(buf.get_ready_page_indices()), 3)
        print("[PASS] Normal session flow transitions to ready.")

    def test_session_flow_with_failed_page(self):
        buf = TranslationStateBuffer()
        buf.start_session([0, 1], timeout_seconds=10.0)

        buf.mark_page_ready(0, [{"id": 1, "text": "Good page"}])
        buf.mark_page_failed(1, "OCR error")

        self.assertTrue(buf.is_batch_ready())
        self.assertEqual(buf.get_ready_page_indices(), [0])
        print("[PASS] Failed page resolved without hanging batch.")

    def test_session_timeout_resolution(self):
        buf = TranslationStateBuffer()
        buf.start_session([0, 1], timeout_seconds=0.1)

        buf.mark_page_ready(0, [{"id": 1, "text": "Ready page"}])
        # Page 1 remains PENDING
        time.sleep(0.15)

        self.assertTrue(buf.is_batch_ready())
        self.assertEqual(buf.get_ready_page_indices(), [0])
        print("[PASS] Timeout resolution triggers batch with ready pages.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
