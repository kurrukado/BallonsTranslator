import threading
import time
import json
from enum import Enum
from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field


class PageState(str, Enum):
    IDLE = "IDLE"
    PENDING = "PENDING"
    OCR_DONE_WAITING = "OCR_DONE_WAITING"
    BATCH_SENT = "BATCH_SENT"
    TRANSLATED = "TRANSLATED"
    FAILED = "FAILED"


class DialogueEntry(BaseModel):
    id: int = Field(..., description="The original numeric ID of the dialogue snippet.")
    text: str = Field(..., description="The source dialogue text.")
    position: Optional[str] = Field(default=None, description="Spatial position on the page (e.g. Top-Right, Center).")
    block_type: Optional[str] = Field(default="DIALOGUE", description="Type: DIALOGUE, THOUGHT, NARRATION, SFX.")
    reading_order: Optional[int] = Field(default=None, description="Spatial reading sequence order.")


class PageBatchEntry(BaseModel):
    page_index: int = Field(..., description="Zero-based index of the page within the chapter.")
    dialogues: List[DialogueEntry] = Field(default_factory=list, description="List of dialogues on this page.")


class ChapterBatchRequest(BaseModel):
    pages: List[PageBatchEntry] = Field(default_factory=list, description="All pages in the chapter.")


class DialogueResponseEntry(BaseModel):
    id: int = Field(..., description="The original numeric ID of the dialogue.")
    text: str = Field(..., description="The natural Vietnamese translated string.")
    emotion_tag: Optional[str] = Field(
        default="normal",
        description="Emotion/lettering tag: 'shout', 'whisper', 'fear', 'surprise', 'normal'."
    )


class PageBatchResponse(BaseModel):
    page_index: int = Field(..., description="Zero-based index of the page.")
    dialogues: List[DialogueResponseEntry] = Field(default_factory=list, description="Translated dialogues for this page.")


class ChapterBatchResponse(BaseModel):
    pages: List[PageBatchResponse] = Field(default_factory=list, description="List of translated pages.")


class TranslationStateBuffer:
    """
    Thread-safe in-memory session state machine for chapter-level batch translation aggregation.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._pages_buffer: Dict[int, List[DialogueEntry]] = {}
        self._page_states: Dict[int, PageState] = {}
        self._session_pages: List[int] = []
        self._session_start_time: float = 0.0
        self._timeout_seconds: float = 60.0

    def start_session(self, expected_page_indices: List[int], timeout_seconds: float = 60.0):
        """
        Initialize a new Run session with a target list of page indices.
        """
        with self._lock:
            self._session_pages = list(expected_page_indices)
            self._pages_buffer.clear()
            self._page_states = {p: PageState.PENDING for p in expected_page_indices}
            self._session_start_time = time.time()
            self._timeout_seconds = max(0.05, timeout_seconds)

    def mark_page_ready(self, page_index: int, dialogues: List[Dict[str, Any]]):
        """
        Record OCR-extracted dialogues for page_index and transition state to OCR_DONE_WAITING.
        """
        with self._lock:
            entries = []
            for d in dialogues:
                d_id = int(d.get("id", 0))
                d_text = str(d.get("text", "")).strip()
                if d_text:
                    entries.append(DialogueEntry(
                        id=d_id,
                        text=d_text,
                        position=d.get("position"),
                        block_type=d.get("block_type", "DIALOGUE"),
                        reading_order=d.get("reading_order")
                    ))
            self._pages_buffer[page_index] = entries
            self._page_states[page_index] = PageState.OCR_DONE_WAITING

    def add_page_dialogues(self, page_index: int, dialogues: List[Dict[str, Any]]):
        """Alias for mark_page_ready."""
        self.mark_page_ready(page_index, dialogues)

    def mark_page_failed(self, page_index: int, error_msg: str = ""):
        """
        Mark a page as FAILED so the batch trigger does not get permanently blocked.
        """
        with self._lock:
            self._page_states[page_index] = PageState.FAILED

    def is_batch_ready(self) -> bool:
        """
        True if all session pages have finished (OCR_DONE_WAITING or FAILED) or timed out.
        """
        with self._lock:
            if not self._session_pages:
                return False
            
            all_resolved = all(
                self._page_states.get(p) in (PageState.OCR_DONE_WAITING, PageState.FAILED)
                for p in self._session_pages
            )
            if all_resolved:
                return True

            # Timeout check to prevent infinite UI deadlock
            if time.time() - self._session_start_time >= self._timeout_seconds:
                # At least one page must have dialogues ready to trigger
                has_any_ready = any(
                    self._page_states.get(p) == PageState.OCR_DONE_WAITING
                    for p in self._session_pages
                )
                return has_any_ready

            return False

    def is_timed_out(self) -> bool:
        with self._lock:
            return (time.time() - self._session_start_time) >= self._timeout_seconds

    def get_ready_page_indices(self) -> List[int]:
        with self._lock:
            return [
                p for p in self._session_pages
                if self._page_states.get(p) == PageState.OCR_DONE_WAITING
            ]

    def set_batch_sent(self):
        with self._lock:
            for p in self._session_pages:
                if self._page_states.get(p) == PageState.OCR_DONE_WAITING:
                    self._page_states[p] = PageState.BATCH_SENT

    def set_page_translated(self, page_index: int):
        with self._lock:
            self._page_states[page_index] = PageState.TRANSLATED

    def get_chapter_payload(self) -> ChapterBatchRequest:
        """
        Serialize all ready pages into a ChapterBatchRequest model.
        """
        with self._lock:
            if self._session_pages:
                ready_indices = sorted([
                    p for p in self._session_pages
                    if p in self._pages_buffer and self._pages_buffer[p]
                ])
            else:
                ready_indices = sorted([
                    p for p in self._pages_buffer.keys()
                    if self._pages_buffer[p]
                ])
            pages_list = [
                PageBatchEntry(page_index=idx, dialogues=self._pages_buffer[idx])
                for idx in ready_indices
            ]
            return ChapterBatchRequest(pages=pages_list)

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._pages_buffer) == 0 or all(len(d) == 0 for d in self._pages_buffer.values())

    def total_dialogues_count(self) -> int:
        with self._lock:
            return sum(len(d) for d in self._pages_buffer.values())

    def clear(self):
        with self._lock:
            self._pages_buffer.clear()
            self._page_states.clear()
            self._session_pages.clear()
            self._session_start_time = 0.0

    def unpack_response(self, response_data: Union[str, Dict[str, Any]]) -> Dict[int, Dict[int, Dict[str, str]]]:
        """
        Parse and unpack ChapterBatchResponse into:
        {page_index: {dialogue_id: {"translation": str, "emotion_tag": str}}}
        """
        if isinstance(response_data, str):
            cleaned = response_data.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
                cleaned = "\n".join(lines).strip()
            parsed_json = json.loads(cleaned)
        else:
            parsed_json = response_data

        validated_response = ChapterBatchResponse(**parsed_json)
        
        result_map: Dict[int, Dict[int, Dict[str, str]]] = {}
        for page in validated_response.pages:
            p_idx = page.page_index
            result_map[p_idx] = {}
            for d in page.dialogues:
                result_map[p_idx][d.id] = {
                    "translation": d.text,
                    "emotion_tag": d.emotion_tag or "normal"
                }
        
        return result_map
