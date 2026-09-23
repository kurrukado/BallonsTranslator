import threading
import time
import json
import re
import random
from enum import Enum
from typing import List, Dict, Optional, Any, Union, Tuple, Set
from pydantic import BaseModel, Field

from utils.logger import logger as LOGGER


class TranslationMode(str, Enum):
    RUN = "RUN"
    DIRECT = "DIRECT"


class PageState(str, Enum):
    IDLE = "IDLE"
    PENDING_OCR_INPAINT = "PENDING_OCR_INPAINT"
    OCR_INPAINT_DONE_WAITING = "OCR_INPAINT_DONE_WAITING"
    BATCH_SENT = "BATCH_SENT"
    TRANSLATED = "TRANSLATED"
    FAILED = "FAILED"


class RequestStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    SUCCESS = "SUCCESS"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    RETRYING = "RETRYING"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


# ---------------------------------------------------------
# Pydantic Schemas for Structured JSON Request & Response
# ---------------------------------------------------------

class DialogueBlock(BaseModel):
    id: Union[int, str] = Field(..., description="Unique dialogue block identifier.")
    text: str = Field(..., description="Source dialogue string.")
    page_index: int = Field(default=0, description="Zero-based page index.")
    reading_order: int = Field(default=0, description="Sequential reading order.")
    position: Optional[str] = Field(default=None, description="Spatial position e.g. Top-Right, Center.")
    block_type: Optional[str] = Field(default="DIALOGUE", description="Type: DIALOGUE, THOUGHT, NARRATION, SFX.")
    speaker_hint: Optional[str] = Field(default=None, description="Estimated speaker name.")
    emotion_hint: Optional[str] = Field(default="normal", description="Emotion/lettering tag.")


class PageBatch(BaseModel):
    page_id: str = Field(..., description="Page identifier e.g. page_001 or 001.jpg.")
    page_index: int = Field(..., description="Zero-based index.")
    blocks: List[DialogueBlock] = Field(default_factory=list, description="All blocks in this page.")


class ContextPayload(BaseModel):
    language_source: str = Field(default="English")
    language_target: str = Field(default="Tiếng Việt")
    glossary: Dict[str, str] = Field(default_factory=dict, description="Character & terminology mapping.")
    pronoun_rules: str = Field(default="", description="Pronoun consistency guidelines.")
    chapter_summary: str = Field(default="", description="Summary of current chapter events.")
    recent_dialogue_context: List[str] = Field(default_factory=list, description="Rolling context of last N blocks.")


class ChapterTranslationPayload(BaseModel):
    job_id: str = Field(default_factory=lambda: f"job_{int(time.time()*1000)}")
    chapter_id: str = Field(default="chapter_001")
    context: ContextPayload = Field(default_factory=ContextPayload)
    pages: List[PageBatch] = Field(default_factory=list)


class DialogueResponseBlock(BaseModel):
    id: Union[int, str] = Field(..., description="The exact matching numeric/string ID from request.")
    translation: str = Field(..., description="Natural Vietnamese translated dialogue.")
    emotion_tag: Optional[str] = Field(default="normal", description="shout, whisper, fear, surprise, normal.")


class PageResponseBatch(BaseModel):
    page_id: Optional[str] = Field(default=None)
    page_index: int = Field(..., description="Matching zero-based page index.")
    dialogues: List[DialogueResponseBlock] = Field(default_factory=list)


class ChapterTranslationResponse(BaseModel):
    pages: List[PageResponseBatch] = Field(default_factory=list)


# ---------------------------------------------------------
# Telemetry & Result Containers
# ---------------------------------------------------------

class TranslationRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: f"req_{int(time.time()*1000)}")
    mode: TranslationMode = Field(default=TranslationMode.RUN)
    job_id: str = Field(default="default_job")
    chapter_id: str = Field(default="default_chapter")
    page_ids: List[str] = Field(default_factory=list)
    block_count: int = Field(default=0)
    attempt: int = Field(default=1)
    model: str = Field(default="gemini-3.5-flash-lite")


class TranslationResult(BaseModel):
    request_id: str
    mode: TranslationMode
    status: RequestStatus
    model_used: str = ""
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    pages_data: Dict[int, Dict[Union[int, str], Dict[str, str]]] = Field(default_factory=dict)
    direct_text: str = ""
    error_message: Optional[str] = None


# ---------------------------------------------------------
# Context Engine & Rolling Memory
# ---------------------------------------------------------

class RollingContextEngine:
    def __init__(self, max_rolling_blocks: int = 6):
        self._lock = threading.Lock()
        self.max_rolling_blocks = max_rolling_blocks
        self.rolling_dialogues: List[str] = []
        self.glossary: Dict[str, str] = {}
        self.pronoun_rules: str = "Giữ nhất quán đại từ nhân xưng theo vai vế nhân vật (anh/em, cậu/tớ, mày/tao, hắn)."
        self.chapter_summary: str = ""

    def set_chapter_summary(self, summary: str):
        with self._lock:
            self.chapter_summary = summary.strip()

    def add_glossary(self, term: str, translation: str):
        with self._lock:
            self.glossary[term.strip()] = translation.strip()

    def update_rolling_history(self, translated_blocks: List[str]):
        with self._lock:
            for tb in translated_blocks:
                if tb and tb.strip():
                    self.rolling_dialogues.append(tb.strip())
            if len(self.rolling_dialogues) > self.max_rolling_blocks:
                self.rolling_dialogues = self.rolling_dialogues[-self.max_rolling_blocks:]

    def get_context_payload(self, src_lang: str, tgt_lang: str) -> ContextPayload:
        with self._lock:
            return ContextPayload(
                language_source=src_lang,
                language_target=tgt_lang,
                glossary=dict(self.glossary),
                pronoun_rules=self.pronoun_rules,
                chapter_summary=self.chapter_summary,
                recent_dialogue_context=list(self.rolling_dialogues)
            )

    def clear(self):
        with self._lock:
            self.rolling_dialogues.clear()
            self.glossary.clear()
            self.chapter_summary = ""


# ---------------------------------------------------------
# Robust JSON Cleaner & Repair Engine
# ---------------------------------------------------------

def clean_manga_punctuation(text: str) -> str:
    """Safely strips unnecessary trailing single periods ('.') from manga translation bubbles
    while carefully preserving ellipses ('...', '…', '..'), question marks ('?'),
    exclamations ('!'), tildes ('~'), and abbreviations."""
    if not text:
        return ""
    text = re.sub(r'[\u200b-\u200f\ufeff\u00ad]', '', str(text)).strip()
    if text.endswith('.') and not text.endswith('..') and not text.endswith('…'):
        lower_trimmed = text.lower()
        if lower_trimmed in {'dr.', 'mr.', 'mrs.', 'ms.', 'vs.', 'etc.', 'v.v.', 'v.v'}:
            return text
        text = text[:-1].rstrip()
    return text


def clean_and_repair_json(raw_text: str) -> Dict[str, Any]:
    """
    Robustly sanitizes, repairs, and extracts structured JSON responses from Gemini LLM:
    - Strips markdown code fences.
    - Removes trailing commas before } or ].
    - Automatically balances unclosed brackets/braces caused by mid-stream token limits.
    - Employs regex block extraction as a graceful fallback.
    """
    if not isinstance(raw_text, str):
        return raw_text

    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    # Fast path: Try valid standard JSON
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Repair 1: Trailing commas
    fixed = re.sub(r",\s*([\]}])", r"\1", cleaned)
    try:
        return json.loads(fixed)
    except Exception:
        pass

    # Repair 2: Balance unclosed brackets and braces
    repaired = fixed.rstrip(", \t\r\n")
    if repaired.count('"') % 2 != 0:
        repaired += '"'
    open_braces = repaired.count("{") - repaired.count("}")
    open_brackets = repaired.count("[") - repaired.count("]")
    repaired += ("]" * max(0, open_brackets)) + ("}" * max(0, open_braces))
    repaired = re.sub(r",\s*([\]}])", r"\1", repaired)
    try:
        return json.loads(repaired)
    except Exception:
        pass

    # Repair 3: Regex structured extraction fallback (Order-independent key matching)
    # Match pages and their dialogues
    pages_list = []
    page_matches = re.finditer(r'"page_index"\s*:\s*(\d+).*?"dialogues"\s*:\s*\[(.*?)\]', cleaned, re.DOTALL)
    found_any = False
    for pm in page_matches:
        p_idx = int(pm.group(1))
        d_block = pm.group(2)
        d_items = []
        for block_str in re.findall(r'\{[^{}]*\}', d_block):
            id_m = re.search(r'"id"\s*:\s*([^,\s}]+)', block_str)
            trans_m = re.search(r'"translation"\s*:\s*"(.*?)"', block_str, re.DOTALL)
            if id_m and trans_m:
                d_id_raw = id_m.group(1).strip('"\'} ')
                d_id = int(d_id_raw) if d_id_raw.isdigit() else d_id_raw
                d_trans = trans_m.group(1)
                emo_m = re.search(r'"emotion_tag"\s*:\s*"(.*?)"', block_str)
                d_emotion = emo_m.group(1) if emo_m else "normal"
                d_items.append({"id": d_id, "translation": d_trans, "emotion_tag": d_emotion})
        if d_items:
            found_any = True
            pages_list.append({"page_index": p_idx, "dialogues": d_items})

    if found_any:
        return {"pages": pages_list}

    # Final attempt: re-raise original JSON error
    return json.loads(cleaned)


# ---------------------------------------------------------
# Translation Orchestrator / Proxy
# ---------------------------------------------------------

class TranslationProxy:
    """
    Central Asynchronous Translation Proxy supporting:
    - Mode 1: Chapter-Batch Aggregator with 1:1 ID Validation & Dynamic Model Fallback.
    - Mode 2: Direct Single-Line Passthrough.
    """

    FALLBACK_MODELS = [
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash",
        "gemini-3.6-flash",
        "gemini-3.7-flash",
        "gemini-2.5-flash-lite",
    ]

    def __init__(self):
        self._lock = threading.Lock()
        self._page_blocks: Dict[int, List[DialogueBlock]] = {}
        self._page_states: Dict[int, PageState] = {}
        self._page_names: Dict[int, str] = {}
        self._session_pages: List[int] = []
        self._session_start_time: float = 0.0
        self._timeout_seconds: float = 60.0
        self._is_cancelled: bool = False
        self.context_engine = RollingContextEngine(max_rolling_blocks=6)

    def start_session(self, page_indices: List[int], page_names: Optional[Dict[int, str]] = None, timeout_seconds: float = 60.0):
        with self._lock:
            self._session_pages = list(page_indices)
            self._page_blocks.clear()
            self._page_states = {p: PageState.PENDING_OCR_INPAINT for p in page_indices}
            self._page_names = page_names or {p: f"page_{p:03d}" for p in page_indices}
            self._session_start_time = time.time()
            self._timeout_seconds = max(0.05, timeout_seconds)
            self._is_cancelled = False
            LOGGER.info(f"🌐 [TranslationProxy] Initialized session for {len(page_indices)} pages.")

    def cancel_session(self):
        with self._lock:
            self._is_cancelled = True
            self._page_blocks.clear()
            self._page_states.clear()
            self._session_pages.clear()
            LOGGER.info("🛑 [TranslationProxy] Session cancelled and cleaned up.")

    def is_cancelled(self) -> bool:
        with self._lock:
            return self._is_cancelled

    def mark_page_completed(self, page_index: int, dialogues: List[Dict[str, Any]], page_name: Optional[str] = None):
        """
        Page Completion Barrier: Called ONLY when OCR == SUCCESS and INPAINT == SUCCESS.
        Prevents duplicate enqueuing.
        """
        with self._lock:
            if self._is_cancelled:
                return

            if self._page_states.get(page_index) == PageState.OCR_INPAINT_DONE_WAITING:
                LOGGER.warning(f"⚠️ [TranslationProxy] Duplicate completion ignored for page {page_index}.")
                return

            if page_name:
                self._page_names[page_index] = page_name

            blocks = []
            for idx, d in enumerate(dialogues, start=1):
                b_id = d.get("id", idx)
                b_text = str(d.get("text", "")).strip()
                if b_text:
                    blocks.append(DialogueBlock(
                        id=b_id,
                        text=b_text,
                        page_index=page_index,
                        reading_order=d.get("reading_order", idx),
                        position=d.get("position"),
                        block_type=d.get("block_type", "DIALOGUE"),
                        speaker_hint=d.get("speaker_hint"),
                        emotion_hint=d.get("emotion_hint", "normal")
                    ))

            self._page_blocks[page_index] = blocks
            self._page_states[page_index] = PageState.OCR_INPAINT_DONE_WAITING
            LOGGER.info(f"⏳ [TranslationProxy] Buffered {len(blocks)} blocks for page {page_index} ({self._page_names.get(page_index)}).")

    def mark_page_failed(self, page_index: int, error_msg: str = ""):
        with self._lock:
            self._page_states[page_index] = PageState.FAILED
            LOGGER.warning(f"❌ [TranslationProxy] Page {page_index} marked as FAILED: {error_msg}")

    def is_batch_ready(self) -> bool:
        with self._lock:
            if not self._session_pages or self._is_cancelled:
                return False

            all_resolved = all(
                self._page_states.get(p) in (PageState.OCR_INPAINT_DONE_WAITING, PageState.FAILED)
                for p in self._session_pages
            )
            if all_resolved:
                return True

            if (time.time() - self._session_start_time) >= self._timeout_seconds:
                return any(
                    self._page_states.get(p) == PageState.OCR_INPAINT_DONE_WAITING
                    for p in self._session_pages
                )

            return False

    def build_sub_batch_payloads(
        self,
        src_lang: str = "English",
        tgt_lang: str = "Tiếng Việt",
        max_pages_per_batch: int = 8,
        max_dialogues_per_batch: int = 40
    ) -> List[ChapterTranslationPayload]:
        """
        Splits chapter pages into safe, manageable sub-batches to prevent token overflow.
        """
        with self._lock:
            ready_pages = sorted([
                p for p in self._session_pages
                if p in self._page_blocks and self._page_blocks[p]
            ]) if self._session_pages else sorted([p for p in self._page_blocks.keys() if self._page_blocks[p]])

            sub_batches: List[List[PageBatch]] = []
            curr_batch: List[PageBatch] = []
            curr_dialogues_count = 0

            for p_idx in ready_pages:
                p_name = self._page_names.get(p_idx, f"page_{p_idx:03d}")
                p_blocks = self._page_blocks[p_idx]
                pb = PageBatch(page_id=p_name, page_index=p_idx, blocks=p_blocks)

                if curr_batch and (len(curr_batch) >= max_pages_per_batch or (curr_dialogues_count + len(p_blocks)) > max_dialogues_per_batch):
                    sub_batches.append(curr_batch)
                    curr_batch = []
                    curr_dialogues_count = 0

                curr_batch.append(pb)
                curr_dialogues_count += len(p_blocks)

            if curr_batch:
                sub_batches.append(curr_batch)

            payloads = []
            for idx, batch in enumerate(sub_batches, start=1):
                ctx = self.context_engine.get_context_payload(src_lang, tgt_lang)
                payloads.append(ChapterTranslationPayload(
                    job_id=f"job_{int(time.time()*1000)}_b{idx}",
                    chapter_id="current_chapter",
                    context=ctx,
                    pages=batch
                ))

            return payloads

    def build_chapter_payload(self, src_lang: str = "English", tgt_lang: str = "Tiếng Việt") -> ChapterTranslationPayload:
        sub_payloads = self.build_sub_batch_payloads(src_lang, tgt_lang, max_pages_per_batch=999, max_dialogues_per_batch=9999)
        return sub_payloads[0] if sub_payloads else ChapterTranslationPayload()

    def get_chapter_payload(self, src_lang: str = "English", tgt_lang: str = "Tiếng Việt") -> ChapterTranslationPayload:
        """Backward compatibility alias for build_chapter_payload."""
        return self.build_chapter_payload(src_lang, tgt_lang)

    def unpack_response(self, response_data: Union[str, Dict[str, Any]]) -> Dict[int, Dict[Union[int, str], Dict[str, str]]]:
        """Backward compatibility helper for unpack_response."""
        payload = self.build_chapter_payload()
        _, result_map, _ = self.validate_and_unpack(payload, response_data)
        return result_map or {}

    def validate_and_unpack(
        self,
        payload: ChapterTranslationPayload,
        response_json: Union[str, Dict[str, Any]]
    ) -> Tuple[bool, Optional[Dict[int, Dict[Union[int, str], Dict[str, str]]]], str]:
        """
        Strict 1:1 ID Validation & Set Equality Check:
        - input_ids == output_ids strictly
        - Reject omitted IDs
        - Reject unknown/hallucinated IDs
        - Restore original sequence order regardless of Gemini output shuffling
        """
        try:
            parsed_data = clean_and_repair_json(response_json) if isinstance(response_json, str) else response_json
            resp_obj = ChapterTranslationResponse(**parsed_data)
        except Exception as e:
            return False, None, f"JSON/Pydantic Parsing Error: {e}"

        # Collect expected vs received IDs per page
        expected_page_ids: Dict[int, Set[Union[int, str]]] = {}
        for pb in payload.pages:
            expected_page_ids[pb.page_index] = {str(b.id) for b in pb.blocks}

        received_page_ids: Dict[int, Set[str]] = {}
        result_map: Dict[int, Dict[Union[int, str], Dict[str, str]]] = {}

        for resp_page in resp_obj.pages:
            p_idx = resp_page.page_index
            if p_idx not in expected_page_ids:
                return False, None, f"Unknown page_index {p_idx} in response."

            received_page_ids[p_idx] = set()
            result_map[p_idx] = {}

            for d in resp_page.dialogues:
                d_id_str = str(d.id)
                if d_id_str not in expected_page_ids[p_idx]:
                    return False, None, f"Unknown/Hallucinated ID '{d.id}' returned for page {p_idx}."
                
                received_page_ids[p_idx].add(d_id_str)
                # Store original key (int or str)
                result_map[p_idx][d.id] = {
                    "translation": clean_manga_punctuation(d.translation),
                    "emotion_tag": d.emotion_tag or "normal"
                }

        # Check for missing/omitted IDs
        for p_idx, exp_ids in expected_page_ids.items():
            rec_ids = received_page_ids.get(p_idx, set())
            missing = exp_ids - rec_ids
            if missing:
                return False, None, f"Validation Failure: Omitted IDs {missing} on page {p_idx}."

        # Order restoration: Re-index to ensure stable sorting & non-empty content
        ordered_result_map: Dict[int, Dict[Union[int, str], Dict[str, str]]] = {}
        for pb in payload.pages:
            p_idx = pb.page_index
            ordered_result_map[p_idx] = {}
            for block in pb.blocks:
                b_id = block.id
                # Match both raw id and string/int variations
                item = result_map[p_idx].get(b_id) or result_map[p_idx].get(str(b_id)) or (result_map[p_idx].get(int(b_id)) if str(b_id).isdigit() else None)
                if not item:
                    return False, None, f"Validation Failure: Missing translation entry for block ID '{b_id}' on page {p_idx}."
                if not str(item.get("translation", "")).strip() and block.text.strip():
                    return False, None, f"Validation Failure: Empty translation returned for block ID '{b_id}' on page {p_idx}."
                ordered_result_map[p_idx][b_id] = item

        return True, ordered_result_map, "OK"

    def total_dialogues_count(self) -> int:
        with self._lock:
            return sum(len(b) for b in self._page_blocks.values())

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._page_blocks) == 0 or all(len(b) == 0 for b in self._page_blocks.values())
