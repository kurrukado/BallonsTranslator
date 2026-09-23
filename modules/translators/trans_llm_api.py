import re
import time
import json
import random
import hashlib
import traceback
from pathlib import Path
from typing import List, Dict, Optional, Type, Any, Union

import httpx

class _LazyOpenAI:
    def __getattr__(self, name):
        import openai
        return getattr(openai, name)

openai = _LazyOpenAI()

from pydantic import BaseModel, Field, ValidationError

from .base import BaseTranslator, register_translator
from utils.quota_tracker import QUOTA_TRACKER
from .context_engine import (
    CharacterProfile,
    CharacterMemory,
    GlossaryManager,
    DialogueItem,
    ContextAssembler,
)
from .translation_state_buffer import (
    TranslationStateBuffer,
    ChapterBatchRequest,
    ChapterBatchResponse,
)


class InvalidNumTranslations(Exception):
    """Exception raised when the number of translations does not match the number of sources."""

    pass


def sanitize_text(text: str) -> str:
    """Safely strips invisible zero-width spaces, soft hyphens, BOMs, and leading/trailing whitespace."""
    if not text:
        return ""
    return re.sub(r'[\u200b-\u200f\ufeff\u00ad]', '', str(text)).strip()


def clean_manga_punctuation(text: str) -> str:
    """Safely strips unnecessary trailing single periods ('.') from manga translation bubbles
    while carefully preserving ellipses ('...', '…', '..'), question marks ('?'),
    exclamations ('!'), tildes ('~'), and abbreviations."""
    if not text:
        return ""
    text = sanitize_text(text)
    if text.endswith('.') and not text.endswith('..') and not text.endswith('…'):
        lower_trimmed = text.lower()
        if lower_trimmed in {'dr.', 'mr.', 'mrs.', 'ms.', 'vs.', 'etc.', 'v.v.', 'v.v'}:
            return text
        text = text[:-1].rstrip()
    return text


class TranslationElement(BaseModel):
    id: int = Field(..., description="The original numeric ID of the text snippet.")
    translation: str = Field(
        ..., description="The translated text corresponding to the id."
    )
    emotion_tag: Optional[str] = Field(
        default="normal",
        description="Emotion/lettering tag: 'shout' (Hét/nhấn mạnh), 'whisper' (Thì thầm/do dự), 'fear' (Sợ hãi/hoảng loạn), 'surprise' (Ngạc nhiên/sốc ?!), 'normal' (Bình thường)."
    )


class TranslationResponse(BaseModel):
    translations: List[TranslationElement] = Field(
        ..., description="The list of translated elements."
    )


@register_translator("LLM_API_Translator")
class LLM_API_Translator(BaseTranslator):
    concate_text = False
    cht_require_convert = True
    params: Dict = {
        "provider": {
            "type": "selector",
            "options": ["Gemini Proxy", "Google", "OpenAI", "OpenRouter", "Ollama"],
            "value": "Gemini Proxy",
            "description": "Select the LLM provider.",
        },
        "apikey": {
            "value": "",
            "description": "Single API key to use if multiple keys are not provided (optional for Gemini Proxy).",
        },
        "multiple_keys": {
            "type": "editor",
            "value": "",
            "description": "API keys separated by semicolons (;). Requests will rotate through these keys.",
        },
        "model": {
            "type": "selector",
            "options": [
                "gemini-3.8-flash",
                "gemini-3.7-flash",
                "gemini-3.6-flash",
                "gemini-3.5-flash",
                "gemini-3.5-flash-lite",
            ],
            "value": "gemini-3.8-flash",
            "description": "Select the Gemini model for translation (Primary: gemini-3.8-flash).",
        },
        "override model": {
            "value": "",
            "description": "Specify a custom model name to override the selected model.",
        },
        "endpoint": {
            "value": "",
            "description": "Base URL for the API. Leave empty for provider default.",
        },
        "system_prompt": {
            "type": "editor",
            "value": (
                "Bạn là Dịch giả Chủ chốt của Nhóm Dịch Manga/Manhwa Scanlation Việt Nam hàng đầu.\n"
                "SỨ MỆNH: Bản dịch phải CỰC KỲ TỰ NHIÊN, LẦY LỘI, HÀI HƯỚC, MẶN MÀ VÀ BỰA ĐÚNG LÚC như phong cách các nhóm dịch truyện tranh nổi tiếng.\n"
                "TUYỆT ĐỐI NÓI KHÔNG VỚI VĂN MẪU KHÔ CỨNG, dịch máy móc hoặc dịch thô từ điển.\n\n"
                "QUY TẮC BẢN ĐỊA HÓA SCANLATION ĐỈNH CAO:\n"
                "1. TỰ ĐỘNG SUY LUẬN VAI VẾ & ĐẠI TỪ (Zero-Config Role & Pronoun Inference):\n"
                "   - Vợ chồng / Người yêu / Romance: BẮT BUỘC xưng hô 'Anh - Em' (hoặc xưng tên thân mật). Tự nhiên, trêu ghẹo, ngọt ngào. TUYỆT ĐỐI KHÔNG dùng 'tớ - cậu' hay 'tôi - cô' trong ngữ cảnh tình cảm/hôn nhân.\n"
                "   - Bạn thân / Đồng trang lứa lầy lội: Linh hoạt 'mày - tao' (khi cà khịa, chửi đùa) hoặc 'cậu - tớ / mình' (thân thiện).\n"
                "   - Tình huống bất lực / Cáu gắt / Tấu hài: Thoại mỉa mai, xéo xắt, chêm từ lóng biểu cảm tự nhiên (như 'toang rồi', 'ối dồi ôi', 'chết dở', 'vãi chưởng', 'ảo ma', 'mất mặt ghê', 'hết nước chấm', 'bó tay').\n"
                "   - Độc thoại nội tâm (THOUGHT): Tự vấn tự nhiên ('mình...', 'quái lạ...', 'trời ạ...').\n"
                "   - Chiến đấu / Kẻ thù: Mày - Tao, Ngươi - Ta, Tên khốn.\n"
                "   - Cổ trang / Kiếm hiệp: Huynh - Đệ, Bổn tọa, Tiểu thư, Công tử.\n"
                "2. KHẨU NGỮ ĐỜI THƯỜNG & TRỢ TỪ CẢM THÁN:\n"
                "   - Dùng linh hoạt trợ từ tiếng Việt để câu thoại có hồn: 'nè', 'cơ chứ', 'chứ lị', 'đấy nhé', 'ơi là trời', 'hả trời', 'nhen', 'chứ sao'.\n"
                "   - Dịch thoát ý câu đùa, thành ngữ, chơi chữ sang tiếng lóng/khẩu ngữ tương đương của giới trẻ Việt Nam.\n"
                "3. NỐI MẠCH CÂU ĐA BONG BÓNG (Multi-Bubble Sentence Stitching):\n"
                "   - Khi 1 câu bị ngắt qua 2-4 bong bóng: Ghép nối ý nghĩa trọn vẹn trước rồi chia vế mượt mà về từng ID, không để câu cụt lủn hay tối nghĩa.\n"
                "4. DẤU CÂU KHUNG THOẠI MANGA:\n"
                "   - TUYỆT ĐỐI KHÔNG để dấu chấm đơn ('.') ở cuối câu thoại trong bong bóng (trừ ba chấm '...' hoặc viết tắt).\n"
                "5. TỰ SỬA LỖI CHÍNH TẢ OCR (Typo Auto-Healing):\n"
                "   - Tự động đoán từ đúng dựa vào ngữ cảnh nếu chữ OCR bị quét lem nhem.\n"
                "6. BẢO TOÀN NỘI DUNG & SCHEMA:\n"
                "   - Bựa và hài hước nhưng KHÔNG bịa đặt sai lệch cốt truyện. Giữ đúng 1:1 ID thoại.\n"
                "   - Trả về JSON hợp lệ: {\"translations\": [{\"id\": 1, \"translation\": \"Thoại dịch\"}]}"
            ),
            "description": "System message to instruct the LLM on context-aware manga translation rules and JSON schema.",
        },
        "invalid repeat count": {
            "value": 2,
            "description": "Number of retries if the count of translations mismatches the source count.",
        },
        "max requests per minute": {
            "value": 10,
            "description": "Maximum requests per minute for EACH API key.",
        },
        "delay": {
            "value": 0.3,
            "description": "Global delay in seconds between requests.",
        },
        "max tokens": {
            "value": 4096,
            "description": "Maximum tokens for the response.",
        },
        "temperature": {
            "value": 0.1,
            "description": "Sampling temperature. Lower values are recommended for structured output.",
        },
        "top p": {
            "value": 1.0,
            "description": "Top P for sampling.",
        },
        "retry attempts": {
            "value": 3,
            "description": "Number of retry attempts on API connection or parsing failures.",
        },
        "retry timeout": {
            "value": 15,
            "description": "Timeout between retry attempts (seconds).",
        },
        "proxy": {
            "value": "",
            "description": "Proxy address (e.g., http(s)://user:password@host:port or socks4/5://user:password@host:port)",
        },
        "frequency penalty": {
            "value": 0.0,
            "description": "Frequency penalty (OpenAI).",
        },
        "presence penalty": {"value": 0.0, "description": "Presence penalty (OpenAI)."},
        "low vram mode": {
            'value': False,
            'description': 'check it if you\'re running it locally on a single device and encountered a crash due to vram OOM',
            'type': 'checkbox',
        }
    }

    def _setup_translator(self):
        self.lang_map = {
            "Auto": "Auto Detect",
            "Japanese": "Japanese",
            "日本語": "Japanese",
            "English": "English",
            "Tiếng Việt": "Vietnamese",
            "Vietnamese": "Vietnamese",
            "简体中文": "Simplified Chinese",
            "繁體中文": "Traditional Chinese",
            "한국어": "Korean",
            "čeština": "Czech",
            "Français": "French",
            "Deutsch": "German",
            "magyar nyelv": "Hungarian",
            "Italiano": "Italian",
            "Polski": "Polish",
            "Português": "Portuguese",
            "limba română": "Romanian",
            "русский язык": "Russian",
            "Español": "Spanish",
            "Türk dili": "Turkish",
            "украї́нська мо́ва": "Ukrainian",
            "Thai": "Thai",
            "Arabic": "Arabic",
            "Malayalam": "Malayalam",
            "Tamil": "Tamil",
            "Hindi": "Hindi",
        }
        if not hasattr(self, "lang_target") or not self.lang_target or self.lang_target in ["简体中文", "Simplified Chinese"]:
            self.lang_target = "Tiếng Việt"
        if not hasattr(self, "lang_source") or not self.lang_source or self.lang_source in ["简体中文", "Simplified Chinese", ""]:
            self.lang_source = "English"
        self.token_count = 0
        self.token_count_last = 0
        self.current_key_index = 0
        self.last_request_time = 0
        self.request_count_minute = 0
        self.minute_start_time = time.time()
        self.key_usage = {}
        self._model_last_request_time: Dict[str, float] = {}
        self.MODEL_RPM_LIMITS: Dict[str, int] = {
            "gemini-3.8-flash": 5,
            "gemini-3.7-flash": 5,
            "gemini-3.6-flash": 5,
            "gemini-3.5-flash": 5,
            "gemini-3.5-flash-lite": 15,
        }
        self.client = None
        self._http_client: Optional[httpx.Client] = None
        self._cached_client: Optional[openai.OpenAI] = None
        self._last_client_config: Optional[tuple] = None
        self._translation_cache: Dict[str, str] = {}
        self.character_memory = CharacterMemory()
        self.glossary_manager = GlossaryManager()
        self.story_genre = "Manga / Comic"
        self.active_endpoint = ""

    def _normalize_model(self, model: str) -> str:
        if not model:
            return "gemini-3.8-flash"
        m = model.strip()
        if ": " in m:
            m = m.split(": ", 1)[1]
        if m.startswith("models/"):
            m = m[7:]
        return m

    def _initialize_client(self, api_key_to_use: str) -> bool:
        if not api_key_to_use or not api_key_to_use.startswith("AIza"):
            try:
                from utils.config import ProgramConfig
                cfg_key = ProgramConfig().get_param("translator", "apikey", "")
                if cfg_key and cfg_key.startswith("AIza"):
                    api_key_to_use = cfg_key
            except Exception:
                pass

        endpoint = self.endpoint
        provider = self.provider
        is_local_proxy = provider == "Gemini Proxy" or (endpoint and ("127.0.0.1:8080" in endpoint or "localhost:8080" in endpoint))

        if is_local_proxy:
            try:
                from utils.gemini_proxy_launcher import ensure_gemini_proxy_running, sync_proxy_api_key, is_proxy_alive
                if not is_proxy_alive():
                    if not ensure_gemini_proxy_running():
                        endpoint = "https://generativelanguage.googleapis.com/v1beta/openai"
                        self.logger.info("[Gemini Proxy] Local proxy offline. Routing directly through Google AI Studio endpoint.")
                    else:
                        endpoint = "http://127.0.0.1:8080/v1"
                else:
                    endpoint = "http://127.0.0.1:8080/v1"
                if is_proxy_alive() and api_key_to_use and api_key_to_use.startswith("AIza"):
                    sync_proxy_api_key(api_key_to_use)
            except Exception as e:
                endpoint = "https://generativelanguage.googleapis.com/v1beta/openai"
        elif not endpoint:
            if provider == "Google":
                endpoint = "https://generativelanguage.googleapis.com/v1beta/openai"
            elif provider == "OpenAI":
                endpoint = "https://api.openai.com/v1"
            elif provider == "OpenRouter":
                endpoint = "https://openrouter.ai/api/v1"
            elif provider == "Grok":
                endpoint = "https://api.x.ai/v1"
            elif provider == "Ollama":
                endpoint = "http://127.0.0.1:11434/v1"
            else:
                endpoint = "https://generativelanguage.googleapis.com/v1beta/openai"

        self.active_endpoint = endpoint
        proxy = self.proxy
        current_config = (provider, endpoint, api_key_to_use, proxy)

        # Reuse existing client if configuration has not changed
        if self._cached_client is not None and self._last_client_config == current_config:
            self.client = self._cached_client
            return True

        if self._http_client is not None:
            try:
                self._http_client.close()
            except Exception:
                pass
            self._http_client = None

        if proxy:
            try:
                proxy_mounts = {
                    "http://": httpx.HTTPTransport(proxy=proxy),
                    "https://": httpx.HTTPTransport(proxy=proxy),
                }
                self._http_client = httpx.Client(mounts=proxy_mounts, timeout=httpx.Timeout(180.0, connect=10.0))
            except Exception as e:
                self.logger.error(
                    f"Failed to initialize proxy '{proxy}': {e}. Proceeding without proxy."
                )
                self._http_client = httpx.Client(timeout=httpx.Timeout(180.0, connect=10.0))
        else:
            self._http_client = httpx.Client(timeout=httpx.Timeout(180.0, connect=10.0))

        masked_key = (
            api_key_to_use[:4] + "..." + api_key_to_use[-4:]
            if len(api_key_to_use) > 8
            else api_key_to_use
        )
        self.logger.debug(
            f"Initializing client for {provider} with key {masked_key} at endpoint {endpoint}"
        )

        try:
            self._cached_client = openai.OpenAI(
                api_key=api_key_to_use, base_url=endpoint, http_client=self._http_client
            )
            self._last_client_config = current_config
            self.client = self._cached_client
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            self.client = None
            self._cached_client = None
            return False

    # --- Property getters ---
    @property
    def provider(self) -> str:
        return self.get_param_value("provider")

    @property
    def apikey(self) -> str:
        key = self.get_param_value("apikey")
        if not key or not key.startswith("AIza"):
            try:
                from utils.config import pcfg
                cfg_key = pcfg.module.translator_params.get("LLM_API_Translator", {}).get("apikey", "")
                if isinstance(cfg_key, dict):
                    cfg_key = cfg_key.get("value", "")
                if cfg_key and cfg_key.startswith("AIza"):
                    return cfg_key
            except Exception:
                pass
            try:
                cfg_path = Path("config/config.json")
                if cfg_path.exists():
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    cfg_key = data.get("module", {}).get("translator_params", {}).get("LLM_API_Translator", {}).get("apikey", "")
                    if cfg_key and cfg_key.startswith("AIza"):
                        return cfg_key
            except Exception:
                pass
        return key

    @property
    def multiple_keys_list(self) -> List[str]:
        keys_str = self.get_param_value("multiple_keys")
        if not isinstance(keys_str, str):
            return []
        return [
            key.strip()
            for key in keys_str.strip().replace("\n", ";").split(";")
            if key.strip()
        ]

    @property
    def model(self) -> str:
        return self.get_param_value("model")

    @property
    def override_model(self) -> Optional[str]:
        return self.get_param_value("override model") or None

    @property
    def endpoint(self) -> Optional[str]:
        return self.get_param_value("endpoint") or None

    @property
    def temperature(self) -> float:
        return float(self.get_param_value("temperature"))

    @property
    def top_p(self) -> float:
        return float(self.get_param_value("top p"))

    @property
    def max_tokens(self) -> int:
        return int(self.get_param_value("max tokens"))

    @property
    def retry_attempts(self) -> int:
        return int(self.get_param_value("retry attempts"))

    @property
    def retry_timeout(self) -> int:
        return int(self.get_param_value("retry timeout"))

    @property
    def proxy(self) -> str:
        return self.get_param_value("proxy")

    @property
    def system_prompt(self) -> str:
        return self.get_param_value("system_prompt")

    @property
    def invalid_repeat_count(self) -> int:
        return int(self.get_param_value("invalid repeat count"))

    @property
    def frequency_penalty(self) -> float:
        return float(self.get_param_value("frequency penalty"))

    @property
    def presence_penalty(self) -> float:
        return float(self.get_param_value("presence penalty"))

    @property
    def max_rpm(self) -> int:
        return int(self.get_param_value("max requests per minute"))

    @property
    def global_delay(self) -> float:
        return float(self.get_param_value("delay"))

    def _assemble_prompts(
        self,
        items_or_queries: Union[List[DialogueItem], List[str]],
        to_lang: str,
        dialogue_history: Optional[List[DialogueItem]] = None,
        max_batch_size: int = 40
    ):
        from_lang = self.lang_map.get(self.lang_source, self.lang_source)

        # Normalize to DialogueItems
        dialogue_items: List[DialogueItem] = []
        for idx, elem in enumerate(items_or_queries):
            if isinstance(elem, DialogueItem):
                dialogue_items.append(elem)
            else:
                dialogue_items.append(DialogueItem(id=idx + 1, source=str(elem)))

        for i in range(0, len(dialogue_items), max_batch_size):
            chunk = dialogue_items[i : i + max_batch_size]

            # Context-Safe Intra-batch deduplication to minimize prompt tokens sent to Gemini
            unique_chunk: List[DialogueItem] = []
            source_to_prompt_id: Dict[Tuple, int] = {}
            for item in chunk:
                dedup_key = (
                    item.source.strip(),
                    item.speaker or "",
                    item.block_type or "DIALOGUE",
                    item.connected_group_id
                )
                if dedup_key not in source_to_prompt_id:
                    p_id = len(unique_chunk) + 1
                    source_to_prompt_id[dedup_key] = p_id
                    unique_chunk.append(
                        DialogueItem(
                            id=p_id,
                            source=item.source,
                            translated=item.translated,
                            position=item.position,
                            direction=item.direction,
                            block_type=item.block_type,
                            speaker=item.speaker,
                            connected_group_id=item.connected_group_id,
                            flow_hint=item.flow_hint,
                        )
                    )

            payload = ContextAssembler.assemble_payload(
                current_items=unique_chunk,
                to_lang=to_lang,
                from_lang=from_lang,
                dialogue_history=dialogue_history,
                character_memory=self.character_memory,
                glossary_manager=self.glossary_manager,
                story_genre=self.story_genre
            )

            if to_lang == "Vietnamese" or self.lang_target in ["Tiếng Việt", "Vietnamese"]:
                target_desc = "Vietnamese (Tiếng Việt)"
                instructions = (
                    "CRITICAL LOCALIZATION INSTRUCTIONS:\n"
                    "1. You MUST translate every text item strictly into natural, fluent, emotionally gripping Vietnamese (Tiếng Việt).\n"
                    "2. COHESIVE MULTI-BUBBLE FLOW: For items marked with 'connected_group', synthesize the compound sentence smoothly, then partition the translated clauses across the corresponding IDs so they read seamlessly.\n"
                    "3. PROPER TONE & PERSONA: Inner monologues (THOUGHT) must use natural self-reflection ('mình', 'cô ấy') instead of robotic 'tôi'. Dialogue must fit character relationships and story atmosphere.\n"
                    "4. DO NOT output Chinese, Japanese, or unlocalized foreign characters in the 'translation' field.\n"
                    "5. Localize sound effects (SFX) into expressive Vietnamese onomatopoeia (e.g., ドン/POW -> RẦM!/BỐP!, ザッ -> XOẸT!, ドキドキ -> THỊCH THỊCH).\n"
                    "6. OCR NOISE RESILIENCE: Text may contain minor OCR/scan errors (e.g., 'vvhat' -> 'what', 'c1ick' -> 'click', rách nét chữ). Use context to intelligently correct these without fabricating content.\n"
                    f"7. Respond strictly with a JSON object containing all {len(unique_chunk)} translated items."
                )
            else:
                target_desc = to_lang
                instructions = (
                    "CRITICAL LOCALIZATION INSTRUCTIONS:\n"
                    f"1. You MUST translate every text item strictly and naturally into {to_lang}.\n"
                    "2. COHESIVE MULTI-BUBBLE FLOW: For items marked with 'connected_group', synthesize the compound sentence smoothly, then partition the translated clauses across the corresponding IDs so they read seamlessly.\n"
                    "3. PROPER TONE & PERSONA: Match character relationships and story atmosphere.\n"
                    "4. OCR NOISE RESILIENCE: Text may contain minor OCR/scan errors. Use context to correct them intelligently.\n"
                    f"5. Respond strictly with a JSON object containing all {len(unique_chunk)} translated items."
                )

            prompt = (
                f"TASK SPECIFICATION:\n"
                f"- SOURCE LANGUAGE: {from_lang}\n"
                f"- TARGET LANGUAGE: {target_desc}\n\n"
                f"{instructions}\n\n"
                f"INPUT CONTEXT & DIALOGUES:\n{payload}"
            )

            yield prompt, len(unique_chunk), chunk, source_to_prompt_id

    def _respect_model_delay(self, model_name: str):
        """Strict Free-Tier safe rate limiting calculated dynamically per Gemini model RPM limit."""
        now = time.time()
        rpm = self.MODEL_RPM_LIMITS.get(model_name, self.max_rpm if self.max_rpm > 0 else 15)
        min_interval = max(60.0 / rpm, self.global_delay)
        
        last_time = getattr(self, "_model_last_request_time", {}).get(model_name, 0.0)
        time_since = now - last_time
        if time_since < min_interval:
            sleep_time = min_interval - time_since
            self.logger.info(f"⏳ [Free-Tier Safety Throttle] Model '{model_name}' (Limit {rpm} RPM): Tạm dừng {sleep_time:.2f}s để bảo đảm không vượt quota...")
            time.sleep(sleep_time)
            
        if not hasattr(self, "_model_last_request_time"):
            self._model_last_request_time = {}
        self._model_last_request_time[model_name] = time.time()

    def _respect_delay(self):
        current_time = time.time()
        rpm = self.max_rpm
        delay = self.global_delay
        if rpm > 0:
            if current_time - self.minute_start_time >= 60:
                self.request_count_minute = 0
                self.minute_start_time = current_time
            if self.request_count_minute >= rpm:
                wait_time = 60.1 - (current_time - self.minute_start_time)
                if wait_time > 0:
                    self.logger.warning(
                        f"Global RPM limit ({rpm}) reached. Waiting {wait_time:.2f} seconds."
                    )
                    time.sleep(wait_time)
                self.request_count_minute = 0
                self.minute_start_time = time.time()

        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < delay:
            sleep_time = delay - time_since_last_request
            if hasattr(self, "debug_mode") and self.debug_mode:
                self.logger.debug(f"Global delay: Waiting {sleep_time:.3f} seconds.")
            time.sleep(sleep_time)

        self.last_request_time = time.time()
        self.request_count_minute += 1

    def _respect_key_limit(self, key: str) -> bool:
        rpm = self.max_rpm
        if rpm <= 0:
            return True
        now = time.time()
        count, start_time = self.key_usage.get(key, (0, now))
        if now - start_time >= 60:
            count, start_time = 0, now
            self.key_usage[key] = (count, start_time)
        if count >= rpm:
            wait_time = 60.1 - (now - start_time)
            if wait_time > 0:
                self.logger.warning(
                    f"RPM limit ({rpm}) reached for key {key[:6]}... Waiting {wait_time:.2f} seconds."
                )
                time.sleep(wait_time)
            self.key_usage[key] = (0, time.time())
            return False
        return True

    def _select_api_key(self) -> Optional[str]:
        api_keys = self.multiple_keys_list
        single_key = self.apikey
        if not api_keys and not single_key:
            if self.provider in ["Gemini Proxy", "LLM Studio", "Ollama"]:
                return "gemini-proxy-key"
            self.logger.error("No API keys provided in parameters.")
            return None

        if not api_keys:
            if self._respect_key_limit(single_key):
                now = time.time()
                count, start_time = self.key_usage.get(single_key, (0, now))
                if now - start_time >= 60:
                    count = 0
                    start_time = now
                self.key_usage[single_key] = (count + 1, start_time)
                return single_key
            return None

        start_index = self.current_key_index
        for i in range(len(api_keys)):
            index = (start_index + i) % len(api_keys)
            key = api_keys[index]
            if self._respect_key_limit(key):
                now = time.time()
                count, start_time = self.key_usage.get(key, (0, now))
                self.key_usage[key] = (count + 1, start_time)
                self.current_key_index = (index + 1) % len(api_keys)
                return key
        self.logger.error("All available API keys are currently rate-limited.")
        return None

    def _get_api_key(self) -> str:
        """Helper to get currently selected or configured API key."""
        return self._select_api_key() or self.apikey or ""

    def _request_translation(self, prompt: str) -> Optional[TranslationResponse]:
        current_api_key = self._select_api_key()
        
        if not current_api_key:
            if self.provider in ["LLM Studio", "Ollama", "Gemini Proxy"]:
                current_api_key = "gemini-proxy-key"
            else:
                raise ConnectionError("No available API key found.")

        if self.provider == "LLM Studio" and not self.endpoint:
            raise ValueError(
                "Endpoint must be specified when using the LLM Studio provider (e.g., http://localhost:1234/v1)."
            )

        if not self._initialize_client(current_api_key):
            raise ConnectionError("Failed to initialize API client.")

        self._respect_delay()

        model_name = self.override_model or self.model
        if ": " in model_name:
            model_name = model_name.split(": ", 1)[1]

        # Map model aliases to official Google Gemini API model IDs
        # Note: gemini-3.7-flash, gemini-3.6-flash, gemini-3.5-flash-lite are REAL model names on Google API
        if self.provider in ["Gemini Proxy", "Google"]:
            target_model = model_name  # Use as-is, these are real Google model names
        else:
            target_model = model_name

        from_lang = self.lang_map.get(self.lang_source, self.lang_source)
        to_lang = self.lang_map.get(self.lang_target, self.lang_target)

        if to_lang == "Vietnamese" or self.lang_target in ["Tiếng Việt", "Vietnamese"]:
            target_desc = "Vietnamese (Tiếng Việt)"
            constraint_rule = "- Translate strictly into natural, fluent Vietnamese (Tiếng Việt).\n- NEVER output unlocalized foreign characters in the 'translation' field."
        else:
            target_desc = to_lang
            constraint_rule = f"- Translate strictly into natural {to_lang}."

        system_instruction = (
            f"{self.system_prompt}\n\n"
            f"ACTIVE TASK CONSTRAINTS:\n"
            f"- SOURCE LANGUAGE: {from_lang}\n"
            f"- TARGET LANGUAGE: {target_desc}\n"
            f"{constraint_rule}"
        )

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ]

        api_args = {
            "model": target_model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }

        if self.provider == "LLM Studio":
            self.logger.debug("Using 'json_schema' mode for LLM Studio.")
            api_args["response_format"] = {
                "type": "json_schema",
                "json_schema": {"schema": TranslationResponse.model_json_schema()},
            }
        elif self.provider in ["OpenAI", "Grok", "Google", "OpenRouter", "Ollama", "Gemini Proxy"]:
            self.logger.debug(f"Using 'json_object' mode for {self.provider}.")
            api_args["response_format"] = {"type": "json_object"}

        if self.provider == "OpenAI":
            api_args["frequency_penalty"] = self.frequency_penalty
            api_args["presence_penalty"] = self.presence_penalty

        completion = None
        retry_after_sec = 60

        # Fallback Chain in exact Free Tier priority order:
        # 1. gemini-3.8-flash (Top quality / 5 RPM / 20 RPD)
        # 2. gemini-3.7-flash (5 RPM / 20 RPD)
        # 3. gemini-3.6-flash (5 RPM / 20 RPD)
        # 4. gemini-3.5-flash (5 RPM / 20 RPD)
        # 5. gemini-3.5-flash-lite (High-RPD fallback / 15 RPM / 500 RPD)
        FALLBACK_CHAIN = [
            "gemini-3.8-flash",
            "gemini-3.7-flash",
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
        ]

        if self.provider in ["Gemini Proxy", "Google"]:
            candidates = list(FALLBACK_CHAIN)
        else:
            candidates = [target_model]

        last_error = None
        for idx, current_model in enumerate(candidates):
            effective_model = self._normalize_model(current_model)
            api_args["model"] = effective_model
            self._respect_model_delay(current_model)
            t0 = time.perf_counter()
            self.logger.info(f"🌐 [Gemini #{idx+1}/{len(candidates)}] Đang gửi yêu cầu dịch đến model '{effective_model}' (Provider: {self.provider})...")

            try:
                completion = self.client.chat.completions.create(**api_args)
                t_elapsed = time.perf_counter() - t0
                self.last_successful_model = effective_model
                if idx > 0:
                    self.logger.info(f"✓ [Fallback #{idx+1} Thành Công] Model dự phòng '{effective_model}' đã phản hồi thành công sau {t_elapsed:.2f}s!")
                else:
                    self.logger.info(f"⚡ [Gemini] Nhận phản hồi từ API sau {t_elapsed:.2f}s (Model: {effective_model})")
                last_error = None
                break
            except Exception as e:
                err_str = str(e)
                is_conn_error = (
                    isinstance(e, (openai.APIConnectionError, httpx.ConnectError, httpx.NetworkError))
                    or "Connection error" in err_str
                    or "actively refused" in err_str
                    or "127.0.0.1:8080" in err_str
                )
                if is_conn_error and "generativelanguage.googleapis.com" not in getattr(self, "active_endpoint", ""):
                    self.logger.warning(
                        "⚠️ [Proxy Offline] Mất kết nối tới local proxy (Connection error). Tự động chuyển toàn bộ sang Google AI Studio Direct API..."
                    )
                    self.endpoint = "https://generativelanguage.googleapis.com/v1beta/openai"
                    self._cached_client = None
                    self._initialize_client(self._get_api_key())
                    # Retry with direct endpoint
                    effective_model = self._normalize_model(current_model)
                    api_args["model"] = effective_model
                    try:
                        completion = self.client.chat.completions.create(**api_args)
                        self.last_successful_model = effective_model
                        last_error = None
                        break
                    except Exception as e_direct:
                        err_str = str(e_direct)
                        e = e_direct

                is_rate_limit = (
                    isinstance(e, openai.RateLimitError)
                    or "429" in err_str
                    or "RESOURCE_EXHAUSTED" in err_str
                    or "quota" in err_str.lower()
                    or "rate_limit" in err_str.lower()
                )
                last_error = e

                if is_rate_limit:
                    if hasattr(e, "response") and e.response and hasattr(e.response, "headers"):
                        retry_hdr = e.response.headers.get("retry-after")
                        if retry_hdr and retry_hdr.isdigit():
                            retry_after_sec = int(retry_hdr)

                    if idx < len(candidates) - 1:
                        next_model = candidates[idx + 1]
                        self.logger.warning(
                            f"⚠️ [Quota Limit 429] Model '{current_model}' chạm hạn mức Rate Limit/Quota. "
                            f"Tự động chuyển tiếp (fallback #{idx+2}) sang model '{next_model}'..."
                        )
                        continue
                    else:
                        self.logger.error(
                            f"❌ [Toàn bộ Quota cạn kiệt] Cả {len(candidates)} model trong chuỗi fallback "
                            f"({' -> '.join(candidates)}) đều chạm hạn mức 429. Vui lòng đợi {retry_after_sec}s trước khi thử lại."
                        )
                        raise openai.RateLimitError(
                            f"Toàn bộ 3 model Gemini ({' -> '.join(candidates)}) đều hết quota/rate limit. "
                            f"Vui lòng đợi {retry_after_sec} giây trước khi gửi yêu cầu tiếp theo.",
                            response=getattr(e, "response", None),
                            body=getattr(e, "body", None)
                        )
                else:
                    if idx < len(candidates) - 1:
                        next_model = candidates[idx + 1]
                        self.logger.warning(
                            f"⚠️ [Lỗi {type(e).__name__}] Model '{current_model}' thất bại: {e}. "
                            f"Tự động chuyển tiếp (fallback #{idx+2}) sang model '{next_model}'..."
                        )
                        continue
                    else:
                        self.logger.error(
                            f"❌ [Toàn bộ model thất bại] Cả {len(candidates)} model trong chuỗi fallback "
                            f"({' -> '.join(candidates)}) đều thất bại. Lỗi cuối cùng: {type(last_error).__name__} - {last_error}"
                        )
                        raise last_error

        if last_error is not None:
            raise last_error

        if completion is None:
            return None

        t_elapsed = time.perf_counter() - t0

        if (
            completion.choices
            and completion.choices[0].message
            and completion.choices[0].message.content
        ):
            raw_content = completion.choices[0].message.content
            json_to_parse = raw_content.strip()

            # Clean markdown code blocks if present
            if "```" in json_to_parse:
                cleaned = re.sub(r"^```(?:json)?\s*", "", json_to_parse, flags=re.MULTILINE)
                cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE).strip()
                json_to_parse = cleaned

            start = json_to_parse.find("{")
            end = json_to_parse.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_to_parse = json_to_parse[start : end + 1]
            try:
                data_to_validate = json.loads(json_to_parse)
                validated_response = TranslationResponse.model_validate(
                    data_to_validate
                )
            except (ValidationError, json.JSONDecodeError) as e:
                self.logger.warning(
                    f"Initial Pydantic validation failed: {e}. Attempting to fix simple dictionary or list format."
                )
                try:
                    # Attempt regex extraction of {"id": X, "translation": "Y"} pairs
                    item_matches = re.findall(
                        r'\{\s*"id"\s*:\s*(\d+)\s*,\s*"translation"\s*:\s*"((?:[^"\\]|\\.)*)"',
                        raw_content
                    )
                    if item_matches:
                        recovered_items = [
                            {"id": int(m[0]), "translation": m[1]}
                            for m in item_matches
                        ]
                        fixed_data = {"translations": recovered_items}
                        validated_response = TranslationResponse.model_validate(fixed_data)
                        self.logger.info(f"Successfully recovered {len(recovered_items)} item(s) from response using regex extraction.")
                    else:
                        simple_data = json.loads(json_to_parse)
                        fixed_translations = []
                        if isinstance(simple_data, dict) and all(k.isdigit() for k in simple_data.keys()):
                            fixed_translations = [{"id": int(k), "translation": v} for k, v in simple_data.items()]
                        elif isinstance(simple_data, list):
                            fixed_translations = simple_data
                        if fixed_translations:
                            validated_response = TranslationResponse.model_validate({"translations": fixed_translations})
                        else:
                            raise e
                except Exception as final_e:
                    self.logger.error(
                        f"Pydantic validation or JSON parsing failed even after attempting fix: {final_e}"
                    )
                    self.logger.debug(f"Raw JSON content from API: {raw_content}")
                    raise
        else:
            self.logger.warning("No valid message content in API response.")
            return None

        if hasattr(completion, "usage") and completion.usage:
            self.token_count += completion.usage.total_tokens
            self.token_count_last = completion.usage.total_tokens
        else:
            self.token_count_last = 0

        return validated_response

    def _compute_cache_key(
        self,
        text: str,
        from_lang: str,
        to_lang: str,
        model_name: str,
        prompt_hash: str,
        context_hash: str = "",
        glossary_hash: str = "",
        char_mem_hash: str = "",
        block_type: str = "DIALOGUE",
        speaker: str = "",
    ) -> str:
        key_raw = f"{text}|{from_lang}|{to_lang}|{model_name}|{prompt_hash}|{context_hash}|{glossary_hash}|{char_mem_hash}|{block_type}|{speaker}"
        return hashlib.md5(key_raw.encode("utf-8")).hexdigest()

    def translate_dialogue_items(
        self,
        items: List[DialogueItem],
        page_context: Optional[dict] = None,
        max_batch_size: int = 40
    ) -> List[str]:
        if not items:
            return []

        model_name = self.override_model or self.model
        if ": " in model_name:
            model_name = model_name.split(": ", 1)[1]

        to_lang = self.lang_map.get(self.lang_target, self.lang_target)
        from_lang = self.lang_map.get(self.lang_source, self.lang_source)

        prompt_hash = hashlib.md5(self.system_prompt.encode("utf-8")).hexdigest()[:8]
        glossary_hash = self.glossary_manager.get_hash()
        char_mem_hash = self.character_memory.get_hash()

        dialogue_history = page_context.get("dialogue_history") if page_context else None
        context_hash_str = ""
        if dialogue_history:
            context_hash_str = hashlib.md5(" ".join(h.source for h in dialogue_history).encode("utf-8")).hexdigest()[:8]

        results: List[Optional[str]] = [None] * len(items)
        uncached_indices: List[int] = []
        uncached_items: List[DialogueItem] = []

        for idx, item in enumerate(items):
            clean_text = sanitize_text(item.source)
            if not clean_text:
                results[idx] = ""
                continue

            # 1. Local zero-token bypass for purely symbolic / punctuation strings
            if not any(c.isalnum() for c in clean_text):
                results[idx] = clean_text
                if "?" in clean_text or "!" in clean_text:
                    item.emotion_tag = "surprise" if ("?" in clean_text and "!" in clean_text) else ("shout" if "!" in clean_text else "normal")
                elif "..." in clean_text or "…" in clean_text:
                    item.emotion_tag = "whisper"
                continue

            cache_key = self._compute_cache_key(
                clean_text, from_lang, to_lang, model_name, prompt_hash,
                context_hash=context_hash_str,
                glossary_hash=glossary_hash,
                char_mem_hash=char_mem_hash,
                block_type=item.block_type or "DIALOGUE",
                speaker=item.speaker or "",
            )
            if cache_key in self._translation_cache:
                results[idx] = self._translation_cache[cache_key]
            else:
                uncached_indices.append(idx)
                uncached_items.append(item)

        if not uncached_items:
            self.logger.info(f"All {len(items)} dialogue blocks resolved locally/cached (0 API calls, 0 token cost).")
            return [r if r is not None else "" for r in results]

        RETRYABLE_EXCEPTIONS = (
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.InternalServerError,
            openai.APIStatusError,
            httpx.RequestError,
        )

        uncached_translations = []

        for prompt, num_unique, chunk_items, src_to_pid in self._assemble_prompts(
            uncached_items,
            to_lang=to_lang,
            dialogue_history=dialogue_history,
            max_batch_size=max_batch_size
        ):
            api_retry_attempt = 0
            mismatch_retry_attempt = 0

            while True:
                try:
                    parsed_response = self._request_translation(prompt)

                    if not parsed_response or not parsed_response.translations:
                        raise ValueError(
                            "Received empty or invalid parsed response from API."
                        )

                    if len(parsed_response.translations) != num_unique:
                        raise InvalidNumTranslations(
                            f"Expected {num_unique}, got {len(parsed_response.translations)}"
                        )

                    translations_dict = {
                        item.id: clean_manga_punctuation(item.translation)
                        for item in parsed_response.translations
                    }
                    emotions_dict = {
                        item.id: (getattr(item, "emotion_tag", "normal") or "normal").lower().strip()
                        for item in parsed_response.translations
                    }

                    chunk_translations = []
                    for item in chunk_items:
                        dedup_key = (
                            item.source.strip(),
                            item.speaker or "",
                            item.block_type or "DIALOGUE",
                            item.connected_group_id
                        )
                        p_id = src_to_pid.get(dedup_key, 1)
                        tr = translations_dict.get(p_id, item.source)
                        em = emotions_dict.get(p_id, "normal")
                        item.emotion_tag = em
                        chunk_translations.append(tr)

                    uncached_translations.extend(chunk_translations)
                    self.logger.info(
                        f"Successfully translated batch of {len(chunk_items)} items ({num_unique} unique). Tokens used: {self.token_count_last}"
                    )
                    break

                except InvalidNumTranslations as e:
                    mismatch_retry_attempt += 1
                    self.logger.warning(
                        f"Translation structure mismatch: {e}. Attempt {mismatch_retry_attempt}/{self.invalid_repeat_count}."
                    )
                    if mismatch_retry_attempt >= self.invalid_repeat_count:
                        self.logger.error(
                            "Fatal Error: Failed to get correct translation structure after retries."
                        )
                        uncached_translations.extend(["[ERROR: Structure Mismatch]"] * len(chunk_items))
                        break
                    time.sleep(self.retry_timeout / 2)

                except openai.RateLimitError as e:
                    self.logger.error(
                        f"🛑 [Quota 429 Hết Lượt] {e}"
                    )
                    uncached_translations.extend([f"[Hết Quota Gemini - Vui lòng đợi 60s]"] * len(chunk_items))
                    break

                except RETRYABLE_EXCEPTIONS as e:
                    api_retry_attempt += 1
                    self.logger.warning(
                        f"API Error (retryable): {type(e).__name__} - {e}. Attempt {api_retry_attempt}/{self.retry_attempts}."
                    )
                    if api_retry_attempt >= self.retry_attempts:
                        self.logger.error(
                            f"Fatal Error: Failed to connect to API after {self.retry_attempts} attempts."
                        )
                        uncached_translations.extend([f"[ERROR: API Failed]"] * len(chunk_items))
                        break
                    time.sleep(self.retry_timeout)

                except (
                    ValidationError,
                    json.JSONDecodeError,
                    openai.BadRequestError,
                    openai.AuthenticationError,
                    ValueError,
                ) as e:
                    self.logger.error(
                        f"Fatal Error: An unrecoverable error occurred: {type(e).__name__} - {e}"
                    )
                    self.logger.debug(traceback.format_exc())
                    uncached_translations.extend([f"[ERROR: {type(e).__name__}]"] * len(chunk_items))
                    break

        # Populate newly translated items into cache and results
        effective_model = getattr(self, "last_successful_model", None) or model_name
        for idx, item, trans in zip(uncached_indices, uncached_items, uncached_translations):
            if not trans.startswith("[ERROR:"):
                cache_key = self._compute_cache_key(
                    item.source.strip(), from_lang, to_lang, effective_model, prompt_hash,
                    context_hash=context_hash_str,
                    glossary_hash=glossary_hash,
                    char_mem_hash=char_mem_hash,
                    block_type=item.block_type or "DIALOGUE",
                    speaker=item.speaker or "",
                )
                self._translation_cache[cache_key] = trans
            results[idx] = trans

        return [r if r is not None else "" for r in results]

    def _translate(self, src_list: List[str]) -> List[str]:
        items = [DialogueItem(id=i + 1, source=text) for i, text in enumerate(src_list)]
        return self.translate_dialogue_items(items)

    def translate_textblk_lst(self, textblk_lst: List[Any], page_context: Optional[dict] = None):
        """Context-Aware translation for a list of TextBlocks on a comic page."""
        if not textblk_lst:
            return

        # Auto-sync glossary from global config if available
        try:
            from utils.config import pcfg
            if hasattr(pcfg, "pre_mt_sublist") or hasattr(pcfg, "mt_sublist"):
                self.glossary_manager.load_from_sublists(
                    getattr(pcfg, "pre_mt_sublist", []),
                    getattr(pcfg, "mt_sublist", [])
                )
        except Exception:
            pass

        non_empty_indices = []
        dialogue_items: List[DialogueItem] = []
        translations = []

        img_shape = page_context.get("img_shape") if page_context else None

        for idx, blk in enumerate(textblk_lst):
            text = blk.get_text() if hasattr(blk, "get_text") else str(blk)
            clean_txt = sanitize_text(text)
            translations.append(clean_txt)
            if clean_txt != "":
                non_empty_indices.append(idx)
                xyxy = getattr(blk, "xyxy", [0, 0, 0, 0])
                position = ContextAssembler.get_spatial_position(
                    xyxy,
                    img_shape[1] if img_shape else 1000,
                    img_shape[0] if img_shape else 1500
                )
                is_vert = getattr(blk, "vertical", True)
                direction = "Vertical" if is_vert else "Horizontal"
                b_type = ContextAssembler.classify_block_type(clean_txt, is_vert)
                
                # Check if block has manual user edit
                is_user_edited = getattr(blk, "user_edited", False) or getattr(blk, "manual_edit", False)
                existing_trans = blk.translation if (hasattr(blk, "translation") and blk.translation and is_user_edited) else None

                dialogue_items.append(
                    DialogueItem(
                        id=len(dialogue_items) + 1,
                        source=clean_txt,
                        translated=existing_trans,
                        position=position,
                        direction=direction,
                        block_type=b_type,
                    )
                )

        if not dialogue_items:
            for blk in textblk_lst:
                if hasattr(blk, "translation") and not getattr(blk, "user_edited", False) and not getattr(blk, "manual_edit", False):
                    blk.translation = ""
            return

        for callback_name, callback in self._preprocess_hooks.items():
            callback(
                translations=translations,
                textblocks=textblk_lst,
                translator=self,
                source_text=[item.source for item in dialogue_items],
            )

        translated_texts = self.translate_dialogue_items(dialogue_items, page_context=page_context)

        for orig_idx, (trans, d_item) in zip(non_empty_indices, zip(translated_texts, dialogue_items)):
            if orig_idx < len(textblk_lst):
                blk = textblk_lst[orig_idx]
                is_user_edited = getattr(blk, "user_edited", False) or getattr(blk, "manual_edit", False)
                if not is_user_edited:
                    translations[orig_idx] = trans
                    if hasattr(blk, "emotion_tag"):
                        blk.emotion_tag = getattr(d_item, "emotion_tag", "normal") or "normal"
                else:
                    # Protect user manual edit from being overwritten
                    translations[orig_idx] = blk.translation if (hasattr(blk, "translation") and blk.translation) else trans

        for callback_name, callback in self._postprocess_hooks.items():
            callback(translations=translations, textblocks=textblk_lst, translator=self)

        for tr, blk in zip(translations, textblk_lst):
            if hasattr(blk, "translation"):
                is_user_edited = getattr(blk, "user_edited", False) or getattr(blk, "manual_edit", False)
                if not is_user_edited:
                    blk.translation = tr

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)

        if param_key in ["proxy", "multiple_keys", "apikey", "provider", "endpoint"]:
            self.client = None
            self._cached_client = None
            self._last_client_config = None
        if param_key in ["model", "override model", "system_prompt"]:
            self._translation_cache.clear()

    def translate_single(
        self,
        text: str,
        src_lang: str = "English",
        tgt_lang: str = "Tiếng Việt",
        context_hints: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Direct Single-Line Passthrough (Mode 2):
        Translates a single bubble with optional surrounding context hints (previous, next, block_type).
        """
        clean_text = sanitize_text(text)
        if not clean_text:
            return ""

        if not any(c.isalnum() for c in clean_text):
            return clean_text

        current_api_key = self._select_api_key() or "gemini-proxy-key"
        if not self._initialize_client(current_api_key):
            return clean_text

        model_name = self.override_model or self.model
        if ": " in model_name:
            model_name = model_name.split(": ", 1)[1]

        target_models = QUOTA_TRACKER.get_candidate_models(preferred_model=model_name, estimated_tokens=1024)
        if not target_models:
            self.logger.warning("⚠️ [Quota Tracker] Toàn bộ model đã hết quota hôm nay cho single-line translation. Giữ nguyên text gốc.")
            return clean_text

        system_instruction = (
            f"You are a Principal Manga/Manhwa Scanlation Localization Translator ({src_lang} -> {tgt_lang}).\n"
            f"TONE & STYLE (Phong cách Nhóm Dịch Lầy Lội / Mặn Mà):\n"
            f"- Cực kỳ tự nhiên, lầy lội, hài hước, mặn mà, bựa đúng lúc; nói KHÔNG với văn mẫu dịch máy khô cứng.\n"
            f"- Tự động suy luận vai vế: Cặp đôi/vợ chồng BẮT BUỘC xưng 'Anh - Em' (hoặc tên thân mật), bạn bè xưng 'Mày - Tao' (khi cà khịa/chửi đùa) hoặc 'Cậu - Tớ / Mình', dùng trợ từ khẩu ngữ tự nhiên (nè, cơ chứ, chứ lị, ối dồi ôi, toang rồi, hết nước chấm).\n"
            f"CORE RULE: Treat text as a visual segment of a comic flow. If it is part of a larger sentence with surrounding context, translate it as a natural, seamless continuation without losing semantic content or dropping verbs/predicates.\n"
            f"If standalone, preserve all meaning (subject, verb, modifiers). Do not truncate full clauses into bare noun phrases. Do not invent missing facts.\n"
            f"Punctuation: Never leave a trailing period ('.') at the end of dialogue bubbles.\n"
            f"Output ONLY the translated text string. No quotes, explanations, JSON, or markdown."
        )

        user_custom_prompt = self.system_prompt.strip() if hasattr(self, "system_prompt") and self.system_prompt else ""
        if user_custom_prompt:
            system_instruction += f"\n\nUSER CUSTOM INSTRUCTIONS & CHARACTER PROFILES (HIGHEST PRIORITY):\n{user_custom_prompt}\n"

        user_content = clean_text
        if context_hints:
            prev_txt = str(context_hints.get("previous", "")).strip()
            next_txt = str(context_hints.get("next", "")).strip()
            b_type = context_hints.get("block_type", "DIALOGUE")
            parts = []
            if prev_txt:
                parts.append(f"[Previous Context: {prev_txt}]")
            parts.append(f"[Target Text to Translate ({b_type}): {clean_text}]")
            if next_txt:
                parts.append(f"[Next Context: {next_txt}]")
            parts.append("Translate ONLY the [Target Text to Translate], ensuring it flows naturally with the surrounding context.")
            user_content = "\n".join(parts)

        for attempt_model in target_models:
            if QUOTA_TRACKER.is_rpd_exhausted(attempt_model):
                continue

            proxy_switched = False
            for retry in range(2):
                try:
                    throttle_delay = QUOTA_TRACKER.wait_for_rpm_slot(attempt_model)
                    if throttle_delay > 0.1:
                        rpm_val = QUOTA_TRACKER.profiles.get(attempt_model, {}).get("rpm", 10)
                        self.logger.info(f"⏳ [RPM Throttle] Đang chờ {throttle_delay:.2f}s theo giới hạn {rpm_val} RPM của model '{attempt_model}'...")

                    self._respect_delay()
                    effective_model = self._normalize_model(attempt_model)
                    completion = self.client.chat.completions.create(
                        model=effective_model,
                        messages=[
                            {"role": "system", "content": system_instruction},
                            {"role": "user", "content": user_content},
                        ],
                        temperature=0.1,
                        max_tokens=1024,
                    )
                    if completion and completion.choices and completion.choices[0].message:
                        res = completion.choices[0].message.content.strip()
                        if res:
                            tokens_used = completion.usage.total_tokens if completion.usage else 0
                            QUOTA_TRACKER.record_call(attempt_model, tokens=tokens_used)
                            # Strip any accidental wrapping quotes
                            if (res.startswith('"') and res.endswith('"')) or (res.startswith("'") and res.endswith("'")):
                                res = res[1:-1].strip()
                            return clean_manga_punctuation(res)
                except Exception as e:
                    err_str = str(e)
                    is_conn_error = (
                        isinstance(e, (openai.APIConnectionError, httpx.ConnectError, httpx.NetworkError))
                        or "Connection error" in err_str
                        or "actively refused" in err_str
                        or "127.0.0.1:8080" in err_str
                    )
                    if is_conn_error and "generativelanguage.googleapis.com" not in getattr(self, "active_endpoint", ""):
                        self.logger.warning(
                            "⚠️ [Proxy Offline] Mất kết nối tới local proxy (Connection error). Tự động chuyển toàn bộ sang Google AI Studio Direct API..."
                        )
                        self.endpoint = "https://generativelanguage.googleapis.com/v1beta/openai"
                        self._cached_client = None
                        self._initialize_client(self._get_api_key())
                        proxy_switched = True
                        break

                    is_rate_limit = (
                        isinstance(e, openai.RateLimitError)
                        or "429" in err_str
                        or "RESOURCE_EXHAUSTED" in err_str
                        or "quota" in err_str.lower()
                        or "rate_limit" in err_str.lower()
                    )
                    if is_rate_limit:
                        QUOTA_TRACKER.record_429_exhaustion(attempt_model, reason=err_str)
                        self.logger.warning(f"translate_single: '{attempt_model}' chạm giới hạn rate limit/quota (429). Tự động chuyển sang model tiếp theo trong chain...")
                        break

                    is_server_overload = (
                        "503" in err_str
                        or "high demand" in err_str.lower()
                        or "overloaded" in err_str.lower()
                        or "unavailable" in err_str.lower()
                    )
                    if is_server_overload:
                        self.logger.warning(f"translate_single: '{attempt_model}' đang quá tải (503 / high demand). Chuyển sang model tiếp theo...")
                        break

                    self.logger.warning(f"translate_single with '{attempt_model}' failed: {e}. Retrying/falling back...")
                    time.sleep(0.5 + random.uniform(0.1, 0.4))

            if proxy_switched:
                continue

        return clean_text

    def translate_chapter_batch(
        self,
        proxy_or_buffer: Any,
        src_lang: str = "English",
        tgt_lang: str = "Tiếng Việt"
    ) -> Dict[int, Dict[Union[int, str], Dict[str, str]]]:
        """
        Chapter-Batch Aggregator (Mode 1):
        Serializes all buffered pages across a chapter into a structured payload,
        dispatches to Gemini with Tiered Model Selection, validates 1:1 ID integrity,
        proactively throttles RPM, and handles quota exhaustion with clean resume checkpoints.
        """
        if hasattr(proxy_or_buffer, "is_empty") and proxy_or_buffer.is_empty():
            return {}

        current_api_key = self._select_api_key() or "gemini-proxy-key"
        if not self._initialize_client(current_api_key):
            raise ConnectionError("Failed to initialize API client for chapter batch translation.")

        # Support both TranslationProxy and legacy TranslationStateBuffer
        is_proxy = hasattr(proxy_or_buffer, "build_sub_batch_payloads")
        if is_proxy:
            sub_payloads = proxy_or_buffer.build_sub_batch_payloads(src_lang, tgt_lang, max_pages_per_batch=6, max_dialogues_per_batch=20)
            total_dialogues = proxy_or_buffer.total_dialogues_count()
        elif hasattr(proxy_or_buffer, "build_chapter_payload"):
            sub_payloads = [proxy_or_buffer.build_chapter_payload(src_lang, tgt_lang)]
            total_dialogues = proxy_or_buffer.total_dialogues_count()
        else:
            sub_payloads = [proxy_or_buffer.get_chapter_payload()]
            total_dialogues = proxy_or_buffer.total_dialogues_count()

        model_name = self.override_model or self.model
        if ": " in model_name:
            model_name = model_name.split(": ", 1)[1]

        system_instruction = (
            f"You are a Principal Manga/Manhwa Scanlation Localization Translator ({src_lang} -> {tgt_lang}).\n\n"
            "CORE LOCALIZATION PRINCIPLE: OCR BLOCKS ARE VISUAL SEGMENTS, NOT ISOLATED SENTENCES.\n"
            "1. Multi-Bubble Sentence Stitching & Continuity:\n"
            "   - Manga scanlations frequently split a single grammatical clause across 2 or more sequential speech bubbles (or across page boundaries).\n"
            "   - Reconstruct the full semantic proposition across neighboring blocks in reading order first, then partition the natural translation across the respective block IDs.\n"
            "   - Never let intermediate blocks sound like broken or abruptly cut-off fragments.\n"
            "2. Strict Anti-Truncation & Full Proposition Preservation:\n"
            "   - NEVER drop subjects, verbs, predicates, tense/modality, negation, or modifiers.\n"
            "   - DO NOT truncate a full clause into a bare noun phrase (e.g. 'their cohabitation life starts a new phase' MUST translate as 'Cuộc sống chung của họ bước sang một giai đoạn mới', NEVER just 'Cuộc sống chung').\n"
            "3. Standalone Titles vs Complete Clauses (Anti-Hallucination):\n"
            "   - If a block is truly a standalone title/phrase without a verb (e.g. 'Their Cohabitation Life'), translate it faithfully as a title ('Cuộc sống chung của họ') without inventing non-existent predicates.\n"
            "   - If a predicate is present, translate the complete proposition.\n"
            "4. False-Merge Prevention:\n"
            "   - Separate standalone utterances (e.g. 'Good morning.' and 'What are you doing?') must remain distinct and translated individually.\n"
            "5. Contextual Pronouns, Slang & Tone Consistency (Phong cách Nhóm Dịch Lầy Lội / Mặn Mà):\n"
            "   - SỨ MỆNH: Thoại cực kỳ tự nhiên, lầy lội, hài hước, mặn mà, bựa đúng lúc; nói KHÔNG với văn mẫu dịch máy khô cứng.\n"
            "   - Couples/Romance/Husband-Wife: BẮT BUỘC xưng hô 'anh - em' (hoặc gọi tên thân mật), ngọt ngào, trêu ghẹo. TUYỆT ĐỐI KHÔNG dùng 'tớ - cậu' hay 'tôi - cô' trong bối cảnh tình cảm.\n"
            "   - Close friends/peers: linh hoạt 'mày - tao' (khi cà khịa, chửi đùa) hoặc 'cậu - tớ / mình' (thân thiện).\n"
            "   - Tình huống bất lực / Cáu gắt / Tấu hài: chêm khẩu ngữ tự nhiên ('toang rồi', 'ối dồi ôi', 'chết dở', 'vãi chưởng', 'ảo ma', 'mất mặt ghê', 'hết nước chấm', 'bó tay').\n"
            "   - Trợ từ cảm thán: dùng linh hoạt 'nè', 'cơ chứ', 'chứ lị', 'đấy nhé', 'ơi là trời', 'nhen', 'chứ sao'.\n"
            "   - Inner thoughts (THOUGHT): tự vấn tự nhiên ('mình...', 'quái lạ...').\n"
            "   - Combat/Enemies: mày/tao, ngươi/ta, tên khốn.\n"
            "   - When Character Profiles or User Instructions are provided, they strictly OVERRIDE all defaults.\n"
            "6. Manga Punctuation Style (Dấu câu khung thoại manga):\n"
            "   - TUYỆT ĐỐI KHÔNG thêm dấu chấm đơn ('.') ở cuối câu thoại/bong bóng trừ phi là dấu ba chấm ('...', '…') hoặc từ viết tắt.\n"
            "7. OCR Noise & Typo Auto-Healing (Tự động phát hiện & sửa lỗi OCR):\n"
            "   - Manga OCR may contain misread characters (e.g. '1' for 'l'/'I', '0' for 'O', 'rn' for 'm', stray hyphens or broken words). Infer the true word from conversational context and translate the intended meaning accurately into Vietnamese.\n"
            "8. Strict JSON Output Schema:\n"
            "   - Preserve every dialogue block ID exactly (1:1 mapping).\n"
            "   - Never omit, rename, merge, duplicate, or add block IDs.\n"
            "   - Output strictly valid JSON matching this schema:\n"
            f'{{"pages": [{{"page_index": 0, "dialogues": [{{"id": 1, "translation": "Bản dịch tiếng Việt"}}]}}]}}'
        )

        user_custom_prompt = self.system_prompt.strip() if hasattr(self, "system_prompt") and self.system_prompt else ""
        if user_custom_prompt:
            system_instruction += (
                f"\n\n9. USER CUSTOM INSTRUCTIONS & CHARACTER PROFILES:\n"
                f"{user_custom_prompt}\n"
                f"\nIMPORTANT OVERRIDE: For Chapter Batch Translation, ALWAYS strictly format your output according to Rule 8 ('pages' array with 'page_index' and 'dialogues'), ignoring any conflicting output format instructions above."
            )

        merged_result_map: Dict[int, Dict[Union[int, str], Dict[str, str]]] = {}

        for b_idx, payload_obj in enumerate(sub_payloads, start=1):
            sub_dialogues_count = sum(len(p.blocks) for p in payload_obj.pages) if hasattr(payload_obj, "pages") else 1
            payload_json_str = payload_obj.model_dump_json(indent=2)
            dynamic_max_tokens = min(16384, max(8192, sub_dialogues_count * 150))

            # Retrieve candidate models prioritizing Tier 1 (High RPD Flash Lite)
            fallback_models = QUOTA_TRACKER.get_candidate_models(preferred_model=model_name, estimated_tokens=dynamic_max_tokens)

            if not fallback_models or QUOTA_TRACKER.are_all_quotas_exhausted():
                resume_file = QUOTA_TRACKER.save_resume_state(
                    chapter_info={"src_lang": src_lang, "tgt_lang": tgt_lang, "total_dialogues": total_dialogues},
                    completed_sub_batches=b_idx - 1,
                    total_sub_batches=len(sub_payloads),
                    results=merged_result_map
                )
                reset_timing = QUOTA_TRACKER.get_reset_timing_info()
                exhaust_msg = (
                    f"Đã dùng hết quota ngày hôm nay cho tất cả model khả dụng (Đã hoàn thành {b_idx-1}/{len(sub_payloads)} sub-batches). "
                    f"Tiến độ đã được lưu an toàn tại '{resume_file}'. Vui lòng tiếp tục vào {reset_timing} hoặc thử lại sau."
                )
                self.logger.error(f"🛑 [Quota Exhausted] {exhaust_msg}")
                raise RuntimeError(exhaust_msg)

            sub_success = False
            last_error = None

            for attempt_model in fallback_models:
                if sub_success:
                    break

                if QUOTA_TRACKER.is_rpd_exhausted(attempt_model):
                    self.logger.info(f"⏭️ [Quota Tracker] Bỏ qua model '{attempt_model}' (RPD hôm nay đã cạn).")
                    continue

                proxy_switched = False
                for retry in range(3):
                    try:
                        throttle_delay = QUOTA_TRACKER.wait_for_rpm_slot(attempt_model)
                        if throttle_delay > 0.1:
                            rpm_val = QUOTA_TRACKER.profiles.get(attempt_model, {}).get("rpm", 10)
                            self.logger.info(f"⏳ [RPM Throttling] Đang chờ {throttle_delay:.2f}s theo giới hạn {rpm_val} RPM của model '{attempt_model}'...")

                        self._respect_delay()
                        effective_model = self._normalize_model(attempt_model)
                        tier_name = QUOTA_TRACKER.profiles.get(attempt_model, {}).get("tier", "standard").upper()
                        rem_rpd = QUOTA_TRACKER.get_remaining_rpd(attempt_model)
                        self.logger.info(
                            f"Dispatching Chapter Batch (Sub-batch {b_idx}/{len(sub_payloads)}, {sub_dialogues_count} dialogues) "
                            f"using model '{effective_model}' [{tier_name} Tier, Rem RPD: {rem_rpd}] (Attempt {retry+1})..."
                        )
                        completion = self.client.chat.completions.create(
                            model=effective_model,
                            messages=[
                                {"role": "system", "content": system_instruction},
                                {"role": "user", "content": payload_json_str},
                            ],
                            response_format={"type": "json_object"},
                            temperature=self.temperature,
                            max_tokens=dynamic_max_tokens,
                        )
                        if completion and completion.choices and completion.choices[0].message:
                            content = completion.choices[0].message.content
                            if content:
                                if is_proxy or hasattr(proxy_or_buffer, "validate_and_unpack"):
                                    valid, sub_res_map, err_msg = proxy_or_buffer.validate_and_unpack(payload_obj, content)
                                    if not valid:
                                        raise ValueError(f"Validation Failure on '{effective_model}': {err_msg}")
                                    # Update rolling memory with translated dialogues
                                    translated_snippets = []
                                    for p_res in sub_res_map.values():
                                        for d_val in p_res.values():
                                            translated_snippets.append(d_val.get("translation", ""))
                                    if hasattr(proxy_or_buffer, "context_engine"):
                                        proxy_or_buffer.context_engine.update_rolling_history(translated_snippets)
                                else:
                                    sub_res_map = proxy_or_buffer.unpack_response(content)

                                for p_k, p_v in sub_res_map.items():
                                    if p_k not in merged_result_map:
                                        merged_result_map[p_k] = {}
                                    merged_result_map[p_k].update(p_v)

                                tokens_used = completion.usage.total_tokens if completion.usage else 0
                                QUOTA_TRACKER.record_call(attempt_model, tokens=tokens_used)
                                self.logger.info(f"✓ Sub-batch {b_idx}/{len(sub_payloads)} validated and translated with model '{effective_model}' (Tokens: {tokens_used}, Rem RPD: {QUOTA_TRACKER.get_remaining_rpd(attempt_model)}).")
                                sub_success = True
                                break
                    except Exception as e:
                        last_error = e
                        err_str = str(e)
                        is_conn_error = (
                            isinstance(e, (openai.APIConnectionError, httpx.ConnectError, httpx.NetworkError))
                            or "Connection error" in err_str
                            or "actively refused" in err_str
                            or "127.0.0.1:8080" in err_str
                        )
                        if is_conn_error and "generativelanguage.googleapis.com" not in getattr(self, "active_endpoint", ""):
                            self.logger.warning(
                                "⚠️ [Proxy Offline] Mất kết nối tới local proxy (Connection error). Tự động chuyển toàn bộ sang Google AI Studio Direct API..."
                            )
                            self.endpoint = "https://generativelanguage.googleapis.com/v1beta/openai"
                            self._cached_client = None
                            self._initialize_client(self._get_api_key())
                            proxy_switched = True
                            break

                        is_rate_limit = (
                            isinstance(e, openai.RateLimitError)
                            or "429" in err_str
                            or "RESOURCE_EXHAUSTED" in err_str
                            or "quota" in err_str.lower()
                            or "rate_limit" in err_str.lower()
                        )
                        if is_rate_limit:
                            QUOTA_TRACKER.record_429_exhaustion(attempt_model, reason=err_str)
                            self.logger.warning(
                                f"⚠️ [Quota Limit 429] Model '{attempt_model}' chạm hạn mức Rate Limit/Quota. "
                                f"Tự động chuyển tiếp ngay sang model dự phòng tiếp theo trong chain..."
                            )
                            break

                        is_server_overload = (
                            "503" in err_str
                            or "high demand" in err_str.lower()
                            or "overloaded" in err_str.lower()
                            or "unavailable" in err_str.lower()
                        )
                        if is_server_overload:
                            self.logger.warning(
                                f"⚠️ [Server Overloaded 503] Model '{attempt_model}' đang quá tải (high demand / 503). "
                                f"Tự động chuyển tiếp ngay sang model dự phòng tiếp theo trong chain..."
                            )
                            break

                        backoff = (1.5 ** retry) + random.uniform(0.1, 0.5)
                        self.logger.warning(f"Chapter sub-batch {b_idx} attempt with '{attempt_model}' failed: {e}. Backing off {backoff:.2f}s...")
                        time.sleep(backoff)

                if proxy_switched:
                    continue

            if not sub_success:
                if QUOTA_TRACKER.are_all_quotas_exhausted():
                    resume_file = QUOTA_TRACKER.save_resume_state(
                        chapter_info={"src_lang": src_lang, "tgt_lang": tgt_lang, "total_dialogues": total_dialogues},
                        completed_sub_batches=b_idx - 1,
                        total_sub_batches=len(sub_payloads),
                        results=merged_result_map
                    )
                    reset_timing = QUOTA_TRACKER.get_reset_timing_info()
                    exhaust_msg = (
                        f"Đã dùng hết quota ngày hôm nay cho tất cả model khả dụng. "
                        f"Tiến độ đã được lưu an toàn tại '{resume_file}'. Vui lòng tiếp tục vào {reset_timing} hoặc thử lại sau."
                    )
                    self.logger.error(f"🛑 [Quota Exhausted] {exhaust_msg}")
                    raise RuntimeError(exhaust_msg)

                self.logger.error(f"Fatal: All model fallbacks failed for chapter sub-batch {b_idx}: {last_error}")
                raise last_error

        return merged_result_map

            