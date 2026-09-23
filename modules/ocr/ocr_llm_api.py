import re
import time
import base64
import json
import cv2
import numpy as np
from typing import List, Optional, Dict

class _LazyOpenAI:
    def __getattr__(self, name):
        import openai
        return getattr(openai, name)

openai = _LazyOpenAI()

import httpx

from .base import register_OCR, OCRBase, TextBlock


@register_OCR("llm_ocr")
class LLM_OCR(OCRBase):
    lang_map = {
        "Auto Detect": None,
        "Afrikaans": "af",
        "Albanian": "sq",
        "Amharic": "am",
        "Arabic": "ar",
        "Armenian": "hy",
        "Assamese": "as",
        "Azerbaijani": "az",
        "Bangla": "bn",
        "Basque": "eu",
        "Belarusian": "be",
        "Bengali": "bn",
        "Bosnian": "bs",
        "Breton": "br",
        "Bulgarian": "bg",
        "Burmese": "my",
        "Catalan": "ca",
        "Cebuano": "ceb",
        "Cherokee": "chr",
        "Chinese (Simplified)": "zh-CN",
        "Chinese (Traditional)": "zh-TW",
        "Corsican": "co",
        "Croatian": "hr",
        "Czech": "cs",
        "Danish": "da",
        "Dutch": "nl",
        "English": "en",
        "Esperanto": "eo",
        "Estonian": "et",
        "Faroese": "fo",
        "Filipino": "fil",
        "Finnish": "fi",
        "French": "fr",
        "Frisian": "fy",
        "Galician": "gl",
        "Georgian": "ka",
        "German": "de",
        "Greek": "el",
        "Gujarati": "gu",
        "Haitian Creole": "ht",
        "Hausa": "ha",
        "Hawaiian": "haw",
        "Hebrew": "he",
        "Hindi": "hi",
        "Hmong": "hmn",
        "Hungarian": "hu",
        "Icelandic": "is",
        "Igbo": "ig",
        "Indonesian": "id",
        "Interlingua": "ia",
        "Irish": "ga",
        "Italian": "it",
        "Japanese": "ja",
        "Javanese": "jv",
        "Kannada": "kn",
        "Kazakh": "kk",
        "Khmer": "km",
        "Korean": "ko",
        "Kurdish": "ku",
        "Kyrgyz": "ky",
        "Lao": "lo",
        "Latin": "la",
        "Latvian": "lv",
        "Lithuanian": "lt",
        "Luxembourgish": "lb",
        "Macedonian": "mk",
        "Malagasy": "mg",
        "Malay": "ms",
        "Malayalam": "ml",
        "Maltese": "mt",
        "Maori": "mi",
        "Marathi": "mr",
        "Mongolian": "mn",
        "Nepali": "ne",
        "Norwegian": "no",
        "Occitan": "oc",
        "Oriya": "or",
        "Pashto": "ps",
        "Persian": "fa",
        "Polish": "pl",
        "Portuguese": "pt",
        "Punjabi": "pa",
        "Quechua": "qu",
        "Romanian": "ro",
        "Russian": "ru",
        "Samoan": "sm",
        "Scots Gaelic": "gd",
        "Serbian (Cyrillic)": "sr-Cyrl",
        "Serbian (Latin)": "sr-Latn",
        "Shona": "sn",
        "Sindhi": "sd",
        "Sinhala": "si",
        "Slovak": "sk",
        "Slovenian": "sl",
        "Somali": "so",
        "Spanish": "es",
        "Sundanese": "su",
        "Swahili": "sw",
        "Swedish": "sv",
        "Tagalog": "tl",
        "Tajik": "tg",
        "Tamil": "ta",
        "Tatar": "tt",
        "Telugu": "te",
        "Thai": "th",
        "Tibetan": "bo",
        "Tigrinya": "ti",
        "Tongan": "to",
        "Turkish": "tr",
        "Ukrainian": "uk",
        "Urdu": "ur",
        "Uyghur": "ug",
        "Uzbek": "uz",
        "Vietnamese": "vi",
        "Welsh": "cy",
        "Xhosa": "xh",
        "Yiddish": "yi",
        "Yoruba": "yo",
        "Zulu": "zu",
    }

    popular_models = [
        "OAI: gpt-4o-mini",
        "OAI: gpt-4-vision-preview",
        "OAI: gpt-4o",
        "OAI: gpt-4",
        "GGL: gemini-3.1-pro-preview",
        "GGL: gemini-3.5-flash-lite",
        "GGL: gemini-3.5-flash",
        "GGL: gemini-3-flash-preview",
    ]

    params = {
        "provider": {
            "type": "selector",
            "options": ["Google", "OpenAI", "OpenRouter", "Ollama"],
            "value": "Google",
            "description": "Select the LLM provider.",
        },
        "api_key": {
            "value": "",
            "description": "API key to use if multiple keys are not provided.",
        },
        "multiple_keys": {
            "type": "editor",
            "value": "",
            "description": "API keys separated by semicolons (;). Requests will rotate.",
        },
        "endpoint": {
            "value": "",
            "description": "Base URL for the API. Leave empty for provider default.",
        },
        "model": {
            "type": "selector",
            "options": popular_models + [
                "OLLAMA: (override model field)"
            ],
            "value": "GGL: gemini-3.6-flash",
            "description": "Select the model to use.",
        },
        "override_model": {
            "value": "",
            "description": "Specify a custom model name to override the selected one.",
        },
        "language": {
            "type": "selector",
            "options": list(lang_map.keys()),
            "value": "English",
            "description": "Language for OCR.",
        },
        "detail_level": {
            "type": "selector",
            "options": ["auto", "low", "high"],
            "value": "auto",
            "description": "Controls image detail level for vision models.",
        },
        "prompt": {
            "type": "editor",
            "value": "Perform strict OCR on the provided manga/comic image crop. The language is **{language}**.\n\nCRITICAL TRANSCRIPTION RULES:\n1. READ EXACTLY WHAT IS VISUALLY PRESENT in the image pixels.\n2. DO NOT translate. DO NOT paraphrase. DO NOT correct grammar or spelling.\n3. DO NOT invent missing words or extrapolate from story context.\n4. PRESERVE exact capitalization: If the comic text is in ALL-CAPS, your output MUST be ALL-CAPS.\n5. PRESERVE all punctuation exactly as visible: apostrophes (I'm, don't), ellipses (...), exclamation/question marks (?!, !?, !).\n6. PRESERVE sound effects (SFX) and character names exactly as lettered.\n7. OUTPUT FORMAT: Return ONLY the exact transcribed text as a single continuous horizontal line. No quotes, no markdown, no conversational filler.",
            "description": "The main prompt for the OCR task. Use {language} placeholder.",
        },
        "system_prompt": {
            "type": "editor",
            "value": "You are an elite, precision computer vision OCR specialist for comic and manga scanlations. Your SOLE duty is verbatim transcription of image text into text strings. You NEVER translate, summarize, normalize, paraphrase, or alter the source text in any way.",
            "description": "Optional system prompt to guide the model's behavior.",
        },
        "proxy": {
            "value": "",
            "description": "Proxy address (e.g., http(s)://user:password@host:port)",
        },
        "delay": {"value": 6.0, "description": "Delay in seconds between requests (10 RPM = 6s min interval)."},
        "requests_per_minute": {
            "value": 10,
            "description": "Maximum number of requests per minute per key.",
        },
        "max_response_tokens": {
            "value": 4096,
            "description": "Maximum number of tokens in the LLM's response.",
        },
        "description": "OCR using various vision-capable LLMs.",
    }

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.last_request_time = 0
        self.client = None
        self.request_count_minute = 0
        self.minute_start_time = time.time()
        self.key_usage = {}
        self.current_key_index = 0
        self._model_last_request_time: Dict[str, float] = {}
        self.MODEL_RPM_LIMITS: Dict[str, int] = {
            "gemini-3.5-flash-lite": 10,
            "gemini-3.6-flash": 5,
            "gemini-3.7-flash": 5,
        }

    def _initialize_client(self, api_key_to_use: str):
        if not api_key_to_use:
            try:
                from utils.config import ProgramConfig
                api_key_to_use = ProgramConfig().get_param("translator", "apikey", "")
            except Exception:
                pass

        endpoint = self.endpoint
        provider = self.provider
        if not endpoint:
            if provider in ["Google", "Gemini Proxy"]:
                endpoint = "https://generativelanguage.googleapis.com/v1beta/openai"
            elif provider == "OpenAI":
                endpoint = "https://api.openai.com/v1"
            elif provider == "OpenRouter":
                endpoint = "https://openrouter.ai/api/v1"
            elif provider == "Ollama":
                endpoint = "http://localhost:11434/v1"

        http_client = None
        if self.proxy:
            try:
                proxy_mounts = {"all://": httpx.HTTPTransport(proxy=self.proxy)}
                http_client = httpx.Client(mounts=proxy_mounts)
            except Exception as e:
                self.logger.error(f"Failed to initialize proxy '{self.proxy}': {e}.")

        masked_key = (
            api_key_to_use[:4] + "..." + api_key_to_use[-4:]
            if len(api_key_to_use) > 8
            else api_key_to_use
        )
        self.logger.debug(
            f"Initializing client for {provider} with key {masked_key} at endpoint {endpoint}"
        )

        self.client = openai.OpenAI(
            api_key=api_key_to_use, base_url=endpoint, http_client=http_client
        )

    # --- Property Getters (similar to translator) ---
    @property
    def provider(self) -> str:
        return self.get_param_value("provider")

    @property
    def api_key(self) -> str:
        return self.get_param_value("api_key")

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
    def endpoint(self) -> Optional[str]:
        return self.get_param_value("endpoint") or None

    @property
    def model(self) -> str:
        return self.get_param_value("model")

    @property
    def override_model(self) -> Optional[str]:
        return self.get_param_value("override_model") or None

    @property
    def language(self) -> str:
        return self.get_param_value("language")

    @property
    def detail_level(self) -> str:
        return self.get_param_value("detail_level")

    @property
    def prompt(self) -> str:
        return self.get_param_value("prompt")

    @property
    def system_prompt(self) -> str:
        return self.get_param_value("system_prompt")

    @property
    def proxy(self) -> str:
        return self.get_param_value("proxy")

    @property
    def requests_per_minute(self) -> int:
        return int(self.get_param_value("requests_per_minute"))

    @property
    def max_response_tokens(self) -> int:
        return int(self.get_param_value("max_response_tokens"))

    @property
    def request_delay(self) -> float:
        try:
            return float(self.get_param_value("delay"))
        except (ValueError, TypeError):
            return 1.0

    def _respect_delay(self):
        # This logic is identical to the one in LLM_API_Translator
        current_time = time.time()
        rpm = self.requests_per_minute
        if rpm > 0:
            if current_time - self.minute_start_time >= 60:
                self.request_count_minute = 0
                self.minute_start_time = current_time
            if self.request_count_minute >= rpm:
                wait_time = 60.1 - (current_time - self.minute_start_time)
                if wait_time > 0:
                    self.logger.warning(
                        f"Global RPM limit ({rpm}) reached. Waiting {wait_time:.2f}s."
                    )
                    time.sleep(wait_time)
                self.request_count_minute = 0
                self.minute_start_time = time.time()

        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.request_delay:
            sleep_time = self.request_delay - time_since_last_request
            if self.debug_mode:
                self.logger.debug(f"Global delay: Waiting {sleep_time:.3f}s.")
            time.sleep(sleep_time)

        self.last_request_time = time.time()
        self.request_count_minute += 1

    def _respect_model_delay(self, model_name: str):
        now = time.time()
        rpm = self.MODEL_RPM_LIMITS.get(model_name, self.requests_per_minute if self.requests_per_minute > 0 else 10)
        min_interval = max(60.0 / rpm, self.request_delay)
        last_time = getattr(self, "_model_last_request_time", {}).get(model_name, 0.0)
        time_since = now - last_time
        if time_since < min_interval:
            sleep_time = min_interval - time_since
            self.logger.info(f"⏳ [OCR Throttle] Model '{model_name}' (Limit {rpm} RPM): Tạm dừng {sleep_time:.2f}s...")
            time.sleep(sleep_time)
        if not hasattr(self, "_model_last_request_time"):
            self._model_last_request_time = {}
        self._model_last_request_time[model_name] = time.time()

    def _respect_key_limit(self, key: str) -> bool:
        # This logic is identical to the one in LLM_API_Translator
        rpm = self.requests_per_minute
        if rpm <= 0:
            return True
        now = time.time()
        count, start_time = self.key_usage.get(key, (0, now))
        if now - start_time >= 60:
            count, start_time = 0, now
        if count >= rpm:
            wait_time = 60.1 - (now - start_time)
            if wait_time > 0:
                self.logger.warning(
                    f"RPM limit ({rpm}) for key {key[:6]}... reached. Waiting {wait_time:.2f}s."
                )
                time.sleep(wait_time)
            self.key_usage[key] = (0, time.time())
            return False
        return True

    def _select_api_key(self) -> Optional[str]:
        # This logic is identical to the one in LLM_API_Translator
        api_keys = self.multiple_keys_list
        single_key = self.api_key
        if not api_keys and not single_key:
            try:
                from utils.gemini_proxy_launcher import get_saved_api_key
                saved_key = get_saved_api_key()
                if saved_key:
                    return saved_key
            except Exception:
                pass
            if self.provider in ["Gemini Proxy", "Google", "LLM Studio", "Ollama"]:
                return "gemini-proxy-key"
            self.logger.error("No API keys provided.")
            return None

        if not api_keys:
            if self._respect_key_limit(single_key):
                now = time.time()
                count, start_time = self.key_usage.get(single_key, (0, now))
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
        self.logger.error("All API keys are rate-limited.")
        return None

    def ocr(self, img_base64: str, prompt_override: str = None) -> str:
        api_key_to_use = self._select_api_key()
        
        if not api_key_to_use:
            if self.provider in ["Gemini Proxy", "LLM Studio", "Ollama"]:
                api_key_to_use = "gemini-proxy-key"
            else:
                return "[ERROR: No available API key]"

        # Re-initialize client if key is different from the last one used
        if not self.client or self.client.api_key != api_key_to_use:
            self._initialize_client(api_key_to_use)

        self._respect_delay()
        try:
            lang_name = self.language
            prompt_text = (prompt_override or self.prompt).format(language=lang_name)

            image_content_part = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
            }

            if self.provider in ["OpenAI", "Google", "OpenRouter"]:
                detail_setting = self.detail_level
                if detail_setting in ["low", "high"]:
                    image_content_part["image_url"]["detail"] = detail_setting

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        image_content_part,
                    ],
                }
            ]
            if self.system_prompt:
                messages.insert(0, {"role": "system", "content": self.system_prompt})

            model_name = self.override_model or self.model
            if ": " in model_name:
                model_name = model_name.split(": ", 1)[1]

            if self.provider in ["Google", "Gemini Proxy"]:
                # Use model names as-is - they are real Google API model names
                target_model = model_name
            else:
                target_model = model_name

            candidates = [target_model]
            if self.provider in ["Google", "Gemini Proxy"]:
                for m in ["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.7-flash"]:
                    if m not in candidates:
                        candidates.append(m)

            max_retries = 3
            response = None
            last_error = None
            for current_model in candidates:
                self._respect_model_delay(current_model)
                for attempt in range(max_retries):
                    try:
                        self.logger.debug(f"OCR request with model: {current_model} (attempt {attempt+1}/{max_retries})")
                        response = self.client.chat.completions.create(
                            model=current_model,
                            messages=messages,
                            max_tokens=self.max_response_tokens,
                        )
                        last_error = None
                        break
                    except Exception as e:
                        err_str = str(e)
                        last_error = e
                        is_rate_limit = ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower())
                        retry_delay = 1.5 + (attempt * 2.0)
                        if is_rate_limit and attempt < max_retries - 1:
                            self.logger.info(f"⏳ OCR 429 Rate Limit - Đang chờ {retry_delay:.0f}s (lần {attempt+2}/{max_retries})...")
                            time.sleep(retry_delay)
                            continue
                        elif len(candidates) > 1:
                            next_idx = candidates.index(current_model) + 1
                            if next_idx < len(candidates):
                                self.logger.warning(f"⚠️ OCR Model '{current_model}' thất bại ({type(e).__name__}) -> Tự động chuyển sang '{candidates[next_idx]}'...")
                                break
                        raise
                if response is not None:
                    break

            if response and response.choices and response.choices[0].message.content:
                full_text = (
                    response.choices[0].message.content.replace("\n", " ").strip()
                )
                self.logger.debug(f"OCR result: {full_text}")
                return full_text
            else:
                self.logger.warning("No text found in OCR response.")
                return ""
        except Exception as e:
            self.logger.error(f"OCR error: {e}")
            return f"[ERROR: {type(e).__name__}]"

    def _build_batched_request(
        self,
        img: np.ndarray,
        blk_list: List[TextBlock],
        prompt: str,
        return_texts: bool = False
    ) -> List[str]:
        """Send ALL cropped text blocks in a SINGLE Gemini Vision request.
        1 page = 1 request (independent of block count) - Free-Tier Safe."""
        valid_blks: List[TextBlock] = []
        crops = []
        for blk in blk_list:
            x1, y1, x2, y2 = blk.xyxy
            if 0 <= x1 < x2 <= img.shape[1] and 0 <= y1 < y2 <= img.shape[0]:
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                valid_blks.append(blk)
                if crop.ndim == 3 and crop.shape[-1] == 4:
                    crop = cv2.cvtColor(crop, cv2.COLOR_RGBA2RGB)
                elif crop.ndim == 2:
                    crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
                _, buffer = cv2.imencode(".jpg", crop)
                crops.append(base64.b64encode(buffer).decode("utf-8"))

        if not valid_blks:
            for blk in blk_list:
                blk.text = ""
            return [""] * len(blk_list)

        lang_name = self.language
        prompt_text = (prompt or self.prompt).format(language=lang_name)
        task_prefix = (
            "You are batch OCR. The following numbered image snippets each correspond to ONE text block.\n"
            "Return results EXACTLY as a JSON object mapping each number to its recognized text.\n"
            'Example: {"1": "Hello world", "2": "NO WAY!"}\n'
            "Recognize ALL text in each snippet (including stylized SFX/brush lettering). "
            "Consolidate each snippet's text into a single horizontal line. No explanations.\n"
            "BATCH INPUT:\n"
        )
        parts = [{"type": "text", "text": task_prefix + prompt_text}]
        for idx, b64 in enumerate(crops):
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

        messages = [{"role": "user", "content": parts}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        model_name = self.override_model or self.model
        if ": " in model_name:
            model_name = model_name.split(": ", 1)[1]

        candidates = [model_name]
        if self.provider in ["Google", "Gemini Proxy"]:
            for m in ["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.7-flash"]:
                if m not in candidates:
                    candidates.append(m)

        api_key_to_use = self._select_api_key()
        if not api_key_to_use:
            api_key_to_use = "gemini-proxy-key"
        if not self.client or self.client.api_key != api_key_to_use:
            self._initialize_client(api_key_to_use)
        self._respect_delay()

        max_retries = 3
        response = None
        for current_model in candidates:
            self._respect_model_delay(current_model)
            for attempt in range(max_retries):
                try:
                    self.logger.info(f"🌐 [Gemini OCR Batch] Gửi {len(crops)} khối chữ trong 1 request duy nhất (Model: {current_model})...")
                    response = self.client.chat.completions.create(
                        model=current_model,
                        messages=messages,
                        max_tokens=self.max_response_tokens,
                        response_format={"type": "json_object"},
                    )
                    break
                except Exception as e:
                    err_str = str(e)
                    is_rate_limit = ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower())
                    retry_delay = 1.5 + (attempt * 2.0)
                    if is_rate_limit and attempt < max_retries - 1:
                        self.logger.info(f"⏳ [Gemini OCR Batch] 429 - Đang chờ {retry_delay:.0f}s (lần {attempt+2}/{max_retries})...")
                        time.sleep(retry_delay)
                        continue
                    elif len(candidates) > 1 and candidates.index(current_model) + 1 < len(candidates):
                        next_model = candidates[candidates.index(current_model) + 1]
                        self.logger.warning(f"⚠️ [Gemini OCR Batch] Model '{current_model}' thất bại ({type(e).__name__}) -> chuyển sang '{next_model}'...")
                        break
                    self.logger.error(f"[Gemini OCR Batch] Fatal: {type(e).__name__}: {e}")
                    for blk in valid_blks:
                        blk.text = []
                    return [""] * len(valid_blks)
            if response is not None:
                break

        results: List[str] = []
        if response and response.choices and response.choices[0].message.content:
            raw = response.choices[0].message.content.strip()
            try:
                import json as _json
                start = raw.find("{")
                end = raw.rfind("}")
                if start != -1 and end > start:
                    raw = raw[start:end + 1]
                data = _json.loads(raw)
                results = [str(data.get(str(i), "")) for i in range(1, len(valid_blks) + 1)]
            except Exception:
                texts = re.findall(r'"(\d+)"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
                mapping = {}
                for tid, ttext in texts:
                    mapping[int(tid)] = ttext
                results = [mapping.get(i + 1, "") for i in range(len(valid_blks))]
        else:
            results = [""] * len(valid_blks)

        for blk, text in zip(valid_blks, results):
            blk.text = [text] if text else []
        return results

    def _ocr_blk_list(
        self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs
    ):
        if len(blk_list) > 1:
            self._build_batched_request(img, blk_list, kwargs.get("prompt"))
            return

        im_h, im_w = img.shape[:2]
        for idx, blk in enumerate(blk_list):
            x1, y1, x2, y2 = blk.xyxy
            if 0 <= x1 < x2 <= im_w and 0 <= y1 < y2 <= im_h:
                if idx > 0:
                    model_name = self.override_model or self.model
                    if ": " in model_name:
                        model_name = model_name.split(": ", 1)[1]
                    self._respect_model_delay(model_name)
                cropped_img = img[y1:y2, x1:x2]
                _, buffer = cv2.imencode(".jpg", cropped_img)
                img_base64 = base64.b64encode(buffer).decode("utf-8")
                blk.text = self.ocr(img_base64, prompt_override=kwargs.get("prompt"))
            else:
                blk.text = ""

    def ocr_img(self, img: np.ndarray, prompt: str = "") -> str:
        _, buffer = cv2.imencode(".jpg", img)
        img_base64 = base64.b64encode(buffer).decode("utf-8")
        return self.ocr(img_base64, prompt_override=prompt)

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key in ["api_key", "multiple_keys", "endpoint", "proxy", "provider"]:
            self.client = None  # Force re-initialization on next call
        if param_key in ["requests_per_minute", "delay"]:
            self.request_count_minute = 0
            self.minute_start_time = time.time()
            self.last_request_time = 0
            