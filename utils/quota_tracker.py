import os
import sys
import time
import json
import datetime
import threading
from typing import Dict, List, Optional, Tuple, Any

# Root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT_DIR, "cache")
QUOTA_STATE_FILE = os.path.join(CACHE_DIR, "quota_state.json")
RESUME_STATE_FILE = os.path.join(CACHE_DIR, "translation_resume_state.json")

# Google AI Studio Free Tier Quota Definitions
# Reference limits based on real measured AI Studio Free Tier specifications
DEFAULT_QUOTA_PROFILES = {
    # 1. PRIMARY GROUP (High RPD - Flash Lite): Primary workhorses for chapter batching
    "gemini-3.1-flash-lite": {
        "rpm": 15,
        "tpm": 250000,
        "rpd": 500,
        "tier": "primary",
        "context_window": 1048576,
        "daily_budget": 500,
    },
    "gemini-3.5-flash-lite": {
        "rpm": 15,
        "tpm": 250000,
        "rpd": 500,
        "tier": "primary",
        "context_window": 1048576,
        "daily_budget": 500,
    },
    "gemini-2.5-flash-lite": {
        "rpm": 10,
        "tpm": 250000,
        "rpd": 20,
        "tier": "primary",
        "context_window": 1048576,
        "daily_budget": 20,
    },
    # 2. RESERVE GROUP (Low RPD / Large Context): Emergency fallback & large payload reserve
    "gemini-3.5-flash": {
        "rpm": 5,
        "tpm": 250000,
        "rpd": 20,
        "tier": "reserve",
        "context_window": 1048576,
        "daily_budget": 10,  # Safety budget to avoid exhausting RPD in one session
    },
    "gemini-3.6-flash": {
        "rpm": 5,
        "tpm": 250000,
        "rpd": 20,
        "tier": "reserve",
        "context_window": 1048576,
        "daily_budget": 10,
    },
    "gemini-3.7-flash": {
        "rpm": 5,
        "tpm": 250000,
        "rpd": 20,
        "tier": "reserve",
        "context_window": 1048576,
        "daily_budget": 10,
    },
    "gemini-3-flash": {
        "rpm": 5,
        "tpm": 250000,
        "rpd": 20,
        "tier": "reserve",
        "context_window": 1048576,
        "daily_budget": 10,
    },
    "gemini-2.5-flash": {
        "rpm": 5,
        "tpm": 250000,
        "rpd": 20,
        "tier": "reserve",
        "context_window": 1048576,
        "daily_budget": 10,
    },
}

# Explicitly Blacklisted / Unavailable models (RPD 0)
BLACKLISTED_MODELS = {
    "gemini-2.5-pro",
    "gemini-3.1-pro",
    "gemini-2-flash",
    "gemini-2-flash-lite",
    "gemini-pro",
    "gemini-1.5-pro",
}

class QuotaTracker:
    """
    Thread-safe Quota & Rate-Limit Tracker for Google AI Studio Free Tier.
    Tracks daily request counts (RPD), per-model request intervals (RPM),
    and enforces tiered model selection with safe persistence across app restarts.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(QuotaTracker, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._mutex = threading.RLock()
        self.profiles = dict(DEFAULT_QUOTA_PROFILES)
        self.usage_today: Dict[str, int] = {}
        self.exhausted_today: Dict[str, bool] = {}
        self.last_request_time: Dict[str, float] = {}
        self.current_pt_date = self._get_current_pt_date()
        self._load_state()

    def _get_current_pt_date(self) -> str:
        """Get current date in US Pacific Time (UTC-8 / UTC-7) where AI Studio quotas reset at 00:00."""
        pt_tz = datetime.timezone(datetime.timedelta(hours=-8))
        return datetime.datetime.now(pt_tz).strftime("%Y-%m-%d")

    def _check_and_reset_daily(self):
        """Check if day boundary in Pacific Time has rolled over, and reset counts if needed."""
        today = self._get_current_pt_date()
        if today != self.current_pt_date:
            self.current_pt_date = today
            self.usage_today.clear()
            self.exhausted_today.clear()
            self._save_state()

    def _normalize_model_name(self, model: str) -> str:
        if not model:
            return "gemini-3.5-flash-lite"
        m = model.strip()
        if ": " in m:
            m = m.split(": ", 1)[1]
        if m.startswith("models/"):
            m = m[7:]
        # Normalize preview suffixes
        if m.endswith("-preview"):
            m = m[:-8]
        return m

    def _load_state(self):
        """Load persistent quota usage state from cache/quota_state.json."""
        with self._mutex:
            os.makedirs(CACHE_DIR, exist_ok=True)
            if os.path.exists(QUOTA_STATE_FILE):
                try:
                    with open(QUOTA_STATE_FILE, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    saved_date = data.get("date", "")
                    if saved_date == self.current_pt_date:
                        self.usage_today = data.get("usage_today", {})
                        self.exhausted_today = data.get("exhausted_today", {})
                    else:
                        # Expired date, start fresh
                        self.usage_today = {}
                        self.exhausted_today = {}
                except Exception:
                    self.usage_today = {}
                    self.exhausted_today = {}
            self._save_state()

    def _save_state(self):
        """Persist state to cache/quota_state.json."""
        with self._mutex:
            try:
                os.makedirs(CACHE_DIR, exist_ok=True)
                payload = {
                    "date": self.current_pt_date,
                    "updated_at": datetime.datetime.now().isoformat(),
                    "reset_time_pt": "00:00 US Pacific Time",
                    "usage_today": self.usage_today,
                    "exhausted_today": self.exhausted_today,
                    "model_profiles": {
                        k: {
                            "rpm": v["rpm"],
                            "rpd": v["rpd"],
                            "tier": v["tier"],
                            "used_today": self.usage_today.get(k, 0),
                            "remaining_today": max(0, v["rpd"] - self.usage_today.get(k, 0)),
                            "is_exhausted": self.is_rpd_exhausted(k)
                        }
                        for k, v in self.profiles.items()
                    }
                }
                with open(QUOTA_STATE_FILE, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

    def is_blacklisted(self, model: str) -> bool:
        norm = self._normalize_model_name(model)
        return norm in BLACKLISTED_MODELS

    def get_remaining_rpd(self, model: str) -> int:
        """Returns remaining daily requests for the given model."""
        with self._mutex:
            self._check_and_reset_daily()
            norm = self._normalize_model_name(model)
            if norm in BLACKLISTED_MODELS:
                return 0
            profile = self.profiles.get(norm)
            if not profile:
                return 0
            if self.exhausted_today.get(norm, False):
                return 0
            used = self.usage_today.get(norm, 0)
            budget = min(profile["rpd"], profile.get("daily_budget", profile["rpd"]))
            return max(0, budget - used)

    def is_rpd_exhausted(self, model: str) -> bool:
        """Checks if a model has exhausted its daily RPD quota or hit a 429 quota error today."""
        with self._mutex:
            self._check_and_reset_daily()
            norm = self._normalize_model_name(model)
            if norm in BLACKLISTED_MODELS:
                return True
            if self.exhausted_today.get(norm, False):
                return True
            return self.get_remaining_rpd(norm) <= 0

    def record_429_exhaustion(self, model: str, reason: str = "429 Quota Exceeded"):
        """Mark model as exhausted for the remainder of the day."""
        with self._mutex:
            norm = self._normalize_model_name(model)
            self.exhausted_today[norm] = True
            self._save_state()

    def get_required_rpm_throttle(self, model: str) -> float:
        """
        Calculate seconds to sleep to respect RPM limit:
        interval = (60.0 / RPM) * safety_margin (1.05)
        """
        with self._mutex:
            norm = self._normalize_model_name(model)
            profile = self.profiles.get(norm, {"rpm": 10})
            rpm = max(1, profile.get("rpm", 10))
            min_interval = (60.0 / rpm) * 1.05  # 5% safety margin
            
            last_time = self.last_request_time.get(norm, 0.0)
            elapsed = time.time() - last_time
            if elapsed < min_interval:
                return min_interval - elapsed
            return 0.0

    def wait_for_rpm_slot(self, model: str) -> float:
        """Proactively sleeps if needed to respect the model's RPM constraint."""
        throttle_sec = self.get_required_rpm_throttle(model)
        if throttle_sec > 0.05:
            time.sleep(throttle_sec)
            return throttle_sec
        return 0.0

    def record_call(self, model: str, tokens: int = 0):
        """Record an API request to the model, updating usage count and timestamps."""
        with self._mutex:
            self._check_and_reset_daily()
            norm = self._normalize_model_name(model)
            self.usage_today[norm] = self.usage_today.get(norm, 0) + 1
            self.last_request_time[norm] = time.time()
            self._save_state()

    def get_candidate_models(self, preferred_model: Optional[str] = None, estimated_tokens: int = 0) -> List[str]:
        """
        Constructs an ordered list of viable models following the Tiered Fallback Strategy:
        1. Tier 1 (Primary - High RPD Flash Lite): 3.1-flash-lite, 3.5-flash-lite, 2.5-flash-lite
        2. Tier 2 (Reserve - Large Context / Low RPD): 3.5-flash, 3.6-flash, 3.7-flash, 2.5-flash
           (Only included if Tier 1 is exhausted OR payload is exceptionally large > 16K tokens)
        3. Excludes all blacklisted models (RPD 0).
        """
        with self._mutex:
            self._check_and_reset_daily()
            candidates = []

            # 1. Evaluate Tier 1 (Primary - High RPD)
            tier1_models = ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-2.5-flash-lite"]
            
            # If preferred model is a valid Tier 1 model, prioritize it
            norm_pref = self._normalize_model_name(preferred_model) if preferred_model else None
            if norm_pref and norm_pref in tier1_models and not self.is_rpd_exhausted(norm_pref):
                candidates.append(norm_pref)

            for m in tier1_models:
                if m not in candidates and not self.is_rpd_exhausted(m):
                    candidates.append(m)

            # 2. Evaluate Tier 2 (Reserve - Low RPD)
            # Only add Tier 2 if Tier 1 is empty OR payload token count is large (> 16,000 tokens)
            tier2_models = ["gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3-flash", "gemini-2.5-flash"]
            
            if not candidates or estimated_tokens > 16000:
                if norm_pref and norm_pref in tier2_models and not self.is_rpd_exhausted(norm_pref) and norm_pref not in candidates:
                    candidates.append(norm_pref)
                for m in tier2_models:
                    if m not in candidates and not self.is_rpd_exhausted(m):
                        candidates.append(m)

            return candidates

    def are_all_quotas_exhausted(self) -> bool:
        """Check if all available models (both Tier 1 and Tier 2) have 0 remaining requests today."""
        with self._mutex:
            self._check_and_reset_daily()
            for m in self.profiles.keys():
                if not self.is_rpd_exhausted(m):
                    return False
            return True

    def get_reset_timing_info(self) -> str:
        """Returns human-readable explanation of when quotas will reset."""
        pt_tz = datetime.timezone(datetime.timedelta(hours=-8))
        now_pt = datetime.datetime.now(pt_tz)
        tomorrow_pt = (now_pt + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        diff_hours = (tomorrow_pt - now_pt).total_seconds() / 3600.0
        return f"00:00 Pacific Time (~{diff_hours:.1f} giờ nữa / 08:00 UTC)"

    def save_resume_state(self, chapter_info: Dict[str, Any], completed_sub_batches: int, total_sub_batches: int, results: Dict[int, Any]) -> str:
        """Save clean checkpoint resume state when quotas are exhausted."""
        with self._mutex:
            os.makedirs(CACHE_DIR, exist_ok=True)
            checkpoint = {
                "saved_at": datetime.datetime.now().isoformat(),
                "quota_reset_estimated": self.get_reset_timing_info(),
                "chapter_info": chapter_info,
                "progress": {
                    "completed_sub_batches": completed_sub_batches,
                    "total_sub_batches": total_sub_batches,
                    "is_complete": completed_sub_batches >= total_sub_batches
                },
                "completed_results": results
            }
            with open(RESUME_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, ensure_ascii=False, indent=2)
            return RESUME_STATE_FILE


# Global singleton helper
QUOTA_TRACKER = QuotaTracker()
