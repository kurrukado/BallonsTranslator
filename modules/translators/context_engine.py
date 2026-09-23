"""
Context Engine for Context-Aware, High-Accuracy Manga/Comic Translation
Provides multi-tier context assembly:
1. Multi-Bubble Cohesion & Sequential Ellipsis Stitching
2. Conversation flow, speaker deduction, and reading order
3. Spatial position, panel layout, and text direction
4. Character memory, proper nouns, and pronoun consistency
5. Relevant glossary and terminology memory
6. Cross-page dialogue continuity & genre tone adaptation
"""

import re
import json
import hashlib
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict


SFX_EN_VN: Dict[str, str] = {
    "POW": "BỐP!", "BAM": "CHÁT!", "BOOM": "ĐÙNG!", "CRASH": "RẦM!",
    "THUD": "ỊCH!", "GASP": "*HÁ HỐC*", "STEP": "BƯỚC", "CREAK": "KÉT",
    "DRIP": "NHỎ GIỌT", "SNAP": "RĂNG RẮC", "BANG": "ĐOÀNG!",
    "SLASH": "XOẸT!", "CLANG": "LOẢNG XOẢNG", "PFFT": "PHỤT",
    "EEK": "Á!", "WHAM": "RẦM!", "SMASH": "CHOANG!", "SPLASH": "TÒM TÒM",
    "CLICK": "TÁCH", "SLAM": "SẦM!", "VROOM": "BÙM BÙM",
    "ZAP": "ZÈN ÉT", "BUZZ": "Ù Ù", "ROAR": "GẦM", "HISS": "XÌ XÌ",
    "WHISPER": "THÌ THẦM", "RUMBLE": "ẦM ẦM", "GROAN": "AI ỐI",
    "SIGH": "THỞ DÀI", "SHATTER": "CHOANG!", "PAT": "VỖ VỖ",
    "TAP": "CỐC CỐC", "RING": "RENG RENG",
    "ドン": "RẦM!", "ドーン": "ĐÙNG!",
    "バン": "ĐOÀNG!", "パン": "BỐP!",
    "ガチャ": "KÉT", "ドキドキ": "THỊCH THỊCH",
    "ゴゴゴ": "ẦM ẦM", "ザッ": "XOẸT!",
    "ガタ": "LẠCH CẠCH", "ピク": "GIẬT",
    "パチ": "TÁCH", "ゾク": "RỢN NGƯỜI",
    "キャ": "Á!", "ハッ": "HẢ!",
    "フッ": "PHỤ", "ニヤ": "CƯỜI KHẨY",
    "ボン": "BÙM", "バチ": "RẠNG",
    "ムク": "NGỒI DẬY", "ドサ": "ỊCH",
    "ピタ": "DỪNG", "クス": "CƯỜI KHẼ",
    "シーン": "IM LẶNG",
}


@dataclass
class CharacterProfile:
    name: str
    gender: str = "Unknown"
    age_group: str = "Young Adult"
    role: str = ""
    relationship_to_others: str = ""
    pronouns_self: str = ""
    pronouns_to_others: Dict[str, str] = field(default_factory=dict)
    speech_style: str = ""
    aliases: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: v for k, v in d.items() if v and v != "Unknown"}


class CharacterMemory:
    def __init__(self, profiles: Optional[List[CharacterProfile]] = None):
        self.profiles: Dict[str, CharacterProfile] = {}
        if profiles:
            for p in profiles:
                self.add_profile(p)

    def add_profile(self, profile: CharacterProfile):
        self.profiles[profile.name] = profile
        for alias in profile.aliases:
            self.profiles[alias] = profile

    def get_relevant_characters(self, text_list: List[str]) -> List[CharacterProfile]:
        combined_text = " ".join(text_list).lower()
        matched: Dict[str, CharacterProfile] = {}
        for name, profile in self.profiles.items():
            if name.lower() in combined_text:
                matched[profile.name] = profile
            else:
                for alias in profile.aliases:
                    if alias.lower() in combined_text:
                        matched[profile.name] = profile
                        break
        if not matched and len(self.profiles) <= 4:
            unique_profiles = {}
            for p in self.profiles.values():
                unique_profiles[p.name] = p
            return list(unique_profiles.values())
        return list(matched.values())

    def get_hash(self) -> str:
        data_str = json.dumps([p.to_dict() for p in self.profiles.values()], sort_keys=True)
        return hashlib.md5(data_str.encode("utf-8")).hexdigest()[:8]


class GlossaryManager:
    def __init__(self, entries: Optional[List[Dict[str, str]]] = None):
        self.entries: List[Dict[str, str]] = entries or []

    def load_from_sublists(self, *sublists: List[Dict]):
        self.entries.clear()
        for sublist in sublists:
            if not sublist:
                continue
            for item in sublist:
                kw = item.get("keyword", "").strip()
                sub = item.get("sub", "").strip()
                if kw and sub:
                    self.entries.append({"term": kw, "translation": sub})

    def add_entry(self, term: str, translation: str, category: str = ""):
        self.entries.append({"term": term, "translation": translation, "category": category})

    def get_relevant_terms(self, text_list: List[str]) -> List[Dict[str, str]]:
        combined_text = " ".join(text_list)
        relevant = []
        seen = set()
        for item in self.entries:
            term = item["term"]
            if term in combined_text and term not in seen:
                seen.add(term)
                relevant.append({"term": term, "translation": item["translation"]})
        return relevant

    def get_hash(self) -> str:
        data_str = json.dumps(self.entries, sort_keys=True)
        return hashlib.md5(data_str.encode("utf-8")).hexdigest()[:8]


@dataclass
class DialogueItem:
    id: int
    source: str
    translated: Optional[str] = None
    speaker: Optional[str] = None
    position: str = "Unknown"
    direction: str = "Vertical"
    block_type: str = "DIALOGUE"
    connected_group_id: Optional[int] = None
    flow_hint: Optional[str] = None
    emotion_tag: Optional[str] = "normal"

    def to_prompt_item(self, include_metadata: bool = True) -> dict:
        item = {"id": self.id, "source": self.source}
        if include_metadata:
            if self.position != "Unknown":
                item["position"] = self.position
            if self.direction != "Unknown":
                item["direction"] = self.direction
            if self.block_type != "DIALOGUE":
                item["type"] = self.block_type
            if self.speaker:
                item["speaker"] = self.speaker
            if self.flow_hint:
                item["flow_hint"] = self.flow_hint
            if self.connected_group_id is not None:
                item["connected_group"] = self.connected_group_id
        return item

    def to_compact_string(self) -> str:
        parts = []
        if self.position != "Unknown":
            parts.append(f"Pos: {self.position}")
        if self.block_type != "DIALOGUE":
            parts.append(f"Type: {self.block_type}")
        if self.speaker:
            parts.append(f"Speaker: {self.speaker}")
        if self.connected_group_id is not None:
            parts.append(f"Group: {self.connected_group_id}")
        if self.flow_hint:
            parts.append(f"({self.flow_hint})")
        meta = ", ".join(parts)
        if meta:
            return f"[{self.id}] ({meta}) \"{self.source}\""
        return f"[{self.id}] \"{self.source}\""


class ContextAssembler:
    @staticmethod
    def get_spatial_position(xyxy: List[int], img_w: int = 1000, img_h: int = 1500) -> str:
        if not xyxy or len(xyxy) < 4 or img_w <= 0 or img_h <= 0:
            return "Center"
        cx = (xyxy[0] + xyxy[2]) / 2.0
        cy = (xyxy[1] + xyxy[3]) / 2.0
        rel_x = cx / img_w
        rel_y = cy / img_h
        v_pos = "Top" if rel_y < 0.33 else ("Bottom" if rel_y > 0.66 else "Middle")
        h_pos = "Right" if rel_x > 0.66 else ("Left" if rel_x < 0.33 else "Center")
        return f"{v_pos}-{h_pos}"

    @staticmethod
    def classify_block_type(text: str, vertical: bool) -> str:
        clean = text.strip()
        if len(clean) <= 6 and re.match(r"^[\u30A0-\u30FF\u3040-\u309F\!\?\~…\uFF01\uFF1F\u30FC\u30FB]+$", clean):
            sfx_chars = ["ド", "バ", "ドン", "ギャ", "ハッ", "ドキ", "キャ", "ゴゴ", "ザッ", "ガタ", "ピク", "パチ", "ゾク", "ボン", "バチ", "ムク", "ドサ", "ピタ", "クス", "シーン", "フッ", "ニヤ"]
            if any(c in clean for c in sfx_chars):
                return "SFX"
        upper = clean.upper().rstrip("!.~ ?…\uFF01\uFF1F-")
        if len(clean) <= 8 and clean.isupper() and upper in SFX_EN_VN:
            return "SFX"
        if len(clean) <= 8 and clean.isupper():
            stripped = clean.rstrip("!.~ ?…\uFF01\uFF1F-")
            if stripped in ["POW", "BAM", "CRASH", "THUD", "GASP", "STEP", "CREAK", "DRIP", "SNAP", "BOOM", "BANG", "SLASH", "CLANG", "PFFT", "EEK", "WHAM", "SMASH", "SPLASH", "CLICK", "SLAM", "VROOM", "ZAP", "BUZZ", "ROAR", "HISS", "WHISPER", "RUMBLE", "GROAN", "SIGH", "SHATTER", "PAT", "TAP", "RING"]:
                return "SFX"
        if (clean.startswith("（") and clean.endswith("）")) or (clean.startswith("(") and clean.endswith(")")):
            return "THOUGHT"
        if (clean.startswith("【") and clean.endswith("】")) or (clean.startswith("[") and clean.endswith("]")):
            return "NARRATION"
        if re.match(r"^(AT THAT TIME|MEANWHILE|THREE DAYS LATER|THE NEXT DAY|ON ANOTHER NOTE|EVENTUALLY|IN THE END|LATER|SOMEWHERE ELSE|THAT NIGHT)", clean, re.IGNORECASE):
            return "NARRATION"
        return "DIALOGUE"

    @classmethod
    def detect_connected_bubbles(cls, items: List[DialogueItem]) -> None:
        current_group = 1
        in_group = False

        for i in range(len(items)):
            item = items[i]
            prev_item = items[i - 1] if i > 0 else None
            next_item = items[i + 1] if i + 1 < len(items) else None

            continues_prev = False
            if prev_item:
                p_text = prev_item.source.strip()
                c_text = item.source.strip()
                if p_text.endswith((",", "-", "—", "...", "…", "、", "ー", "〜", "~")):
                    continues_prev = True
                elif re.match(r"^(and|but|or|so|because|where|who|which|whose|that|to|for|with|than|yet|though|even|nhưng|mà|và|vậy|thì)\b", c_text, re.IGNORECASE):
                    continues_prev = True
                elif prev_item.block_type == "THOUGHT" and item.block_type == "THOUGHT" and prev_item.position.split("-")[0] == item.position.split("-")[0]:
                    if not p_text.endswith((".", "!", "?", "。")):
                        continues_prev = True

            leads_next = False
            if next_item:
                c_text = item.source.strip()
                n_text = next_item.source.strip()
                if c_text.endswith((",", "-", "—", "...", "…", "、", "ー", "〜", "~")):
                    leads_next = True
                elif re.match(r"^(and|but|or|so|because|where|who|which|whose|that|to|for|with|than|yet|though|even)\b", n_text, re.IGNORECASE):
                    leads_next = True

            if continues_prev:
                if not in_group:
                    current_group += 1
                    if prev_item:
                        prev_item.connected_group_id = current_group
                        prev_item.flow_hint = "Start of multi-bubble sentence"
                    in_group = True
                item.connected_group_id = current_group
                item.flow_hint = "Continuation of multi-bubble sentence"
            else:
                if in_group and not leads_next:
                    in_group = False

    @classmethod
    def infer_story_genre(cls, texts: List[str]) -> str:
        combined = " ".join(texts).lower()
        if any(w in combined for w in ["murder", "culprit", "police", "investigation", "crime", "victim", "corpse", "explode", "evidence", "suspect", "bizarre", "犯人", "事件", "警察", "捜査", "死体", "殺害"]):
            return "Mystery / Crime Thriller / Detective (Tense, analytical, suspenseful)"
        if any(w in combined for w in ["magic", "attack", "power", "demon", "sword", "enemy", "destroy", "kill", "defeat", "guild", "monster", "魔法", "攻撃", "魔王", "敵", "倒す", "ギルド", "勝負"]):
            return "Action / Supernatural Fantasy (Bold, dramatic, confrontational)"
        if any(w in combined for w in ["senpai", "kouhai", "school", "love", "feelings", "confess", "date", "class", "先輩", "後輩", "放課後", "好き", "告白", "学校", "デート"]):
            return "Romance / Slice of Life / School (Tender, natural, intimate)"
        return "Manga / Comic Drama (Expressive, emotional, natural)"

    @classmethod
    def build_dialogue_items(
        cls,
        textblock_list: Any,
        img_shape: Optional[Tuple[int, int]] = None
    ) -> List[DialogueItem]:
        items: List[DialogueItem] = []
        img_w = img_shape[1] if img_shape and len(img_shape) >= 2 else 1000
        img_h = img_shape[0] if img_shape and len(img_shape) >= 2 else 1500

        for i, blk in enumerate(textblock_list):
            src_text = blk.get_text() if hasattr(blk, "get_text") else str(blk)
            if not src_text.strip():
                continue
            xyxy = getattr(blk, "xyxy", [0, 0, 0, 0])
            position = cls.get_spatial_position(xyxy, img_w, img_h)
            is_vert = getattr(blk, "vertical", True)
            direction = "Vertical" if is_vert else "Horizontal"
            b_type = cls.classify_block_type(src_text, is_vert)
            items.append(
                DialogueItem(
                    id=i + 1,
                    source=src_text,
                    translated=getattr(blk, "translation", None),
                    position=position,
                    direction=direction,
                    block_type=b_type,
                )
            )
        cls.detect_connected_bubbles(items)
        return items

    @staticmethod
    def extract_pronoun_locks(
        dialogue_history: Optional[List[DialogueItem]] = None,
        character_memory: Optional[CharacterMemory] = None
    ) -> List[str]:
        locks: List[str] = []
        seen = set()

        if character_memory:
            for p in character_memory.profiles.values():
                if p.pronouns_self and p.name not in seen:
                    locks.append(f"{p.name}: Xưng '{p.pronouns_self}'")
                    seen.add(p.name)
                for other, pro in p.pronouns_to_others.items():
                    pair_key = f"{p.name}->{other}"
                    if pair_key not in seen:
                        locks.append(f"{p.name} gọi {other} là '{pro}'")
                        seen.add(pair_key)

        if dialogue_history:
            for h in dialogue_history:
                tr = (h.translated or "").lower()
                if ("cậu" in tr or "tớ" in tr or "mình" in tr) and "cậu - tớ" not in seen:
                    locks.append("PRONOUN LOCK: 'cậu - tớ / mình' (Friends/peers)")
                    seen.add("cậu - tớ")
                elif ("anh" in tr and "em" in tr) and "anh - em" not in seen:
                    locks.append("PRONOUN LOCK: 'anh - em' (Intimate / Senior-Junior)")
                    seen.add("anh - em")
                elif ("mày" in tr or "tao" in tr) and "mày - tao" not in seen:
                    locks.append("PRONOUN LOCK: 'mày - tao' (Hostile/combat)")
                    seen.add("mày - tao")
                elif ("tôi" in tr and "ông" in tr) and "tôi - ông" not in seen:
                    locks.append("PRONOUN LOCK: 'tôi - ông' (Formal/respectful)")
                    seen.add("tôi - ông")

        return locks

    @classmethod
    def assemble_payload(
        cls,
        current_items: List[DialogueItem],
        to_lang: str,
        from_lang: str,
        dialogue_history: Optional[List[DialogueItem]] = None,
        character_memory: Optional[CharacterMemory] = None,
        glossary_manager: Optional[GlossaryManager] = None,
        story_genre: Optional[str] = None
    ) -> str:
        source_texts = [item.source for item in current_items]
        inferred_genre = story_genre or cls.infer_story_genre(source_texts)
        relevant_chars = character_memory.get_relevant_characters(source_texts) if character_memory else []
        relevant_terms = glossary_manager.get_relevant_terms(source_texts) if glossary_manager else []
        pronoun_locks = cls.extract_pronoun_locks(dialogue_history, character_memory)

        lines = []
        lines.append(f"SCENE: {inferred_genre}")

        if relevant_chars:
            lines.append("CHARACTERS & PRONOUNS:")
            for c in relevant_chars:
                desc = c.name
                if c.age_group and c.age_group != "Unknown":
                    desc += f" ({c.age_group}"
                    if c.role:
                        desc += f", {c.role}"
                    desc += ")"
                if c.pronouns_self:
                    desc += f": Xưng '{c.pronouns_self}'"
                if c.pronouns_to_others:
                    pairs = ", ".join(f"gọi {k} là '{v}'" for k, v in c.pronouns_to_others.items())
                    desc += f", {pairs}"
                if c.speech_style:
                    desc += f" [{c.speech_style}]"
                lines.append(f"- {desc}")

        if pronoun_locks:
            lines.append("PRONOUN LOCKS:")
            for lock in pronoun_locks:
                lines.append(f"- {lock}")

        if relevant_terms:
            lines.append("RELEVANT TERMS:")
            for t in relevant_terms:
                lines.append(f"- {t['term']} = {t['translation']}")

        if dialogue_history:
            recent = [h for h in dialogue_history[-4:] if h.source.strip()]
            if recent:
                lines.append("PREVIOUS CONVERSATION CONTEXT:")
                for h in recent:
                    src = h.source.replace("\n", " ").strip()
                    tr = (h.translated or "").replace("\n", " ").strip()
                    if tr:
                        lines.append(f"[P{h.id}] \"{src}\" -> \"{tr}\"")
                    else:
                        lines.append(f"[P{h.id}] \"{src}\"")

        lines.append("CURRENT DIALOGUES TO TRANSLATE:")
        for item in current_items:
            lines.append(item.to_compact_string())

        connected_groups = {}
        for item in current_items:
            if item.connected_group_id is not None:
                connected_groups.setdefault(item.connected_group_id, []).append(item.id)

        if connected_groups:
            lines.append("COHESION NOTES:")
            for gid, ids in connected_groups.items():
                lines.append(f"- Group {gid} (IDs: {', '.join(map(str, ids))}): These bubbles form a SINGLE continuous sentence. Translate the whole thought cohesively, then partition clauses across IDs without awkward breaks.")

        return "\n".join(lines)