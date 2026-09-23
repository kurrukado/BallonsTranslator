import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import openai
import json
from utils.gemini_proxy_launcher import ensure_gemini_proxy_running, sync_proxy_api_key

ensure_gemini_proxy_running()
_api_key = os.getenv("GEMINI_API_KEY", "")
if not _api_key:
    try:
        with open("config/config.json", "r", encoding="utf-8") as _f:
            _api_key = json.load(_f).get("module", {}).get("translator_params", {}).get("LLM_API_Translator", {}).get("apikey", "")
    except Exception:
        pass
sync_proxy_api_key(_api_key)

client = openai.OpenAI(
    api_key=_api_key or "dummy_key_for_proxy",
    base_url="http://127.0.0.1:8080/v1"
)

dialogues = [
    {
        "id": 1,
        "panel": "Panel 1 (Top)",
        "type": "DIALOGUE (Police to MC)",
        "source": "ON ANOTHER NOTE, THERE HAVE ALSO BEEN A SERIES OF MURDERS WHERE THE BODIES WERE FOUND AND APPEARED TO HAVE EXPLODED FROM THE INSIDE."
    },
    {
        "id": 2,
        "panel": "Panel 1 (Top)",
        "type": "DIALOGUE (Police to MC)",
        "source": "BUT, BECAUSE OF PRESSURE FROM SOMEONE WITH PRETTY HIGH INFLUENCE, THE INVESTIGATION COULDN'T PROCEED."
    },
    {
        "id": 3,
        "panel": "Panel 2 (Middle)",
        "type": "THOUGHT (MC Inner Monologue - Connected Part 1)",
        "source": "I DON'T KNOW ANYONE OTHER THAN YUREA WHO HAS THAT KIND OF POWER,"
    },
    {
        "id": 4,
        "panel": "Panel 2 (Middle)",
        "type": "THOUGHT (MC Inner Monologue - Connected Part 2)",
        "source": "BUT JUDGING BY THE SITUATION, IT'S UNLIKELY TO BE HER."
    },
    {
        "id": 5,
        "panel": "Panel 2 (Middle)",
        "type": "THOUGHT (MC Inner Monologue - Connected Part 3)",
        "source": "AT THE VERY LEAST, SHE WAS WITH ME THE ENTIRE NIGHT A CRIME WAS COMMITTED, SO I KNEW THAT YUREA WASN'T THE CULPRIT."
    },
    {
        "id": 6,
        "panel": "Panel 3 (Bottom)",
        "type": "THOUGHT (MC Inner Monologue - Deductive Reasoning Part 1)",
        "source": "IN THE FIRST PLACE, HOW DO YOU EVEN MURDER SOMEONE BY MAKING THEIR BODY EXPLODE FROM THE INSIDE?"
    },
    {
        "id": 7,
        "panel": "Panel 3 (Bottom)",
        "type": "THOUGHT (MC Inner Monologue - Deductive Reasoning Part 2)",
        "source": "WHO IN THE WORLD HAS ENOUGH INFLUENCE TO PRESSURE THE POLICE INTO SUPPRESSING SUCH A BIZARRE MURDER CASE?"
    }
]

system_prompt = (
    "You are an elite Manga/Manhwa/Webtoon localization translator specializing in Vietnamese (Tiếng Việt).\n"
    "Your goal: Produce the most natural, suspenseful, emotionally resonant, and cohesive translation possible.\n\n"
    "CRITICAL LOCALIZATION RULES:\n"
    "1. COHESIVE MULTI-BUBBLE FLOW (Ghép mạch câu đa bong bóng):\n"
    "   - Notice that bubbles 3, 4, 5 form one continuous train of thought. Translate them so the sentence transition feels completely natural in Vietnamese, not like disjointed fragments.\n"
    "   - Bubbles 6 and 7 form a sharp deductive reasoning sequence in the character's mind.\n"
    "2. NATURAL VIETNAMESE LOCALIZATION:\n"
    "   - Use evocative, cinematic thriller phrasing ('vụ án mạng kỳ quái', 'nổ tung từ bên trong', 'sức ép từ kẻ có tầm ảnh hưởng lớn', 'bưng bít vụ án').\n"
    "   - Character inner monologue: Use natural self-reflection ('mình', 'cô ấy', or subjectless flow) instead of robotic 'Tôi'.\n"
    "   - Keep character name 'Yurea' consistent.\n"
    "3. JSON SCHEMA:\n"
    "   - Respond strictly with JSON: {\"translations\": [{\"id\": 1, \"translation\": \"...\"}]}"
)

user_prompt = (
    "Please translate the following sequential manga speech bubbles into natural Vietnamese:\n\n"
    f"{json.dumps(dialogues, ensure_ascii=False, indent=2)}"
)

resp = client.chat.completions.create(
    model="gemini-3.5-flash-lite",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    response_format={"type": "json_object"},
    timeout=30.0
)

result = json.loads(resp.choices[0].message.content)

print("=" * 80)
print("  HIGH-QUALITY CONTEXT-AWARE LOCALIZATION FOR USER MANGA SAMPLE")
print("=" * 80)
for item in result.get("translations", []):
    item_id = item["id"]
    orig = dialogues[item_id - 1]
    print(f"[{item_id}] {orig['panel']} | {orig['type']}")
    print(f"     EN: {orig['source']}")
    print(f"     VI: {item['translation']}\n")
print("=" * 80)
