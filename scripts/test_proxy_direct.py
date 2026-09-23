import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import openai
from utils.gemini_proxy_launcher import restart_gemini_proxy

restart_gemini_proxy()

client = openai.OpenAI(
    api_key="gemi...-key",
    base_url="http://127.0.0.1:8080/v1"
)

resp = client.chat.completions.create(
    model="gemini-3.5-flash-lite",
    messages=[
        {"role": "system", "content": "You are a Japanese to Vietnamese manga translator. Respond with JSON: {\"translations\": [{\"id\": 1, \"translation\": \"text\"}]}"},
        {"role": "user", "content": "Translate strictly into Vietnamese: [{\"id\": 1, \"source\": \"おい！油断するなよ！\"}]"}
    ],
    response_format={"type": "json_object"},
    timeout=20.0
)

print("[SUCCESS] Gemini 3.5 Flash Lite proxy call:")
print(resp.choices[0].message.content)
