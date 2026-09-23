import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.gemini_proxy_launcher import restart_gemini_proxy
restart_gemini_proxy()

import openai
client = openai.OpenAI(
    api_key="gemi...-key",
    base_url="http://127.0.0.1:8080/v1"
)

try:
    resp = client.chat.completions.create(
        model="gemini-3.5-flash-lite",
        messages=[
            {"role": "system", "content": "You are a Japanese to Vietnamese manga translator. Output JSON."},
            {"role": "user", "content": "{\"translations\": [{\"id\": 1, \"source\": \"おい！油断するなよ！\"}]}"}
        ],
        response_format={"type": "json_object"},
        timeout=15.0
    )
    print("Proxy Success!")
    print("Response:", resp.choices[0].message.content)
except Exception as e:
    print("Proxy Error:", type(e), e)
