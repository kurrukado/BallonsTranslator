import urllib.request
import json
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

key = open('d:/gemini-proxy/api-key.txt').read().strip()
models_to_test = [
    'gemini-2.5-flash',
    'gemini-2.5-flash-lite',
    'gemini-2.0-flash',
    'gemini-2.0-flash-lite',
    'gemini-1.5-flash'
]

for m in models_to_test:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={key}"
    payload = {"contents": [{"parts": [{"text": "Hello"}]}]}
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            resp_txt = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            print(f"Model {m:22s} : [SUCCESS] -> {resp_txt}")
    except Exception as e:
        print(f"Model {m:22s} : [FAILED]  -> {e}")
