import urllib.request
import json
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

key = open('d:/gemini-proxy/api-key.txt').read().strip()
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={key}"

payload = {
    "contents": [{"parts": [{"text": "Translate to Vietnamese in 1 line: おい！油断するなよ！"}]}]
}

req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        print("[SUCCESS] Direct Gemini response:", data["candidates"][0]["content"]["parts"][0]["text"])
except Exception as e:
    print("[ERROR] Google API failed:", e)
