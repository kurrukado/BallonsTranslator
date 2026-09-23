import urllib.request
import json

key = open('d:/gemini-proxy/api-key.txt').read().strip()
url = 'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions'

# Test WITH response_format
payload_with_rf = {
    'model': 'gemini-2.5-flash-lite',
    'messages': [{'role': 'user', 'content': 'Respond with JSON: {\"greeting\": \"hello\"}'}],
    'response_format': {'type': 'json_object'}
}
req = urllib.request.Request(url, data=json.dumps(payload_with_rf).encode(), headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {key}'})
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        print('WITH response_format status:', resp.status)
except urllib.error.HTTPError as e:
    print('WITH response_format HTTP ERROR:', e.code, e.read().decode())
except Exception as e:
    print('WITH response_format ERROR:', e)

# Test WITHOUT response_format
payload_without_rf = {
    'model': 'gemini-2.5-flash-lite',
    'messages': [{'role': 'user', 'content': 'Respond with JSON: {\"greeting\": \"hello\"}'}]
}
req2 = urllib.request.Request(url, data=json.dumps(payload_without_rf).encode(), headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {key}'})
try:
    with urllib.request.urlopen(req2, timeout=10) as resp:
        print('WITHOUT response_format status:', resp.status)
except urllib.error.HTTPError as e:
    print('WITHOUT response_format HTTP ERROR:', e.code, e.read().decode())
except Exception as e:
    print('WITHOUT response_format ERROR:', e)
