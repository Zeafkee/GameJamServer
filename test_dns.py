import requests
try:
    r = requests.get("https://api.openrouter.ai/v1/chat/completions", timeout=8)
    print("status:", r.status_code)
    print("headers:", r.headers)
    print("body (partial):", r.text[:500])
except Exception as e:
    print("ERROR:", repr(e))