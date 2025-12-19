# OpenRouter Scoring Server (Local) — Rubric Integrated

This is a local FastAPI server that evaluates player counseling responses using an explicit rubric and OpenRouter LLM fallback. It performs a safety-first blacklist check, calls OpenRouter to get a structured JSON evaluation (score + sub-scores + feedback), post-processes results (deterministic formula), caches them, and returns them to the client (Unity or anything that can POST JSON).

Important:
- This server is designed to run locally (your machine). Do NOT store OpenRouter API keys in the Unity client.
- If you don't set an OpenRouter key, the server falls back to a heuristic evaluator so the game remains playable offline.

Files:
- `server_openrouter.py` — main FastAPI server (rubric + blacklist + OpenRouter integration)
- `requirements.txt` — Python deps
- `.env.example` — example env file

Quick start (Windows / macOS / Linux)
1. Create and activate a Python venv:
   - Windows (PowerShell):
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
   - macOS / Linux:
     python -m venv .venv
     source .venv/bin/activate

2. Install dependencies:
   pip install -r requirements.txt

3. Add your OpenRouter API key:
   - Copy `.env.example` to `.env` and set `OPENROUTER_KEY=sk_your_openrouter_key_here`
   - If you do NOT want to use OpenRouter, leave it unset — server uses heuristics.

4. Run server:
   uvicorn server_openrouter:app --host 127.0.0.1 --port 8000 --reload

5. Test with curl (example):
   curl -X POST "http://127.0.0.1:8000/score" -H "Content-Type: application/json" -d "{\"contextId\":\"c1\",\"playerText\":\"Apologize sincerely, make sure she is safe and seek counseling.\"}"

5.1  Test with Powershell (example)
$body = @{
    contextId = "c1"
    playerText = "Apologize sincerely, make sure she is safe and seek counseling."
} | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/score" -Method Post -ContentType "application/json" -Body $body

Endpoints
- GET /health — basic status
- POST /score — body: { "contextId": "string", "playerText": "string" } → returns JSON evaluation:
  {
    "score": int,
    "empathy": int,
    "actionability": int,
    "safety": int,
    "accountability": int,
    "spiritual": int,
    "feedback": "string",
    "flagged": bool,
    "flag_reason": "string"
  }

Notes on rubric & safety
- Server performs a quick regex-based blacklist check for obviously harmful / violent instructions and immediately returns flagged response (score 0).
- Otherwise it constructs a compact prompt with few-shot examples and calls OpenRouter (temperature=0, deterministic).
- The server recomputes the final score using fixed weights to ensure consistency.

If you want, next I will:
- Add a small set of unit tests / example inputs (C step).
- Provide a Unity sample scene that posts answers and displays returned JSON.

Would you like me to prepare the Unity demo next or refine the few-shot examples further?