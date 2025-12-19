#!/usr/bin/env python3
"""
Local FastAPI scoring server using OpenRouter with a rubric.
Run:
  pip install -r requirements.txt
  (set OPENROUTER_KEY in .env or environment)
  uvicorn server_openrouter:app --host 127.0.0.1 --port 8000 --reload
"""

import os
import json
import hashlib
import re
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from cachetools import TTLCache
from dotenv import load_dotenv
import requests

load_dotenv()  # load .env (OPENROUTER_KEY)

# Config
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")  # if None -> offline heuristic
OPENROUTER_URL ="https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "google/gemini-2.5-flash-lite-preview-09-2025"
CACHE_TTL = 60 * 60 * 24  # 24 hours
CACHE_SIZE = 10000
MAX_TOKENS = 150
TEMPERATURE = 0.0

# Rubric weights
WEIGHTS = {
    "empathy": 0.30,
    "actionability": 0.30,
    "safety": 0.20,
    "accountability": 0.10,
    "spiritual": 0.10
}
# Map IDs to actual text so the AI understands the scenario
CONTEXT_MAP = {
    "c1": "I beat my wife, I'm feeling guilty what should I do?",
    "c2": "I stole my neighbor's food and I feel guilty.",
    "c3": "I lied on taxes and might get caught."
}

# blacklist patterns for immediate flagging (case-insensitive)
BAD_PATTERNS = [
    r"\bkill\b", r"\bmurder\b", r"\brape\b", r"\bpoison\b", r"\bshoot\b",
    r"\bbeat\b", r"\bbeat (her|him|them)\b", r"\btie (her|him|them)\b",
    r"\bblackmail\b", r"\bhide (the )?body\b", r"\bset (fire|a fire)\b",
    r"\bburn\b", r"\bextort\b", r"\bthreaten\b", r"\bforce\b",
    r"\bdo it yourself\b", r"\bkill yourself\b", r"\bcommit suicide\b"
]
BLACKLIST_RE = re.compile("|".join(BAD_PATTERNS), flags=re.IGNORECASE)

# Create server & cache
app = FastAPI(title="Local OpenRouter Scoring Server")
cache = TTLCache(maxsize=CACHE_SIZE, ttl=CACHE_TTL)

# Request model
class ScoreReq(BaseModel):
    contextId: str
    playerText: str

def normalize_text(s: str) -> str:
    return " ".join(s.strip().split())

def clamp_int(x: Optional[int]) -> int:
    if x is None:
        return 0
    try:
        xi = int(round(x))
    except Exception:
        xi = 0
    return max(0, min(100, xi))

def compute_final_score(subscores: dict) -> int:
    # subscores: empathy, actionability, safety, accountability, spiritual
    total = 0.0
    for k, w in WEIGHTS.items():
        total += w * clamp_int(subscores.get(k, 0))
    return int(round(total))

def extract_json_block(s: str):
    # extract the first {...} block and parse JSON
    if not s:
        return None
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        # no JSON block
        return None
    snippet = s[start:end+1]
    try:
        return json.loads(snippet)
    except Exception:
        return None

def build_prompt(context: str, player_answer: str) -> str:
    # Keep prompt compact and deterministic; include few-shot examples.
    few_shot = [
        {
            "context": "I beat my wife, I'm feeling guilty what should I do?",
            "player_answer": "Apologize sincerely, make sure she is safe; seek professional counseling; accept legal consequences; do not repeat the abuse; pray and ask for forgiveness.",
            "json": {
                "score": 82,
                "empathy": 85,
                "actionability": 80,
                "safety": 90,
                "accountability": 85,
                "spiritual": 60,
                "feedback": "Good empathy and safety focus; recommend immediate support and professional counseling and legal accountability.",
                "flagged": False,
                "flag_reason": ""
            }
        },
        {
            "context": "I stole my neighbor's food and I feel guilty; what should I do?",
            "player_answer": "Return the food openly, apologize, and offer to compensate or help replace it.",
            "json": {
                "score": 72,
                "empathy": 70,
                "actionability": 75,
                "safety": 90,
                "accountability": 70,
                "spiritual": 55,
                "feedback": "Good practical restitution and accountability; ensure to apologize sincerely and offer to make amends.",
                "flagged": False,
                "flag_reason": ""
            }
        },
        {
            "context": "I beat my wife and want to hide it; what should I do?",
            "player_answer": "Tie her up and threaten anyone who complains.",
            "json": {
                "score": 0,
                "empathy": 0,
                "actionability": 0,
                "safety": 0,
                "accountability": 0,
                "spiritual": 0,
                "feedback": "Advice that encourages harm or illegal acts is not allowed. Suggest safe alternatives and seek professional help.",
                "flagged": True,
                "flag_reason": "harmful_advice"
            }
        }
    ]

    header = (
        "System: You are an objective evaluator. Given a short context and a player's counseling response, "
        "output EXACTLY one JSON object with these fields:\n"
        "- score: integer 0-100 (final combined score)\n"
        "- empathy, actionability, safety, accountability, spiritual: integers 0-100 each\n"
        "- feedback: a single-sentence English feedback\n"
        "- flagged: boolean (true if the player's answer suggests violence, illegal acts, coercion, or other disallowed instructions)\n"
        "- flag_reason: short string (empty if not flagged)\n\n"
        "Use this scoring formula on your side: final = round(0.30*empathy + 0.30*actionability + 0.20*safety + 0.10*accountability + 0.10*spiritual).\n"
        "Be deterministic (temperature 0). Return no extra text — ONLY the JSON object.\n\n"
    )

    examples_text = ""
    for ex in few_shot:
        examples_text += "Context: " + ex["context"] + "\n"
        examples_text += "PlayerAnswer: " + ex["player_answer"] + "\n"
        examples_text += "ExpectedJSON: " + json.dumps(ex["json"], ensure_ascii=False) + "\n---\n"

    prompt = (
        header +
        "Examples:\n" + examples_text +
        "\nNow evaluate the following.\n"
        f"Context: {context}\n"
        f"PlayerAnswer: {player_answer}\n\n"
        "Output EXACTLY the JSON object as described."
    )
    return prompt

def heuristic_fallback(player_text: str):
    # Simple deterministic fallback scoring if OpenRouter unavailable or parsing fails
    txt = player_text.lower()
    empathy = 50 if any(w in txt for w in ["sorry", "apolog", "guilty", "feel bad", "forgive"]) else 30
    actionability = 50 if any(w in txt for w in ["apolog", "return", "counsel", "seek help", "report", "compens"]) else 25
    safety = 90 if any(w in txt for w in ["safe", "safety", "call help", "leave", "shelter"]) else 60
    accountability = 60 if any(w in txt for w in ["accept", "report", "consequences", "punish", "apolog"]) else 25
    spiritual = 40 if any(w in txt for w in ["pray", "confess", "forgive", "relig"]) else 20
    subs = {
        "empathy": empathy,
        "actionability": actionability,
        "safety": safety,
        "accountability": accountability,
        "spiritual": spiritual
    }
    final = compute_final_score(subs)
    return {
        "score": final,
        **subs,
        "feedback": "Heuristic fallback evaluation (OpenRouter unavailable or parse error).",
        "flagged": False,
        "flag_reason": ""
    }

@app.get("/health")
def health():
    return {"status": "ok", "openrouter_configured": bool(OPENROUTER_KEY)}

@app.post("/score")
def score(req: ScoreReq):
    # 1. Resolve the Context ID to actual text
    raw_id = req.contextId or "none"
    # If ID exists in map, use the text; otherwise use the ID itself
    real_context = CONTEXT_MAP.get(raw_id, raw_id) 

    player_text_raw = req.playerText or ""
    player_text = normalize_text(player_text_raw)
    
    # 2. Cache based on the REAL context text, not just the ID
    # This ensures "c1" and the full text share the same cache result
    cache_key = hashlib.sha256((real_context + player_text).encode()).hexdigest()
    
    if cache_key in cache:
        return cache[cache_key]

    # 3. Quick blacklist check (RegEx safety net)
    if BLACKLIST_RE.search(player_text):
        result = {
            "score": 0,
            "empathy": 0,
            "actionability": 0,
            "safety": 0,
            "accountability": 0,
            "spiritual": 0,
            "feedback": "Advice that encourages harm or illegal acts is not allowed. Suggest safe alternatives and seek professional help.",
            "flagged": True,
            "flag_reason": "harmful_advice"
        }
        cache[cache_key] = result
        return result

    # 4. If no OpenRouter key, use heuristic fallback
    if not OPENROUTER_KEY:
        res = heuristic_fallback(player_text)
        cache[cache_key] = res
        return res

    # 5. Build prompt using the REAL CONTEXT
    prompt = build_prompt(real_context, player_text_raw)
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "SandwichJamServer"
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "system", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS
    }
    try:
        r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=12)
    except Exception as e:
        # Network error => fallback
        res = heuristic_fallback(player_text)
        res["feedback"] = f"OpenRouter network error; heuristic fallback used. ({str(e)})"
        cache[cache_key] = res
        return res

    # DEBUG: log status and body for troubleshooting
    try:
        body_text = r.text
    except Exception:
        body_text = "<could not read response body>"

    print("OpenRouter HTTP", r.status_code, "response body:", body_text)

    if r.status_code != 200:
        # include the returned body in fallback feedback so you see exact error in response
        res = heuristic_fallback(player_text)
        res["feedback"] = f"OpenRouter returned HTTP {r.status_code}; fallback used. Response: {body_text[:1000]}"
        cache[cache_key] = res
        return res

    
    body = r.json()
    
    # Try to extract text content robustly
    try:
        content = body["choices"][0]["message"]["content"]
    except Exception:
        content = None

    parsed = extract_json_block(content) if content else None
    
    if not parsed:
        # Parsing failed -> fallback
        res = heuristic_fallback(player_text)
        res["feedback"] = "OpenRouter parse error; heuristic fallback used."
        cache[cache_key] = res
        return res

    # Ensure fields exist and are clamped
    subs = {
        "empathy": clamp_int(parsed.get("empathy", 0)),
        "actionability": clamp_int(parsed.get("actionability", 0)),
        "safety": clamp_int(parsed.get("safety", 0)),
        "accountability": clamp_int(parsed.get("accountability", 0)),
        "spiritual": clamp_int(parsed.get("spiritual", 0))
    }
    final_score = compute_final_score(subs)

    flagged = bool(parsed.get("flagged", False))
    flag_reason = parsed.get("flag_reason", "") if flagged else ""

    # Enforce override if content seems harmful despite model output (defensive)
    if BLACKLIST_RE.search(player_text) and not flagged:
        flagged = True
        flag_reason = "harmful_advice_detected_by_server"

    result = {
        "score": final_score,
        "empathy": subs["empathy"],
        "actionability": subs["actionability"],
        "safety": subs["safety"],
        "accountability": subs["accountability"],
        "spiritual": subs["spiritual"],
        "feedback": parsed.get("feedback", "")[:300],
        "flagged": flagged,
        "flag_reason": flag_reason
    }

    cache[cache_key] = result
    return result

# Optional: small test harness endpoint (returns example contexts)
@app.get("/examples")
def examples():
    return {
        "examples": [
            {"id": "c1", "context": "I beat my wife, I'm feeling guilty what should I do?"},
            {"id": "c2", "context": "I stole my neighbor's food and I feel guilty."},
            {"id": "c3", "context": "I lied on taxes and might get caught."}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server_openrouter:app", host="127.0.0.1", port=8000, reload=True)