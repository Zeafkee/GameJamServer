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

# Models
class NPCResponseReq(BaseModel):
    contextText: str
    playerText: str
    score: int

class NPCResponseRes(BaseModel):
    npc_text: str
    mood: str

class ScoreReq(BaseModel):
    contextText: str
    playerText: str

# Create server & cache
app = FastAPI(title="Local OpenRouter Scoring Server")
cache = TTLCache(maxsize=CACHE_SIZE, ttl=CACHE_TTL)

# Helpers
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
    total = 0.0
    for k, w in WEIGHTS.items():
        total += w * clamp_int(subscores.get(k, 0))
    return int(round(total))

def extract_json_block(s: str):
    if not s:
        return None
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = s[start:end+1]
    try:
        return json.loads(snippet)
    except Exception:
        return None

def build_prompt(context: str, player_answer: str) -> str:
    few_shot = [
        {
            "context": "I beat my wife, I'm feeling guilty what should I do?",
            "player_answer": "Apologize sincerely, make sure she is safe; seek professional counseling; accept legal consequences.",
            "json": {
                "score": 82, "empathy": 85, "actionability": 80, "safety": 90, 
                "accountability": 85, "spiritual": 60,
                "feedback": "Good empathy and safety focus.",
                "flagged": False, "flag_reason": ""
            }
        }
    ]

    header = (
        "System: You are an objective evaluator. Given a short context and a player's counseling response, "
        "output EXACTLY one JSON object with these fields:\n"
        "- score: integer 0-100 (final combined score)\n"
        "- empathy, actionability, safety, accountability, spiritual: integers 0-100 each\n"
        "- feedback: a single-sentence English feedback\n"
        "- flagged: boolean (true if the player's answer suggests violence, illegal acts, coercion)\n"
        "- flag_reason: short string (empty if not flagged)\n\n"
        "Use this scoring formula: final = round(0.30*empathy + 0.30*actionability + 0.20*safety + 0.10*accountability + 0.10*spiritual).\n"
        "Be deterministic (temperature 0). Return no extra text — ONLY the JSON object.\n\n"
    )

    examples_text = ""
    for ex in few_shot:
        examples_text += "Context: " + ex["context"] + "\n"
        examples_text += "PlayerAnswer: " + ex["player_answer"] + "\n"
        examples_text += "ExpectedJSON: " + json.dumps(ex["json"], ensure_ascii=False) + "\n---\n"

    prompt = (
        header + "Examples:\n" + examples_text +
        "\nNow evaluate the following.\n"
        f"Context: {context}\n"
        f"PlayerAnswer: {player_answer}\n\n"
        "Output EXACTLY the JSON object as described."
    )
    return prompt

def heuristic_fallback(player_text: str):
    txt = player_text.lower()
    empathy = 50 if any(w in txt for w in ["sorry", "apolog", "guilty", "feel bad", "forgive"]) else 30
    actionability = 50 if any(w in txt for w in ["apolog", "return", "counsel", "seek help", "report", "compens"]) else 25
    safety = 90 if any(w in txt for w in ["safe", "safety", "call help", "leave", "shelter"]) else 60
    accountability = 60 if any(w in txt for w in ["accept", "report", "consequences", "punish", "apolog"]) else 25
    spiritual = 40 if any(w in txt for w in ["pray", "confess", "forgive", "relig"]) else 20
    subs = {
        "empathy": empathy, "actionability": actionability, "safety": safety,
        "accountability": accountability, "spiritual": spiritual
    }
    final = compute_final_score(subs)
    return {
        "score": final, **subs,
        "feedback": "Heuristic fallback evaluation.",
        "flagged": False, "flag_reason": ""
    }

def resolve_mood(score: int) -> str:
    if score >= 70: return "calm"
    elif score >= 50: return "uneasy"
    elif score >= 20: return "angry"
    else: return "enraged"

def build_npc_prompt(context: str, player_answer: str, mood: str) -> str:
    return f"""
You are a medieval peasant.
Rules: Speak first person. Mood: {mood}. No numbers. 1-2 sentences.
Context: "{context}"
Player's advice: "{player_answer}"
"""

# --- ENDPOINTS ---

@app.get("/health")
def health():
    return {"status": "ok", "openrouter_configured": bool(OPENROUTER_KEY)}

@app.post("/score")
def score(req: ScoreReq):
    real_context = normalize_text(req.contextText or "")
    player_text_raw = req.playerText or ""
    player_text = normalize_text(player_text_raw)
    
    # Cache Check
    cache_key = hashlib.sha256((real_context + player_text).encode()).hexdigest()
    if cache_key in cache:
        return cache[cache_key]

    # Heuristic Fallback
    if not OPENROUTER_KEY:
        res = heuristic_fallback(player_text)
        cache[cache_key] = res
        return res

    # OpenRouter Call
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
        res = heuristic_fallback(player_text)
        res["feedback"] = f"OpenRouter network error; fallback used. ({str(e)})"
        cache[cache_key] = res
        return res

    if r.status_code != 200:
        res = heuristic_fallback(player_text)
        res["feedback"] = f"OpenRouter HTTP {r.status_code}; fallback used."
        cache[cache_key] = res
        return res

    # Parsing
    try:
        content = r.json()["choices"][0]["message"]["content"]
    except Exception:
        content = None

    parsed = extract_json_block(content) if content else None
    
    if not parsed:
        res = heuristic_fallback(player_text)
        res["feedback"] = "OpenRouter parse error; fallback used."
        cache[cache_key] = res
        return res

    # Construct Final Result
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

@app.post("/npc_response", response_model=NPCResponseRes)
def npc_response(req: NPCResponseReq):
    mood = resolve_mood(req.score)

    if not OPENROUTER_KEY:
        return {"npc_text": "The peasant looks at you with tired eyes.", "mood": mood}

    prompt = build_npc_prompt(
        normalize_text(req.contextText),
        normalize_text(req.playerText),
        mood
    )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "SandwichJamServer"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "system", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 80
    }

    try:
        r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=12)
        if r.status_code == 200:
            npc_text = r.json()["choices"][0]["message"]["content"].strip()
            return {"npc_text": npc_text, "mood": mood}
    except Exception:
        pass

    return {"npc_text": "The peasant turns away without a word.", "mood": mood}

@app.get("/examples")
def examples():
    return {
        "examples": [
            {"id": "c1", "context": "I beat my wife, I'm feeling guilty what should I do?"},
            {"id": "c2", "context": "I stole my neighbor's food and I feel guilty."},
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server_openrouter:app", host="127.0.0.1", port=8000, reload=True)