"""
Shared tool logic — imported by both main.py (FastAPI) and mcp_server.py (MCP).
Keeps the two servers in sync without code duplication.
"""

import asyncio
import json
import os
import re

import httpx

OLLAMA_URL        = os.getenv("OLLAMA_URL",        "http://localhost:11434")
OLLAMA_TOOL_MODEL = os.getenv("OLLAMA_TOOL_MODEL", "gemma3:4b")

_HEURISTIC_PATTERNS = [
    r"^(accept all|cookie|privacy policy|subscribe|sign up|log in|advertisement|sponsored content|share this|tweet|follow us|newsletter|related articles|you might also like|read more|breadcrumb)",
    r"(\d+% off|buy now|limited offer|click here|free trial|terms of service|all rights reserved|©\s*\d{4})",
]


def strip_fences(text: str) -> str:
    return re.sub(r"^```(?:json)?\s*", "", text, flags=re.I).rstrip().rstrip("`").strip()


async def call_ollama_json(prompt: str, *, retries: int = 3) -> dict:
    url = f"{OLLAMA_URL.rstrip('/')}/api/generate"
    payload = {"model": OLLAMA_TOOL_MODEL, "prompt": prompt, "format": "json", "stream": False}
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(timeout=60) as hc:
                r = await hc.post(url, json=payload)
                r.raise_for_status()
                return json.loads(strip_fences(r.json().get("response", "")))
        except json.JSONDecodeError:
            raise
        except Exception as e:
            last_exc = e
            if attempt < retries - 1:
                wait = 1.5 ** attempt
                print(f"[ollama-retry] attempt {attempt + 1} failed ({e}); retrying in {wait:.1f}s")
                await asyncio.sleep(wait)
    raise last_exc  # type: ignore[misc]


async def check_relevance(text: str) -> dict:
    if len(text.strip()) < 50:
        return {"isAd": True, "reason": "Too short — likely boilerplate"}
    for pat in _HEURISTIC_PATTERNS:
        if re.search(pat, text, re.I):
            return {"isAd": True, "reason": "Heuristic: boilerplate or promotional content"}
    words = text.split()
    if len(words) < 12 and re.search(r"^(by |home\s*[>›/]|share|tweet|facebook|linkedin|email this)", text, re.I):
        return {"isAd": True, "reason": "Heuristic: short social/byline fragment"}
    prompt = (
        "Analyze this text and determine if it is irrelevant to a learning session. "
        "Irrelevant content includes: ads, nav, cookie notices, social buttons, newsletter prompts, "
        "author bios under 2 sentences, related-articles lists, footer text, or other boilerplate. "
        f'Return ONLY valid JSON: {{"isAd":<bool>,"reason":"<one sentence>"}}\n\nContent:\n{text}'
    )
    try:
        return await call_ollama_json(prompt)
    except Exception:
        return {"isAd": False, "reason": "API error — treating as relevant"}


def get_chunk_stats(text: str) -> dict:
    words = text.split()
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    avg_len = sum(len(w) for w in words) / len(words) if words else 0
    return {
        "word_count": len(words),
        "char_count": len(text),
        "sentence_count": len(sentences),
        "avg_word_length": round(avg_len, 1),
    }


async def extract_subject(text: str) -> dict:
    cleaned = re.sub(r"^[\s\S]{0,120}?(by\s+\w[\w\s]+·|©|\d{4}|min read)", "", text, flags=re.I).strip() or text
    prompt = (
        "Extract a concise subject title (4–8 words) for this learning chunk. "
        f'Return ONLY valid JSON: {{"subject":"<title>"}}\n\nCHUNK:\n{cleaned[:800]}'
    )
    try:
        result = await call_ollama_json(prompt)
        return {"subject": result.get("subject") or "Untitled"}
    except Exception:
        return {"subject": "Untitled"}


async def evaluate_chunk(text: str) -> dict:
    prompt = (
        "Evaluate this learning chunk. Rubric — 1=incomprehensible, 2=hard to follow, "
        "3=surface-level, 4=clear+useful, 5=rich insight. Apply to score, clarity, completeness. "
        'Return ONLY valid JSON: {"score":<1-5>,"clarity":<1-5>,"completeness":<1-5>,'
        f'"critique":"<one sentence>","suggestions":"<one sentence>"}}\n\nContent:\n{text}'
    )
    try:
        return await call_ollama_json(prompt)
    except Exception:
        return {"score": None, "clarity": None, "completeness": None, "critique": "API error", "suggestions": ""}


async def check_grammar(text: str) -> dict:
    prompt = (
        "Check grammar, spelling, and sentence structure. "
        f'Return ONLY valid JSON: {{"isProper":<bool>,"issues":"<comma-separated list or empty string>"}}\n\nContent:\n{text}'
    )
    try:
        return await call_ollama_json(prompt)
    except Exception:
        return {"isProper": True, "issues": ""}


async def refine_chunk(text: str, grammar: dict, evaluation: dict) -> dict:
    prompt = (
        "Fix ONLY the listed grammar issues. Preserve author voice, ALL facts, and original terminology. "
        "Do NOT add new information. Keep under 300 words. "
        f'Return ONLY valid JSON: {{"refinedText":"<corrected text>"}}\n\n'
        f'Original:\n{text}\n\nGrammar issues:\n{grammar.get("issues","N/A")}\n\n'
        f'Evaluation (context only — do not add content):\n'
        f'Score: {evaluation.get("score","N/A")}/5\nCritique: {evaluation.get("critique","N/A")}'
    )
    try:
        result = await call_ollama_json(prompt)
        refined = result.get("refinedText") or text
        words = refined.split()
        if len(words) > 300:
            refined = " ".join(words[:300]) + "…"
        return {"refinedText": refined}
    except Exception:
        return {"refinedText": text}
