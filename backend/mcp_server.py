"""
FastMCP server — exposes the TypingFlow analysis tools as MCP tools.

Run:
  uv run fastmcp run mcp_server.py
or mount alongside the FastAPI app for stdio transport.
"""

import asyncio
import json
import os
import re

import httpx
from dotenv import load_dotenv
from fastmcp import FastMCP
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv()

_api_key = os.getenv("GEMINI_API_KEY")
if not _api_key:
    raise RuntimeError("GEMINI_API_KEY is not set — add it to backend/.env")

client = genai.Client(api_key=_api_key)

STRUCTURE_MODEL   = os.getenv("STRUCTURE_MODEL",   "gemini-3.1-flash-lite")
OLLAMA_URL        = os.getenv("OLLAMA_URL",        "http://localhost:11434")
OLLAMA_TOOL_MODEL = os.getenv("OLLAMA_TOOL_MODEL", "gemma3:4b")

mcp = FastMCP("TypingFlow Tools")

# ── Shared helpers (duplicated from main.py to keep servers independent) ───────

_HEURISTIC_PATTERNS = [
    r"^(accept all|cookie|privacy policy|subscribe|sign up|log in|advertisement|sponsored content|share this|tweet|follow us|newsletter|related articles|you might also like|read more|breadcrumb)",
    r"(\d+% off|buy now|limited offer|click here|free trial|terms of service|all rights reserved|©\s*\d{4})",
]


def _strip_fences(text: str) -> str:
    return re.sub(r"^```(?:json)?\s*", "", text, flags=re.I).rstrip().rstrip("`").strip()


async def _call_ollama_json(prompt: str, *, retries: int = 3) -> dict:
    url = f"{OLLAMA_URL.rstrip('/')}/api/generate"
    payload = {"model": OLLAMA_TOOL_MODEL, "prompt": prompt, "format": "json", "stream": False}
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(timeout=60) as hc:
                r = await hc.post(url, json=payload)
                r.raise_for_status()
                return json.loads(_strip_fences(r.json().get("response", "")))
        except json.JSONDecodeError:
            raise
        except Exception as e:
            last_exc = e
            if attempt < retries - 1:
                await asyncio.sleep(1.5 ** attempt)
    raise last_exc  # type: ignore[misc]


async def _call_gemini_json(prompt: str, *, retries: int = 3) -> dict:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            response = await client.aio.models.generate_content(
                model=STRUCTURE_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            return json.loads(_strip_fences(response.text))
        except json.JSONDecodeError:
            raise
        except Exception as e:
            last_exc = e
            if attempt < retries - 1:
                await asyncio.sleep(1.5 ** attempt)
    raise last_exc  # type: ignore[misc]


# ── MCP Tool input schemas ─────────────────────────────────────────────────────

class RelevanceInput(BaseModel):
    text: str

class SubjectInput(BaseModel):
    text: str

class EvaluateInput(BaseModel):
    text: str

class GrammarInput(BaseModel):
    text: str

class RefineInput(BaseModel):
    text: str
    grammar_issues: str = ""
    evaluation_score: int | None = None
    evaluation_critique: str = ""

class StatsInput(BaseModel):
    text: str

class StructureInput(BaseModel):
    content_json: str  # JSON-encoded list of ContentItem dicts
    system_prompt: str = ""


# ── MCP Tools ─────────────────────────────────────────────────────────────────

@mcp.tool()
async def check_relevance(text: str) -> dict:
    """
    Determine whether a text chunk is boilerplate/ad content or real learning material.
    Returns {isAd: bool, reason: str}.
    """
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
        return await _call_ollama_json(prompt)
    except Exception:
        return {"isAd": False, "reason": "API error — treating as relevant"}


@mcp.tool()
async def extract_subject(text: str) -> dict:
    """
    Extract a concise 4–8 word subject title for a learning chunk.
    Returns {subject: str}.
    """
    cleaned = re.sub(r"^[\s\S]{0,120}?(by\s+\w[\w\s]+·|©|\d{4}|min read)", "", text, flags=re.I).strip() or text
    prompt = (
        "Extract a concise subject title (4–8 words) for this learning chunk. "
        f'Return ONLY valid JSON: {{"subject":"<title>"}}\n\nCHUNK:\n{cleaned[:800]}'
    )
    try:
        result = await _call_ollama_json(prompt)
        return {"subject": result.get("subject") or "Untitled"}
    except Exception:
        return {"subject": "Untitled"}


@mcp.tool()
async def evaluate_chunk(text: str) -> dict:
    """
    Evaluate a learning chunk on score, clarity, and completeness (each 1–5).
    Returns {score, clarity, completeness, critique, suggestions}.
    """
    prompt = (
        "Evaluate this learning chunk. Rubric — 1=incomprehensible, 2=hard to follow, "
        "3=surface-level, 4=clear+useful, 5=rich insight. Apply to score, clarity, completeness. "
        'Return ONLY valid JSON: {"score":<1-5>,"clarity":<1-5>,"completeness":<1-5>,'
        f'"critique":"<one sentence>","suggestions":"<one sentence>"}}\n\nContent:\n{text}'
    )
    try:
        return await _call_ollama_json(prompt)
    except Exception:
        return {"score": None, "clarity": None, "completeness": None, "critique": "API error", "suggestions": ""}


@mcp.tool()
async def check_grammar(text: str) -> dict:
    """
    Check grammar, spelling, and sentence structure of a text chunk.
    Returns {isProper: bool, issues: str}.
    """
    prompt = (
        "Check grammar, spelling, and sentence structure. "
        f'Return ONLY valid JSON: {{"isProper":<bool>,"issues":"<comma-separated list or empty string>"}}\n\nContent:\n{text}'
    )
    try:
        return await _call_ollama_json(prompt)
    except Exception:
        return {"isProper": True, "issues": ""}


@mcp.tool()
async def refine_chunk(text: str, grammar_issues: str = "", evaluation_score: int | None = None, evaluation_critique: str = "") -> dict:
    """
    Fix grammar issues in a chunk while preserving the author's voice and facts.
    Returns {refinedText: str}.
    """
    prompt = (
        "Fix ONLY the listed grammar issues. Preserve author voice, ALL facts, and original terminology. "
        "Do NOT add new information. Keep under 300 words. "
        f'Return ONLY valid JSON: {{"refinedText":"<corrected text>"}}\n\n'
        f"Original:\n{text}\n\nGrammar issues:\n{grammar_issues or 'N/A'}\n\n"
        f"Evaluation (context only — do not add content):\n"
        f"Score: {evaluation_score or 'N/A'}/5\nCritique: {evaluation_critique or 'N/A'}"
    )
    try:
        result = await _call_ollama_json(prompt)
        refined = result.get("refinedText") or text
        words = refined.split()
        if len(words) > 300:
            refined = " ".join(words[:300]) + "…"
        return {"refinedText": refined}
    except Exception:
        return {"refinedText": text}


@mcp.tool()
def get_chunk_stats(text: str) -> dict:
    """
    Compute word count, char count, sentence count, and avg word length for a text chunk.
    Returns {word_count, char_count, sentence_count, avg_word_length}.
    """
    words = text.split()
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    avg_len = sum(len(w) for w in words) / len(words) if words else 0
    return {
        "word_count": len(words),
        "char_count": len(text),
        "sentence_count": len(sentences),
        "avg_word_length": round(avg_len, 1),
    }


@mcp.tool()
async def structure_content(content_json: str, system_prompt: str = "") -> dict:
    """
    Structure raw scraped content (JSON array of {type, content, src} items) into
    a learning payload with tldr, tags, nuggets, star_rating, and coverage_pct.
    """
    from main import SYSTEM_PROMPT as DEFAULT_SYSTEM_PROMPT  # noqa: PLC0415
    sp = system_prompt or DEFAULT_SYSTEM_PROMPT
    prompt = sp + "\n\nRAW SCRAPED CONTENT:\n" + content_json
    return await _call_gemini_json(prompt)


if __name__ == "__main__":
    mcp.run()
