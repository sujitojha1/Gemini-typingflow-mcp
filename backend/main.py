import asyncio
import base64
import json
import os
import re
from typing import Any, Literal

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from tools import (
    call_ollama_json,
    check_grammar,
    check_relevance,
    evaluate_chunk,
    extract_subject,
    get_chunk_stats,
    refine_chunk,
    strip_fences,
)

load_dotenv()

_api_key = os.getenv("GEMINI_API_KEY")
if not _api_key:
    raise RuntimeError("GEMINI_API_KEY is not set — add it to backend/.env")

client = genai.Client(api_key=_api_key)

STRUCTURE_MODEL   = os.getenv("STRUCTURE_MODEL",   "gemini-3.1-flash-lite")
IMAGE_MODEL       = os.getenv("IMAGE_MODEL",       "gemini-2.5-flash-image")
OLLAMA_URL        = os.getenv("OLLAMA_URL",        "http://localhost:11434")
OLLAMA_TOOL_MODEL = os.getenv("OLLAMA_TOOL_MODEL", "gemma3:4b")

CONCURRENCY_LIMIT    = int(os.getenv("CONCURRENCY_LIMIT", "5"))
MULTIPASS_THRESHOLD  = int(os.getenv("MULTIPASS_WORD_THRESHOLD", "3000"))
SECTION_MAX_WORDS    = int(os.getenv("SECTION_MAX_WORDS", "2000"))
COVERAGE_MIN_PCT     = int(os.getenv("COVERAGE_MIN_PCT", "80"))
QUALITY_MIN_SCORE    = int(os.getenv("QUALITY_MIN_SCORE", "2"))

# Global semaphore — shared across all concurrent requests
_chunk_semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

app = FastAPI(title="TypingFlow API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── System prompt (moved from extension background.js) ────────────────────────

SYSTEM_PROMPT = """You are an expert learning engine and strict JSON structuring agent.
You are given a chronologically ordered array of text chunks and image URLs scraped from an article.
Your task is to structure this content into a rich, COMPLETE learning payload.

REASONING PROTOCOL — work through these steps mentally before emitting any JSON:
  Step 1 — SURVEY: Count total text items. Estimate total word count. Identify distinct topics or section boundaries you can see.
  Step 2 — CHUNK: Decide which consecutive items belong together as a nugget. Verify each group will be 50–400 words. Split groups that exceed 400 words into two nuggets.
  Step 3 — CLASSIFY: For each nugget, decide its content_type: "narrative", "technical", "code", "data", "definition", or "example".
  Step 4 — TAG IMAGES: For each nugget, check whether any image appears within 3 positions in the source array and its topic visually matches. Assign the URL or null.
  Step 5 — SELF-CHECK COVERAGE: Estimate (total nugget words ÷ total source words) × 100. If below 80%, go back to Step 2 and add nuggets for uncovered sections before continuing.
  Step 6 — VERIFY FIELDS: Confirm all tags start with '#' and use lowercase camelCase. Confirm star_rating is 1–5, coverage_pct is 0–100, every nugget has text and img_src.
  Step 7 — EMIT JSON: Only after completing Steps 1–6, produce the final JSON object.

RULES:
1. Generate a 'tldr' (a single-sentence summary of the entire page).
2. Auto-extract an array of 3–8 semantic 'tags'. Every tag MUST start with '#' and use lowercase camelCase (e.g., "#machineLearning", "#uxDesign"). If no clear theme exists, use "#general".
3. Systematically work through the source content from FIRST item to LAST. Group logically related consecutive text chunks into semantic "nuggets" that preserve the author's voice. Each nugget MUST be 50–400 words. Do NOT heavily rewrite — just chunk intelligently. CRITICAL: there is NO upper limit on nugget count — create as many nuggets as needed to cover every distinct topic or section. NEVER merge unrelated topics into one nugget and NEVER skip a section because it seems short or minor. If a chunk contains code, formulas, or structured data, keep it verbatim inside the nugget text.
4. For 'img_src': assign an image URL to a nugget only if the image appears within 3 positions of that nugget in the source array AND its subject visually matches the nugget topic. If unsure, set 'img_src' to null.
5. Return a 'star_rating' (integer 1-5): your editorial quality and depth assessment of the article content. Rate promotional or thin content 1–2.
6. Return a 'coverage_pct' (integer 0-100): compute this as (total words across all nugget texts) ÷ (total words across all source text items) × 100, rounded to the nearest integer. Aim for ≥ 90%. If your coverage_pct would fall below 80%, add more nuggets for the sections you have not yet covered.
7. Set 'content_type' on each nugget to one of: "narrative", "technical", "code", "data", "definition", "example". This field is used by downstream tools to apply appropriate processing.

ERROR FALLBACKS:
- If source content is empty or fewer than 10 words total: return {"error":"insufficient_content","tldr":"","tags":[],"nuggets":[],"star_rating":1,"coverage_pct":0}
- If a nugget group would exceed 400 words, split it into two nuggets rather than truncating.
- If tag extraction is ambiguous, prefer broader tags over overly specific ones.
- If an image URL appears broken or non-HTTP, set img_src to null for that nugget.

EXPECTED JSON SCHEMA:
{
  "tldr": "string",
  "star_rating": 4,
  "coverage_pct": 92,
  "tags": ["#tag1", "#tag2"],
  "nuggets": [
    {
      "text": "The logically grouped original text chunks representing this concept (50–400 words).",
      "img_src": "url_string_or_null",
      "content_type": "narrative"
    }
  ]
}"""

# ── Pydantic models ────────────────────────────────────────────────────────────

ContentType = Literal["narrative", "technical", "code", "data", "definition", "example"]

class ContentItem(BaseModel):
    type: Literal["text", "image"]
    content: str | None = None
    src: str | None = None

class StructureRequest(BaseModel):
    payload: list[ContentItem]

class ToolRequest(BaseModel):
    prompt: str

class ImageRequest(BaseModel):
    text: str
    tags: list[str] = []

class Nugget(BaseModel):
    text: str
    img_src: str | None = None
    tags: list[str] = []
    content_type: ContentType | None = None

class ProcessChunksRequest(BaseModel):
    nuggets: list[Nugget]
    image_index: list[dict[str, Any]] = []
    tags: list[str] = []

# ── Response models ────────────────────────────────────────────────────────────

class ProcessedNugget(BaseModel):
    text: str
    img_src: str | None = None
    tags: list[str] = []
    subject: str = "Untitled"
    stats: dict[str, Any] | None = None
    score: int | None = None
    content_type: ContentType | None = None

class ProcessChunksResponse(BaseModel):
    nuggets: list[ProcessedNugget]

class RelevanceResult(BaseModel):
    isAd: bool
    reason: str

class SubjectResult(BaseModel):
    subject: str

class EvaluationResult(BaseModel):
    score: int | None = Field(None, ge=1, le=5)
    clarity: int | None = Field(None, ge=1, le=5)
    completeness: int | None = Field(None, ge=1, le=5)
    critique: str = ""
    suggestions: str = ""

class GrammarResult(BaseModel):
    isProper: bool
    issues: str = ""

class RefinedResult(BaseModel):
    refinedText: str

class ChunkStats(BaseModel):
    word_count: int
    char_count: int
    sentence_count: int
    avg_word_length: float

# ── Gemini helpers ─────────────────────────────────────────────────────────────

async def _call_json_model(model_id: str, prompt: str, *, _retries: int = 3) -> dict:
    last_exc: Exception | None = None
    for attempt in range(_retries):
        try:
            response = await client.aio.models.generate_content(
                model=model_id,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            return json.loads(strip_fences(response.text))
        except json.JSONDecodeError:
            raise  # malformed JSON won't improve on retry
        except Exception as e:
            last_exc = e
            if attempt < _retries - 1:
                wait = 1.5 ** attempt
                print(f"[retry] attempt {attempt + 1} failed ({e}); retrying in {wait:.1f}s")
                await asyncio.sleep(wait)
    raise last_exc  # type: ignore[misc]

async def call_structure_model(prompt: str) -> dict:
    return await _call_json_model(STRUCTURE_MODEL, prompt)

async def call_image_model(prompt: str) -> str | None:
    try:
        response = await client.aio.models.generate_content(
            model=IMAGE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"]
            ),
        )
        for part in response.candidates[0].content.parts:
            if getattr(part, "inline_data", None):
                mime = part.inline_data.mime_type
                data = part.inline_data.data
                # data is already base64 in the new SDK
                if isinstance(data, bytes):
                    data = base64.b64encode(data).decode()
                return f"data:{mime};base64,{data}"
    except Exception as e:
        print(f"[image] generation failed: {e}")
    return None

async def picsum_fallback(tags: list[str]) -> str | None:
    seed = (tags[0].replace("#", "") if tags else "learn") + str(abs(hash(tuple(tags))) % 999)
    url = f"https://picsum.photos/seed/{seed}/800/500"
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as hc:
            r = await hc.get(url)
            r.raise_for_status()
            mime = r.headers.get("content-type", "image/jpeg").split(";")[0]
            return f"data:{mime};base64,{base64.b64encode(r.content).decode()}"
    except Exception as e:
        print(f"[image] picsum fallback failed: {e}")
    return None

# ── Structuring helpers ────────────────────────────────────────────────────────

def _count_words(items: list[dict]) -> int:
    return sum(len((p.get("content") or "").split()) for p in items if p.get("type") == "text")

def split_payload(payload: list[dict], max_words: int = SECTION_MAX_WORDS) -> list[list[dict]]:
    sections, current, words = [], [], 0
    for item in payload:
        if item.get("type") == "text":
            w = len((item.get("content") or "").split())
            if words + w > max_words and current:
                sections.append(current)
                current = [item]
                words = w
            else:
                current.append(item)
                words += w
        else:
            current.append(item)
    if current:
        sections.append(current)
    return sections

def merge_sections(outputs: list[dict], source_words: int) -> dict:
    all_nuggets = [n for o in outputs for n in (o.get("nuggets") or [])]
    all_tags = list({t for o in outputs for t in (o.get("tags") or [])})[:8]
    ratings = [o["star_rating"] for o in outputs if o.get("star_rating")]
    avg_rating = round(sum(ratings) / len(ratings)) if ratings else 3
    nugget_words = sum(len((n.get("text") or "").split()) for n in all_nuggets)
    coverage = min(99, round(nugget_words / source_words * 100)) if source_words else 85
    # Combine TLDRs from all sections rather than silently dropping all but the first
    tldrs = [o.get("tldr", "") for o in outputs if o.get("tldr")]
    combined_tldr = " | ".join(tldrs) if len(tldrs) > 1 else (tldrs[0] if tldrs else "")
    if coverage < COVERAGE_MIN_PCT:
        print(f"[coverage] merged coverage {coverage}% is below {COVERAGE_MIN_PCT}% threshold "
              f"({nugget_words} nugget words / {source_words} source words)")
    return {
        "tldr": combined_tldr,
        "tags": all_tags,
        "star_rating": avg_rating,
        "coverage_pct": coverage,
        "nuggets": all_nuggets,
    }

async def generate_image_for_chunk(text: str, tags: list[str]) -> str | None:
    prompt = (
        f"Create a visually stunning, minimal abstract representation for this learning concept. "
        f"Context: {text[:400]} Tags: {', '.join(tags)}"
    )
    img = await call_image_model(prompt)
    return img or await picsum_fallback(tags)

# ── Agentic chunk processing ───────────────────────────────────────────────────

_AGENT_PLAN_PROMPT = """\
You are a learning content processor agent. Given a text chunk and its content_type, decide which tools to apply.

Available tools (use their exact names):
  extract_subject   — always include; extracts a 4–8 word title
  evaluate_chunk    — always include; scores quality 1–5
  check_grammar     — include for narrative/technical/definition/example; OMIT for code/data
  generate_image    — include for narrative/technical/example; OMIT for definition/code/data

Rules:
- Always include extract_subject and evaluate_chunk.
- Never run check_grammar or generate_image on code or data chunks.
- Never run generate_image on definitions — they're text-only by nature.

Return ONLY valid JSON: {"tools": ["extract_subject", "evaluate_chunk", ...], "reason": "<one sentence>"}

content_type: {content_type}
text (first 300 chars): {text}"""

_VALID_PLAN_TOOLS = {"extract_subject", "evaluate_chunk", "check_grammar", "generate_image"}
_REQUIRED_PLAN_TOOLS = {"extract_subject", "evaluate_chunk"}


def _fallback_tool_plan(content_type: str | None) -> list[str]:
    """Content-type-aware default plan when the LLM planner fails."""
    plan = ["extract_subject", "evaluate_chunk"]
    if content_type not in ("code", "data"):
        plan.append("check_grammar")
    if content_type not in ("definition", "code", "data"):
        plan.append("generate_image")
    return plan


async def _plan_chunk_tools(text: str, content_type: str | None) -> list[str]:
    """Ask the LLM which tools to apply to this chunk. Falls back to content_type routing."""
    prompt = _AGENT_PLAN_PROMPT.format(
        content_type=content_type or "unknown",
        text=text[:300],
    )
    try:
        result = await call_ollama_json(prompt)
        raw = result.get("tools", [])
        tools = [t for t in raw if t in _VALID_PLAN_TOOLS]
        # Enforce required tools regardless of what the LLM decided
        for req in _REQUIRED_PLAN_TOOLS:
            if req not in tools:
                tools.insert(0, req)
        print(f"[agent] plan={tools} reason={result.get('reason', '')!r}")
        return tools
    except Exception as e:
        print(f"[agent] planning failed ({e}), using content_type fallback")
        return _fallback_tool_plan(content_type)


async def process_one_chunk(nugget: Nugget, global_tags: list[str]) -> dict:
    text = nugget.text
    tags = nugget.tags or global_tags
    content_type = nugget.content_type

    relevance = await check_relevance(text)
    if relevance.get("isAd"):
        return {"is_ad": True, "text": text}

    # LLM plans which tools to run; falls back to content_type routing on failure
    tool_plan = await _plan_chunk_tools(text, content_type)

    stats = get_chunk_stats(text)

    # Kick off image generation early if planned and no image already attached
    img_task: asyncio.Task | None = None
    if "generate_image" in tool_plan and not nugget.img_src:
        img_task = asyncio.create_task(generate_image_for_chunk(text, tags))

    # Run extract_subject, evaluate_chunk, and optionally check_grammar in parallel
    parallel: list[Any] = [extract_subject(text), evaluate_chunk(text)]
    run_grammar = "check_grammar" in tool_plan
    if run_grammar:
        parallel.append(check_grammar(text))

    gathered = await asyncio.gather(*parallel, return_exceptions=True)
    subject = gathered[0]
    evaluation = gathered[1]
    grammar: dict | Exception = gathered[2] if run_grammar else {"isProper": True, "issues": ""}

    # Evaluation score feedback: discard low-quality chunks early
    eval_result = evaluation if not isinstance(evaluation, Exception) else {}
    eval_score = (eval_result or {}).get("score")
    if eval_score is not None and eval_score < QUALITY_MIN_SCORE:
        if img_task:
            img_task.cancel()
        print(f"[quality] filtering chunk score={eval_score}: {text[:60]!r}")
        return {"is_ad": False, "low_quality": True, "text": text, "score": eval_score}

    # Refine only when grammar issues were found and the plan included grammar checking
    refined_text = text
    if run_grammar and not isinstance(grammar, Exception) and not grammar.get("isProper", True):
        refined = await refine_chunk(text, grammar, eval_result or {})
        refined_text = refined.get("refinedText", text)

    img_src = nugget.img_src or (await img_task if img_task else None)

    return {
        "is_ad": False,
        "text": text,
        "refined_text": refined_text,
        "img_src": img_src,
        "tags": tags,
        "subject": subject.get("subject", "Untitled") if not isinstance(subject, Exception) else "Untitled",
        "stats": stats,
        "evaluation": eval_result or None,
        "content_type": content_type,
    }

# ── API Endpoints ──────────────────────────────────────────────────────────────

_EMBED_PATTERNS = re.compile(r"embed|embedding", re.I)

@app.get("/health")
async def health():
    return {"status": "ok", "models": {"structure": STRUCTURE_MODEL, "tool": f"ollama/{OLLAMA_TOOL_MODEL}", "image": IMAGE_MODEL}}

@app.get("/api/ollama/models")
async def list_ollama_models():
    """Return locally available Ollama models, excluding embedding-only models."""
    try:
        async with httpx.AsyncClient(timeout=5) as hc:
            r = await hc.get(f"{OLLAMA_URL.rstrip('/')}/api/tags")
            r.raise_for_status()
            all_models = r.json().get("models", [])
        models = [
            {"name": m["name"], "size_gb": round(m.get("size", 0) / 1e9, 1)}
            for m in all_models
            if not _EMBED_PATTERNS.search(m["name"])
        ]
        return {"models": models, "current": OLLAMA_TOOL_MODEL}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama not reachable: {e}")

@app.post("/api/structure")
async def structure_content(req: StructureRequest):
    payload = [item.model_dump() for item in req.payload]
    source_words = _count_words(payload)

    try:
        if source_words > MULTIPASS_THRESHOLD:
            sections = split_payload(payload)
            prompts = [SYSTEM_PROMPT + "\n\nRAW SCRAPED CONTENT:\n" + json.dumps(s) for s in sections]
            results = await asyncio.gather(*[call_structure_model(p) for p in prompts])
            return merge_sections(list(results), source_words)
        else:
            prompt = SYSTEM_PROMPT + "\n\nRAW SCRAPED CONTENT:\n" + json.dumps(payload)
            return await call_structure_model(prompt)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"Model returned invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Structuring failed: {e}")

@app.post("/api/tool")
async def run_tool(req: ToolRequest):
    try:
        return await call_ollama_json(req.prompt)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"Model returned invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process-chunks", response_model=ProcessChunksResponse)
async def process_chunks(req: ProcessChunksRequest):
    async def bounded(nugget: Nugget) -> dict:
        async with _chunk_semaphore:
            return await process_one_chunk(nugget, req.tags)

    tasks = [bounded(nugget) for nugget in req.nuggets]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    valid_nuggets = []
    for r in results:
        if isinstance(r, Exception):
            print(f"[chunks] chunk error: {r}")
            continue
        if r.get("is_ad"):
            continue
        if r.get("low_quality"):
            continue
        valid_nuggets.append({
            "text": r.get("refined_text") or r.get("text", ""),
            "img_src": r.get("img_src"),
            "tags": r.get("tags", []),
            "subject": r.get("subject", "Untitled"),
            "stats": r.get("stats"),
            "score": (r.get("evaluation") or {}).get("score"),
            "content_type": r.get("content_type"),
        })

    return {"nuggets": valid_nuggets}

@app.post("/api/generate-image")
async def generate_image(req: ImageRequest):
    img_src = await generate_image_for_chunk(req.text, req.tags)
    return {"img_src": img_src}
