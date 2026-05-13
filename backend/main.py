import asyncio
import base64
import json
import os
import re
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv()

_api_key = os.getenv("GEMINI_API_KEY")
if not _api_key:
    raise RuntimeError("GEMINI_API_KEY is not set — add it to backend/.env")

client = genai.Client(api_key=_api_key)

STRUCTURE_MODEL = os.getenv("STRUCTURE_MODEL", "gemini-3.1-flash-lite")
TOOL_MODEL      = os.getenv("TOOL_MODEL",      "gemini-2.5-flash-lite")
IMAGE_MODEL     = os.getenv("IMAGE_MODEL",     "gemini-2.0-flash-preview-image-generation")

CONCURRENCY_LIMIT    = int(os.getenv("CONCURRENCY_LIMIT", "5"))
MULTIPASS_THRESHOLD  = int(os.getenv("MULTIPASS_WORD_THRESHOLD", "3000"))
SECTION_MAX_WORDS    = int(os.getenv("SECTION_MAX_WORDS", "2000"))

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

class ContentItem(BaseModel):
    type: str
    content: str | None = None
    src: str | None = None

class StructureRequest(BaseModel):
    payload: list[ContentItem]

class ToolRequest(BaseModel):
    prompt: str

class ImageRequest(BaseModel):
    text: str
    tags: list[str] = []

class ProcessChunksRequest(BaseModel):
    nuggets: list[dict[str, Any]]
    image_index: list[dict[str, Any]] = []
    tags: list[str] = []

# ── Gemini helpers ─────────────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    return re.sub(r"^```(?:json)?\s*", "", text, flags=re.I).rstrip().rstrip("`").strip()

async def _call_json_model(model_id: str, prompt: str) -> dict:
    response = await client.aio.models.generate_content(
        model=model_id,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        ),
    )
    return json.loads(_strip_fences(response.text))

async def call_structure_model(prompt: str) -> dict:
    return await _call_json_model(STRUCTURE_MODEL, prompt)

async def call_tool_model(prompt: str) -> dict:
    return await _call_json_model(TOOL_MODEL, prompt)

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
    return {
        "tldr": outputs[0].get("tldr", "") if outputs else "",
        "tags": all_tags,
        "star_rating": avg_rating,
        "coverage_pct": coverage,
        "nuggets": all_nuggets,
    }

# ── Tool implementations ───────────────────────────────────────────────────────

_HEURISTIC_PATTERNS = [
    r"^(accept all|cookie|privacy policy|subscribe|sign up|log in|advertisement|sponsored content|share this|tweet|follow us|newsletter|related articles|you might also like|read more|breadcrumb)",
    r"(\d+% off|buy now|limited offer|click here|free trial|terms of service|all rights reserved|©\s*\d{4})",
]

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
        f'Analyze this text and determine if it is irrelevant to a learning session. '
        f'Irrelevant content includes: ads, nav, cookie notices, social buttons, newsletter prompts, '
        f'author bios under 2 sentences, related-articles lists, footer text, or other boilerplate. '
        f'Return ONLY valid JSON: {{"isAd":<bool>,"reason":"<one sentence>"}}\n\nContent:\n{text}'
    )
    try:
        return await call_tool_model(prompt)
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
        f'Extract a concise subject title (4–8 words) for this learning chunk. '
        f'Return ONLY valid JSON: {{"subject":"<title>"}}\n\nCHUNK:\n{cleaned[:800]}'
    )
    try:
        result = await call_tool_model(prompt)
        return {"subject": result.get("subject") or "Untitled"}
    except Exception:
        return {"subject": "Untitled"}

async def evaluate_chunk(text: str) -> dict:
    prompt = (
        f'Evaluate this learning chunk. Rubric — 1=incomprehensible, 2=hard to follow, '
        f'3=surface-level, 4=clear+useful, 5=rich insight. Apply to score, clarity, completeness. '
        f'Return ONLY valid JSON: {{"score":<1-5>,"clarity":<1-5>,"completeness":<1-5>,'
        f'"critique":"<one sentence>","suggestions":"<one sentence>"}}\n\nContent:\n{text}'
    )
    try:
        return await call_tool_model(prompt)
    except Exception:
        return {"score": None, "clarity": None, "completeness": None, "critique": "API error", "suggestions": ""}

async def check_grammar(text: str) -> dict:
    prompt = (
        f'Check grammar, spelling, and sentence structure. '
        f'Return ONLY valid JSON: {{"isProper":<bool>,"issues":"<comma-separated list or empty string>"}}\n\nContent:\n{text}'
    )
    try:
        return await call_tool_model(prompt)
    except Exception:
        return {"isProper": True, "issues": ""}

async def refine_chunk(text: str, grammar: dict, evaluation: dict) -> dict:
    prompt = (
        f'Fix ONLY the listed grammar issues. Preserve author voice, ALL facts, and original terminology. '
        f'Do NOT add new information. Keep under 300 words. '
        f'Return ONLY valid JSON: {{"refinedText":"<corrected text>"}}\n\n'
        f'Original:\n{text}\n\nGrammar issues:\n{grammar.get("issues","N/A")}\n\n'
        f'Evaluation (context only — do not add content):\n'
        f'Score: {evaluation.get("score","N/A")}/5\nCritique: {evaluation.get("critique","N/A")}'
    )
    try:
        result = await call_tool_model(prompt)
        refined = result.get("refinedText") or text
        words = refined.split()
        if len(words) > 300:
            refined = " ".join(words[:300]) + "…"
        return {"refinedText": refined}
    except Exception:
        return {"refinedText": text}

async def generate_image_for_chunk(text: str, tags: list[str]) -> str | None:
    prompt = (
        f"Create a visually stunning, minimal abstract representation for this learning concept. "
        f"Context: {text[:400]} Tags: {', '.join(tags)}"
    )
    img = await call_image_model(prompt)
    return img or await picsum_fallback(tags)

# ── Single chunk processing ────────────────────────────────────────────────────

async def process_one_chunk(
    nugget: dict, idx: int, total: int, image_index: list[dict], global_tags: list[str]
) -> dict:
    text = nugget.get("text") or ""
    tags = nugget.get("tags") or global_tags or []
    content_type = nugget.get("content_type")

    relevance = await check_relevance(text)
    if relevance.get("isAd"):
        return {"is_ad": True, "text": text}

    stats_coro     = asyncio.get_event_loop().run_in_executor(None, get_chunk_stats, text)
    subject_coro   = extract_subject(text)
    eval_coro      = evaluate_chunk(text)
    grammar_coro   = check_grammar(text)

    stats, subject, evaluation, grammar = await asyncio.gather(
        stats_coro, subject_coro, eval_coro, grammar_coro,
        return_exceptions=True,
    )

    refined_text = text
    if not isinstance(grammar, Exception) and not grammar.get("isProper", True):
        refined = await refine_chunk(text, grammar, evaluation if not isinstance(evaluation, Exception) else {})
        refined_text = refined.get("refinedText", text)

    img_src = nugget.get("img_src") or None
    if not img_src:
        img_src = await generate_image_for_chunk(text, tags)

    return {
        "is_ad": False,
        "text": text,
        "refined_text": refined_text,
        "img_src": img_src,
        "tags": tags,
        "subject": subject.get("subject", "Untitled") if not isinstance(subject, Exception) else "Untitled",
        "stats": stats if not isinstance(stats, Exception) else None,
        "evaluation": evaluation if not isinstance(evaluation, Exception) else None,
        "content_type": content_type,
    }

# ── API Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "models": {"structure": STRUCTURE_MODEL, "tool": TOOL_MODEL, "image": IMAGE_MODEL}}

@app.post("/api/structure")
async def structure_content(req: StructureRequest):
    payload = [item.model_dump() for item in req.payload]
    source_words = _count_words(payload)

    try:
        if source_words > MULTIPASS_THRESHOLD:
            sections = split_payload(payload)
            results = []
            for section in sections:
                prompt = SYSTEM_PROMPT + "\n\nRAW SCRAPED CONTENT:\n" + json.dumps(section)
                result = await call_structure_model(prompt)
                results.append(result)
            return merge_sections(results, source_words)
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
        return await call_tool_model(req.prompt)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"Model returned invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process-chunks")
async def process_chunks(req: ProcessChunksRequest):
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async def bounded(nugget: dict, idx: int) -> dict:
        async with semaphore:
            return await process_one_chunk(nugget, idx, len(req.nuggets), req.image_index, req.tags)

    tasks = [bounded(nugget, i) for i, nugget in enumerate(req.nuggets)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    valid_nuggets = []
    for r in results:
        if isinstance(r, Exception):
            print(f"[chunks] chunk error: {r}")
            continue
        if r.get("is_ad"):
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
