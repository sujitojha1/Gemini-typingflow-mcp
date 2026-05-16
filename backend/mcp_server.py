"""
FastMCP server — exposes the TypingFlow analysis tools as MCP tools.

Run:
  uv run fastmcp run mcp_server.py
or mount alongside the FastAPI app for stdio transport.
"""

import json
import os

from dotenv import load_dotenv
from fastmcp import FastMCP
from google import genai
from google.genai import types
from tools import (
    check_grammar,
    check_relevance,
    evaluate_chunk,
    extract_subject,
    get_chunk_stats,
    refine_chunk,
    strip_fences as _strip_fences,
)

load_dotenv()

_api_key = os.getenv("GEMINI_API_KEY")
if not _api_key:
    raise RuntimeError("GEMINI_API_KEY is not set — add it to backend/.env")

client = genai.Client(api_key=_api_key)

STRUCTURE_MODEL = os.getenv("STRUCTURE_MODEL", "gemini-3.1-flash-lite")

mcp = FastMCP("TypingFlow Tools")


async def _call_gemini_json(prompt: str, *, retries: int = 3) -> dict:
    import asyncio
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


# ── MCP Tools — thin wrappers that expose shared tools via the MCP protocol ────

@mcp.tool()
async def mcp_check_relevance(text: str) -> dict:
    """Determine whether a text chunk is boilerplate/ad content or real learning material. Returns {isAd, reason}."""
    return await check_relevance(text)


@mcp.tool()
async def mcp_extract_subject(text: str) -> dict:
    """Extract a concise 4–8 word subject title for a learning chunk. Returns {subject}."""
    return await extract_subject(text)


@mcp.tool()
async def mcp_evaluate_chunk(text: str) -> dict:
    """Evaluate a learning chunk on score, clarity, and completeness (each 1–5). Returns {score, clarity, completeness, critique, suggestions}."""
    return await evaluate_chunk(text)


@mcp.tool()
async def mcp_check_grammar(text: str) -> dict:
    """Check grammar, spelling, and sentence structure of a text chunk. Returns {isProper, issues}."""
    return await check_grammar(text)


@mcp.tool()
async def mcp_refine_chunk(text: str, grammar_issues: str = "", evaluation_score: int | None = None, evaluation_critique: str = "") -> dict:
    """Fix grammar issues in a chunk while preserving the author's voice and facts. Returns {refinedText}."""
    return await refine_chunk(text, {"issues": grammar_issues}, {"score": evaluation_score, "critique": evaluation_critique})


@mcp.tool()
def mcp_get_chunk_stats(text: str) -> dict:
    """Compute word count, char count, sentence count, and avg word length for a text chunk."""
    return get_chunk_stats(text)


@mcp.tool()
async def mcp_structure_content(content_json: str, system_prompt: str = "") -> dict:
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
