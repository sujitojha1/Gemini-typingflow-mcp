# Gemini TypingFlow — Agentic

A Chrome extension + FastAPI backend that transforms dense web articles into active-recall typing sessions. It extracts page content, structures it into semantic nuggets via LLM, generates contextual visuals, and drops you into a focused typing interface — burning key ideas into memory through doing, not passive reading.

---

## Architecture Overview

```
Chrome Extension (extension/)          FastAPI Backend (backend/)
──────────────────────────────         ──────────────────────────
popup.js        — DOM scan,            /api/structure     — LLM chunking
                  user actions         /api/process-chunks — agentic pipeline
background.js   — orchestration,      /api/generate-image — image generation
                  API routing          /api/tool           — Ollama tool calls
content.js      — overlay UI,         /api/ollama/models  — list local models
                  typing interface     /health             — status check
options.js      — settings UI
```

All AI calls (Gemini + Ollama) happen in the backend. The extension communicates with it at `http://127.0.0.1:8000`. The API key never touches the browser.

---

## How It Works

```
Popup opens
  → instant DOM scan → word count + image count displayed

User clicks "Process Page Intelligence"
  → content.js extracts text blocks + image URLs from the page
  → Track 1 (Normal Process): POST /api/structure → nuggets returned → gallery opens
  → Track 2 (Agentic Process): POST /api/process-chunks → parallel pipeline in backend
      → for each nugget (up to 5 concurrent):
          checkRelevance → extractSubject + evaluateChunk + checkGrammar (parallel)
          → refineChunk (conditional) → image generation → updateCoverage
      → on completion → extension switches to refined chunks
      → toast: "✦ Agent refined your nuggets" → gallery updates in place

User clicks any nugget card
  → typing overlay opens
  → Gemini image model generates contextual visuals per nugget (via /api/generate-image)

Session complete → one-click Markdown export
```

---

## Features

### Two-Track Processing Pipeline

1. **Track 1 — Normal Process (Fast)**
   - Sends scraped content to `/api/structure`.
   - For articles > 3 000 words, the backend splits into sections and calls the structure model in parallel, then merges results.
   - Gallery mounts in ~2 s for immediate interaction.

2. **Track 2 — Agentic Process (Background)**
   - Sends initial nuggets to `/api/process-chunks`.
   - Backend runs a semaphore-limited parallel pipeline (5 concurrent chunks).
   - Each chunk: `checkRelevance` → `getChunkStats` + `extractSubject` + `evaluateChunk` + `checkGrammar` (parallel) → `refineChunk` (conditional on grammar issues) → image generation.
   - Tool calls route to Ollama (`OLLAMA_TOOL_MODEL`) for low-latency local inference.
   - ~20–30 s total; gallery refreshes in place on completion.

### Multi-Provider Model Support

- **Gemini** — structure model and image generation, configured via `STRUCTURE_MODEL` and `IMAGE_MODEL` env vars.
- **Ollama** — local inference for all agent tool calls (`OLLAMA_TOOL_MODEL`). Options page lets you set the Ollama URL and fetch available models.

### Nugget Gallery

- `[01] — click to type ›` card per nugget with amber left-border accent
- Thumbnail image or hexagon placeholder + 200-char text preview
- Star rating (★★★★☆) and coverage % progress bar in header
- `· ✦ refined by Agent` suffix once the background pipeline completes

### Active Recall Typing Interface

- Character-by-character validation — green correct, red wrong
- Segmented pip progress bar — done (green) / active (blue glow) / pending (dark), with "2 of 4" counter
- Web Audio feedback — soft bandpass noise on correct key, low sine-wave thud on wrong key (synthesized, no external files)
- Fixed bottom metrics bar — WPM · Accuracy · Chars with live gradient fill
- Prev / Next navigation; auto-advances on perfect nugget completion

### Hybrid Visual Context

- Page images matched to nuggets by the LLM are displayed immediately
- For nuggets without a page image, `/api/generate-image` calls `gemini-2.5-flash-preview-image-generation`
- Picsum fallback if image generation fails
- All images returned as `data:` base64 URIs — no CSP issues

### Second Brain Markdown Export

Exports an Obsidian/Notion-ready `.md` file on session completion:
- YAML frontmatter: date, tags, source URL
- TL;DR block quote
- Each nugget as a section with embedded image

---

## Setup

### 1. Backend

```bash
cd backend
cp .env.example .env          # or create .env manually
# Edit .env and set GEMINI_API_KEY
uv sync                       # or: pip install -r requirements.txt
uvicorn main:app --reload     # runs on http://127.0.0.1:8000
```

**`backend/.env` variables:**

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | *(required)* | Google AI API key |
| `STRUCTURE_MODEL` | `gemini-3.1-flash-lite` | Model for page structuring |
| `IMAGE_MODEL` | `gemini-2.5-flash-preview-image-generation` | Model for image generation |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server base URL |
| `OLLAMA_TOOL_MODEL` | `gemma3:4b` | Ollama model for agent tool calls |
| `CONCURRENCY_LIMIT` | `5` | Max parallel chunks in `/api/process-chunks` |
| `MULTIPASS_WORD_THRESHOLD` | `3000` | Word count above which multi-pass structuring kicks in |
| `SECTION_MAX_WORDS` | `2000` | Words per section in multi-pass mode |

### 2. Chrome Extension

1. Open `chrome://extensions` → enable **Developer Mode**
2. Click **Load unpacked** → select the `extension/` folder
3. Click the extension icon → **⚙** → configure Ollama URL if needed → **Save Settings**
4. Make sure the backend is running before clicking **Process Page Intelligence**

---

## File Structure

```
.
├── backend/
│   ├── main.py              # FastAPI app — all AI logic lives here
│   ├── requirements.txt
│   ├── pyproject.toml
│   └── .env                 # GEMINI_API_KEY (not committed)
└── extension/
    ├── manifest.json        # MV3 — host_permissions for localhost:8000
    ├── background.js        # Service worker — orchestrates tracks, routes to backend
    ├── content.js           # Full overlay UI: gallery, typing interface, audio, export
    ├── popup.html/js        # Popup — DOM scan on open, triggers extraction
    └── options.html/js      # Settings — Ollama URL + model, API key display
```

### Backend API

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Returns model config and status |
| `/api/structure` | POST | Chunks scraped content into nuggets via Gemini |
| `/api/process-chunks` | POST | Runs agentic pipeline on nuggets (parallel, Ollama tools) |
| `/api/generate-image` | POST | Generates a contextual image for one nugget |
| `/api/tool` | POST | Raw Ollama tool call (used for ad-hoc prompts) |
| `/api/ollama/models` | GET | Lists locally available Ollama models |

---

## Models Used

| Model | Purpose | Provider |
|---|---|---|
| `gemini-3.1-flash-lite` | Default structure model | Google AI |
| `gemini-2.5-flash-preview-image-generation` | Per-nugget image generation | Google AI |
| `gemma3:4b` (default) | Agent tool calls | Ollama (local) |
| *(any Ollama model)* | Tool calls when configured | Ollama (local) |

---

## Security

- `GEMINI_API_KEY` lives only in `backend/.env` — never in the extension or browser storage
- Extension communicates with the backend over localhost; `host_permissions` in `manifest.json` scoped to `127.0.0.1:8000` and `localhost:8000`
- DOM extraction uses `textContent` and `createElement` — no `innerHTML` on external content
- Image URLs validated before assignment to `img.src`
- Page stats scan uses an inline `func:` in `executeScript` — no `eval`, no string injection

---

## Prompt Engineering

The core structuring prompt (`SYSTEM_PROMPT` in [backend/main.py](backend/main.py)) was evaluated against a 9-criterion rubric for structured LLM reasoning.

### Rubric Results

| Criterion | Original | Final |
|---|---|---|
| explicit_reasoning | ✗ | ✓ 7-step REASONING PROTOCOL |
| structured_output | ✓ | ✓ |
| tool_separation | ✗ | ✓ steps 1–6 reason; step 7 emits |
| conversation_loop | ✗ | ✓ step 5 loops if coverage < 80% |
| instructional_framing | ✓ | ✓ |
| internal_self_checks | ✗ | ✓ coverage + field validity checks |
| reasoning_type_awareness | ✗ | ✓ `content_type` field per nugget |
| fallbacks | ✗ | ✓ ERROR FALLBACKS block |
| overall_clarity | 3/9 met | 9/9 met |

### Before vs After (2 400-word article)

| Metric | Original | Final |
|---|---|---|
| Nugget count | 6 (skipped 3 sections) | 9 (full coverage) |
| coverage_pct | 74% | 91% |
| content_type field | absent | present on all nuggets |
| Malformed tags | 1 | 0 |
| Empty-page error shape | unstructured | `{"error":"insufficient_content",...}` |

---

## Development

See [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) for the full phase-by-phase build log and agentic tweaks roadmap.
