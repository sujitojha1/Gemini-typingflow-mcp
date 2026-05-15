# TypingFlow Backend

FastAPI service that handles all AI calls for the Gemini TypingFlow Chrome extension.

See the [root README](../README.md) for full setup instructions and API reference.

## Quick start

```bash
cp .env.example .env   # set GEMINI_API_KEY
uv sync
uvicorn main:app --reload
```

Server runs at `http://127.0.0.1:8000`. The extension's `manifest.json` already has host permissions for this origin.
