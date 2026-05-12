// tool_helper.js — Forwards individual tool prompts to the backend /api/tool endpoint.
// The tool_*.js files in this directory are retained as readable documentation of
// what each tool does; the actual LLM execution happens in backend/main.py.
//
// Depends on: background.js globals — fetchWithTimeout, getBackendUrl

async function callToolModel(prompt, defaultResult, timeoutMs = 20000) {
    let backendUrl;
    try {
        backendUrl = await getBackendUrl();
    } catch (_) {
        backendUrl = 'http://localhost:8000';
    }

    try {
        const res = await fetchWithTimeout(`${backendUrl}/api/tool`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt }),
        }, timeoutMs);

        if (!res.ok) {
            const errBody = await res.json().catch(() => ({}));
            const lastError = `Backend ${res.status}: ${errBody.detail || res.statusText}`;
            console.warn(`[tool] callToolModel HTTP error — ${lastError}`);
            return { ...defaultResult, error: lastError };
        }

        return await res.json();
    } catch (e) {
        const lastError = e.name === 'AbortError'
            ? `timed out after ${timeoutMs}ms`
            : e.message;
        console.warn('[tool] callToolModel failed —', lastError);
        return { ...defaultResult, error: lastError };
    }
}
