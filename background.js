// Load backend URL constant
importScripts('shared_config.js');

// ── Utilities ──────────────────────────────────────────────────────────────────

async function fetchWithTimeout(url, options, timeoutMs = 25000) {
    const controller = new AbortController();
    const timer = setTimeout(() => {
        console.error(`[typingflow] fetchWithTimeout: aborting after ${timeoutMs}ms`);
        controller.abort();
    }, timeoutMs);
    try {
        return await fetch(url, { ...options, signal: controller.signal });
    } finally {
        clearTimeout(timer);
    }
}

function isValidHttpUrl(str) {
    try { const u = new URL(str); return u.protocol === 'https:' || u.protocol === 'http:'; }
    catch { return false; }
}

async function getBackendUrl() {
    const stored = await chrome.storage.sync.get('backendUrl');
    return (stored.backendUrl || BACKEND_URL).replace(/\/$/, '');
}

// ── Image generation (proxied through backend) ─────────────────────────────────

async function generateContextualImage({ text, tags }) {
    const { imageQuotaExceeded } = await chrome.storage.session.get('imageQuotaExceeded');
    if (imageQuotaExceeded) return { success: true, img_src: null };

    const backendUrl = await getBackendUrl();
    try {
        const res = await fetchWithTimeout(`${backendUrl}/api/generate-image`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: (text || '').slice(0, 500), tags: tags || [] }),
        }, 30000);
        if (!res.ok) {
            if (res.status === 429) await chrome.storage.session.set({ imageQuotaExceeded: true });
            return { success: true, img_src: null };
        }
        const data = await res.json();
        return { success: true, img_src: data.img_src };
    } catch (e) {
        console.error('[typingflow] generateContextualImage failed:', e.message);
        return { success: true, img_src: null };
    }
}

// ── Load pipeline module ───────────────────────────────────────────────────────
importScripts('agentic_flow.js');

// ── Message router ─────────────────────────────────────────────────────────────

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'agent_start') {
        runAgentPipeline(request.tabId);
        sendResponse({ started: true });
        return true;
    }
    if (request.action === 'generate_image_asset') {
        generateContextualImage(request.payload).then(sendResponse);
        return true;
    }
});
