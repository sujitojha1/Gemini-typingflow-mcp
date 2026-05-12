const BACKEND_URL = 'http://localhost:8000';

// ── Utilities ──────────────────────────────────────────────────────────────────

async function fetchWithTimeout(url, options, timeoutMs = 25000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
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
    const { backendUrl } = await chrome.storage.sync.get('backendUrl');
    return (backendUrl || BACKEND_URL).replace(/\/$/, '');
}

// ── Image generation via backend ───────────────────────────────────────────────

async function generateContextualImage({ text, tags }) {
    const { imageQuotaExceeded } = await chrome.storage.session.get('imageQuotaExceeded');
    if (imageQuotaExceeded) return { success: true, img_src: null };

    const base = await getBackendUrl();
    try {
        const res = await fetchWithTimeout(`${base}/api/generate-image`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: (text || '').slice(0, 500), tags: tags || [] }),
        }, 30000);
        if (!res.ok) {
            if (res.status === 429) await chrome.storage.session.set({ imageQuotaExceeded: true });
            return { success: true, img_src: null };
        }
        return { success: true, img_src: (await res.json()).img_src };
    } catch (e) {
        return { success: true, img_src: null };
    }
}

// ── Pipeline ───────────────────────────────────────────────────────────────────

function normalizeForTyping(text) {
    return text
        .replace(/[""«»‹›]/g, '"')
        .replace(/[''‚‛`]/g, "'")
        .replace(/[–—―−﹘﹣－]/g, '-')
        .replace(/…/g, '...')
        .replace(/[     　]/g, ' ')
        .replace(/•/g, '*')
        .normalize('NFD').replace(/[̀-ͯ]/g, '');
}

function agentBroadcast(tabId, task, model, detail = '') {
    const msg = { action: 'agent_status', task, model, detail };
    chrome.tabs.sendMessage(tabId, msg, () => { chrome.runtime.lastError; });
    chrome.runtime.sendMessage(msg, () => { chrome.runtime.lastError; });
}

function tabMessage(tabId, msg) {
    return new Promise((resolve, reject) => {
        chrome.tabs.sendMessage(tabId, msg, resp => {
            if (chrome.runtime.lastError) reject(new Error(chrome.runtime.lastError.message));
            else resolve(resp);
        });
    });
}

async function extractFromTab(tabId) {
    const unwrap = d => !d ? null : Array.isArray(d) ? d : (d.payload || null);
    try {
        const r = await tabMessage(tabId, { action: 'extract_content' });
        const v = unwrap(r?.payload);
        if (v) return v;
    } catch (_) {}
    await new Promise((resolve, reject) => {
        chrome.scripting.executeScript({ target: { tabId }, files: ['content.js'] }, r => {
            if (chrome.runtime.lastError) reject(new Error(chrome.runtime.lastError.message));
            else resolve(r);
        });
    });
    for (let i = 0; i < 5; i++) {
        await new Promise(r => setTimeout(r, 250));
        try {
            const r = await tabMessage(tabId, { action: 'extract_content' });
            const v = unwrap(r?.payload);
            if (v) return v;
        } catch (_) {}
    }
    return null;
}

async function runAgentPipeline(tabId) {
    const t0 = Date.now();
    const sessionId = 'session_' + Date.now();
    const base = await getBackendUrl();

    agentBroadcast(tabId, '[1/4] Connecting', 'backend');

    // Purge stale session data
    chrome.storage.local.get(null, items => {
        if (chrome.runtime.lastError) return;
        const stale = Object.keys(items).filter(k => k.startsWith('tf_') && !k.includes(sessionId));
        if (stale.length) chrome.storage.local.remove(stale);
    });

    // Health check
    try {
        const h = await fetchWithTimeout(`${base}/health`, {}, 5000);
        if (!h.ok) throw new Error(`status ${h.status}`);
    } catch (e) {
        agentBroadcast(tabId, 'error', null, `Backend not reachable at ${base} — is it running? (${e.message})`);
        return;
    }

    // Extract DOM content
    agentBroadcast(tabId, '[2/4] Extracting', '—', `tab ${tabId}`);
    let payload;
    try { payload = await extractFromTab(tabId); } catch (_) {
        agentBroadcast(tabId, 'error', null, 'injection blocked'); return;
    }
    if (!payload?.length) {
        agentBroadcast(tabId, 'error', null, 'no content extracted'); return;
    }
    agentBroadcast(tabId, '[2/4] Extracted', '—', `${payload.length} blocks · ${Date.now() - t0}ms`);

    const imageIndex = payload
        .filter(p => p.type === 'image' && isValidHttpUrl(p.src))
        .map((img, idx) => ({ idx, src: img.src }));

    await chrome.storage.local.set({ [`tf_agent_payload_${sessionId}`]: payload, tf_agent_tab: tabId });

    // ── Track 1: Structure ────────────────────────────────────────────────────
    agentBroadcast(tabId, '[3/4] Structuring', 'backend');
    let structured;
    try {
        const res = await fetchWithTimeout(`${base}/api/structure`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ payload }),
        }, 90000);
        if (!res.ok) {
            const e = await res.json().catch(() => ({}));
            agentBroadcast(tabId, 'error', null, `Structuring failed: ${e.detail || res.statusText}`);
            return;
        }
        structured = await res.json();
    } catch (e) {
        agentBroadcast(tabId, 'error', null, `Structure request failed: ${e.message}`); return;
    }

    const nuggets = structured.nuggets || [];
    nuggets.forEach(n => { if (n.text) n.text = normalizeForTyping(n.text); });
    agentBroadcast(tabId, '[4/4] Mounting', 'backend', `${nuggets.length} nuggets`);

    await tabMessage(tabId, { action: 'mount_ui', data: structured }).catch(() => {});
    await tabMessage(tabId, { action: 'open_overlay' }).catch(() => {});
    chrome.runtime.sendMessage({ action: 'agent_close_popup' }, () => { chrome.runtime.lastError; });

    await chrome.storage.local.set({
        tf_agent_nuggets: nuggets,
        tf_agent_session: {
            timestamp: Date.now(), model: 'backend', nuggetCount: nuggets.length,
            tldr: structured.tldr, tags: structured.tags, star_rating: structured.star_rating,
        },
    });
    agentBroadcast(tabId, 'Done [1-4]', 'backend', `${nuggets.length} nuggets · ${Date.now() - t0}ms`);

    // ── Track 2: Parallel enhancement ────────────────────────────────────────
    const keepAlive = setInterval(() => chrome.runtime.getPlatformInfo(() => {}), 20000);
    try {
        agentBroadcast(tabId, '[Agent] Enhancing', 'backend', `${nuggets.length} chunks`);
        const res = await fetchWithTimeout(`${base}/api/process-chunks`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ nuggets, image_index: imageIndex, tags: structured.tags || [] }),
        }, 180000);

        if (!res.ok) { console.warn('[agent] Track 2 failed:', res.status); return; }

        const refined = await res.json();
        refined.nuggets?.forEach(n => { if (n.text) n.text = normalizeForTyping(n.text); });
        Object.assign(refined, {
            sessionId, isAgentRefined: true,
            tldr: structured.tldr, tags: structured.tags,
            star_rating: structured.star_rating, coverage_pct: structured.coverage_pct,
        });

        await chrome.storage.local.set({ tf_pt_refined: refined });
        chrome.tabs.sendMessage(tabId, { action: 'update_nuggets', data: refined }, () => { chrome.runtime.lastError; });
        agentBroadcast(tabId, 'refined', 'backend', `${refined.nuggets?.length || 0} nuggets enhanced`);
    } catch (e) {
        agentBroadcast(tabId, 'error', null, `Enhancement failed: ${e.message}`);
    } finally {
        clearInterval(keepAlive);
    }
}

// ── Message router ─────────────────────────────────────────────────────────────

chrome.runtime.onMessage.addListener((request, _sender, sendResponse) => {
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
