// ─────────────────────────────────────────────────────────────────────────────
// agentic_flow.js — Agent pipeline (frontend orchestration only)
//
// Architecture:
//   Track 1 (fast): extractFromTab → POST /api/structure → mount_ui → open_overlay
//   Track 2 (background): POST /api/process-chunks → update_nuggets
//
// All LLM calls, model selection, and multi-pass splitting live in the backend.
// This file handles only DOM extraction, backend communication, and UI updates.
//
// Depends on: background.js globals — fetchWithTimeout, isValidHttpUrl,
//             generateContextualImage, getBackendUrl
// ─────────────────────────────────────────────────────────────────────────────

// Normalizes typographic characters to keyboard-typeable equivalents.
function normalizeForTyping(text) {
    return text
        .replace(/[""«»‹›]/g, '"')
        .replace(/[''‚‛`]/g, "'")
        .replace(/[–—―−﹘﹣－]/g, '-')
        .replace(/…/g, '...')
        .replace(/[     　]/g, ' ')
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
        chrome.tabs.sendMessage(tabId, msg, (resp) => {
            if (chrome.runtime.lastError) reject(new Error(chrome.runtime.lastError.message));
            else resolve(resp);
        });
    });
}

async function extractFromTab(tabId) {
    function unwrap(data) {
        if (!data) return null;
        return Array.isArray(data) ? data : (data.payload || null);
    }
    try {
        const resp = await tabMessage(tabId, { action: 'extract_content' });
        const result = unwrap(resp?.payload);
        if (result) return result;
    } catch (_) {}
    await new Promise((resolve, reject) => {
        chrome.scripting.executeScript({ target: { tabId }, files: ['content.js'] }, r => {
            if (chrome.runtime.lastError) reject(new Error(chrome.runtime.lastError.message));
            else resolve(r);
        });
    });
    for (let attempt = 0; attempt < 5; attempt++) {
        await new Promise(r => setTimeout(r, 250));
        try {
            const resp = await tabMessage(tabId, { action: 'extract_content' });
            const result = unwrap(resp?.payload);
            if (result) return result;
        } catch (_) {}
    }
    return null;
}

// ── Main pipeline ──────────────────────────────────────────────────────────────

async function runAgentPipeline(tabId) {
    const t0 = Date.now();
    const sessionId = 'session_' + Date.now();
    const backendUrl = await getBackendUrl();

    agentBroadcast(tabId, '[1/4] Session init', '—');

    // Clean up stale session data
    chrome.storage.local.get(null, (allItems) => {
        if (chrome.runtime.lastError) return;
        const staleKeys = Object.keys(allItems).filter(k =>
            k.startsWith('tf_') && !k.includes(sessionId)
        );
        if (staleKeys.length) chrome.storage.local.remove(staleKeys);
    });

    // Backend health check
    agentBroadcast(tabId, '[1/4] Connecting', 'backend');
    try {
        const health = await fetchWithTimeout(`${backendUrl}/health`, {}, 5000);
        if (!health.ok) throw new Error(`status ${health.status}`);
    } catch (e) {
        agentBroadcast(tabId, 'error', null,
            `Backend not reachable at ${backendUrl} — is it running? (${e.message})`);
        return;
    }

    // Extract DOM content
    agentBroadcast(tabId, '[2/4] Extracting content', '—', `tab ${tabId}`);
    let payload;
    try {
        payload = await extractFromTab(tabId);
    } catch (e) {
        agentBroadcast(tabId, 'error', null, 'injection blocked');
        return;
    }
    if (!payload?.length) {
        agentBroadcast(tabId, 'error', null, 'no content extracted from page');
        return;
    }
    agentBroadcast(tabId, '[2/4] Extracted', '—', `${payload.length} blocks in ${Date.now() - t0}ms`);

    const imageIndex = payload
        .filter(p => p.type === 'image' && isValidHttpUrl(p.src))
        .map((img, idx) => ({ idx, src: img.src }));

    await chrome.storage.local.set({ [`tf_agent_payload_${sessionId}`]: payload, tf_agent_tab: tabId });

    // ── Track 1: Structure via backend ─────────────────────────────────────────
    agentBroadcast(tabId, '[3/4] Structuring', 'backend');
    let structureResult;
    try {
        const res = await fetchWithTimeout(`${backendUrl}/api/structure`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ payload }),
        }, 90000);
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            agentBroadcast(tabId, 'error', null, `Structuring failed: ${err.detail || res.statusText}`);
            return;
        }
        structureResult = await res.json();
    } catch (e) {
        agentBroadcast(tabId, 'error', null, `Structure request failed: ${e.message}`);
        return;
    }

    const nuggets = structureResult.nuggets || [];
    nuggets.forEach(n => { if (n.text) n.text = normalizeForTyping(n.text); });
    agentBroadcast(tabId, '[4/4] Mounting', 'backend', `${nuggets.length} nuggets`);

    await tabMessage(tabId, { action: 'mount_ui', data: structureResult }).catch(() => {});
    await tabMessage(tabId, { action: 'open_overlay' }).catch(() => {});
    chrome.runtime.sendMessage({ action: 'agent_close_popup' }, () => { chrome.runtime.lastError; });

    await chrome.storage.local.set({
        tf_agent_nuggets: nuggets,
        tf_agent_session: {
            timestamp: Date.now(),
            model: 'backend',
            nuggetCount: nuggets.length,
            tldr: structureResult.tldr,
            tags: structureResult.tags,
            star_rating: structureResult.star_rating,
        },
    });

    agentBroadcast(tabId, 'Done [1-4]', 'backend',
        `${nuggets.length} nuggets | ${Date.now() - t0}ms total`);

    // ── Track 2: Parallel enhancement via backend ──────────────────────────────
    const keepAlive = setInterval(() => chrome.runtime.getPlatformInfo(() => {}), 20000);
    try {
        agentBroadcast(tabId, '[Agent] Enhancing', 'backend', `${nuggets.length} chunks`);
        const res = await fetchWithTimeout(`${backendUrl}/api/process-chunks`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                nuggets,
                image_index: imageIndex,
                tags: structureResult.tags || [],
            }),
        }, 180000);

        if (!res.ok) {
            console.warn('[agent] Track 2 enhance failed:', res.status);
            return;
        }

        const refinedData = await res.json();
        if (refinedData.nuggets) {
            refinedData.nuggets.forEach(n => { if (n.text) n.text = normalizeForTyping(n.text); });
        }

        Object.assign(refinedData, {
            sessionId,
            isAgentRefined: true,
            tldr: structureResult.tldr,
            tags: structureResult.tags,
            star_rating: structureResult.star_rating,
            coverage_pct: structureResult.coverage_pct,
        });

        await chrome.storage.local.set({ tf_pt_refined: refinedData });
        chrome.tabs.sendMessage(tabId,
            { action: 'update_nuggets', data: refinedData },
            () => { chrome.runtime.lastError; }
        );
        agentBroadcast(tabId, 'refined', 'backend',
            `${refinedData.nuggets?.length || 0} nuggets enhanced`);
    } catch (e) {
        console.error('[agent] Track 2 failed:', e.message);
        agentBroadcast(tabId, 'error', null, `Enhancement failed: ${e.message}`);
    } finally {
        clearInterval(keepAlive);
    }
}
