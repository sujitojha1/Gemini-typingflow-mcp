const BACKEND_URL = 'http://localhost:8000';

const backendUrlInput = document.getElementById('backendUrl');
const testBtn         = document.getElementById('testBtn');
const testStatus      = document.getElementById('testStatus');
const saveBtn         = document.getElementById('saveBtn');
const saveStatus      = document.getElementById('saveStatus');

// ── Load saved settings ────────────────────────────────────────────────────────

chrome.storage.sync.get('backendUrl', (result) => {
    backendUrlInput.value = result.backendUrl || BACKEND_URL;
});

// ── Test connection ────────────────────────────────────────────────────────────

testBtn.addEventListener('click', async () => {
    testStatus.textContent = 'Testing…';
    testStatus.className = '';
    const url = (backendUrlInput.value.trim() || BACKEND_URL).replace(/\/$/, '');
    try {
        const res = await fetch(`${url}/health`, { signal: AbortSignal.timeout(5000) });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        testStatus.textContent = `Connected — structure: ${data.models?.structure || '?'}`;
        testStatus.className = 'status-ok';
    } catch (e) {
        testStatus.textContent = `Failed: ${e.message}`;
        testStatus.className = 'status-err';
    }
});

// ── Save ──────────────────────────────────────────────────────────────────────

saveBtn.addEventListener('click', () => {
    const url = backendUrlInput.value.trim() || BACKEND_URL;
    chrome.storage.sync.set({ backendUrl: url }, () => {
        saveStatus.textContent = 'Settings saved.';
        setTimeout(() => { saveStatus.textContent = ''; }, 3000);
    });
});
