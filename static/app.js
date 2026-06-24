// State
let dbPage = 1;
let dbTable = 'enrollments';

// Initialization

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initDebugPanel();
    initEnroll();
    initEncrypt();
    initDecrypt();
    initDatabaseExplorer();
});

// Tab Navigation

function initTabs() {
    const btns = document.querySelectorAll('.tab-btn');
    const indicator = document.getElementById('tab-indicator');

    function activateTab(tabName) {
        // Update buttons
        btns.forEach(b => b.classList.toggle('active', b.dataset.tab === tabName));

        // Update content
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        const target = document.getElementById('tab-' + tabName);
        if (target) target.classList.add('active');

        // Move indicator
        const activeBtn = document.querySelector(`.tab-btn[data-tab="${tabName}"]`);
        if (activeBtn && indicator) {
            indicator.style.left = activeBtn.offsetLeft + 'px';
            indicator.style.width = activeBtn.offsetWidth + 'px';
        }

        // Refresh lists
        if (tabName === 'encrypt' || tabName === 'decrypt') loadUsers();
    }

    btns.forEach(btn => btn.addEventListener('click', () => activateTab(btn.dataset.tab)));

    // Set initial position
    requestAnimationFrame(() => activateTab('enroll'));
}

// Debug Panel

function initDebugPanel() {
    const panel = document.getElementById('debug-panel');
    const toggle = document.getElementById('debug-toggle');

    toggle.addEventListener('click', () => {
        panel.classList.toggle('open');
    });
}

function getSelectedEngine() {
    const checked = document.querySelector('input[name="engine"]:checked');
    return checked ? checked.value : 'gru';
}

function updateDebugPanel(debug) {
    if (!debug) return;

    // Open panel
    document.getElementById('debug-panel').classList.add('open');

    // Fuzzy Vectors
    if (debug.fuzzy_vectors) {
        const fv = debug.fuzzy_vectors;
        document.getElementById('vectors-content').innerHTML = `
            <div class="debug-row"><span class="debug-key">positive</span><span class="debug-val">${fv.mu_mean.toFixed(6)} ± ${fv.mu_std.toFixed(6)}</span></div>
            <div class="debug-row"><span class="debug-key">neutral</span><span class="debug-val">${fv.eta_mean.toFixed(6)} ± ${fv.eta_std.toFixed(6)}</span></div>
            <div class="debug-row"><span class="debug-key">negative</span><span class="debug-val">${fv.nu_mean.toFixed(6)} ± ${fv.nu_std.toFixed(6)}</span></div>
            <div class="debug-row"><span class="debug-key">raw pixel mean</span><span class="debug-val">${fv.raw_mean.toFixed(4)} ± ${fv.raw_std.toFixed(4)}</span></div>
        `;
    }

    // Risk Evaluation
    if (debug.risk) {
        const r = debug.risk;
        const valClass = r.s_anomaly < 0.3 ? 'success' : r.s_anomaly < 0.7 ? 'warning' : 'error';
        document.getElementById('risk-content').innerHTML = `
            <div class="debug-row"><span class="debug-key">engine</span><span class="debug-val">${debug.engine || '—'}</span></div>
            <div class="debug-row"><span class="debug-key">S_anomaly</span><span class="debug-val ${valClass}">${r.s_anomaly.toFixed(4)}</span></div>
            <div class="debug-row"><span class="debug-key">α (alpha)</span><span class="debug-val">${r.alpha.toFixed(4)}</span></div>
            <div class="debug-row"><span class="debug-key">τ_ν base</span><span class="debug-val">${r.tau_nu_base}</span></div>
            <div class="debug-row"><span class="debug-key">max noise allowed</span><span class="debug-val">${r.max_noise_allowed.toFixed(4)}</span></div>
            <div class="debug-row"><span class="debug-key">actual biometric noise</span><span class="debug-val">${r.mean_biometric_noise.toFixed(4)}</span></div>
        `;
    }

    // Telemetry
    if (debug.telemetry) {
        const t = debug.telemetry;
        const timeStr = t.time ? new Date(t.time * 1000).toLocaleString() : '—';
        document.getElementById('telemetry-content').innerHTML = `
            <div class="debug-row"><span class="debug-key">IP</span><span class="debug-val">${t.ip}</span></div>
            <div class="debug-row"><span class="debug-key">geo (lat, lon)</span><span class="debug-val">${t.geo[0]}, ${t.geo[1]}</span></div>
            <div class="debug-row"><span class="debug-key">device</span><span class="debug-val">${t.device_hash}</span></div>
            <div class="debug-row"><span class="debug-key">timestamp</span><span class="debug-val">${timeStr}</span></div>
        `;
    }

    // Decrypted Text
    if (debug.decrypted_text !== undefined) {
        if (debug.decrypted_text) {
            const title = debug.title ? `<div style="color:var(--text-dim);margin-bottom:4px;font-size:11px">${debug.title}</div>` : '';
            document.getElementById('text-content').innerHTML = `${title}<div style="color:var(--success);white-space:pre-wrap">${escapeHtml(debug.decrypted_text)}</div>`;
        } else {
            const reason = debug.denial_reason || 'Authentication denied';
            document.getElementById('text-content').innerHTML = `<div style="color:var(--error)">${escapeHtml(reason)}</div>`;
        }
    }

    // History
    if (debug.history_count !== undefined) {
        document.getElementById('history-count').textContent = `${debug.history_count} entries`;
    }

    if (debug.history && debug.history.length > 0) {
        let html = '<table class="db-table"><thead><tr><th>IP</th><th>Geo</th><th>Device</th><th>Time</th></tr></thead><tbody>';
        debug.history.forEach(h => {
            const ts = new Date(h.time * 1000).toLocaleString();
            html += `<tr><td>${h.ip}</td><td>${h.geo[0]}, ${h.geo[1]}</td><td>${h.device_hash}</td><td>${ts}</td></tr>`;
        });
        html += '</tbody></table>';
        document.getElementById('history-table-wrap').innerHTML = html;
    } else if (debug.history_count === 0) {
        document.getElementById('history-table-wrap').innerHTML = '<span class="debug-empty">No login history (first attempt)</span>';
    }

    // Public key
    if (debug.public_key) {
        setHeaderStatus(`PK: ${debug.public_key.substring(0, 24)}…`);
    }
}

// Enroll

function initEnroll() {
    const select = document.getElementById('enroll-fingerprint');
    const btn = document.getElementById('btn-enroll');
    const preview = document.getElementById('enroll-preview');

    // Load fingerprint list
    fetch('/api/fingerprints')
        .then(r => r.json())
        .then(data => {
            select.innerHTML = '<option value="">— Select fingerprint —</option>';
            data.fingerprints.forEach(name => {
                const opt = document.createElement('option');
                opt.value = name;
                opt.textContent = name.replace('.BMP', '');
                select.appendChild(opt);
            });
        });

    // Preview on selection
    select.addEventListener('change', () => {
        const name = select.value;
        btn.disabled = !name;
        if (name) {
            preview.innerHTML = `<img src="/api/fingerprint-image/${name}" alt="Fingerprint preview">`;
        } else {
            preview.innerHTML = '<span class="preview-placeholder">Select a fingerprint</span>';
        }
    });

    // Enroll action
    btn.addEventListener('click', async () => {
        const name = select.value;
        if (!name) return;

        setLoading(btn, true);
        clearResult('enroll-result');

        try {
            const resp = await fetch('/api/enroll', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ fingerprint: name })
            });
            const data = await resp.json();
            showResult('enroll-result', data.status, data.message, data.debug?.public_key);
            if (data.debug) updateDebugPanel(data.debug);
        } catch (err) {
            showResult('enroll-result', 'error', 'Network error: ' + err.message);
        } finally {
            setLoading(btn, false);
        }
    });
}

// Encrypt

function initEncrypt() {
    const userSelect = document.getElementById('encrypt-user');
    const titleInput = document.getElementById('encrypt-title');
    const contentInput = document.getElementById('encrypt-content');
    const btn = document.getElementById('btn-encrypt');

    function validateEncrypt() {
        btn.disabled = !userSelect.value || !contentInput.value.trim();
    }

    userSelect.addEventListener('change', validateEncrypt);
    contentInput.addEventListener('input', validateEncrypt);

    btn.addEventListener('click', async () => {
        const userId = userSelect.value;
        const title = titleInput.value.trim() || 'Untitled';
        const content = contentInput.value.trim();
        if (!userId || !content) return;

        setLoading(btn, true);
        clearResult('encrypt-result');

        try {
            const resp = await fetch('/api/encrypt', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_id: userId, title, content })
            });
            const data = await resp.json();
            showResult('encrypt-result', data.status, data.message, data.debug ? `CT size: ${data.debug.ciphertext_bytes} bytes` : null);
            if (data.debug) updateDebugPanel(data.debug);
        } catch (err) {
            showResult('encrypt-result', 'error', 'Network error: ' + err.message);
        } finally {
            setLoading(btn, false);
        }
    });

    // Load users
    loadUsers();
}

// Decrypt

function initDecrypt() {
    const userSelect = document.getElementById('decrypt-user');
    const docSection = document.getElementById('decrypt-doc-section');
    const docSelect = document.getElementById('decrypt-message');
    const noiseInput = document.getElementById('decrypt-noise');
    const preview = document.getElementById('decrypt-preview');
    const btn = document.getElementById('btn-decrypt');
    const inlineContent = document.getElementById('inline-decrypted-content');

    userSelect.addEventListener('change', () => {
        inlineContent.style.display = 'none';
        btn.disabled = true;

        if (userSelect.value) {
            const name = userSelect.value + '.BMP';
            preview.innerHTML = `<img src="/api/fingerprint-image/${name}" alt="Fingerprint">`;

            // Load docs
            const user = window.enrolledUsers.find(u => u.user_id === userSelect.value);
            docSelect.innerHTML = '<option value="">— Select document —</option>';
            if (user && user.messages && user.messages.length > 0) {
                user.messages.forEach(m => {
                    const opt = document.createElement('option');
                    opt.value = m.id;
                    opt.textContent = m.title || 'Untitled Document';
                    docSelect.appendChild(opt);
                });
                docSection.style.display = 'block';
            } else {
                docSection.style.display = 'none';
            }
        } else {
            preview.innerHTML = '<span class="preview-placeholder">Select a user</span>';
            docSection.style.display = 'none';
            docSelect.innerHTML = '<option value="">— Select document —</option>';
        }
    });

    docSelect.addEventListener('change', () => {
        btn.disabled = !docSelect.value;
        inlineContent.style.display = 'none';
    });

    btn.addEventListener('click', async () => {
        const userId = userSelect.value;
        const msgId = docSelect.value;
        if (!userId || !msgId) return;

        const engine = getSelectedEngine();
        const noise_override = noiseInput.value;
        const payload = { user_id: userId, message_id: msgId, engine, noise_override };

        const ip = document.getElementById('override-ip').value.trim();
        const device = document.getElementById('override-device').value.trim();
        const lat = document.getElementById('override-lat').value.trim();
        const lon = document.getElementById('override-lon').value.trim();

        if (ip) payload.ip_override = ip;
        if (device) payload.device_hash_override = device;
        if (lat) payload.lat_override = lat;
        if (lon) payload.lon_override = lon;

        setLoading(btn, true);
        clearResult('decrypt-result');
        inlineContent.style.display = 'none';

        try {
            const resp = await fetch('/api/decrypt', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await resp.json();
            showResult('decrypt-result', data.status, data.message);
            if (data.status === 'success' && data.decrypted_content) {
                inlineContent.textContent = data.decrypted_content;
                inlineContent.style.display = 'block';
            }
            if (data.debug) updateDebugPanel(data.debug);
        } catch (err) {
            showResult('decrypt-result', 'error', 'Network error: ' + err.message);
        } finally {
            setLoading(btn, false);
        }
    });
}

// User List Loader

window.enrolledUsers = [];

async function loadUsers() {
    try {
        const resp = await fetch('/api/users');
        const data = await resp.json();
        const users = data.users || [];
        window.enrolledUsers = users;

        // Encrypt options
        const encSelect = document.getElementById('encrypt-user');
        encSelect.innerHTML = '<option value="">— Select enrolled user —</option>';
        users.forEach(u => {
            const opt = document.createElement('option');
            opt.value = u.user_id;
            opt.textContent = `${u.user_id} (pk: ${u.public_key.substring(0, 12)}…)`;
            encSelect.appendChild(opt);
        });

        // Decrypt options
        const decSelect = document.getElementById('decrypt-user');
        decSelect.innerHTML = '<option value="">— Select user with encrypted data —</option>';
        users.filter(u => u.messages && u.messages.length > 0).forEach(u => {
            const opt = document.createElement('option');
            opt.value = u.user_id;
            opt.textContent = `${u.user_id} (${u.messages.length} documents) (pk: ${u.public_key.substring(0, 12)}…)`;
            decSelect.appendChild(opt);
        });
    } catch (err) {
        console.error('Failed to load users:', err);
    }
}

// Database Explorer

function initDatabaseExplorer() {
    const tableSelect = document.getElementById('db-table-select');
    const searchInput = document.getElementById('db-search');
    const btnRefresh = document.getElementById('btn-db-refresh');
    const btnPrev = document.getElementById('btn-db-prev');
    const btnNext = document.getElementById('btn-db-next');

    tableSelect.addEventListener('change', () => {
        dbTable = tableSelect.value;
        dbPage = 1;
        loadDatabaseTable();
    });

    let searchTimeout;
    searchInput.addEventListener('input', () => {
        clearTimeout(searchTimeout);
        searchTimeout = setTimeout(() => {
            dbPage = 1;
            loadDatabaseTable();
        }, 400);
    });

    btnRefresh.addEventListener('click', () => loadDatabaseTable());
    btnPrev.addEventListener('click', () => { if (dbPage > 1) { dbPage--; loadDatabaseTable(); } });
    btnNext.addEventListener('click', () => { dbPage++; loadDatabaseTable(); });
}

async function loadDatabaseTable() {
    const wrap = document.getElementById('db-table-wrap');
    const pageInfo = document.getElementById('db-page-info');
    const btnPrev = document.getElementById('btn-db-prev');
    const btnNext = document.getElementById('btn-db-next');
    const searchTerm = document.getElementById('db-search').value.trim();

    try {
        const resp = await fetch(`/api/database/${dbTable}?page=${dbPage}&per_page=15&search=${encodeURIComponent(searchTerm)}`);
        const data = await resp.json();

        pageInfo.textContent = `Page ${data.page} of ${data.total_pages} (${data.total} rows)`;
        btnPrev.disabled = data.page <= 1;
        btnNext.disabled = data.page >= data.total_pages;

        if (data.rows.length === 0) {
            wrap.innerHTML = '<span class="debug-empty">No data found in this table</span>';
            return;
        }

        let html = '<table class="db-table"><thead><tr>';
        html += '<th>ID</th><th>Public Key</th><th>Lat</th><th>Lon</th><th>IP</th><th>Device</th><th>Time</th>';
        html += '</tr></thead><tbody>';

        data.rows.forEach(r => {
            const ts = r.timestamp ? new Date(r.timestamp * 1000).toLocaleString() : '—';
            html += `<tr>
                <td>${r.id}</td>
                <td>${r.public_key}</td>
                <td>${r.latitude}</td>
                <td>${r.longitude}</td>
                <td>${r.ip_address}</td>
                <td>${r.device_hash}</td>
                <td>${ts}</td>
            </tr>`;
        });

        html += '</tbody></table>';
        wrap.innerHTML = html;
    } catch (err) {
        wrap.innerHTML = `<span class="debug-empty">Error loading data: ${err.message}</span>`;
    }
}

// Helpers

function setLoading(btn, loading) {
    btn.classList.toggle('loading', loading);
    btn.disabled = loading;
}

function clearResult(elementId) {
    document.getElementById(elementId).innerHTML = '';
}

function showResult(elementId, status, message, detail) {
    const icons = { success: '✓', error: '✗', info: 'ℹ', denied: '⛔' };
    const icon = icons[status] || '•';
    let html = `<div class="result-banner ${status}">
        <span class="result-icon">${icon}</span>
        <div>
            <div>${escapeHtml(message)}</div>
            ${detail ? `<div class="result-detail">${escapeHtml(String(detail))}</div>` : ''}
        </div>
    </div>`;
    document.getElementById(elementId).innerHTML = html;
}

function setHeaderStatus(text) {
    const el = document.getElementById('header-status-text');
    if (el) el.textContent = text;
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}
