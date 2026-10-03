let queue = [];
let selectedId = null;
const decided = {};

const $ = (id) => document.getElementById(id);

// All server values are untrusted (wallet IDs, locations, device strings flow
// from POST /score into timelines). Escape everything interpolated into HTML;
// textContent assignments elsewhere are inherently safe.
function esc(v) {
  return String(v ?? '').replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function showError(msg) {
  const bar = $('errorBar');
  bar.hidden = false;
  bar.textContent = msg;
}
function clearError() { $('errorBar').hidden = true; }

// Per-endpoint last outcome, always visible: tells a dead server
// (all FAIL) apart from a sick one (mixed results) with no devtools.
const diag = { health: '…', alerts: '…' };
function renderDiag() {
  const el = $('diagLine');
  if (el) el.textContent = `health: ${diag.health} · alerts: ${diag.alerts}`;
}

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    const txt = await r.text().catch(() => '');
    throw new Error(`${path} -> ${r.status} ${txt.slice(0, 160)}`);
  }
  return r.json();
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

let queueLoaded = false;

async function checkHealth(retries = 5) {
  // Free-tier hosts sleep and redeploy; attempts often race the wake-up, so
  // retry with backoff and always report the real reason. The /health link
  // lets you tell a browser/network problem (link fails too) from a console
  // bug (link works) without any tooling.
  let lastErr = '';
  for (let i = 0; i < retries; i++) {
    try {
      const h = await api('/health');
      $('health').textContent =
        `online · ${h.history_rows} txns · ${h.graph_nodes} graph nodes · queue ${h.queue_size}`;
      $('health').className = 'health ok';
      diag.health = `ok (${h.queue_size} queued)`;
      renderDiag();
      clearError();
      if (!queueLoaded) loadQueue();  // we arrived during an outage; catch up
      return true;
    } catch (err) {
      lastErr = err.message;
      diag.health = `FAIL (${err.message})`;
      renderDiag();
      $('health').textContent = `contacting API (try ${i + 1}/${retries})…`;
      $('health').className = 'health bad';
      await sleep(3000 * (i + 1));
    }
  }
  $('health').innerHTML =
    `API offline (${esc(lastErr)}), re-checking every 30s. Self-hosting? start: ` +
    `<code>uvicorn api.main:app --port 8000</code>. Hosted demo? open ` +
    `<a href="/health" target="_blank" rel="noopener">/health</a> directly: ` +
    `if it fails too, the network or host is down; if it works, hard-refresh this page.`;
  $('health').className = 'health bad';
  // Keep polling while offline so a load during a restart heals itself
  // instead of freezing on a stale message. Skipped in hidden tabs.
  setTimeout(() => { if (!document.hidden) checkHealth(2); }, 30000);
  return false;
}

document.addEventListener('visibilitychange', () => {
  if (!document.hidden) checkHealth(1);
});

function badge(level) {
  return `<span class="badge ${esc(level)}">${esc(level)}</span>`;
}

async function loadQueue() {
  clearError();
  $('queueBody').innerHTML = '<tr><td colspan="6" class="muted">loading queue…</td></tr>';
  try {
    const level = $('levelFilter').value;
    const q = level ? `?limit=200&level=${level}` : '?limit=200';
    const data = await api(`/alerts${q}`);
    queue = data.alerts || [];
    diag.alerts = `ok (${queue.length})`;
    renderDiag();
    renderQueue();
  } catch (err) {
    diag.alerts = `FAIL (${err.message})`;
    renderDiag();
    $('queueBody').innerHTML = '<tr><td colspan="6" class="muted">queue failed to load</td></tr>';
    showError(`Queue failed: ${err.message}`);
  }
}

function renderQueue() {
  const f = $('searchBox').value.toLowerCase();
  const minRisk = parseFloat($('minRisk').value || '0');
  $('minRiskVal').textContent = minRisk.toFixed(2);
  const sortBy = $('sortSelect').value;
  let rows = queue.filter((r) =>
    (r.risk_score >= minRisk) &&
    (!f || r.txn_id.toLowerCase().includes(f) ||
      (r.sender_id || '').toLowerCase().includes(f) ||
      (r.receiver_id || '').toLowerCase().includes(f))
  );
  rows = rows.slice().sort((a, b) =>
    sortBy === 'amount' ? b.amount - a.amount : b.risk_score - a.risk_score);
  $('queueCount').textContent = `· ${rows.length} shown`;
  if (!rows.length) {
    $('queueBody').innerHTML = '<tr><td colspan="6" class="muted">no alerts match filters</td></tr>';
    return;
  }
  queueLoaded = true;
  $('queueBody').innerHTML = rows.map((r) => `
    <tr data-id="${esc(r.txn_id)}" class="${r.txn_id === selectedId ? 'sel' : ''}">
      <td><strong>${Number(r.risk_score).toFixed(2)}</strong>${decided[r.txn_id] ? ` <span class="muted">(${esc(decided[r.txn_id])})</span>` : ''}</td>
      <td><code>${esc(r.txn_id)}</code></td>
      <td><code>${esc(r.sender_id)} → ${esc(r.receiver_id)}</code></td>
      <td>৳${Number(r.amount).toLocaleString()}</td>
      <td>${badge(r.risk_level)}</td>
      <td class="muted">${esc(r.recommended_action)}</td>
    </tr>`).join('');
  document.querySelectorAll('#queueBody tr[data-id]').forEach((tr) =>
    tr.addEventListener('click', () => openCase(tr.dataset.id)));
}

async function openCase(id) {
  selectedId = id;
  clearError();
  renderQueue();
  $('caseEmpty').hidden = true;
  $('caseBox').hidden = false;
  $('caseNarrative').textContent = 'loading case…';
  try {
    const lang = $('langSelect').value;
    const c = await api(`/case/${encodeURIComponent(id)}?lang=${lang}`);
    $('caseId').textContent = c.txn_id;
    $('caseMeta').textContent =
      `${c.sender_id} → ${c.receiver_id} · ৳${Number(c.amount).toLocaleString()} · ${c.channel} · ${c.timestamp}`;
    $('caseLevel').textContent = `${c.risk_level} · ${Number(c.risk_score).toFixed(2)}`;
    $('caseLevel').className = `badge ${c.risk_level}`;
    $('caseNarrative').textContent = c.narrative || '(no narrative)';
    $('caseReasons').innerHTML = (c.top_3_reasons || []).map((x) => `<li>${esc(x)}</li>`).join('');
    const comp = c.components || { p_fraud: c.p_fraud, anomaly: c.anomaly, graph_boost: c.graph_boost };
    $('caseComponents').innerHTML = Object.entries({
      'P(fraud)': comp.p_fraud ?? c.p_fraud,
      anomaly: comp.anomaly ?? c.anomaly,
      graph_boost: comp.graph_boost ?? c.graph_boost,
      fraud_neighbors_2hop: c.fraud_neighbors_2hop,
      latency_ms: comp.latency_ms,
    }).filter(([, v]) => v !== undefined).map(([k, v]) => `<div><b>${esc(k)}</b><span>${esc(v)}</span></div>`).join('');
    const sample = c.fraud_neighbor_sample || [];
    $('caseGraph').innerHTML = c.fraud_neighbors_2hop
      ? `<code>${esc(c.receiver_id)}</code> ↔ <span class="muted">${esc(c.fraud_neighbors_2hop)} fraud within 2 hops</span><br/>` +
        (sample.length ? sample.map((n) => `<code>${esc(n)}</code>`).join(' · ') : '<span class="muted">no sample</span>')
      : '<span class="muted">no fraud proximity — receiver is clean in the 2-hop graph</span>';
    $('caseTimeline').innerHTML = (c.timeline || []).map((t) =>
      `<li><code>${esc(t.sender)} → ${esc(t.receiver)}</code> ৳${Number(t.amount).toLocaleString()} <span class="muted">${esc(t.timestamp)} · ${esc(t.location)}</span></li>`).join('') || '<li class="muted">no history</li>';
    $('decStatus').textContent = decided[id] ? `logged: ${decided[id]}` : '';
  } catch (err) {
    $('caseNarrative').textContent = 'case failed to load';
    showError(`Case failed: ${err.message}`);
  }
}

async function sendDecision(decision) {
  if (!selectedId) return;
  try {
    await api('/decision', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ txn_id: selectedId, decision, analyst: 'analyst-1' }),
    });
    decided[selectedId] = decision;
    $('decStatus').textContent = `logged: ${decision}`;
    renderQueue();
  } catch (err) {
    showError(`Decision failed: ${err.message}`);
  }
}

$('refreshBtn').addEventListener('click', () => { checkHealth(); loadQueue(); });
$('levelFilter').addEventListener('change', loadQueue);
$('sortSelect').addEventListener('change', renderQueue);
$('minRisk').addEventListener('input', renderQueue);
$('langSelect').addEventListener('change', () => selectedId && openCase(selectedId));
$('searchBox').addEventListener('input', renderQueue);
document.querySelectorAll('.decisions button').forEach((b) =>
  b.addEventListener('click', () => sendDecision(b.dataset.dec)));

$('scoreForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  clearError();
  const fd = new FormData(e.target);
  const body = {
    sender_id: fd.get('sender_id'), receiver_id: fd.get('receiver_id'),
    amount: Number(fd.get('amount')), channel: fd.get('channel'),
    device_id: fd.get('device_id'), location: fd.get('location'),
    timestamp: fd.get('timestamp'), type: fd.get('type'), lang: $('langSelect').value,
    password_reset_flag: fd.get('pwd_reset') ? 1 : 0,
  };
  $('scoreOut').textContent = 'scoring…';
  try {
    const out = await api('/score', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    $('scoreOut').textContent = JSON.stringify(out, null, 2);
    $('scoreOut').className = '';
  } catch (err) {
    $('scoreOut').textContent = `error: ${err.message}`;
    showError(`Score failed: ${err.message}`);
  }
});

// First-run walkthrough, remembered per browser.
try {
  if (!localStorage.getItem('vigil_walkthrough')) $('walkthrough').hidden = false;
} catch { /* private mode: stay quiet, console still works */ }
$('walkDismiss').addEventListener('click', () => {
  $('walkthrough').hidden = true;
  try { localStorage.setItem('vigil_walkthrough', 'done'); } catch { /* ignore */ }
});

function fillForm(values) {
  const form = $('scoreForm');
  for (const [name, value] of Object.entries(values)) {
    const el = form.elements[name];
    if (!el) continue;
    if (el.type === 'checkbox') el.checked = Boolean(value);
    else el.value = value;
  }
}
$('fillScam').addEventListener('click', () => fillForm({
  sender_id: 'C000001', receiver_id: 'C000002', amount: 45000, channel: 'app',
  device_id: 'DX999', location: 'Dhaka', timestamp: '2026-08-15T23:10:00',
  type: 'P2P', pwd_reset: false,
}));
$('fillNormal').addEventListener('click', () => fillForm({
  sender_id: 'C000001', receiver_id: 'C000002', amount: 1800, channel: 'app',
  device_id: 'D000001', location: 'Dhaka', timestamp: '2026-08-15T14:10:00',
  type: 'merchant', pwd_reset: false,
}));

checkHealth();
loadQueue();
