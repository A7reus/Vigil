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

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    const txt = await r.text().catch(() => '');
    throw new Error(`${path} -> ${r.status} ${txt.slice(0, 160)}`);
  }
  return r.json();
}

async function checkHealth() {
  try {
    const h = await api('/health');
    $('health').textContent =
      `online · ${h.history_rows} txns · ${h.graph_nodes} graph nodes · queue ${h.queue_size}`;
    $('health').className = 'health ok';
  } catch (err) {
    $('health').textContent = 'API offline — start with: uvicorn api.main:app --port 8000';
    $('health').className = 'health bad';
  }
}

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
    renderQueue();
  } catch (err) {
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

$('refreshBtn').addEventListener('click', loadQueue);
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

checkHealth();
loadQueue();
