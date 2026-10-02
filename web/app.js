let queue = [];
let selectedId = null;

const $ = (id) => document.getElementById(id);

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${path} -> ${r.status}`);
  return r.json();
}

async function checkHealth() {
  try {
    const h = await api('/health');
    $('health').textContent = `online · ${h.history_rows} txns · ${h.graph_nodes} graph nodes`;
    $('health').className = 'health ok';
  } catch {
    $('health').textContent = 'API offline — start with: uvicorn api.main:app --port 8000';
    $('health').className = 'health bad';
  }
}

function badge(level) {
  return `<span class="badge ${level}">${level}</span>`;
}

async function loadQueue() {
  const level = $('levelFilter').value;
  const q = level ? `?limit=200&level=${level}` : '?limit=200';
  const data = await api(`/alerts${q}`);
  queue = data.alerts || [];
  renderQueue();
}

function renderQueue() {
  const f = $('searchBox').value.toLowerCase();
  const rows = queue.filter((r) =>
    !f ||
    r.txn_id.toLowerCase().includes(f) ||
    (r.sender_id || '').toLowerCase().includes(f) ||
    (r.receiver_id || '').toLowerCase().includes(f)
  );
  $('queueCount').textContent = `· ${rows.length} shown`;
  $('queueBody').innerHTML = rows.map((r) => `
    <tr data-id="${r.txn_id}" class="${r.txn_id === selectedId ? 'sel' : ''}">
      <td><strong>${r.risk_score.toFixed(2)}</strong></td>
      <td><code>${r.txn_id}</code></td>
      <td><code>${r.sender_id} → ${r.receiver_id}</code></td>
      <td>৳${Number(r.amount).toLocaleString()}</td>
      <td>${badge(r.risk_level)}</td>
      <td class="muted">${r.recommended_action}</td>
    </tr>`).join('');
  document.querySelectorAll('#queueBody tr').forEach((tr) =>
    tr.addEventListener('click', () => openCase(tr.dataset.id)));
}

async function openCase(id) {
  selectedId = id;
  renderQueue();
  const lang = $('langSelect').value;
  const c = await api(`/case/${id}?lang=${lang}`);
  $('caseEmpty').hidden = true;
  $('caseBox').hidden = false;
  $('caseId').textContent = c.txn_id;
  $('caseMeta').textContent =
    `${c.sender_id} → ${c.receiver_id} · ৳${Number(c.amount).toLocaleString()} · ${c.channel} · ${c.timestamp}`;
  $('caseLevel').textContent = `${c.risk_level} · ${c.risk_score.toFixed(2)}`;
  $('caseLevel').className = `badge ${c.risk_level}`;
  $('caseNarrative').textContent = c.narrative || '(no narrative)';
  $('caseReasons').innerHTML = (c.top_3_reasons || []).map((x) => `<li>${x}</li>`).join('');
  const comp = c.components || { p_fraud: c.p_fraud, anomaly: c.anomaly, graph_boost: c.graph_boost };
  $('caseComponents').innerHTML = Object.entries({
    'P(fraud)': comp.p_fraud ?? c.p_fraud,
    anomaly: comp.anomaly ?? c.anomaly,
    graph_boost: comp.graph_boost ?? c.graph_boost,
    fraud_neighbors_2hop: c.fraud_neighbors_2hop,
    latency_ms: comp.latency_ms,
  }).filter(([, v]) => v !== undefined).map(([k, v]) => `<div><b>${k}</b><span>${v}</span></div>`).join('');
  $('caseTimeline').innerHTML = (c.timeline || []).map((t) =>
    `<li><code>${t.txn_id}</code> → <code>${t.receiver}</code> ৳${Number(t.amount).toLocaleString()} <span class="muted">${t.timestamp} · ${t.location}</span></li>`).join('') || '<li class="muted">no history</li>';
  $('decStatus').textContent = '';
}

async function sendDecision(decision) {
  if (!selectedId) return;
  await api('/decision', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ txn_id: selectedId, decision, analyst: 'analyst-1' }),
  });
  $('decStatus').textContent = `logged: ${decision}`;
}

$('refreshBtn').addEventListener('click', loadQueue);
$('levelFilter').addEventListener('change', loadQueue);
$('langSelect').addEventListener('change', () => selectedId && openCase(selectedId));
$('searchBox').addEventListener('input', renderQueue);
document.querySelectorAll('.decisions button').forEach((b) =>
  b.addEventListener('click', () => sendDecision(b.dataset.dec)));

$('scoreForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const body = {
    sender_id: fd.get('sender_id'), receiver_id: fd.get('receiver_id'),
    amount: Number(fd.get('amount')), channel: fd.get('channel'),
    device_id: fd.get('device_id'), location: fd.get('location'),
    timestamp: fd.get('timestamp'), type: fd.get('type'), lang: $('langSelect').value,
  };
  try {
    const out = await api('/score', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    $('scoreOut').textContent = JSON.stringify(out, null, 2);
    $('scoreOut').className = '';
  } catch (err) {
    $('scoreOut').textContent = `error: ${err.message}`;
  }
});

checkHealth();
loadQueue();
