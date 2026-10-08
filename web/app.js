/* Vigil analyst console v2.
   API contract (unchanged): GET /health, GET /alerts, GET /case/{id}?lang=,
   POST /score, POST /decision. */

const state = {
  queue: [],
  selectedId: null,
  decided: {},
  level: '',
  lang: 'en',
  loaded: false,
  loading: false,
  caseToken: 0,
  lastCase: null,
};

const $ = (id) => document.getElementById(id);
const NARROW = window.matchMedia('(max-width: 1000px)');

/* All server values are untrusted (wallet IDs, locations, device strings flow
   from POST /score into timelines). Escape everything interpolated into HTML;
   textContent assignments are inherently safe. */
function esc(v) {
  return String(v ?? '').replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
const money = (n) => '৳' + Number(n || 0).toLocaleString();
const num = (v) => (Number.isFinite(Number(v)) ? Number(v) : 0);
const clamp01 = (v) => Math.min(1, Math.max(0, num(v)));
const LEVELS = ['High', 'Medium', 'Low'];
const safeLevel = (l) => (LEVELS.includes(l) ? l : 'Low');

/* ---------- Network ---------- */
function authHeaders() {
  let t = null;
  try { t = sessionStorage.getItem('vigil_token'); } catch (e) {}
  return t ? { 'Authorization': `Bearer ${t}` } : {};
}
async function api(path, opts) {
  opts = { ...(opts || {}), headers: { ...(opts && opts.headers), ...authHeaders() } };
  const r = await fetch(path, opts);
  if (!r.ok) {
    const txt = await r.text().catch(() => '');
    throw new Error(`${path} -> ${r.status} ${txt.slice(0, 160)}`);
  }
  return r.json();
}

/* ---------- Toasts ---------- */
function toast(msg, kind = '', ms = 3500) {
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  const span = document.createElement('span');
  span.textContent = msg;
  const x = document.createElement('button');
  x.type = 'button';
  x.setAttribute('aria-label', 'Dismiss');
  x.textContent = '✕';
  const close = () => el.remove();
  x.addEventListener('click', close);
  el.append(span, x);
  $('toasts').appendChild(el);
  if (ms) setTimeout(close, kind === 'error' ? Math.max(ms, 7000) : ms);
}

/* ---------- Theme ---------- */
function setTheme(t) {
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem('vigil_theme', t); } catch (e) {}
}
$('themeBtn').addEventListener('click', () =>
  setTheme(document.documentElement.dataset.theme === 'light' ? 'dark' : 'light'));

/* ---------- Health ---------- */
async function checkHealth() {
  const box = $('health');
  try {
    const h = await api('/health');
    $('healthText').textContent =
      `online · ${h.history_rows} txns · ${h.graph_nodes} graph nodes · queue ${h.queue_size}`;
    box.className = 'health ok';
  } catch (err) {
    $('healthText').textContent = 'API offline — start with: uvicorn api.main:app --port 8000';
    box.className = 'health bad';
  }
}

/* ---------- KPIs + counts ---------- */
const REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)');
function tween(el, to) {
  const from = Number(el.dataset.v);
  el.dataset.v = to;
  cancelAnimationFrame(el._raf);
  if (REDUCED.matches || !Number.isFinite(from) || from === to) { el.textContent = to.toLocaleString(); return; }
  const t0 = performance.now(), dur = 350;
  const step = (t) => {
    const p = Math.min(1, (t - t0) / dur);
    const e = 1 - Math.pow(1 - p, 3);
    el.textContent = Math.round(from + (to - from) * e).toLocaleString();
    if (p < 1) el._raf = requestAnimationFrame(step);
  };
  el._raf = requestAnimationFrame(step);
}
function renderStats() {
  const q = state.queue;
  const count = (l) => q.filter((r) => r.risk_level === l).length;
  const high = count('High'), med = count('Medium'), low = count('Low');
  tween($('kTotal'), q.length);
  $('kTotalSub').textContent = state.loaded ? `${low} low · ${med} medium · ${high} high` : '\u00a0';
  tween($('kHigh'), high);
  tween($('kMed'), med);
  const tot = q.length || 1;
  const dist = $('kDist');
  dist.querySelector('.h').style.width = (high / tot * 100) + '%';
  dist.querySelector('.m').style.width = (med / tot * 100) + '%';
  dist.querySelector('.l').style.width = (low / tot * 100) + '%';
  const exposure = q.filter((r) => r.risk_level !== 'Low').reduce((s, r) => s + num(r.amount), 0);
  $('kExposure').textContent = money(exposure);
  const done = Object.keys(state.decided).length;
  $('kDone').textContent = done;
  $('kDoneBar').style.width = (q.length ? Math.min(100, (done / q.length) * 100) : 0) + '%';
  $('cAll').textContent = q.length;
  $('cHigh').textContent = high;
  $('cMed').textContent = med;
  $('cLow').textContent = low;
}

/* ---------- Queue ---------- */
function skeletonRows(n = 7) {
  return Array.from({ length: n }, () =>
    `<tr class="state-row skel-row"><td><span class="skel" style="width:3.5rem"></span></td>
      <td><span class="skel" style="width:70%"></span></td>
      <td><span class="skel" style="width:4rem;margin-left:auto"></span></td>
      <td><span class="skel" style="width:4.5rem"></span></td>
      <td class="col-action"><span class="skel" style="width:60%"></span></td></tr>`).join('');
}

async function loadQueue(opts = {}) {
  if (state.loading) return;
  state.loading = true;
  $('refreshBtn').disabled = true;
  if (!opts.silent && !state.queue.length) $('queueBody').innerHTML = skeletonRows();
  try {
    const data = await api('/alerts?limit=200');
    state.queue = data.alerts || [];
    state.loaded = true;
    renderQueue();
    if (opts.announce) toast(`Queue updated · ${state.queue.length} alerts`, 'ok', 2000);
  } catch (err) {
    state.loaded = true;
    if (!state.queue.length) {
      $('queueBody').innerHTML =
        `<tr class="state-row"><td colspan="5">Queue failed to load.<br/><span class="small">${esc(err.message)}</span><br/>
        <button type="button" class="btn" id="retryQueue">Try again</button></td></tr>`;
      $('retryQueue').addEventListener('click', () => loadQueue());
    }
    toast(`Queue failed: ${err.message}`, 'error');
  } finally {
    state.loading = false;
    $('refreshBtn').disabled = false;
  }
}

function visibleRows() {
  const f = $('searchBox').value.trim().toLowerCase();
  const minRisk = num($('minRisk').value);
  const sortBy = $('sortSelect').value;
  const hideDone = $('hideDone').checked;
  const rows = state.queue.filter((r) =>
    (!state.level || r.risk_level === state.level) &&
    num(r.risk_score) >= minRisk &&
    !(hideDone && state.decided[r.txn_id]) &&
    (!f || String(r.txn_id).toLowerCase().includes(f) ||
      String(r.sender_id || '').toLowerCase().includes(f) ||
      String(r.receiver_id || '').toLowerCase().includes(f)));
  return rows.sort((a, b) =>
    sortBy === 'amount' ? num(b.amount) - num(a.amount) : num(b.risk_score) - num(a.risk_score));
}

function renderQueue(focusSelected = false) {
  renderStats();
  const minRisk = num($('minRisk').value);
  $('minRiskVal').textContent = minRisk.toFixed(2);
  const rows = visibleRows();
  $('queueCount').textContent = `· ${rows.length} of ${state.queue.length} shown`;
  if (!rows.length) {
    $('queueBody').innerHTML = `<tr class="state-row"><td colspan="5">${
      state.queue.length
        ? 'No alerts match these filters.<br/><button type="button" class="btn" id="resetFilters">Clear filters</button>'
        : 'The queue is empty. Nothing needs review right now.'}</td></tr>`;
    const rf = $('resetFilters');
    if (rf) rf.addEventListener('click', resetFilters);
    return;
  }
  $('queueBody').innerHTML = rows.map((r) => {
    const lvl = safeLevel(r.risk_level);
    const d = state.decided[r.txn_id];
    const sel = r.txn_id === state.selectedId;
    return `<tr data-id="${esc(r.txn_id)}" tabindex="0" aria-selected="${sel}" class="risk-${lvl}${sel ? ' sel' : ''}${d ? ' done' : ''}">
      <td class="c-risk"><div class="score-cell"><strong class="score-num">${num(r.risk_score).toFixed(2)}</strong><div class="mini-bar"><i style="width:${(clamp01(r.risk_score) * 100).toFixed(0)}%"></i></div></div></td>
      <td class="c-txn"><div class="txn-cell"><code>${esc(r.txn_id)}</code><span class="route">${esc(r.sender_id)} → ${esc(r.receiver_id)}</span>${d ? `<span class="tag-done">${esc(d)}</span>` : ''}</div></td>
      <td class="num c-amt">${money(r.amount)}</td>
      <td class="c-level"><span class="badge ${lvl}">${esc(r.risk_level)}</span></td>
      <td class="col-action">${esc(r.recommended_action)}</td>
    </tr>`;
  }).join('');
  if (focusSelected) {
    const row = $('queueBody').querySelector('tr.sel');
    if (row) { row.focus({ preventScroll: true }); row.scrollIntoView({ block: 'nearest' }); }
  }
}

function resetFilters() {
  state.level = '';
  document.querySelectorAll('.chip[data-level]').forEach((x) =>
    x.setAttribute('aria-pressed', x.dataset.level === ''));
  $('searchBox').value = '';
  $('minRisk').value = 0;
  $('hideDone').checked = false;
  renderQueue();
}

function markSelected() {
  document.querySelectorAll('#queueBody tr[data-id]').forEach((tr) => {
    const on = tr.dataset.id === state.selectedId;
    tr.classList.toggle('sel', on);
    tr.setAttribute('aria-selected', on);
  });
}

$('queueBody').addEventListener('click', (e) => {
  const tr = e.target.closest('tr[data-id]');
  if (tr) openCase(tr.dataset.id);
});
$('queueBody').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') {
    const tr = e.target.closest('tr[data-id]');
    if (tr) { e.preventDefault(); openCase(tr.dataset.id); }
  }
});

function stepSelection(dir, fromKeyboard = true) {
  const rows = visibleRows();
  if (!rows.length) return;
  let i = rows.findIndex((r) => r.txn_id === state.selectedId);
  i = i === -1 ? (dir > 0 ? 0 : rows.length - 1) : Math.min(rows.length - 1, Math.max(0, i + dir));
  openCase(rows[i].txn_id, { focusRow: fromKeyboard });
}

/* Filter wiring */
document.querySelectorAll('.chip[data-level]').forEach((b) =>
  b.addEventListener('click', () => {
    state.level = b.dataset.level;
    document.querySelectorAll('.chip[data-level]').forEach((x) =>
      x.setAttribute('aria-pressed', x === b));
    renderQueue();
  }));
['sortSelect', 'hideDone'].forEach((id) => $(id).addEventListener('change', () => renderQueue()));
['minRisk', 'searchBox'].forEach((id) => $(id).addEventListener('input', () => renderQueue()));
$('refreshBtn').addEventListener('click', () => loadQueue({ announce: true, silent: true }).then(checkHealth));

/* Auto refresh */
let autoTimer = null;
$('autoRefresh').addEventListener('change', (e) => {
  clearInterval(autoTimer);
  if (e.target.checked) {
    autoTimer = setInterval(() => { if (!document.hidden) { loadQueue({ silent: true }); checkHealth(); } }, 30000);
    toast('Auto-refresh on · every 30s', '', 2000);
  }
});

/* Language */
document.querySelectorAll('.seg-btn[data-lang]').forEach((b) =>
  b.addEventListener('click', () => {
    state.lang = b.dataset.lang;
    document.querySelectorAll('.seg-btn[data-lang]').forEach((x) =>
      x.setAttribute('aria-pressed', x === b));
    document.documentElement.lang = state.lang === 'bn' ? 'bn' : 'en';
    if (state.selectedId) openCase(state.selectedId);
  }));

/* ---------- Case ---------- */
let sheetPushed = false;
function openSheet() {
  if (!NARROW.matches) return;
  $('casePane').classList.add('open');
  document.body.classList.add('sheet-open');
  if (!sheetPushed) { history.pushState({ vigilSheet: 1 }, ''); sheetPushed = true; }
}
function closeSheet(fromPop = false) {
  $('casePane').classList.remove('open');
  document.body.classList.remove('sheet-open');
  if (sheetPushed) {
    sheetPushed = false;
    if (!fromPop) history.back();
  }
  const row = $('queueBody').querySelector('tr.sel');
  if (row && !fromPop) row.focus({ preventScroll: true });
}
window.addEventListener('popstate', () => {
  if ($('casePane').classList.contains('open')) closeSheet(true);
});
NARROW.addEventListener('change', (e) => {
  if (!e.matches) {
    $('casePane').classList.remove('open');
    document.body.classList.remove('sheet-open');
    sheetPushed = false;
  }
});
$('caseClose').addEventListener('click', () => closeSheet());
$('casePrev').addEventListener('click', () => stepSelection(-1, false));
$('caseNext').addEventListener('click', () => stepSelection(1, false));

function showCaseShell(on) {
  $('caseEmpty').hidden = on;
  $('caseBody').hidden = !on;
  $('caseNav').hidden = !on;
  $('decisionBar').hidden = !on;
}

async function openCase(id, opts = {}) {
  state.selectedId = id;
  const token = ++state.caseToken;
  markSelected();
  showCaseShell(true);
  openSheet();
  $('caseBody').innerHTML =
    '<div class="hero"><span class="skel" style="width:40%;height:2.6rem"></span><span class="skel" style="margin-top:1rem"></span></div>' +
    '<span class="skel" style="width:90%;margin-bottom:.6rem"></span><span class="skel" style="width:75%;margin-bottom:.6rem"></span><span class="skel" style="width:85%"></span>';
  $('caseScroll').scrollTop = 0;
  syncDecisionUI();
  if (opts.focusRow) {
    const row = $('queueBody').querySelector('tr.sel');
    if (row) { row.focus({ preventScroll: true }); row.scrollIntoView({ block: 'nearest' }); }
  }
  try {
    const c = await api(`/case/${encodeURIComponent(id)}?lang=${state.lang}`);
    if (token !== state.caseToken) return; // a newer request superseded this one
    state.lastCase = c;
    renderCase(c);
  } catch (err) {
    if (token !== state.caseToken) return;
    $('caseBody').innerHTML = `<div class="err-inline">Case failed to load.<br/><span class="small">${esc(err.message)}</span><br/>
      <button type="button" class="btn" id="retryCase">Try again</button></div>`;
    $('retryCase').addEventListener('click', () => openCase(id));
    toast(`Case failed: ${err.message}`, 'error');
  }
}

function meter(name, v) {
  const pct = (clamp01(v) * 100).toFixed(0);
  return `<div class="signal"><span class="name">${esc(name)}</span>
    <div class="meter" role="img" aria-label="${esc(name)} ${pct} percent"><i style="width:${pct}%"></i></div>
    <span class="val">${num(v).toFixed(2)}</span></div>`;
}

function scoreHero(c, action) {
  const lvl = safeLevel(c.risk_level);
  const pct = (clamp01(c.risk_score) * 100).toFixed(1);
  return `<div class="hero ${lvl}">
    <div class="hero-top">
      <div class="score"><span class="big">${num(c.risk_score).toFixed(2)}</span><span class="max">/1.00</span></div>
      <span class="badge ${lvl}">${esc(c.risk_level)} risk</span>
    </div>
    <div class="band" aria-hidden="true">
      <div class="zone z-allow"></div><div class="zone z-review"></div><div class="zone z-hold"></div>
      <div class="marker ${lvl}" style="left:${pct}%"></div>
    </div>
    <div class="band-labels" aria-hidden="true"><span>allow</span><span>review</span><span>hold + step-up</span></div>
    ${action ? `<div class="next-step"><b>What next</b><span>${esc(action)}</span></div>` : ''}
  </div>`;
}

function renderCase(c) {
  const comp = c.components || { p_fraud: c.p_fraud, anomaly: c.anomaly, graph_boost: c.graph_boost };
  const action = c.recommended_action ||
    (state.queue.find((r) => r.txn_id === c.txn_id) || {}).recommended_action;
  const signals = [
    ['P(fraud)', comp.p_fraud ?? c.p_fraud],
    ['Anomaly', comp.anomaly ?? c.anomaly],
    ['Graph boost', comp.graph_boost ?? c.graph_boost],
  ].filter(([, v]) => v !== undefined && v !== null);
  const extras = [
    ['Fraud neighbors (2-hop)', c.fraud_neighbors_2hop],
    ['Latency', comp.latency_ms !== undefined ? `${comp.latency_ms} ms` : undefined],
  ].filter(([, v]) => v !== undefined && v !== null);
  const sample = c.fraud_neighbor_sample || [];
  const hits = num(c.fraud_neighbors_2hop);
  const reasons = c.top_3_reasons || [];
  const timeline = c.timeline || [];

  $('caseBody').innerHTML = `
    <div class="case-id"><code>${esc(c.txn_id)}</code>
      <button type="button" class="btn copy-btn" id="copyId" aria-label="Copy transaction ID">Copy ID</button></div>
    ${scoreHero(c, action)}
    <dl class="facts">
      <dt>Route</dt><dd><code>${esc(c.sender_id)} → ${esc(c.receiver_id)}</code></dd>
      <dt>Amount</dt><dd><strong>${money(c.amount)}</strong></dd>
      <dt>Channel</dt><dd>${esc(c.channel)}</dd>
      <dt>Time</dt><dd>${esc(c.timestamp)}</dd>
    </dl>

    <section class="sect"><h3>What happened</h3>
      <p class="narrative">${esc(c.narrative || '(no narrative)')}</p></section>

    <section class="sect"><h3>Why it is risky</h3>
      ${reasons.length ? `<ol class="reasons">${reasons.map((x) => `<li>${esc(x)}</li>`).join('')}</ol>` : '<p class="muted small">No reasons were returned for this case.</p>'}</section>

    <section class="sect"><h3>Signals</h3>
      <div class="signals">${signals.map(([k, v]) => meter(k, v)).join('') || '<p class="muted small">No component scores.</p>'}</div>
      ${extras.length ? `<div class="chips-plain">${extras.map(([k, v]) => `<span class="stat"><b>${esc(k)}</b>${esc(v)}</span>`).join('')}</div>` : ''}</section>

    <section class="sect"><h3>Mule-ring proximity</h3>
      <div class="graph-box${hits ? ' alert' : ''}">${hits
        ? `<code>${esc(c.receiver_id)}</code> sits within 2 hops of <strong>${esc(hits)}</strong> known fraud wallet${hits === 1 ? '' : 's'}.
           <div class="wallets">${sample.length ? sample.map((n) => `<code>${esc(n)}</code>`).join('') : '<span class="muted">no sample available</span>'}</div>`
        : '<span class="muted">No fraud proximity. The receiver is clean in the 2-hop graph.</span>'}</div></section>

    <section class="sect"><h3>Wallet timeline <span class="muted" style="text-transform:none;letter-spacing:0;font-weight:400">· last 10 sends + receives</span></h3>
      ${timeline.length ? `<ul class="timeline">${timeline.map((t) =>
        `<li><div class="tl-main"><code>${esc(t.sender)} → ${esc(t.receiver)}</code><span class="tl-amt">${money(t.amount)}</span></div>
         <div class="tl-sub">${esc(t.timestamp)} · ${esc(t.location)}</div></li>`).join('')}</ul>`
        : '<p class="muted small">No history for this wallet.</p>'}</section>`;

  $('copyId').addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(c.txn_id); toast('Transaction ID copied', 'ok', 1800); }
    catch (e) { toast('Copy is not available in this browser', 'error'); }
  });
  syncDecisionUI();
}

/* ---------- Decisions ---------- */
let freezeArmed = null;
function disarmFreeze() {
  clearTimeout(freezeArmed);
  freezeArmed = null;
  const b = document.querySelector('.dec.freeze');
  if (b) { b.classList.remove('confirm'); b.textContent = 'Freeze + review'; }
}
function syncDecisionUI() {
  disarmFreeze();
  const d = state.decided[state.selectedId];
  $('decStatus').textContent = d ? `Logged: ${d}` : 'Not yet decided';
  document.querySelectorAll('.decision-btns button').forEach((b) =>
    b.classList.toggle('chosen', b.dataset.dec === d));
}

async function sendDecision(decision) {
  const id = state.selectedId;
  if (!id) return;
  if (decision === 'freeze' && !freezeArmed) {
    const b = document.querySelector('.dec.freeze');
    b.classList.add('confirm');
    b.textContent = 'Confirm freeze?';
    freezeArmed = setTimeout(disarmFreeze, 3500);
    return;
  }
  disarmFreeze();
  const btns = document.querySelectorAll('.decision-btns button');
  btns.forEach((b) => (b.disabled = true));
  try {
    await api('/decision', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ txn_id: id, decision, analyst: 'analyst-1' }),
    });
    const rowsBefore = visibleRows();
    const idx = rowsBefore.findIndex((r) => r.txn_id === id);
    state.decided[id] = decision;
    toast(`${id}: ${decision} logged`, 'ok', 2200);
    syncDecisionUI();
    renderQueue();
    if ($('autoAdvance').checked) {
      const rows = visibleRows();
      const next = rows.find((r) => !state.decided[r.txn_id] && rowsBefore.slice(idx + 1).some((x) => x.txn_id === r.txn_id)) ||
        rows.find((r) => !state.decided[r.txn_id]);
      if (next) openCase(next.txn_id);
    }
  } catch (err) {
    toast(`Decision failed: ${err.message}`, 'error');
  } finally {
    btns.forEach((b) => (b.disabled = false));
  }
}
document.querySelectorAll('.decision-btns button').forEach((b) =>
  b.addEventListener('click', () => sendDecision(b.dataset.dec)));

/* ---------- Score dialog ---------- */
const dlg = $('scoreDialog');
const PRESETS = {
  suspicious: { sender_id: 'C000001', receiver_id: 'C000002', amount: 45000, type: 'P2P', channel: 'app', location: 'Dhaka', device_id: 'DX999', timestamp: '2026-08-15T23:10:00', pwd: true },
  typical: { sender_id: 'C000003', receiver_id: 'C000004', amount: 850, type: 'merchant', channel: 'app', location: 'Dhaka', device_id: '', timestamp: '2026-08-15T14:20:00', pwd: false },
};
function applyPreset(name) {
  const p = PRESETS[name];
  if (!p) return;
  const f = $('scoreForm');
  Object.entries(p).forEach(([k, v]) => {
    if (k === 'pwd') f.elements.pwd_reset.checked = v;
    else if (f.elements[k]) f.elements[k].value = v;
  });
}
function openScore() {
  if (typeof dlg.showModal === 'function') dlg.showModal(); else dlg.setAttribute('open', '');
  const first = $('scoreForm').elements.sender_id;
  if (first) first.focus();
}
function closeScore() { if (dlg.open) dlg.close(); }
$('openScore').addEventListener('click', openScore);
$('scoreClose').addEventListener('click', closeScore);
$('scoreCancel').addEventListener('click', closeScore);
dlg.addEventListener('click', (e) => { if (e.target === dlg) closeScore(); });
document.querySelectorAll('[data-preset]').forEach((b) =>
  b.addEventListener('click', () => applyPreset(b.dataset.preset)));

function showFormError(msg) {
  const el = $('formError');
  el.hidden = !msg;
  el.textContent = msg || '';
}

$('scoreForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  showFormError('');
  const f = e.target;
  const fd = new FormData(f);
  const amount = Number(fd.get('amount'));
  let ts = String(fd.get('timestamp') || '');
  if (ts.length === 16) ts += ':00'; // datetime-local omits seconds
  const problems = [];
  if (!String(fd.get('sender_id')).trim()) problems.push('Sender wallet is required.');
  if (!String(fd.get('receiver_id')).trim()) problems.push('Receiver wallet is required.');
  if (!(amount > 0)) problems.push('Amount must be greater than zero.');
  if (!ts) problems.push('Time is required.');
  ['sender_id', 'receiver_id', 'amount', 'timestamp'].forEach((n) =>
    f.elements[n].classList.toggle('invalid', problems.some((p) => p.toLowerCase().startsWith(n.slice(0, 4)))));
  if (problems.length) { showFormError(problems.join(' ')); return; }

  const body = {
    sender_id: String(fd.get('sender_id')).trim(), receiver_id: String(fd.get('receiver_id')).trim(),
    amount, channel: fd.get('channel'),
    device_id: fd.get('device_id'), location: fd.get('location'),
    timestamp: ts, type: fd.get('type'), lang: state.lang,
    password_reset_flag: fd.get('pwd_reset') ? 1 : 0,
  };
  const btn = $('scoreSubmit');
  btn.disabled = true;
  btn.textContent = 'Scoring…';
  try {
    const out = await api('/score', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    renderScoreResult(out);
  } catch (err) {
    showFormError(`Scoring failed: ${err.message}`);
    toast(`Score failed: ${err.message}`, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Score transaction';
  }
});

function renderScoreResult(o) {
  const comp = o.components || {};
  const signals = [
    ['P(fraud)', comp.p_fraud ?? o.p_fraud],
    ['Anomaly', comp.anomaly ?? o.anomaly],
    ['Graph boost', comp.graph_boost ?? o.graph_boost],
  ].filter(([, v]) => v !== undefined && v !== null);
  const reasons = o.top_3_reasons || o.reasons || [];
  const hasScore = o.risk_score !== undefined;
  $('scoreEmpty').hidden = true;
  $('scoreOutBox').hidden = false;
  $('scoreOutBox').innerHTML = `
    ${hasScore ? scoreHero(o, o.recommended_action) : ''}
    ${o.narrative ? `<section class="sect" style="margin-top:0"><h3>Investigator note</h3><p class="narrative">${esc(o.narrative)}</p></section>` : ''}
    ${reasons.length ? `<section class="sect"><h3>Why</h3><ol class="reasons">${reasons.map((x) => `<li>${esc(x)}</li>`).join('')}</ol></section>` : ''}
    ${signals.length ? `<section class="sect"><h3>Signals</h3><div class="signals">${signals.map(([k, v]) => meter(k, v)).join('')}</div></section>` : ''}
    <details class="raw"><summary>Raw response</summary><pre>${esc(JSON.stringify(o, null, 2))}</pre></details>`;
}

/* ---------- Guided tour ---------- */
const TOUR_KEY = 'vigil_tour_seen_v1';
const TOUR_STEPS = [
  { title: 'Welcome to the Vigil console',
    body: 'Analysts review risky transfers here. This short tour covers the three jobs: review the queue, decide on a case, and test a transfer.' },
  { title: 'See the picture at a glance', sel: '#kpis',
    body: 'These tiles summarize the queue: how many alerts, how many are high or medium, the money at stake, and your progress this session.' },
  { title: 'Find the right alerts', sel: '#queueTools',
    body: 'Filter by level, search by wallet or transaction, sort, or hide what you have already reviewed. Counts update as you work.' },
  { title: 'Review the queue', sel: '#queueTable',
    body: 'Rows are scored by the model, riskiest first. Click one, or use J and K to move and Enter to open.' },
  { title: 'Investigate a case', sel: '#casePane', ensureCase: true,
    body: 'The score sits on the policy band, followed by what happened, why it is risky, the signals, network proximity, and wallet history.' },
  { title: 'Decide', sel: '#decisionBar', ensureCase: true,
    body: 'Allow it, step it up, or freeze it. Freeze asks for a second click. Keys A, S and F work too. Decisions are logged for retraining.' },
  { title: 'Test a transaction', sel: '#openScore',
    body: 'Score any transfer on demand. Try the password-reset checkbox to watch an account-takeover signal fire.' },
  { title: 'You are set',
    body: 'High risk never blocks money on its own. Replay this tour anytime with the ? button.' },
];
let tourIndex = 0;
let tourAutoTried = false;
let tourTarget = null;
const tourOpen = () => !$('tourOverlay').hidden;

function tourSeen() { try { return !!localStorage.getItem(TOUR_KEY); } catch (e) { return true; } }
function markTourSeen() { try { localStorage.setItem(TOUR_KEY, '1'); } catch (e) {} }
function clearSpot() { document.querySelectorAll('.tour-spot').forEach((el) => el.classList.remove('tour-spot')); }

function placeCard(target) {
  tourTarget = target || null;
  const card = $('tourCard');
  card.classList.toggle('center', !target);
  card.style.left = '';
  card.style.top = '';
  if (!target) return;
  target.scrollIntoView({ block: 'nearest' });
  const m = 12;
  const r = target.getBoundingClientRect();
  const cw = card.offsetWidth || 340;
  const ch = card.offsetHeight || 280;
  const left = Math.min(Math.max(r.left, m), Math.max(m, window.innerWidth - cw - m));
  let top = r.bottom + m;
  if (top + ch > window.innerHeight - m) top = r.top - ch - m;
  if (top < m) top = Math.max(m, window.innerHeight - ch - m); // target is tall: float the card over its lower edge
  top = Math.max(m, Math.min(top, window.innerHeight - ch - m));
  card.style.left = left + 'px';
  card.style.top = top + 'px';
}
window.addEventListener('resize', () => { if (tourOpen()) placeCard(tourTarget); });

async function showStep(i) {
  tourIndex = i;
  const step = TOUR_STEPS[i];
  clearSpot();
  let target = step.sel ? document.querySelector(step.sel) : null;
  let body = step.body;
  if (!step.ensureCase && $('casePane').classList.contains('open')) closeSheet();
  if (step.ensureCase) {
    if (!state.queue.length) {
      body = 'No alerts are queued right now. Come back when the queue has rows, then click one to open its case.';
      target = null;
    } else if (!state.selectedId) {
      const first = visibleRows()[0] || state.queue[0];
      if (first) await openCase(first.txn_id);
    }
  }
  if (target) target.classList.add('tour-spot');
  $('tourTitle').textContent = step.title;
  $('tourBody').textContent = body;
  $('tourStep').textContent = `Step ${i + 1} of ${TOUR_STEPS.length}`;
  $('tourDots').innerHTML = TOUR_STEPS.map((_, n) => `<span class="${n === i ? 'on' : ''}"></span>`).join('');
  $('tourNext').textContent = i === TOUR_STEPS.length - 1 ? 'Finish' : 'Next';
  placeCard(target);
  $('tourNext').focus();
}
function startTour() {
  $('tourOverlay').hidden = false;
  $('tourCard').hidden = false;
  showStep(0);
}
function endTour() {
  $('tourOverlay').hidden = true;
  $('tourCard').hidden = true;
  clearSpot();
  markTourSeen();
  $('tourBtn').focus();
}
$('tourBtn').addEventListener('click', startTour);
$('tourNext').addEventListener('click', () => {
  if (tourIndex >= TOUR_STEPS.length - 1) endTour(); else showStep(tourIndex + 1);
});
$('tourSkip').addEventListener('click', endTour);

/* ---------- Keyboard shortcuts ---------- */
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    if (tourOpen()) { endTour(); return; }
    if (dlg.open) return; // native dialog handles Escape
    if (NARROW.matches && $('casePane').classList.contains('open')) closeSheet();
    return;
  }
  if (e.metaKey || e.ctrlKey || e.altKey || tourOpen() || dlg.open) return;
  const t = e.target;
  const typing = t && (t.tagName === 'INPUT' || t.tagName === 'SELECT' || t.tagName === 'TEXTAREA' || t.isContentEditable);
  if (typing) {
    if (e.key === 'Escape' && t.id === 'searchBox') t.blur();
    return;
  }
  switch (e.key) {
    case 'j': case 'ArrowDown':
      if (e.key === 'ArrowDown' && !t.closest('#queueBody')) return;
      e.preventDefault(); stepSelection(1); break;
    case 'k': case 'ArrowUp':
      if (e.key === 'ArrowUp' && !t.closest('#queueBody')) return;
      e.preventDefault(); stepSelection(-1); break;
    case '/': e.preventDefault(); $('searchBox').focus(); $('searchBox').select(); break;
    case 'r': loadQueue({ announce: true, silent: true }); break;
    case 't': e.preventDefault(); openScore(); break;
    case '?': startTour(); break;
    case 'a': if (state.selectedId) sendDecision('allow'); break;
    case 's': if (state.selectedId) sendDecision('step-up'); break;
    case 'f': if (state.selectedId) sendDecision('freeze'); break;
    default:
  }
});

/* ---------- Boot ---------- */
checkHealth();
loadQueue().then(() => {
  if (!tourAutoTried) {
    tourAutoTried = true;
    if (!tourSeen()) startTour();
  }
});

/* ---------- Auth (guest by default; sign-in attributes decisions) ---------- */
let me = null;
async function refreshMe() {
  let t = null;
  try { t = sessionStorage.getItem('vigil_token'); } catch (e) {}
  me = null;
  if (t) {
    try { me = (await api('/auth/me')).user; }
    catch (e) { try { sessionStorage.removeItem('vigil_token'); } catch (e2) {} }
  }
  const signed = !!me, admin = signed && me.role === 'admin';
  $('userChip').hidden = !signed;
  $('logoutBtn').hidden = !signed;
  $('signinBtn').hidden = signed;
  $('adminTab').hidden = !admin;
  if (signed) {
    $('userName').textContent = me.username;
    $('userRole').textContent = me.role;
  }
  if (!admin) { $('adminView').hidden = true; }
}
function showAuth(tab) {
  $('authView').hidden = false;
  $('tabLogin').setAttribute('aria-pressed', String(tab !== 'register'));
  $('tabRegister').setAttribute('aria-pressed', String(tab === 'register'));
  $('loginForm').hidden = tab === 'register';
  $('registerForm').hidden = tab !== 'register';
}
$('signinBtn').addEventListener('click', () => showAuth('login'));
$('authClose').addEventListener('click', () => { $('authView').hidden = true; });
$('tabLogin').addEventListener('click', () => showAuth('login'));
$('tabRegister').addEventListener('click', () => showAuth('register'));
$('logoutBtn').addEventListener('click', async () => {
  try { await api('/auth/logout', { method: 'POST' }); } catch (e) {}
  try { sessionStorage.removeItem('vigil_token'); } catch (e) {}
  await refreshMe();
  toast('Signed out — continuing as guest');
});
async function submitAuth(form, errEl, path, okMsg) {
  const fd = new FormData(form);
  try {
    const out = await api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: fd.get('username'), password: fd.get('password') }) });
    $(errEl).hidden = true;
    if (out.token) {
      try { sessionStorage.setItem('vigil_token', out.token); } catch (e) {}
      $('authView').hidden = true;
      await refreshMe();
      toast(okMsg);
    } else {
      $('pendingNote').hidden = false;
      toast('Registered — an admin must approve you first');
    }
  } catch (e) { const p = $(errEl); p.textContent = String(e.message || e); p.hidden = false; }
}
$('loginForm').addEventListener('submit', (e) => { e.preventDefault(); submitAuth(e.target, 'authError', '/auth/login', 'Signed in'); });
$('registerForm').addEventListener('submit', (e) => { e.preventDefault(); submitAuth(e.target, 'regError', '/auth/register', 'Registered'); });

/* ---------- Admin dashboard ---------- */
let retrainTimer = null;
function adminRow(cells) {
  const tr = document.createElement('tr');
  for (const c of cells) {
    const td = document.createElement('td');
    if (typeof c === 'string') td.textContent = c; else td.appendChild(c);
    tr.appendChild(td);
  }
  return tr;
}
function adminBtn(label, fn) {
  const b = document.createElement('button');
  b.type = 'button'; b.className = 'btn'; b.textContent = label;
  b.addEventListener('click', async () => { try { await fn(); await loadAdmin(); } catch (e) { toast(String(e.message || e), 'error'); } });
  return b;
}
async function loadAdmin() {
  $('adminView').hidden = false;
  const users = (await api('/admin/users')).users;
  const pend = users.filter((u) => u.status === 'pending');
  $('pendingCount').textContent = pend.length ? `(${pend.length})` : '';
  const pb = $('pendingBody'); pb.replaceChildren();
  for (const u of pend) {
    pb.appendChild(adminRow([u.username, (u.created_at || '').slice(0, 10), adminBtn('Approve', () =>
      api(`/admin/users/${u.id}/approve`, { method: 'POST' }))]));
  }
  const ub = $('usersBody'); ub.replaceChildren();
  for (const u of users) {
    const act = u.status === 'active' && u.username !== me.username
      ? adminBtn('Disable', () => api(`/admin/users/${u.id}/disable`, { method: 'POST' }))
      : (u.status === 'disabled' ? adminBtn('Enable', () => api(`/admin/users/${u.id}/enable`, { method: 'POST' })) : document.createTextNode('—'));
    ub.appendChild(adminRow([u.username, u.role, u.status, act]));
  }
  const revs = (await api('/admin/decisions?limit=200')).decisions;
  const rb = $('reviewsBody'); rb.replaceChildren();
  for (const r of revs.slice(0, 100)) {
    rb.appendChild(adminRow([r.txn_id, r.analyst, r.decision, (r.at || '').slice(0, 19).replace('T', ' ')]));
  }
  await refreshRetrain(true);
}
$('adminTab').addEventListener('click', async () => {
  try { await loadAdmin(); } catch (e) { toast(String(e.message || e), 'error'); }
});
$('consoleTab').addEventListener('click', () => { $('adminView').hidden = true; });

/* ---------- Admin retraining ---------- */
async function refreshRetrain(silent) {
  try {
    const s = await api('/admin/retrain/status');
    const el = $('retrainState');
    el.textContent = `state: ${s.state}`;
    const out = $('retrainOut');
    if (s.result) {
      out.hidden = false;
      out.textContent = JSON.stringify(s.result, null, 2);
      if (s.state === 'done' && !silent) {
        toast(`Retrain ${s.result.verdict || 'finished'} — see verdict below`, s.result.verdict === 'SHIP' ? '' : 'error');
      }
    }
    return s.state;
  } catch (e) { if (!silent) toast(String(e.message || e), 'error'); return 'error'; }
}
async function startRetrain(apply) {
  await api(`/admin/retrain?apply=${apply ? 'true' : 'false'}`, { method: 'POST' });
  $('retrainState').textContent = 'state: running…';
  if (retrainTimer) clearInterval(retrainTimer);
  retrainTimer = setInterval(async () => {
    const st = await refreshRetrain(false);
    if (st === 'done' || st === 'error') clearInterval(retrainTimer);
  }, 5000);
}
$('retrainBtn').addEventListener('click', () => startRetrain(false).catch((e) => toast(String(e.message || e), 'error')));
$('retrainApplyBtn').addEventListener('click', () => startRetrain(true).catch((e) => toast(String(e.message || e), 'error')));

refreshMe();
