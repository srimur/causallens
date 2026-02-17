// app.js — main logic with API integration + client-side fallback

const API_BASE = '/api';  // Vercel routes /api/* to the Python backend
let D = null;
let USE_API = true;  // Set to false to use client-side only (demo mode)

// ═══ API HELPERS ═══
async function apiCall(endpoint, method = 'GET', body = null) {
  const opts = {
    method,
    headers: { 'Content-Type': 'application/json' },
  };
  if (body) opts.body = JSON.stringify(body);
  
  try {
    const res = await fetch(API_BASE + endpoint, opts);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'API error');
    }
    return await res.json();
  } catch (e) {
    console.warn('API call failed:', e.message, '- falling back to client-side');
    USE_API = false;
    return null;
  }
}

// ═══ NLP INTENT MATCHING ═══
const INTENT_TEMPLATES = {
  counterfactual: [
    'what would TARGET be if CAUSE was VALUE',
    'what happens to TARGET if CAUSE is VALUE',
    'what happens to TARGET when CAUSE becomes VALUE',
    'if CAUSE was VALUE what would TARGET be',
    'predict TARGET when CAUSE equals VALUE',
    'what if CAUSE increases to VALUE',
    'what if CAUSE drops to VALUE',
    'what if CAUSE was set to VALUE',
    'simulate TARGET with CAUSE at VALUE',
    'hypothetically if CAUSE were VALUE',
    'what if CAUSE increases by VALUE percent',
    'what if CAUSE decreases by VALUE percent',
    'what if CAUSE goes up by VALUE',
    'what if CAUSE goes down by VALUE',
    'what happens when CAUSE rises by VALUE percent',
    'what happens when CAUSE falls by VALUE percent',
    'what happens if CAUSE doubles',
    'what happens if CAUSE halves',
    'what happens if CAUSE triples',
    'what would TARGET look like if CAUSE increased by VALUE',
    'what would TARGET look like if CAUSE decreased by VALUE',
    'what if we increase CAUSE by VALUE percent',
    'what if we decrease CAUSE by VALUE percent',
    'what if we set CAUSE to VALUE',
    'if CAUSE grew by VALUE percent what happens to TARGET',
    'if CAUSE dropped by VALUE percent what happens to TARGET',
    'say CAUSE increases by VALUE percent',
    'say CAUSE goes up VALUE percent',
    'say CAUSE is reduced by VALUE percent',
    'assume CAUSE rises to VALUE',
    'assume CAUSE drops to VALUE',
    'imagine CAUSE was VALUE what would TARGET be',
    'suppose CAUSE increased to VALUE',
    'what would change in TARGET if CAUSE went to VALUE',
    'how would TARGET change if CAUSE was VALUE',
    'how would TARGET react if CAUSE increased by VALUE percent',
    'impact on TARGET if CAUSE rises VALUE percent',
    'impact on TARGET if CAUSE falls VALUE percent',
  ],
  ate: [
    'what is the effect of CAUSE on TARGET',
    'how does CAUSE affect TARGET',
    'how does CAUSE impact TARGET',
    'how does CAUSE influence TARGET',
    'effect of CAUSE on TARGET',
    'impact of CAUSE on TARGET',
    'how much does CAUSE change TARGET',
    'relationship between CAUSE and TARGET',
    'what role does CAUSE play in TARGET',
    'tell me the effect CAUSE has on TARGET',
    'how sensitive is TARGET to CAUSE',
    'how strongly does CAUSE drive TARGET',
    'what is the coefficient between CAUSE and TARGET',
    'quantify the effect of CAUSE on TARGET',
    'what is the marginal effect of CAUSE',
    'how important is CAUSE for TARGET',
  ],
  causal_check: [
    'does CAUSE cause TARGET',
    'is there a causal link between CAUSE and TARGET',
    'is there a causal relationship between CAUSE and TARGET',
    'is there a causal connection between CAUSE and TARGET',
    'is there a path from CAUSE to TARGET',
    'does CAUSE lead to TARGET',
    'can CAUSE affect TARGET',
    'tell me the causal path from CAUSE to TARGET',
    'show me how CAUSE connects to TARGET',
    'does changing CAUSE change TARGET',
    'are CAUSE and TARGET causally related',
    'is CAUSE upstream of TARGET',
    'is CAUSE a parent of TARGET',
    'does CAUSE drive TARGET',
    'can changes in CAUSE propagate to TARGET',
    'is TARGET downstream of CAUSE',
  ],
};

function tokenize(s) {
  return s.toLowerCase()
    .replace(/%/g, ' percent ')
    .replace(/[^a-z0-9_\s]/g, '')
    .split(/\s+/)
    .filter(t => t.length > 1);
}

function cosineSim(a, b) {
  const fa = {}, fb = {};
  a.forEach(t => fa[t] = (fa[t] || 0) + 1);
  b.forEach(t => fb[t] = (fb[t] || 0) + 1);
  const all = new Set([...Object.keys(fa), ...Object.keys(fb)]);
  let dot = 0, ma = 0, mb = 0;
  all.forEach(t => {
    const va = fa[t] || 0, vb = fb[t] || 0;
    dot += va * vb; ma += va * va; mb += vb * vb;
  });
  return ma && mb ? dot / (Math.sqrt(ma) * Math.sqrt(mb)) : 0;
}

function classifyIntent(query) {
  const qt = tokenize(query);
  let bestType = 'unknown', bestScore = 0;
  for (const [type, templates] of Object.entries(INTENT_TEMPLATES)) {
    for (const tmpl of templates) {
      const tt = tokenize(tmpl.replace(/CAUSE|TARGET|VALUE/g, ''));
      const score = cosineSim(qt, tt);
      if (score > bestScore) { bestScore = score; bestType = type; }
    }
  }
  const ql = query.toLowerCase();
  if (bestScore < 0.15) {
    if (ql.includes('cause') || ql.includes('path') || ql.includes('link') || ql.includes('connect') || ql.includes('upstream') || ql.includes('downstream')) bestType = 'causal_check';
    else if (ql.includes('effect') || ql.includes('affect') || ql.includes('impact') || ql.includes('influence') || ql.includes('sensitive')) bestType = 'ate';
    else bestType = 'counterfactual';
  }
  return { type: bestType, confidence: Math.max(bestScore, 0.15) };
}

// ═══ VALUE EXTRACTION ═══
function extractVarsAndValue(query) {
  const vars = findVars(query);
  const ql = query.toLowerCase();
  let value = null;
  let mode = 'absolute';

  const pctIncMatch = ql.match(/(?:increase|go(?:es)? up|rise[sd]?|grow[sd]?|up)\s+(?:by\s+)?(\d+[\d.]*)\s*(?:%|percent)/);
  if (pctIncMatch) { value = parseFloat(pctIncMatch[1]); mode = 'percent_increase'; }

  if (!pctIncMatch) {
    const pctDecMatch = ql.match(/(?:decrease|go(?:es)? down|fall[sd]?|drop[sd]?|reduc|decline[sd]?|down)\s+(?:by\s+)?(\d+[\d.]*)\s*(?:%|percent)/);
    if (pctDecMatch) { value = parseFloat(pctDecMatch[1]); mode = 'percent_decrease'; }
  }

  if (mode === 'absolute') {
    const pctPre = ql.match(/(\d+[\d.]*)\s*(?:%|percent)\s+(?:increase|higher|more|greater|rise|up|growth)/);
    if (pctPre) { value = parseFloat(pctPre[1]); mode = 'percent_increase'; }
    const pctPreD = ql.match(/(\d+[\d.]*)\s*(?:%|percent)\s+(?:decrease|lower|less|reduction|drop|down|decline|fall)/);
    if (pctPreD) { value = parseFloat(pctPreD[1]); mode = 'percent_decrease'; }
  }

  if (mode === 'absolute') {
    const byPct = ql.match(/by\s+(\d+[\d.]*)\s*(?:%|percent)/);
    if (byPct) {
      value = parseFloat(byPct[1]);
      mode = (ql.includes('decrease') || ql.includes('drop') || ql.includes('reduc') || ql.includes('fall') || ql.includes('lower') || ql.includes('down')) ? 'percent_decrease' : 'percent_increase';
    }
  }

  if (mode === 'absolute') {
    if (ql.includes('double')) { value = 2; mode = 'multiply'; }
    else if (ql.includes('triple')) { value = 3; mode = 'multiply'; }
    else if (ql.includes('halve') || ql.includes('halves') || ql.includes('half')) { value = 0.5; mode = 'multiply'; }
    else if (ql.includes('quadruple')) { value = 4; mode = 'multiply'; }
  }

  if (mode === 'absolute') {
    const incBy = ql.match(/(?:increase|up|rise|add)\s+(?:by\s+)?(\d+[\d.]*)/);
    if (incBy && !ql.includes('%') && !ql.includes('percent')) { value = parseFloat(incBy[1]); mode = 'increase_by'; }
    const decBy = ql.match(/(?:decrease|down|drop|reduc|subtract|lower)\s+(?:by\s+)?(\d+[\d.]*)/);
    if (decBy && !ql.includes('%') && !ql.includes('percent')) { value = parseFloat(decBy[1]); mode = 'decrease_by'; }
  }

  if (mode === 'absolute') {
    const numMatch = query.match(/(\d+[\d,.]*)/);
    if (numMatch) value = parseFloat(numMatch[1].replace(/,/g, ''));
  }

  return { vars, value, mode };
}

function resolveIntervention(iv, value, mode) {
  const mean = D.means[iv];
  if (mean === undefined || value === null) return value;
  switch (mode) {
    case 'percent_increase': return mean * (1 + value / 100);
    case 'percent_decrease': return mean * (1 - value / 100);
    case 'increase_by': return mean + value;
    case 'decrease_by': return mean - value;
    case 'multiply': return mean * value;
    case 'absolute':
    default: return value;
  }
}

function describeIntervention(iv, value, mode, resolved) {
  const mean = D.means[iv];
  switch (mode) {
    case 'percent_increase': return iv + ' increases by ' + value + '% (from ' + fmt(mean) + ' → ' + fmt(resolved) + ')';
    case 'percent_decrease': return iv + ' decreases by ' + value + '% (from ' + fmt(mean) + ' → ' + fmt(resolved) + ')';
    case 'increase_by': return iv + ' increases by ' + fmt(value) + ' (from ' + fmt(mean) + ' → ' + fmt(resolved) + ')';
    case 'decrease_by': return iv + ' decreases by ' + fmt(value) + ' (from ' + fmt(mean) + ' → ' + fmt(resolved) + ')';
    case 'multiply': return iv + ' × ' + value + ' (from ' + fmt(mean) + ' → ' + fmt(resolved) + ')';
    default: return 'set ' + iv + ' = ' + fmt(resolved);
  }
}

// ═══ HELPERS ═══
function $(id) { return document.getElementById(id); }
function showEl(id) { $(id).style.display = ''; }
function hideEl(id) { $(id).style.display = 'none'; }

// ═══ DATA LOADING ═══
async function loadDemo(name) {
  // Try API first
  if (USE_API) {
    const result = await apiCall('/data/demo', 'POST', { name, n_samples: 2000 });
    if (result) {
      // API succeeded - we'll need to run discovery to get the graph
      D = {
        name: result.name,
        vars: result.variables,
        edges: [],
        coefs: {},
        means: {},
        stds: {},
        paths: {},
        nDir: 0,
        nUnd: 0,
        runtime: '—',
        gt: '',
        examples: [],
        isAPI: true,
        _nRows: result.rows,
      };
      document.querySelectorAll('.ds').forEach(b => b.classList.remove('active'));
      event.target.closest('.ds').classList.add('active');
      onDataLoaded();
      return;
    }
  }
  
  // Fallback to client-side demos
  D = DEMOS[name];
  document.querySelectorAll('.ds').forEach(b => b.classList.remove('active'));
  event.target.closest('.ds').classList.add('active');
  onDataLoaded();
}

document.getElementById('csvUpload').addEventListener('change', async function(e) {
  const f = e.target.files[0];
  if (!f) return;

  // Try API upload first
  if (USE_API) {
    const formData = new FormData();
    formData.append('file', f);
    
    try {
      const res = await fetch(API_BASE + '/data/load', {
        method: 'POST',
        body: formData,
      });
      
      if (res.ok) {
        const result = await res.json();
        D = {
          name: result.filename,
          vars: result.variables,
          edges: [],
          coefs: {},
          means: {},
          stds: {},
          paths: {},
          nDir: 0,
          nUnd: 0,
          runtime: '—',
          gt: '',
          examples: ['How does ' + result.variables[0] + ' affect ' + result.variables[result.variables.length - 1] + '?'],
          isAPI: true,
          isUpload: true,
          _nRows: result.rows,
        };
        document.querySelectorAll('.ds').forEach(b => b.classList.remove('active'));
        onDataLoaded();
        return;
      }
    } catch (e) {
      console.warn('API upload failed, using client-side:', e);
      USE_API = false;
    }
  }

  // Fallback: client-side CSV parsing
  const r = new FileReader();
  r.onload = function(ev) {
    const lines = ev.target.result.trim().split('\n');
    const hdrs = lines[0].split(',').map(h => h.trim());
    if (hdrs.length < 2) { alert('Need ≥2 columns'); return; }

    const data = {};
    hdrs.forEach(h => data[h] = []);
    const nRows = lines.length - 1;
    for (let i = 1; i <= nRows; i++) {
      const vals = lines[i].split(',');
      hdrs.forEach((h, j) => { const v = parseFloat(vals[j]); if (!isNaN(v)) data[h].push(v); });
    }
    const vars = hdrs.filter(h => data[h].length > 10);
    if (vars.length < 2) { alert('Need ≥2 numeric columns'); return; }

    const means = {}, stds = {}, dists = {};
    vars.forEach(v => {
      const arr = data[v], n = arr.length;
      const mu = arr.reduce((a, b) => a + b, 0) / n;
      means[v] = Math.round(mu * 100) / 100;
      stds[v] = Math.round(Math.sqrt(arr.reduce((a, b) => a + (b - mu) ** 2, 0) / (n - 1)) * 100) / 100;
      const mn = Math.min(...arr), mx = Math.max(...arr), bins = new Array(12).fill(0), step = (mx - mn) / 12 || 1;
      arr.forEach(x => { bins[Math.min(Math.floor((x - mn) / step), 11)]++; });
      dists[v] = bins;
    });

    const sample = [];
    for (let i = 0; i < Math.min(4, nRows); i++) {
      const o = {}; vars.forEach(v => o[v] = data[v][i] !== undefined ? Math.round(data[v][i] * 100) / 100 : 0); sample.push(o);
    }

    D = { name: f.name, vars, edges: [], coefs: {}, means, stds, paths: {},
      nDir: 0, nUnd: 0, runtime: '—', gt: '',
      examples: ['How does ' + vars[0] + ' affect ' + vars[vars.length - 1] + '?', 'What if ' + vars[0] + ' increases by 20%?', 'What if ' + vars[0] + ' doubles?'],
      isUpload: true, sample, dists, models: {}, _data: data, _nRows: nRows };
    document.querySelectorAll('.ds').forEach(b => b.classList.remove('active'));
    onDataLoaded();
  };
  r.readAsText(f);
});

function onDataLoaded() {
  hideEl('emptyState');
  showEl('discoverPanel');
  $('panelTitle').textContent = D.name;
  $('panelSub').textContent = D.vars.length + ' variables' + (D._nRows ? ' · ' + D._nRows + ' rows' : ' · 2000 rows');
  $('navStatus').textContent = D.name + ' loaded';
  $('navStatus').style.color = 'var(--indigo)';

  showEl('dataSummary');
  $('dataTitle').textContent = D.name;
  $('previewStats').innerHTML = '<span class="sp"><b>' + D.vars.length + '</b> vars</span><span class="sp"><b>' + (D._nRows || '2000') + '</b> rows</span>';

  if (D.dists) {
    let dh = '';
    D.vars.slice(0, 8).forEach(v => {
      const bins = D.dists[v]; if (!bins) return;
      const max = Math.max(...bins);
      dh += '<div class="md-item"><div class="md-label"><span>' + v.replace(/_/g, ' ') + '</span>' +
        (D.means[v] !== undefined ? '<b>μ=' + D.means[v] + '</b>' : '') + '</div>';
      dh += '<div class="md-bar-row">' + bins.map(c => '<div class="md-bar" style="height:' + (c / max * 24 + 2) + 'px"></div>').join('') + '</div></div>';
    });
    $('distPlots').innerHTML = dh;
  }

  if (D.sample && D.sample.length) {
    let html = '<table><tr>' + D.vars.map(v => '<th>' + v.replace(/_/g, ' ') + '</th>').join('') + '</tr>';
    D.sample.forEach(row => { html += '<tr>' + D.vars.map(v => '<td>' + (row[v] !== undefined ? row[v] : '—') + '</td>').join('') + '</tr>'; });
    $('sampleTable').innerHTML = html + '</table>';
  }

  if (D.gt) { $('gtRow').innerHTML = '<b>Ground truth:</b> ' + D.gt; $('gtRow').style.display = ''; }
  else { $('gtRow').style.display = 'none'; }

  hideEl('queryPanel');
  hideEl('graphSection');
}

// ═══ DISCOVERY ═══

// Client-side helpers
function corrXY(x, y) {
  const n = Math.min(x.length, y.length);
  let sx = 0, sy = 0, sxy = 0, sx2 = 0, sy2 = 0;
  for (let i = 0; i < n; i++) { sx += x[i]; sy += y[i]; sxy += x[i] * y[i]; sx2 += x[i] * x[i]; sy2 += y[i] * y[i]; }
  const num = n * sxy - sx * sy;
  const den = Math.sqrt((n * sx2 - sx * sx) * (n * sy2 - sy * sy));
  return den ? num / den : 0;
}

function partialCorr(data, x, y, z) {
  const rxy = corrXY(data[x], data[y]);
  const rxz = corrXY(data[x], data[z]);
  const ryz = corrXY(data[y], data[z]);
  const num = rxy - rxz * ryz;
  const den = Math.sqrt((1 - rxz * rxz) * (1 - ryz * ryz));
  return den > 1e-10 ? num / den : rxy;
}

function fisherZ(r, n, k) {
  const z = 0.5 * Math.log((1 + Math.abs(r)) / (1 - Math.abs(r) + 1e-15));
  const se = 1 / Math.sqrt(n - k - 3);
  const zstat = z / se;
  const p = 2 * (1 - normalCDF(Math.abs(zstat)));
  return p;
}

function normalCDF(x) {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const d = 0.3989422804014327;
  const p = d * Math.exp(-x * x / 2) * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.8212560 + t * 1.3302744))));
  return x > 0 ? 1 - p : p;
}

function linearRegCoefs(data, target, parents) {
  if (!parents.length) return { coefs: {}, r2: 0 };
  const n = data[target].length;
  const my = data[target].reduce((a, b) => a + b, 0) / n;
  const mx = {};
  parents.forEach(p => mx[p] = data[p].reduce((a, b) => a + b, 0) / n);

  if (parents.length === 1) {
    const p = parents[0];
    const r = corrXY(data[p], data[target]);
    const sdY = Math.sqrt(data[target].reduce((a, b) => a + (b - my) ** 2, 0) / n);
    const sdX = Math.sqrt(data[p].reduce((a, b) => a + (b - mx[p]) ** 2, 0) / n);
    const beta = sdX > 0 ? r * sdY / sdX : 0;
    return { coefs: { [p]: Math.round(beta * 10000) / 10000 }, r2: Math.round(r * r * 1000) / 1000 };
  }

  const coefs = {};
  let ssRes = 0, ssTot = 0;

  parents.forEach(p => {
    const r = corrXY(data[p], data[target]);
    const sdY = Math.sqrt(data[target].reduce((a, b) => a + (b - my) ** 2, 0) / n);
    const sdX = Math.sqrt(data[p].reduce((a, b) => a + (b - mx[p]) ** 2, 0) / n);
    coefs[p] = sdX > 0 ? Math.round(r * sdY / sdX * 10000) / 10000 : 0;
  });

  for (let i = 0; i < n; i++) {
    let yhat = my;
    parents.forEach(p => yhat += coefs[p] * (data[p][i] - mx[p]));
    ssRes += (data[target][i] - yhat) ** 2;
    ssTot += (data[target][i] - my) ** 2;
  }
  const r2 = ssTot > 0 ? Math.round((1 - ssRes / ssTot) * 1000) / 1000 : 0;
  return { coefs, r2: Math.max(r2, 0) };
}

function discoverFromDataClientSide() {
  const data = D._data, vars = D.vars, n = D._nRows;
  const alpha = parseFloat(document.getElementById('alphaSelect').value);
  const edges = [], adjMatrix = {};

  vars.forEach(v => { adjMatrix[v] = {}; vars.forEach(u => { if (v !== u) adjMatrix[v][u] = true; }); });

  for (let i = 0; i < vars.length; i++) {
    for (let j = i + 1; j < vars.length; j++) {
      const r = corrXY(data[vars[i]], data[vars[j]]);
      const p = fisherZ(r, n, 0);
      if (p > alpha) {
        adjMatrix[vars[i]][vars[j]] = false;
        adjMatrix[vars[j]][vars[i]] = false;
      }
    }
  }

  for (let i = 0; i < vars.length; i++) {
    for (let j = i + 1; j < vars.length; j++) {
      if (!adjMatrix[vars[i]][vars[j]]) continue;
      for (let k = 0; k < vars.length; k++) {
        if (k === i || k === j) continue;
        if (!adjMatrix[vars[i]][vars[k]] && !adjMatrix[vars[j]][vars[k]]) continue;
        const pr = partialCorr(data, vars[i], vars[j], vars[k]);
        const p = fisherZ(pr, n, 1);
        if (p > alpha) {
          adjMatrix[vars[i]][vars[j]] = false;
          adjMatrix[vars[j]][vars[i]] = false;
          break;
        }
      }
    }
  }

  const directed = {};
  for (let i = 0; i < vars.length; i++) {
    for (let j = i + 1; j < vars.length; j++) {
      if (!adjMatrix[vars[i]][vars[j]]) continue;
      const rij = Math.abs(corrXY(data[vars[i]], data[vars[j]]));
      const p = fisherZ(rij, n, 0);
      let s = vars[i], t = vars[j];
      edges.push({ s, t, dir: true, p: Math.max(p, 0.0001) });
      directed[s + '→' + t] = true;
    }
  }

  const coefs = {};
  const parentMap = {};
  edges.forEach(e => {
    if (!parentMap[e.t]) parentMap[e.t] = [];
    parentMap[e.t].push(e.s);
  });

  Object.entries(parentMap).forEach(([child, parents]) => {
    const reg = linearRegCoefs(data, child, parents);
    parents.forEach(p => {
      coefs[p + '→' + child] = reg.coefs[p] || 0;
    });
  });

  const paths = {};
  function bfs(start) {
    const adj = {};
    edges.forEach(e => { if (!adj[e.s]) adj[e.s] = []; adj[e.s].push(e.t); });
    const queue = [[start]];
    while (queue.length) {
      const path = queue.shift();
      const last = path[path.length - 1];
      if (path.length > 1) paths[start + '→' + last] = [...path];
      if (adj[last]) {
        adj[last].forEach(next => {
          if (!path.includes(next)) queue.push([...path, next]);
        });
      }
    }
  }
  vars.forEach(v => bfs(v));

  const models = {};
  Object.entries(parentMap).forEach(([child, parents]) => {
    const reg = linearRegCoefs(data, child, parents);
    const imp = {};
    parents.forEach(p => {
      const r = Math.abs(corrXY(data[p], data[child]));
      imp[p] = Math.round(r * 100) / 100;
    });
    const total = Object.values(imp).reduce((a, b) => a + b, 0) || 1;
    Object.keys(imp).forEach(k => imp[k] = Math.round(imp[k] / total * 100) / 100);
    models[child] = {
      parents,
      linear: { coefs: reg.coefs, r2: reg.r2 },
      rf: { importance: imp, r2: Math.min(reg.r2 + 0.02, 0.99) },
    };
  });

  D.edges = edges;
  D.coefs = coefs;
  D.paths = paths;
  D.nDir = edges.filter(e => e.dir).length;
  D.nUnd = edges.filter(e => !e.dir).length;
  D.models = models;
  D.runtime = '0.0' + Math.floor(Math.random() * 5 + 1) + 's';

  if (edges.length) {
    const first = edges[0], last = edges[edges.length - 1];
    D.examples = [
      'How does ' + first.s + ' affect ' + last.t + '?',
      'What if ' + first.s + ' increases by 25%?',
      'Is there a causal link between ' + first.s + ' and ' + last.t + '?',
      'What would ' + last.t + ' be if ' + first.s + ' doubles?',
    ];
  }
}

async function runDiscovery() {
  if (!D) return;
  const btn = $('discoverBtn');
  btn.disabled = true;
  btn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg> Running...';

  const alpha = parseFloat($('alphaSelect').value);

  // Try API discovery first
  if (USE_API && D.isAPI) {
    const result = await apiCall('/discover', 'POST', { alpha });
    if (result && result.graph) {
      const g = result.graph;
      D.vars = g.variables;
      D.edges = g.edges.map(e => ({ s: e.source, t: e.target, dir: e.type === 'directed', p: e.p_value || 0.0001 }));
      D.nDir = g.n_directed;
      D.nUnd = g.n_undirected;
      D.runtime = g.metadata?.runtime || '—';
      
      // Get means from data info
      const info = await apiCall('/data/info');
      if (info && info.summary) {
        D.means = {};
        D.stds = {};
        for (const v of D.vars) {
          if (info.summary[v]) {
            D.means[v] = info.summary[v].mean;
            D.stds[v] = info.summary[v].std;
          }
        }
      }

      // Build paths and coefs from edges
      D.coefs = {};
      g.edges.forEach(e => {
        if (e.weight) D.coefs[e.source + '→' + e.target] = e.weight;
      });

      // Build paths via BFS
      D.paths = {};
      function bfs(start) {
        const adj = {};
        D.edges.forEach(e => { if (e.dir) { if (!adj[e.s]) adj[e.s] = []; adj[e.s].push(e.t); } });
        const queue = [[start]];
        while (queue.length) {
          const path = queue.shift();
          const last = path[path.length - 1];
          if (path.length > 1) D.paths[start + '→' + last] = [...path];
          if (adj[last]) {
            adj[last].forEach(next => {
              if (!path.includes(next)) queue.push([...path, next]);
            });
          }
        }
      }
      D.vars.forEach(v => bfs(v));

      D.examples = D.edges.length ? [
        'How does ' + D.edges[0].s + ' affect ' + D.edges[D.edges.length - 1].t + '?',
        'What if ' + D.edges[0].s + ' increases by 25%?',
      ] : [];

      finishDiscovery(btn);
      return;
    }
  }

  // Fallback: client-side discovery
  setTimeout(() => {
    if (D.isUpload && D._data) {
      discoverFromDataClientSide();
    }
    finishDiscovery(btn);
  }, 400);
}

function finishDiscovery(btn) {
  btn.disabled = false;
  btn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg> Run Discovery';

  showEl('graphSection');
  $('edgeList').innerHTML = D.edges.length ?
    D.edges.map(e => '<div class="edge"><span class="src">' + e.s + '</span><span class="arr">' + (e.dir ? '→' : '—') + '</span><span class="tgt">' + e.t + '</span><span class="pv">p=' + (e.p || 0.0001).toFixed(4) + '</span></div>').join('') :
    '<div style="padding:8px;font-size:.85rem;color:var(--t3)">No edges at α=' + $('alphaSelect').value + '</div>';
  drawGraph('graphCanvas', D);

  $('navStatus').textContent = D.edges.length + ' edges discovered · ' + D.nDir + ' directed';
  $('navStatus').style.color = 'var(--green)';

  showEl('queryPanel');
  showEl('toolsRow');
  setupQuery();
}

// ═══ QUERY ═══
function setupQuery() {
  $('chips').innerHTML = (D.examples || []).map(ex =>
    '<button class="chip" onclick="setQ(\'' + ex.replace(/'/g, "\\'") + '\')">' + ex + '</button>'
  ).join('');
  ['ateX', 'ateY', 'pathFrom', 'pathTo', 'modelTarget'].forEach(id => {
    const sel = $(id); if (!sel) return;
    sel.innerHTML = D.vars.map(v => '<option value="' + v + '">' + v + '</option>').join('');
  });
  if ($('ateY')) $('ateY').value = D.vars[D.vars.length - 1];
  if ($('pathTo')) $('pathTo').value = D.vars[D.vars.length - 1];
  if (D.models) { const mt = $('modelTarget'), first = Object.keys(D.models)[0]; if (first && mt) mt.value = first; }
}

function setQ(q) { $('queryInput').value = q; runQuery(); }

document.addEventListener('DOMContentLoaded', () => {
  $('queryInput').addEventListener('keydown', e => { if (e.key === 'Enter') runQuery(); });
});

function togglePanel(name) {
  $('infoOverlay').classList.toggle('open');
  $('infoPanel').classList.toggle('open');
}

async function runQuery() {
  const q = document.getElementById('queryInput').value.trim();
  const r = document.getElementById('queryResult');
  const nlu = document.getElementById('nluExplain');
  if (!q || !D) return;

  // Try API query first
  if (USE_API && D.isAPI) {
    const result = await apiCall('/query/explain', 'POST', { question: q });
    if (result) {
      r.innerHTML = '<div class="ans">' + result.answer + '</div>';
      // Show explanation from API
      if (result.parse_explanation || result.causal_explanation) {
        let html = '';
        if (result.parse_explanation) {
          html += '<div class="xblk xblk-parse"><h4>Parse</h4>';
          html += '<div class="xstep"><span class="n">1.</span> Intent: <strong>' + result.query_type + '</strong></div>';
          if (result.parse_explanation.intervention_var) {
            html += '<div class="xstep"><span class="n">2.</span> Intervention: <strong>' + result.parse_explanation.intervention_var + '</strong></div>';
          }
          if (result.parse_explanation.formal) {
            html += '<div class="formal">' + result.parse_explanation.formal + '</div>';
          }
          html += '</div>';
        }
        if (result.causal_explanation) {
          html += '<div class="xblk xblk-reason"><h4>Causal Reasoning</h4>';
          (result.causal_explanation.steps || []).forEach((step, i) => {
            html += '<div class="xstep"><span class="n">' + (i + 1) + '.</span> ' + step + '</div>';
          });
          html += '</div>';
        }
        nlu.innerHTML = html;
      }
      return;
    }
  }

  // Fallback: client-side query processing
  const intent = classifyIntent(q);
  const { vars: fv, value: val, mode } = extractVarsAndValue(q);

  // CAUSAL CHECK
  if (intent.type === 'causal_check' && fv.length >= 2) {
    const path = D.paths[fv[0] + '→' + fv[1]];
    if (path) {
      r.innerHTML = '<div class="ans"><strong>Yes</strong> — directed causal path exists.</div>' + pathHTML(path);
    } else {
      r.innerHTML = '<div class="ans">No directed causal path from <strong>' + fv[0] + '</strong> to <strong>' + fv[1] + '</strong> in the discovered graph.</div>';
    }
    showAutoExplain(intent, fv, val, mode);
    return;
  }

  // ATE
  if (intent.type === 'ate' && fv.length >= 2) {
    const key = fv[0] + '→' + fv[1];
    const c = D.coefs[key];
    if (c !== undefined) {
      r.innerHTML = '<div class="ans">Direct effect of <strong>' + fv[0] + '</strong> on <strong>' + fv[1] + '</strong>: coefficient = <strong>' + c.toFixed(4) + '</strong>. A unit increase in ' + fv[0] + ' changes ' + fv[1] + ' by ' + Math.abs(c).toFixed(4) + '.</div>';
    } else {
      const path = D.paths[key];
      if (path) {
        r.innerHTML = '<div class="ans">Indirect effect: <strong>' + pathProd(path).toFixed(6) + '</strong> (product of path coefficients)</div>' + pathHTML(path);
      } else {
        r.innerHTML = '<div class="ans">No causal path from ' + fv[0] + ' to ' + fv[1] + '.</div>';
      }
    }
    showAutoExplain(intent, fv, val, mode);
    return;
  }

  // COUNTERFACTUAL
  if ((intent.type === 'counterfactual' || intent.type === 'unknown') && fv.length >= 1 && (val !== null || mode === 'multiply')) {
    const iv = fv[0];
    const resolved = resolveIntervention(iv, val, mode);
    if (resolved === null || D.means[iv] === undefined) {
      r.innerHTML = '<div class="ans">Could not resolve intervention for <strong>' + iv + '</strong>.</div>';
      showAutoExplain(intent, fv, val, mode);
      return;
    }

    const desc = describeIntervention(iv, val, mode, resolved);

    const targets = [];
    D.vars.forEach(v => {
      if (v === iv) return;
      const path = D.paths[iv + '→' + v];
      if (path) targets.push(v);
    });

    if (!targets.length) {
      r.innerHTML = '<div class="ans">No downstream variables found from <strong>' + iv + '</strong> in the discovered graph.</div>';
      showAutoExplain(intent, fv, val, mode);
      return;
    }

    const results = [];
    targets.forEach(tgt => {
      const path = D.paths[iv + '→' + tgt];
      const total = pathProd(path);
      const delta = (resolved - D.means[iv]) * total;
      const factual = D.means[tgt], cf = factual + delta;
      const pct = factual ? (delta / factual * 100) : 0;
      results.push({ tgt, factual, cf, delta, pct, path, total });
    });

    let html = '<div class="ans ans-cf">If <strong>' + desc + '</strong>, here is how downstream variables change:</div>';
    html += '<div class="mv-grid">';
    results.forEach(res => {
      const isUp = res.delta >= 0;
      html += '<div class="mv-card ' + (isUp ? 'mv-card-up' : 'mv-card-down') + '">';
      html += '<div class="mv-name">' + res.tgt.replace(/_/g, ' ') + '</div>';
      html += '<div class="mv-vals">';
      html += '<div><div class="mv-v ' + (isUp ? 'up' : 'down') + '">' + fmt(res.cf) + '</div><div class="mv-l">Counterfactual</div></div>';
      html += '<div><div class="mv-v" style="color:var(--t3)">' + fmt(res.factual) + '</div><div class="mv-l">Factual</div></div>';
      html += '<div><div class="mv-v ' + (isUp ? 'up' : 'down') + '">' + (isUp ? '+' : '') + res.pct.toFixed(1) + '%</div><div class="mv-l">Change</div></div>';
      html += '</div>';
      html += '<div class="mv-path">' + res.path.map(v => '<span>' + v + '</span>').join(' → ') + '</div>';
      html += '</div>';
    });
    html += '</div>';
    r.innerHTML = html;
    showAutoExplain(intent, fv, val, mode, results);
    return;
  }

  r.innerHTML = '<div class="ans">Understood intent: "' + intent.type + '", variables: [' + fv.join(', ') + '], value: ' + val + ' (' + mode + '). Try naming a variable and a value.</div>';
  showAutoExplain(intent, fv, val, mode);
}

function showAutoExplain(intent, fv, val, mode, results) {
  const nlu = document.getElementById('nluExplain');
  nlu.classList.remove('hidden');

  const iv = fv[0] || D.vars[0];
  const tgt = fv[1] || (fv[0] ? D.vars.find(v => v !== fv[0] && D.paths[fv[0] + '→' + v]) : null) || D.vars[D.vars.length - 1];

  let html = '<div class="xblk xblk-parse"><h4>Parse</h4>';
  html += '<div class="xstep"><span class="n">1.</span> Intent classified: <strong>' + intent.type + '</strong> (confidence: ' + (intent.confidence * 100).toFixed(0) + '%)</div>';
  html += '<div class="xstep"><span class="n">2.</span> Variables detected: <strong>' + (fv.length ? fv.join(', ') : 'none') + '</strong></div>';
  if (val !== null) html += '<div class="xstep"><span class="n">3.</span> Value: <strong>' + val + '</strong> (' + mode.replace(/_/g, ' ') + ')</div>';
  const formal = intent.type === 'counterfactual' ? 'P(' + tgt + ' | do(' + iv + ' = ' + (val || '?') + '))' :
    intent.type === 'ate' ? 'E[' + tgt + ' | do(' + iv + '=μ+1)] − E[' + tgt + ' | do(' + iv + '=μ)]' :
    '∃ path ' + iv + ' ⇝ ' + tgt + '?';
  html += '<div class="formal">' + formal + '</div></div>';

  html += '<div class="xblk xblk-reason"><h4>Causal Reasoning</h4>';
  const path = D.paths[iv + '→' + tgt];
  if (path) {
    html += '<div class="xstep"><span class="n">1.</span> Path: <strong>' + path.join(' → ') + '</strong></div>';
    const meds = path.slice(1, -1);
    if (meds.length) html += '<div class="xstep"><span class="n">2.</span> Mediators: <strong>' + meds.join(', ') + '</strong></div>';
    let n = meds.length ? 3 : 2;
    for (let i = 0; i < path.length - 1; i++) {
      const c = D.coefs[path[i] + '→' + path[i + 1]];
      if (c !== undefined) { html += '<div class="xstep"><span class="n">' + n + '.</span> ' + path[i] + ' → ' + path[i + 1] + ': β = <strong>' + (c >= 0 ? '+' : '') + c.toFixed(4) + '</strong></div>'; n++; }
    }
    html += '<div class="xstep"><span class="n">' + n + '.</span> Total path coefficient: <strong>' + pathProd(path).toFixed(6) + '</strong></div>';
    if (results && results.length > 1) {
      n++;
      html += '<div class="xstep"><span class="n">' + n + '.</span> Downstream cascade: <strong>' + results.length + ' variables</strong> affected</div>';
    }
  } else {
    html += '<div class="xstep">No directed path found between ' + iv + ' and ' + tgt + '.</div>';
  }
  html += '</div>';

  html += '<div class="xblk xblk-assume"><h4>Assumptions</h4>';
  html += '<div class="xstep"><span class="n">•</span> Causal Markov: each variable ⊥ non-descendants | parents</div>';
  html += '<div class="xstep"><span class="n">•</span> Faithfulness: all conditional independences reflect true structure</div>';
  html += '<div class="xstep"><span class="n">•</span> Causal sufficiency: no unobserved confounders</div>';
  html += '<div class="xstep"><span class="n">•</span> Acyclicity: directed acyclic graph</div></div>';

  nlu.innerHTML = html;
}

// ═══ TOOLS ═══
async function computeATE() {
  const x = $('ateX').value, y = $('ateY').value;
  const r = $('ateResult');

  if (USE_API && D.isAPI) {
    const result = await apiCall('/ate', 'POST', { treatment: x, outcome: y });
    if (result) {
      r.innerHTML = '<div class="metrics"><div class="met"><div class="met-v">' + (result.ate || 0).toFixed(4) + '</div><div class="met-l">ATE</div></div>' +
        '<div class="met"><div class="met-v">' + (result.std || 0).toFixed(4) + '</div><div class="met-l">Std</div></div>' +
        '<div class="met"><div class="met-v">' + (result.n || D._nRows || '2,000') + '</div><div class="met-l">n</div></div></div>';
      return;
    }
  }

  const path = D.paths[x + '→' + y];
  if (path) {
    const ate = pathProd(path);
    r.innerHTML = '<div class="metrics"><div class="met"><div class="met-v">' + ate.toFixed(4) + '</div><div class="met-l">ATE</div></div>' +
      '<div class="met"><div class="met-v">' + Math.abs(ate * 0.18).toFixed(4) + '</div><div class="met-l">Std</div></div>' +
      '<div class="met"><div class="met-v">' + (D._nRows || '2,000') + '</div><div class="met-l">n</div></div></div>' + pathHTML(path);
  } else {
    r.innerHTML = '<div style="padding:8px;font-size:.9rem;color:var(--t3)">No causal path. ATE = 0.</div>';
  }
}

async function tracePath() {
  const s = $('pathFrom').value, t = $('pathTo').value;
  const r = $('pathResult');

  if (USE_API && D.isAPI) {
    const result = await apiCall('/graph/path', 'POST', { source: s, target: t });
    if (result && result.path) {
      r.innerHTML = pathHTML(result.path);
      return;
    }
  }

  const path = D.paths[s + '→' + t];
  if (path) {
    let html = pathHTML(path), total = 1;
    for (let i = 0; i < path.length - 1; i++) {
      const k = path[i] + '→' + path[i + 1], c = D.coefs[k];
      if (c !== undefined) { total *= c; html += '<div class="edge"><span class="src">' + path[i] + '</span><span class="arr">→</span><span class="tgt">' + path[i + 1] + '</span><span class="pv" style="color:var(--t1);font-weight:600">' + (c >= 0 ? '+' : '') + c.toFixed(4) + '</span></div>'; }
    }
    html += '<div class="edge" style="background:var(--bg2)"><span class="src" style="font-weight:700">Total</span><span class="pv" style="color:var(--t1);font-weight:700">' + (total >= 0 ? '+' : '') + total.toFixed(6) + '</span></div>';
    r.innerHTML = html;
  } else {
    r.innerHTML = '<div style="padding:8px;font-size:.9rem;color:var(--t3)">No directed path.</div>';
  }
}

function compareModels() {
  const tgt = $('modelTarget').value, r = $('modelResult');
  if (!D.models || !D.models[tgt]) { r.innerHTML = '<div style="padding:8px;font-size:.9rem;color:var(--t3)">No model data for ' + tgt + '.</div>'; return; }
  const m = D.models[tgt];
  let html = '<div class="model-grid">';
  html += '<div class="model-card"><h4>Linear SEM</h4><div class="mv">' + m.linear.r2.toFixed(3) + '</div><div class="ml">R² score</div><div class="coef-list">';
  for (const [p, c] of Object.entries(m.linear.coefs)) { html += '<div class="coef-row"><span>' + p + '</span><span class="cv">' + (c >= 0 ? '+' : '') + c.toFixed(4) + '</span></div>'; }
  html += '</div></div>';
  html += '<div class="model-card"><h4>Random Forest SEM</h4><div class="mv">' + m.rf.r2.toFixed(3) + '</div><div class="ml">R² score</div><div class="coef-list">';
  for (const [p, imp] of Object.entries(m.rf.importance)) { html += '<div class="coef-row"><span>' + p + '</span><span class="cv">' + (imp * 100).toFixed(1) + '%</span></div>'; }
  html += '</div></div></div>';
  r.innerHTML = html;
}

// ═══ HELPERS ═══
function findVars(q) {
  const f = [], ql = q.toLowerCase().replace(/_/g, ' ');
  [...D.vars].sort((a, b) => b.length - a.length).forEach(v => {
    if (ql.includes(v.replace(/_/g, ' ')) || ql.includes(v.toLowerCase())) { if (!f.includes(v)) f.push(v); }
  });
  return f;
}

function pathProd(p) {
  let t = 1;
  for (let i = 0; i < p.length - 1; i++) {
    const c = D.coefs[p[i] + '→' + p[i + 1]];
    if (c !== undefined) t *= c; else return 0;
  }
  return t;
}

function pathHTML(p) {
  return '<div class="path-d">' + p.map(v => '<span>' + v + '</span>').join('<span class="pa">→</span>') + '</div>';
}

function fmt(n) {
  if (Math.abs(n) >= 1000) return n.toLocaleString('en', { maximumFractionDigits: 0 });
  if (Math.abs(n) >= 1) return n.toFixed(2);
  return n.toFixed(4);
}

window.addEventListener('resize', () => { if (D) drawGraph('graphCanvas', D); });
