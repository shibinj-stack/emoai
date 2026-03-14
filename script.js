/* ============================================================
   KeyMood — script.js
   Detection logic:
     1. Keyword scan  → clear winner?  show result
     2. Mixed/none    → LSTM fallback  show result
   No UI changes — purely internal smart fallback logic.
   ============================================================ */

const API_URL = 'https://emoai.onrender.com/api/predict';

const EMOTIONS = {
  happy:    { color: '#f9c74f', emoji: '😊', label: 'Happy' },
  calm:     { color: '#4ecdc4', emoji: '😌', label: 'Calm' },
  sad:      { color: '#6895d4', emoji: '😔', label: 'Sad' },
  stressed: { color: '#f4504a', emoji: '😤', label: 'Stressed' },
};

const INSIGHTS = {
  happy:    `Your typing shows <strong>quick, rhythmic keystrokes</strong> with minimal hesitation — elevated speed and positive language are strong markers of a happy state.`,
  calm:     `Your typing exhibits <strong>steady, consistent rhythm</strong> — low variability and peaceful language patterns indicate a calm, focused mindset.`,
  sad:      `Your typing shows <strong>slower pace with frequent pauses</strong> — reduced keystroke frequency and low-energy language are behavioral markers of sadness.`,
  stressed: `Your typing reveals <strong>irregular rhythm and frequent corrections</strong> — high variability and anxious language indicate cognitive stress.`,
};

// ── State ─────────────────────────────────────────────────────
const state = {
  keyEvents: [], keyDownTimes: {}, flightTimes: [], dwellTimes: [],
  pauses: 0, backspaces: 0, startTime: null, lastKeyTime: null, lastKeyUpTime: null, totalKeys: 0,
  PAUSE_THRESHOLD: 800,
};

// ── Server health check ───────────────────────────────────────
window.addEventListener('load', () => {
  fetch('https://emoai.onrender.com/health')
    .then(r => r.json())
    .then(() => {
      document.getElementById('serverStatus').textContent = 'ML Engine Ready';
      const dot = document.getElementById('statusDot');
      if (dot) { dot.style.background = '#4ecdc4'; dot.style.boxShadow = '0 0 10px #4ecdc4'; dot.style.animation = 'pulse 2s ease-in-out infinite'; }
    })
    .catch(() => {
      document.getElementById('serverStatus').textContent = 'Server Offline';
      const dot = document.getElementById('statusDot');
      if (dot) { dot.style.background = '#f4504a'; dot.style.boxShadow = '0 0 8px #f4504a'; }
    });
});

// ── Event capture ─────────────────────────────────────────────
const textarea = document.getElementById('typingInput');

textarea.addEventListener('keydown', (e) => {
  const now = performance.now();
  if (!state.startTime) state.startTime = now;
  if (state.lastKeyTime && (now - state.lastKeyTime) > state.PAUSE_THRESHOLD) state.pauses++;
  state.lastKeyTime = now;
  state.totalKeys++;
  if (e.key === 'Backspace') state.backspaces++;
  if (state.lastKeyUpTime) {
    const flight = now - state.lastKeyUpTime;
    if (flight > 0 && flight < 2000) state.flightTimes.push(flight);
  }
  state.keyDownTimes[e.code] = now;
  state.keyEvents.push({ key: e.key, downTime: now, upTime: null });
  updateMetrics();
  addVizBar();
});

textarea.addEventListener('keyup', (e) => {
  const now = performance.now();
  const dt  = state.keyDownTimes[e.code];
  if (dt) {
    state.dwellTimes.push(now - dt);
    for (let i = state.keyEvents.length - 1; i >= 0; i--) {
      if (state.keyEvents[i].key === e.key && !state.keyEvents[i].upTime) {
        state.keyEvents[i].upTime = now; break;
      }
    }
    delete state.keyDownTimes[e.code];
  }
  state.lastKeyUpTime = now;
  updateMetrics();
});

// ── Visualizer ────────────────────────────────────────────────
function addVizBar() {
  const viz = document.getElementById('keystrokeViz');
  const bar = document.createElement('div');
  bar.className    = 'keystroke-bar';
  bar.style.height = (10 + Math.random() * 28) + 'px';
  const dwell = state.dwellTimes[state.dwellTimes.length - 1] || 100;
  bar.style.opacity = String(0.4 + Math.min(dwell / 200, 1) * 0.5);
  viz.appendChild(bar);
  if (viz.children.length > 80) viz.removeChild(viz.firstChild);
  viz.scrollLeft = viz.scrollWidth;
}

// ── Metrics ───────────────────────────────────────────────────
function updateMetrics() {
  document.getElementById('keyCount').textContent = state.totalKeys + ' keystrokes captured';

  if (state.startTime && state.totalKeys > 5) {
    const mins = (performance.now() - state.startTime) / 60000;
    document.getElementById('wpm').textContent =
      Math.min(Math.round((state.totalKeys / 5) / mins), 250);
  }
  if (state.dwellTimes.length > 2) {
    document.getElementById('dwellTime').textContent =
      Math.round(state.dwellTimes.reduce((a, b) => a + b, 0) / state.dwellTimes.length);
  }
  document.getElementById('pauseCount').textContent = state.pauses;

  if (state.flightTimes.length > 5) {
    const mean     = state.flightTimes.reduce((a, b) => a + b, 0) / state.flightTimes.length;
    const variance = state.flightTimes.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / state.flightTimes.length;
    const cv       = Math.sqrt(variance) / mean;
    document.getElementById('rhythmScore').textContent =
      Math.max(0, Math.min(100, Math.round(100 - cv * 50)));
  }

  document.getElementById('analyzeBtn').disabled = state.totalKeys < 10;
}

// ── Build LSTM payload ────────────────────────────────────────
function buildPayload() {
  const elapsed        = (performance.now() - state.startTime) / 60000;
  const wpm            = Math.min((state.totalKeys / 5) / elapsed, 250);
  const burstScore     = state.flightTimes.length > 0
    ? state.flightTimes.filter(t => t < 100).length / state.flightTimes.length : 0.3;

  return {
    flightTimes:    state.flightTimes.slice(-100),
    dwellTimes:     state.dwellTimes.slice(-100),
    wpm:            parseFloat(wpm.toFixed(2)),
    pauseRate:      parseFloat((state.pauses / Math.max(elapsed, 0.1)).toFixed(4)),
    backspaceRatio: parseFloat((state.backspaces / Math.max(state.totalKeys, 1)).toFixed(4)),
    burstScore:     parseFloat(burstScore.toFixed(4)),
    totalKeys:      state.totalKeys,
  };
}

// ═══════════════════════════════════════════════════════════════
//  MAIN ANALYZE — keyword first, LSTM fallback
// ═══════════════════════════════════════════════════════════════

async function analyzeEmotion() {
  const text    = textarea.value.trim();
  const overlay = document.getElementById('loadingOverlay');
  const loadTxt = document.getElementById('loadingText');

  // ── LAYER 1: Keyword detection ────────────────────────────
  if (text.length > 0) {
    const kw = detectByKeywords(text);

    if (kw.matched) {
      // Clear keyword winner → show result, skip LSTM entirely
      showResults({
        emotion:    kw.emotion,
        confidence: kw.confidence,
        scores:     kw.scores,
        features:   null,
      });
      return;
    }
    // Mixed or no keywords → fall through to LSTM below
  }

  // ── LAYER 2: LSTM fallback ────────────────────────────────
  overlay.classList.add('visible');
  if (loadTxt) loadTxt.textContent = 'Running LSTM analysis...';

  try {
    const res = await fetch(API_URL, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(buildPayload()),
    });
    if (!res.ok) throw new Error((await res.json()).error || 'Server error');
    const result = await res.json();
    overlay.classList.remove('visible');
    showResults(result);
  } catch (err) {
    overlay.classList.remove('visible');
    alert('Flask server not reachable.\nRun: python app.py\n\nError: ' + err.message);
  }
}

// ── Render results ────────────────────────────────────────────
function showResults(result) {
  const em = EMOTIONS[result.emotion];

  // Badge
  const badge = document.getElementById('emotionBadge');
  badge.textContent      = em.emoji;
  badge.style.background = em.color + '22';
  badge.style.border     = `1px solid ${em.color}44`;

  // Label + color
  const label = document.getElementById('emotionLabel');
  label.textContent = em.label;
  label.style.color = em.color;

  // Confidence bar
  const fill = document.getElementById('confidenceFill');
  fill.style.background = em.color;
  document.getElementById('confidencePct').textContent = result.confidence + '%';
  setTimeout(() => { fill.style.width = result.confidence + '%'; }, 50);

  // Breakdown bars
  const breakdown = document.getElementById('emotionBreakdown');
  breakdown.innerHTML = '';
  if (result.scores) {
    Object.entries(result.scores).sort((a, b) => b[1] - a[1]).forEach(([key, pct]) => {
      const e2       = EMOTIONS[key];
      const isActive = key === result.emotion;
      const row      = document.createElement('div');
      row.className   = 'emotion-row' + (isActive ? ' active' : '');
      row.style.color = isActive ? e2.color : 'var(--muted)';
      row.innerHTML = `
        <div class="emotion-row-top">
          <span class="emotion-name">${e2.emoji} ${e2.label}</span>
          <span class="emotion-pct">${pct}%</span>
        </div>
        <div class="bar-row">
          <div class="bar-inner" style="background:${e2.color};width:0%"></div>
        </div>`;
      breakdown.appendChild(row);
      setTimeout(() => { row.querySelector('.bar-inner').style.width = pct + '%'; }, 100);
    });
  }

  // Insight
  document.getElementById('insightText').innerHTML = INSIGHTS[result.emotion];

  // Feature panel (LSTM only — only shown when features returned from server)
  const fp = document.getElementById('featurePanel');
  if (fp) {
    if (result.features) {
      fp.innerHTML = '<div class="feature-panel-title">// Feature Vector sent to LSTM</div>' +
        Object.entries(result.features).map(([k, v]) =>
          `<div class="feature-row"><span class="fname">${k}</span><span class="fval">${v}</span></div>`
        ).join('');
    } else {
      fp.innerHTML = '';
    }
  }

  document.getElementById('resultPanel').classList.add('visible');
  document.getElementById('resultPanel').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── Reset ─────────────────────────────────────────────────────
function resetAll() {
  Object.assign(state, {
    keyEvents: [], keyDownTimes: {}, flightTimes: [], dwellTimes: [],
    pauses: 0, backspaces: 0, startTime: null, lastKeyTime: null, lastKeyUpTime: null, totalKeys: 0,
  });
  textarea.value = '';
  document.getElementById('keyCount').textContent    = '0 keystrokes captured';
  document.getElementById('wpm').textContent         = '—';
  document.getElementById('dwellTime').textContent   = '—';
  document.getElementById('pauseCount').textContent  = '0';
  document.getElementById('rhythmScore').textContent = '—';
  document.getElementById('keystrokeViz').innerHTML  = '';
  document.getElementById('analyzeBtn').disabled     = true;
  document.getElementById('resultPanel').classList.remove('visible');
  textarea.focus();
}
