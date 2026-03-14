/* ============================================================
   KeyMood — keywords.js  (calm-fixed)
   - Removed calm words that overlap with happy
   - Added stronger calm-exclusive keywords
   - Negation, neither, smart fallback all preserved
   ============================================================ */

const KEYWORD_DICT = {

  happy: [
    'happy','happiness','joy','joyful','excited','excitement',
    'great','amazing','awesome','wonderful','fantastic','brilliant',
    'excellent','superb','love','loving','loved',
    'cheerful','delighted','elated','thrilled','ecstatic',
    'glad','pleased','grateful','thankful','blessed',
    'proud','confident','motivated','inspired','energetic',
    'fun','laugh','laughing','smile','smiling','grin',
    'celebrate','celebrating','celebration','winning','success',
    'perfect','beautiful','incredible','outstanding',
    'yay','woohoo','wow','haha','lol',
    'hope','hopeful','optimistic','positive','bright',
    'enjoy','enjoying','enjoyed','pleasure','pleasant',
  ],

  calm: [
    // Strong calm-exclusive words only — no overlap with happy
    'calm','calming','calmed',
    'peaceful','peace',
    'relaxed','relaxing','relax','relaxation',
    'serene','serenity',
    'tranquil','tranquility',
    'quiet','stillness','silence','silent',
    'breathe','breathing','breath',
    'meditate','meditating','meditation',
    'mindful','mindfulness',
    'gentle','slow','steady',
    'at ease','ease',
    'unwind','unwinding','unwound',
    'soothing','soothed','soothe',
    'mellow','undisturbed','undisturbing',
    'laid back','laid-back',
    'no rush','no stress','stress free','stress-free',
    'taking it slow','slowing down',
    'deep breath','slow breath',
    'nothing to worry','nothing to fear',
    'not worried','not anxious','not nervous',
    'not stressed','not panicking',
  ],

  sad: [
    'sad','sadness','unhappy','unhappiness',
    'depressed','depression','down','low','blue',
    'cry','crying','cried','tears','tearful','weeping',
    'grief','grieving','grieve','mourn','mourning',
    'lonely','loneliness','alone','isolated','empty',
    'hopeless','hopelessness','helpless','helplessness',
    'miss','missing','lost','loss','heartbroken','broken',
    'pain','painful','hurt','hurting','suffering',
    'tired','exhausted','drained','numb','lifeless',
    'regret','regretful','sorry','guilt','guilty',
    'disappointed','disappointment','fail','failed','failure',
    'useless','worthless','dark','gloomy','miserable',
  ],

  stressed: [
    'stress','stressed','stressful','stressing',
    'anxious','anxiety','nervous','nervousness','panic','panicking',
    'overwhelmed','overwhelm','overloaded',
    'worried','worry','worrying','fear','fearful','scared',
    'tense','tension','pressure','pressured',
    'deadline','deadlines','late','behind',
    'busy','rushed','rush','hurry','hurrying','urgent',
    'distracted','distraction','confused','confusion',
    'struggling','struggle','angry','anger','mad',
    'frustrated','frustration','annoyed','irritated','rage',
    'too much','not enough time','fed up','had enough',
    'headache','heart racing',
  ],

};

// ── Negation flip map ─────────────────────────────────────────
const NEGATION_FLIP = {
  happy:    'sad',
  calm:     'stressed',
  sad:      'happy',
  stressed: 'calm',
};

// ── Negation trigger words ────────────────────────────────────
const NEGATION_WORDS = [
  "not", "don't", "dont", "doesn't", "doesnt",
  "didn't", "didnt", "can't", "cant", "cannot",
  "never", "no", "nothing", "neither", "nor",
  "hardly", "barely", "scarcely", "without",
  "isn't", "isnt", "wasn't", "wasnt",
  "aren't", "arent", "weren't", "werent",
  "couldn't", "couldnt", "shouldn't", "shouldnt",
  "stop", "stopped", "less", "little",
];

// ── Neither / neutral phrases → LSTM ─────────────────────────
const NEITHER_PHRASES = [
  'neither', 'not sure', 'not really', 'i don\'t know',
  'i dont know', 'no idea', 'nothing in particular',
  'not anything', 'none of these', 'not any',
  'somewhere in between', 'in between', 'mixed feelings',
  'not sure how', 'hard to say', 'difficult to say',
  'can\'t tell', 'cant tell', 'not certain',
];

// ── Token scanner with negation window ───────────────────────
function scanTokens(text) {
  const lower  = text.toLowerCase().replace(/[^a-z\s']/g, ' ');
  const tokens = lower.split(/\s+/).filter(t => t.length > 0);
  const hits   = [];

  for (const emotion of Object.keys(KEYWORD_DICT)) {
    for (const keyword of KEYWORD_DICT[emotion]) {
      const isPhrase = keyword.includes(' ');

      if (isPhrase) {
        const idx = lower.indexOf(keyword);
        if (idx === -1) continue;
        const before  = lower.slice(0, idx).trim().split(/\s+/).slice(-5);
        const negated = before.some(w => NEGATION_WORDS.includes(w));
        hits.push({ keyword, emotion, negated, weight: 2 });
      } else {
        const pos = tokens.indexOf(keyword);
        if (pos === -1) continue;
        const before  = tokens.slice(Math.max(0, pos - 5), pos);
        const negated = before.some(w => NEGATION_WORDS.includes(w));
        hits.push({ keyword, emotion, negated, weight: 1 });
      }
    }
  }

  return hits;
}

// ── Main: detectByKeywords(text) ─────────────────────────────
function detectByKeywords(text) {
  const lower = text.toLowerCase();

  // Check neither/neutral first
  if (NEITHER_PHRASES.some(p => lower.includes(p))) {
    return { matched: false, reason: 'neither' };
  }

  const hits = scanTokens(text);
  if (hits.length === 0) return { matched: false, reason: 'no_keywords' };

  // Build effective scores with negation flipping
  const scores = { happy: 0, calm: 0, sad: 0, stressed: 0 };
  for (const hit of hits) {
    const target = hit.negated ? NEGATION_FLIP[hit.emotion] : hit.emotion;
    scores[target] += hit.weight;
  }

  const total = Object.values(scores).reduce((a, b) => a + b, 0);
  if (total === 0) return { matched: false, reason: 'no_keywords' };

  // Normalize
  const pct = {};
  Object.keys(scores).forEach(e => {
    pct[e] = parseFloat(((scores[e] / total) * 100).toFixed(1));
  });

  const sorted = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  const top    = sorted[0];
  const second = sorted[1];

  // Rule 1: only one emotion scored → clear winner
  if (second[1] === 0) {
    return {
      matched:      true,
      emotion:      top[0],
      confidence:   100,
      scores:       pct,
      matchedWords: hits.map(h => h.negated ? `not ${h.keyword}` : h.keyword),
    };
  }

  // Rule 2: top ≥ 2× second → dominant winner
  if (top[1] >= second[1] * 2) {
    return {
      matched:      true,
      emotion:      top[0],
      confidence:   parseFloat(pct[top[0]]),
      scores:       pct,
      matchedWords: hits.map(h => h.negated ? `not ${h.keyword}` : h.keyword),
    };
  }

  // Rule 3: mixed → LSTM fallback
  return { matched: false, reason: 'mixed' };
}