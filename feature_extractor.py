# ============================================================
#  KeyMood LSTM — feature_extractor.py  (realistic calm)
# ============================================================

import numpy as np

FEATURE_NAMES = [
    'wpm',
    'mean_flight',
    'std_flight',
    'cv_flight',
    'mean_dwell',
    'std_dwell',
    'backspace_ratio',
    'pause_rate',
    'burst_score',
]

N_FEATURES     = len(FEATURE_NAMES)
TIMESTEPS      = 10
EMOTION_LABELS = ['happy', 'calm', 'sad', 'stressed']
N_CLASSES      = len(EMOTION_LABELS)


def extract_features_from_raw(data: dict) -> dict:
    flight = np.array(data.get('flightTimes', [200.0] * 10), dtype=float)
    dwell  = np.array(data.get('dwellTimes',  [130.0] * 10), dtype=float)

    if len(flight) < 3:
        flight = np.pad(flight, (0, 3 - len(flight)), constant_values=180.0)
    if len(dwell) < 3:
        dwell  = np.pad(dwell,  (0, 3 - len(dwell)),  constant_values=130.0)

    mean_f = float(np.mean(flight))
    std_f  = float(np.std(flight))
    cv_f   = std_f / mean_f if mean_f > 0 else 0.0

    mean_d = float(np.mean(dwell))
    std_d  = float(np.std(dwell))

    wpm             = float(data.get('wpm',            45.0))
    backspace_ratio = float(data.get('backspaceRatio', 0.02))
    pause_rate      = float(data.get('pauseRate',      1.2))

    # Compute burst score directly from flightTimes
    # (ratio of gaps under 120ms — calm should be near zero)
    burst_score = float(np.mean(flight < 150)) if len(flight) > 0 else 0.1

    return {
        'wpm':             max(wpm,             0.0),
        'mean_flight':     max(mean_f,          0.0),
        'std_flight':      max(std_f,           0.0),
        'cv_flight':       max(cv_f,            0.0),
        'mean_dwell':      max(mean_d,          0.0),
        'std_dwell':       max(std_d,           0.0),
        'backspace_ratio': max(backspace_ratio, 0.0),
        'pause_rate':      max(pause_rate,      0.0),
        'burst_score':     max(burst_score,     0.0),
    }


def features_to_vector(features: dict) -> np.ndarray:
    return np.array([features[f] for f in FEATURE_NAMES], dtype=np.float32)


def build_lstm_sequence(feature_vec: np.ndarray, timesteps: int = TIMESTEPS) -> np.ndarray:
    seq   = np.tile(feature_vec, (timesteps, 1))
    noise = np.random.normal(0, 0.02, seq.shape)
    seq   = seq + seq * noise
    seq   = np.clip(seq, 0, None)
    return seq.astype(np.float32)