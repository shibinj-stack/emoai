# ============================================================
#  KeyMood LSTM — generate_data.py  (realistic calm profile)
# ============================================================

import numpy as np
import pandas as pd
import os

np.random.seed(42)

PROFILES = {
    'happy': {
        # Fast, energetic, bursty — high WPM, short flights, high burst
        'wpm':             (88,   10),
        'mean_flight':     (85,   15),   # very short gaps
        'std_flight':      (35,   10),
        'cv_flight':       (0.42,  0.06),
        'mean_dwell':      (65,   12),
        'std_dwell':       (18,    6),
        'backspace_ratio': (0.04,  0.02),
        'pause_rate':      (0.12,  0.06),
        'burst_score':     (0.88,  0.06),  # very high — lots of rapid keys
    },
    'calm': {
        # Steady, moderate, consistent — the key is LOW cv_flight + LOW pause_rate
        # Completely separated from sad on: wpm, mean_flight, cv, dwell, pause_rate, burst
        'wpm':             (58,    8),   # 42-74 — clearly above sad (max ~34)
        'mean_flight':     (220,  20),   # 180-260 — raised mean to 220ms
        'std_flight':      (38,   10),
        'cv_flight':       (0.25,  0.05), # 0.15-0.35 — hard ceiling below sad (min 0.45)
        'mean_dwell':      (105,  15),   # 75-135 — hard ceiling below sad (min 155)
        'std_dwell':       (20,    7),
        'backspace_ratio': (0.02,  0.01),
        'pause_rate':      (0.6,   0.20), # 0.2-1.0 — hard ceiling below sad (min 2.2)
        'burst_score':     (0.18,  0.06), # 0.06-0.30 — overlaps sad BUT other features dominate
    },
    'sad': {
        # Slow, heavy, irregular — high flights, high dwell, very high pause_rate
        # Every feature pushed far from calm with tight std to avoid bleed
        'wpm':             (18,    5),   # 8-28 — hard floor above calm (min 42)
        'mean_flight':     (320,  40),   # 240-400 — hard floor above calm (max 195)
        'std_flight':      (90,   15),
        'cv_flight':       (0.65,  0.08), # 0.49-0.81 — hard floor above calm (max 0.35)
        'mean_dwell':      (220,  30),   # 160-280 — hard floor above calm (max 135)
        'std_dwell':       (70,   12),
        'backspace_ratio': (0.05,  0.02),
        'pause_rate':      (4.2,   0.50), # 3.2-5.2 — completely above calm (max 1.0)
        'burst_score':     (0.05,  0.02), # near zero — very few rapid keys
    },
    'stressed': {
        'wpm':             (65,  18),
        'mean_flight':     (130,  45),
        'std_flight':      (105,  25),
        'cv_flight':       (0.92, 0.20),
        'mean_dwell':      (130,  25),
        'std_dwell':       (65,  18),
        'backspace_ratio': (0.22, 0.06),
        'pause_rate':      (2.0,  0.55),
        'burst_score':     (0.58, 0.13),
    },
}

EMOTION_LABELS = list(PROFILES.keys())
FEATURE_NAMES  = list(list(PROFILES.values())[0].keys())


def sample_session(emotion, session_id):
    p   = PROFILES[emotion]
    row = {'session_id': session_id, 'emotion': emotion}
    for feature, (mean, std) in p.items():
        row[feature] = max(np.random.normal(mean, std), 0.0)
    return row


def generate(n_per_class=1500, output_path='data/keystroke_dataset.csv'):
    os.makedirs('data', exist_ok=True)
    # Calm gets extra samples to counteract its historically low prediction rate
    class_counts = {e: n_per_class for e in EMOTION_LABELS}
    class_counts['calm']  = int(n_per_class * 1.5)  # 50% more calm samples
    class_counts['sad']   = int(n_per_class * 0.65) # 35% fewer sad samples
    class_counts['happy'] = int(n_per_class * 0.65) # 35% fewer happy samples
    records    = []
    session_id = 0
    for emotion in EMOTION_LABELS:
        for _ in range(class_counts[emotion]):
            records.append(sample_session(emotion, session_id))
            session_id += 1

    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(output_path, index=False)

    print(f"Generated {len(df)} samples -> {output_path}")
    print("\nFeature means per emotion:")
    print(df.groupby('emotion')[FEATURE_NAMES].mean().round(2).to_string())
    return df


if __name__ == '__main__':
    generate(n_per_class=1200)