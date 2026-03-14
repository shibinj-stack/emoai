# ============================================================
#  KeyMood LSTM — app.py
#
#  Flask REST API backend.
#  Loads trained LSTM model + scaler, accepts keystroke
#  feature data from the frontend, returns emotion prediction.
#
#  Run:  python app.py
#  API:  POST http://localhost:5000/api/predict
# ============================================================

import os
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

import tensorflow as tf

from feature_extractor import (
    extract_features_from_raw,
    features_to_vector,
    build_lstm_sequence,
    FEATURE_NAMES,
    EMOTION_LABELS,
    TIMESTEPS,
)

app = Flask(__name__)
CORS(app)   # Allow all origins (frontend can call freely)

# ── Load model and scaler on startup ─────────────────────────
MODEL_PATH  = os.path.join('model', 'lstm_model.keras')
SCALER_PATH = os.path.join('model', 'scaler.pkl')

print("Loading LSTM model...")

if not os.path.exists(MODEL_PATH):
    print("ERROR: model/lstm_model.keras not found.")
    print("   Run: python train_lstm.py")
    exit(1)

if not os.path.exists(SCALER_PATH):
    print("ERROR: model/scaler.pkl not found.")
    print("   Run: python train_lstm.py")
    exit(1)

model  = tf.keras.models.load_model(MODEL_PATH)
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

print("Model loaded successfully")


# ═══════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    POST /api/predict

    Request JSON (from script.js):
    {
        "flightTimes":    [120, 95, 210, 88, ...],
        "dwellTimes":     [80, 110, 75, 95, ...],
        "wpm":            68.4,
        "pauseRate":      0.8,
        "backspaceRatio": 0.04,
        "totalKeys":      145
    }

    Response JSON:
    {
        "emotion":    "calm",
        "confidence": 74.3,
        "scores": {
            "happy":    12.1,
            "calm":     74.3,
            "sad":       8.2,
            "stressed":  5.4
        },
        "features": { ... }
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON body received'}), 400

    try:
        # Step 1: Extract 9 features from raw keystroke data
        features_dict = extract_features_from_raw(data)
        feature_vec   = features_to_vector(features_dict)    # shape (9,)

        # Step 2: Scale using training scaler
        feature_vec_scaled = scaler.transform(
            feature_vec.reshape(1, -1)
        ).flatten()

        # Step 3: Build LSTM sequence (10 x 9)
        sequence = build_lstm_sequence(feature_vec_scaled)   # (10, 9)
        X = sequence[np.newaxis, :, :]                       # (1, 10, 9)

        # Step 4: LSTM inference
        probabilities    = model.predict(X, verbose=0)[0]   # shape (4,)
        predicted_idx    = int(np.argmax(probabilities))
        dominant_emotion = EMOTION_LABELS[predicted_idx]
        confidence       = round(float(probabilities[predicted_idx]) * 100, 1)

        scores = {
            label: round(float(prob) * 100, 1)
            for label, prob in zip(EMOTION_LABELS, probabilities)
        }

        return jsonify({
            'emotion':    dominant_emotion,
            'confidence': confidence,
            'scores':     scores,
            'features':   {k: round(float(v), 3) for k, v in features_dict.items()},
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':       'ok',
        'model':        'LSTM Bidirectional',
        'emotions':     EMOTION_LABELS,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
