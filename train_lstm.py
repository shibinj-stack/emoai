# ============================================================
#  KeyMood LSTM — train_lstm.py  (compatible with TF 2.x & 3.x)
#  Works with Python 3.10 on Windows
# ============================================================

import os
import sys
import numpy as np
import pandas as pd

# ── Keras import: works on ALL TF versions ────────────────────
try:
    import keras
    from keras.models   import Sequential
    from keras.layers   import LSTM, Bidirectional, Dense, Dropout, Input, BatchNormalization
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from keras.optimizers import Adam
    print(f"Using keras {keras.__version__} (standalone)")
except ImportError:
    try:
        from tensorflow import keras
        from tensorflow.keras.models   import Sequential
        from tensorflow.keras.layers   import LSTM, Bidirectional, Dense, Dropout, Input, BatchNormalization
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        from tensorflow.keras.optimizers import Adam
        print("Using tensorflow.keras")
    except ImportError:
        print("ERROR: Neither 'keras' nor 'tensorflow' found.")
        print("Run:  pip install tensorflow==2.12.0")
        sys.exit(1)

import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")

import matplotlib
matplotlib.use('Agg')   # non-interactive backend (safe on Windows)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection    import train_test_split
from sklearn.preprocessing      import StandardScaler, LabelEncoder
from sklearn.metrics            import classification_report, confusion_matrix

from feature_extractor import (
    FEATURE_NAMES, N_FEATURES, TIMESTEPS,
    N_CLASSES, EMOTION_LABELS, build_lstm_sequence
)

# ── Reproducibility ───────────────────────────────────────────
np.random.seed(42)
tf.random.set_seed(42)

os.makedirs('model', exist_ok=True)
os.makedirs('plots', exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════

def load_data(csv_path='data/keystroke_dataset.csv'):
    if not os.path.exists(csv_path):
        print("Dataset not found. Run first:  python generate_data.py")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"\nLoaded {len(df)} samples")
    print(df['emotion'].value_counts())

    X_raw = df[FEATURE_NAMES].values.astype(np.float32)

    le = LabelEncoder()
    le.fit(EMOTION_LABELS)
    y  = le.transform(df['emotion'])

    return X_raw, y, le


def scale_and_reshape(X_raw, scaler=None, fit=True):
    if fit:
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
    else:
        X_scaled = scaler.transform(X_raw)

    X_seq = np.array([
        build_lstm_sequence(X_scaled[i]) for i in range(len(X_scaled))
    ])
    return X_seq, scaler


# ═══════════════════════════════════════════════════════════════
# 2. BUILD MODEL
# ═══════════════════════════════════════════════════════════════

def build_model():
    """
    Bidirectional LSTM architecture:

      Input  (10 timesteps x 9 features)
        -> BiLSTM (64 units, return_sequences=True)
        -> Dropout(0.4)
        -> LSTM (32 units, return_sequences=False)
        -> Dropout(0.3)
        -> Dense(64, relu)
        -> BatchNormalization
        -> Dropout(0.2)
        -> Dense(4, softmax)   <- Happy / Calm / Sad / Stressed
    """
    model = Sequential([
        Input(shape=(TIMESTEPS, N_FEATURES)),

        Bidirectional(LSTM(
            64,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.1,
        )),
        Dropout(0.4),

        LSTM(
            32,
            return_sequences=False,
            dropout=0.2,
            recurrent_dropout=0.1,
        ),
        Dropout(0.3),

        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(N_CLASSES, activation='softmax'),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


# ═══════════════════════════════════════════════════════════════
# 3. TRAIN
# ═══════════════════════════════════════════════════════════════

def train(X_train, y_train, X_val, y_val):
    model = build_model()
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath='model/lstm_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
        ),
    ]

    # Boost calm class weight so LSTM is forced to learn it
    class_weight = {
        0: 0.6,   # happy    <- reduced
        1: 2.5,   # calm     <- boosted further
        2: 0.6,   # sad      <- reduced
        3: 1.0,   # stressed <- unchanged
    }

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )
    return model, history


# ═══════════════════════════════════════════════════════════════
# 4. EVALUATE & PLOT
# ═══════════════════════════════════════════════════════════════

def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('LSTM Training History', fontsize=14, fontweight='bold')

    axes[0].plot(history.history['accuracy'],     label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Val')
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['loss'],     label='Train')
    axes[1].plot(history.history['val_loss'], label='Val')
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/training_history.png', dpi=150)
    plt.close()
    print("Plot saved -> plots/training_history.png")


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=EMOTION_LABELS,
        yticklabels=EMOTION_LABELS,
    )
    plt.title('Confusion Matrix', fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix.png', dpi=150)
    plt.close()
    print("Plot saved -> plots/confusion_matrix.png")


def evaluate(model, X_test, y_test):
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy : {acc*100:.2f}%")
    print(f"Test Loss     : {loss:.4f}")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=EMOTION_LABELS))
    plot_confusion_matrix(y_test, y_pred)


def save_scaler(scaler):
    import pickle
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler saved -> model/scaler.pkl")


# ═══════════════════════════════════════════════════════════════
# 5. MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 50)
    print("  KeyMood — LSTM Training")
    print("=" * 50)

    X_raw, y, le = load_data()

    # 70 / 15 / 15 split
    X_temp, X_test_raw, y_temp, y_test = train_test_split(
        X_raw, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
    )

    print(f"\nTrain: {len(X_train_raw)} | Val: {len(X_val_raw)} | Test: {len(X_test_raw)}")

    X_train, scaler = scale_and_reshape(X_train_raw, fit=True)
    X_val,   _      = scale_and_reshape(X_val_raw,   scaler=scaler, fit=False)
    X_test,  _      = scale_and_reshape(X_test_raw,  scaler=scaler, fit=False)

    print(f"Input shape: {X_train.shape}  (samples, timesteps, features)")

    model, history = train(X_train, y_train, X_val, y_val)
    plot_history(history)
    evaluate(model, X_test, y_test)
    save_scaler(scaler)

    print("\nDone!")
    print("  Model  -> model/lstm_model.keras")
    print("  Scaler -> model/scaler.pkl")
    print("  Plots  -> plots/")
    print("\nNext: python app.py")