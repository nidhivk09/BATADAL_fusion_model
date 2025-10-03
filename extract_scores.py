import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from data_loader import load_test_data  # use the same loader as training

# ---------------------------
# Config
# ---------------------------
DATASET = "BATADAL"
WINDOW_CNN = 200          # <- your CNN "history" during training
WINDOW_DNN = 1            # <- your DNN "history" during training

MODEL_PATHS = {
    "autoencoder": "/Users/nidhikulkarni/ics-anomaly-detection/models/results/AE-BATADAL-l5-cf2.5.h5",
    "cnn":         "/Users/nidhikulkarni/ics-anomaly-detection/models/results/CNN-BATADAL-l8-hist200-kern3-units32.h5",
    "dnn":         "/Users/nidhikulkarni/ics-anomaly-detection/models/results/DNN-BATADAL-l4-units64.h5",
}

# ---------------------------
# Utils
# ---------------------------
def create_sliding_windows(data: np.ndarray, window: int) -> np.ndarray:
    """
    Turn (N, F) into (N - window + 1, window, F) using a simple sliding window.
    """
    N = len(data)
    if N < window:
        raise ValueError(f"Not enough rows ({N}) for window={window}.")
    # Efficient rolling view could be used, but a simple loop is fine here:
    out = np.empty((N - window + 1, window, data.shape[1]), dtype=data.dtype)
    for i in range(N - window + 1):
        out[i] = data[i:i + window]
    return out

def ensure_prob_vector(y):
    """
    Convert model outputs to a 1D anomaly score vector:
    - If output has 2 columns, take [:,1] (prob of 'anomaly')
    - If output has 1 column, take [:,0] (sigmoid)
    """
    y = np.asarray(y)
    if y.ndim == 1:
        return y
    if y.shape[1] == 1:
        return y[:, 0]
    return y[:, 1]  # assume [:,1] is anomaly prob for softmax

# ---------------------------
# Load models (skip if missing)
# ---------------------------
models = {}
for name, path in MODEL_PATHS.items():
    if os.path.exists(path):
        models[name] = load_model(path)
        print(f"✅ Loaded {name} model")
    else:
        print(f"⚠️ Model file not found: {path}, skipping {name}")

# ---------------------------
# Load preprocessed test data
# ---------------------------
X_test, Y_test, sensor_cols = load_test_data(DATASET)  # X_test shape (N, 43)
N, F = X_test.shape
print("✅ Final test data shape:", X_test.shape)

results = {}

# ---------------------------
# AE: reconstruction error on (N, F)
# ---------------------------
if "autoencoder" in models:
    recon = models["autoencoder"].predict(X_test)
    ae_scores_full = np.mean((X_test - recon) ** 2, axis=1)  # shape (N,)
else:
    ae_scores_full = None

# ---------------------------
# CNN: needs (N - Wc + 1, Wc, F)
# ---------------------------
if "cnn" in models:
    X_test_cnn = create_sliding_windows(X_test, window=WINDOW_CNN)
    print("CNN test shape:", X_test_cnn.shape)  # (N - Wc + 1, Wc, F)
    proba_cnn = models["cnn"].predict(X_test_cnn)
    cnn_scores = ensure_prob_vector(proba_cnn)  # shape (N - Wc + 1,)
else:
    cnn_scores = None

# ---------------------------
# DNN: expects (batch, 1, F)
# We can either add a time axis of 1 for each row OR use sliding windows of size 1.
# ---------------------------
if "dnn" in models:
    # Option A: simple expand dims -> (N, 1, F)
    # X_test_dnn = np.expand_dims(X_test, axis=1)
    # Option B (equivalent when history=1): sliding windows of size 1
    X_test_dnn = create_sliding_windows(X_test, window=WINDOW_DNN)  # (N, 1, F)
    proba_dnn = models["dnn"].predict(X_test_dnn)
    dnn_scores_full = ensure_prob_vector(proba_dnn)  # shape (N,)
else:
    dnn_scores_full = None

# ---------------------------
# Align lengths for CSV
# We align everything to the CNN timeline (if CNN exists):
#   CNN length = N - Wc + 1
#   Use AE[DNN] scores from index (Wc - 1) onward to match CNN’s last-index-of-window convention.
# If CNN is missing, we fall back to the longest available vector and truncate the others.
# ---------------------------
columns = {}
if cnn_scores is not None:
    L = cnn_scores.shape[0]                 # target length
    start = WINDOW_CNN - 1                  # align to window end
    if ae_scores_full is not None:
        columns["autoencoder_score"] = ae_scores_full[start:start + L]
    if dnn_scores_full is not None:
        columns["dnn_score"] = dnn_scores_full[start:start + L]
    columns["cnn_score"] = cnn_scores
else:
    # No CNN → pick the longest available, then truncate others to match
    candidates = []
    if ae_scores_full is not None:
        candidates.append(("autoencoder_score", ae_scores_full))
    if dnn_scores_full is not None:
        candidates.append(("dnn_score", dnn_scores_full))
    if not candidates:
        raise SystemExit("❌ No model scores available to save.")
    # pick max length
    target_name, target_vec = max(candidates, key=lambda kv: kv[1].shape[0])
    L = target_vec.shape[0]
    for name, vec in candidates:
        columns[name] = vec[:L]

# Safety check: all columns same length
lens = {k: len(v) for k, v in columns.items()}
if len(set(lens.values())) != 1:
    raise RuntimeError(f"Length mismatch after alignment: {lens}")

df_scores = pd.DataFrame(columns)
df_scores.to_csv("batadal_scores.csv", index=False)
print(f"✅ Saved anomaly scores to batadal_scores.csv with columns {list(columns.keys())} and length {len(df_scores)}")
