import streamlit as st
import pandas as pd
import numpy as np
import time
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

# --- Configuration (CRITICAL: REPLACE PLACEHOLDERS WITH TRAINING DATA STATS!) ---
DATA_PATH = "test_whitebox_attack.csv"
AE_MODEL_PATH = "AE-BATADAL-l5-cf2.5.h5"
CNN_MODEL_PATH = "CNN-BATADAL-l8-hist200-kern3-units32.h5"
DNN_MODEL_PATH = "DNN-BATADAL-l4-units64.h5"

WINDOW_SIZE = 200

# Standard Scaling (Mean/Std) for the 43 sensor features.
# REPLACE WITH YOUR TRAINING DATA MEAN/STD!
SCALER_MEAN = np.array([3.24, 2.84, 3.77, 3.58, 4.48, 5.38, 3.34, 2.98, 33.73, 27.60,
                        87.28, 27.69, 84.97, 24.32, 99.65, 24.21, 68.62, 33.10, 29.82,
                        98.79, 98.81, 0.77, 16.48, 0.04, 1.17, 36.20, 0.04, 49.34,
                        23.11, 0.04, 30.83, 0.04, 78.44, 0.96, 0.77, 0.04, 0.96,
                        0.04, 0.04, 0.69, 1.00, 0.04, 0.73])
SCALER_STD = np.array([1.34, 1.52, 1.33, 1.03, 0.02, 0.22, 1.04, 0.01, 5.70, 3.43,
                       15.02, 13.91, 7.10, 3.90, 17.15, 16.92, 6.10, 7.69, 6.86,
                       8.01, 8.01, 3.34, 12.34, 0.33, 1.82, 14.85, 0.28, 4.21,
                       3.34, 0.26, 2.25, 0.26, 21.05, 0.19, 0.42, 0.19, 0.19,
                       0.28, 0.28, 0.46, 0.46, 0.19, 0.44])

# SCORE MIN/MAX for Fusion Normalization
# These are still placeholders, but the logic below now INVERTS AE/DNN scores
# to ensure low raw score maps to low normalized score.
SCORE_MIN = np.array([0.0000, 0.0000, 0.0000])
SCORE_MAX = np.array([0.1, 1.0, 0.5])

FUSION_THRESHOLD = 0.3352


# --- Custom Keras Loss for Autoencoder (MSE) ---
def mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


CUSTOM_OBJECTS = {'mse': mse}


# --- Model Loading and Prediction Functions ---

@st.cache_resource
def load_all_models():
    """Loads all Keras models."""
    st.info("Loading models... This happens only once.")
    try:
        ae_model = load_model(AE_MODEL_PATH, custom_objects=CUSTOM_OBJECTS, compile=False)
        cnn_model = load_model(CNN_MODEL_PATH, compile=False)
        dnn_model = load_model(DNN_MODEL_PATH, compile=False)
        return ae_model, cnn_model, dnn_model
    except Exception as e:
        st.error(
            f"Error loading models. Please check the file paths and ensure the models were saved correctly in Keras format: {e}")
        return None, None, None


# ... (preprocess_sample remains the same) ...
def preprocess_sample(df_data, start_index):
    if start_index + WINDOW_SIZE > len(df_data):
        start_index = len(df_data) - WINDOW_SIZE

    window_df = df_data.iloc[start_index: start_index + WINDOW_SIZE].copy()
    sensor_data = window_df.iloc[:, 1:44]
    timestamp = window_df['DATETIME'].iloc[-1]
    att_flag = window_df['ATT_FLAG'].iloc[-1]
    X_raw = sensor_data.values
    X_scaled = (X_raw - SCALER_MEAN) / SCALER_STD  # Shape (200, 43)
    X_cnn_input = X_scaled[np.newaxis, :, :]  # Shape (1, 200, 43)

    return X_cnn_input, X_scaled, timestamp, att_flag


def predict_and_fuse(models, X_cnn_input, X_scaled_window):
    """
    Runs predictions on all models and calculates a normalized fusion score.
    """
    ae_model, cnn_model, dnn_model = models

    # --- Derive Specific Inputs from the last timestep (index -1) ---
    X_ae_input = X_scaled_window[-1, :][np.newaxis, :]
    X_dnn_input = X_scaled_window[-1, :][np.newaxis, np.newaxis, :]

    # --- Raw Predictions ---
    ae_error = np.mean(np.square(ae_model.predict(X_ae_input, verbose=0) - X_ae_input), axis=1)[0]
    cnn_score = cnn_model.predict(X_cnn_input, verbose=0)[0][0]
    dnn_score = dnn_model.predict(X_dnn_input, verbose=0)[0][0]

    raw_scores = np.array([ae_error, dnn_score, cnn_score])

    # 4. Score Normalization (Min-Max Scaling)
    # Standard Min-Max normalization for all
    df_norm_scores_std = (raw_scores - SCORE_MIN) / (SCORE_MAX - SCORE_MIN)
    df_norm_scores_std = np.clip(df_norm_scores_std, 0, 1)  # Clamp to [0, 1]

    # FIX: INVERTING SCORES: If low raw score (e.g., 0.0001) maps to 1.0,
    # it means the model is inverted or its MIN/MAX are severely wrong.
    # We will assume a low raw score should result in a low normalized anomaly score.

    # Invert normalization for AE and DNN (Indices 0 and 1)
    # The CNN score (Index 2) remains standard as it seems to be outputting closer to 0 for normal

    # 0: AE, 1: DNN, 2: CNN
    df_norm_scores = np.zeros(3)

    # Invert AE and DNN normalization: Normalized Score = 1 - Standard Normalized Score
    df_norm_scores[0] = 1 - df_norm_scores_std[0]  # AE
    df_norm_scores[1] = 1 - df_norm_scores_std[1]  # DNN
    df_norm_scores[2] = df_norm_scores_std[2]  # CNN (Standard)

    df_norm_scores = np.clip(df_norm_scores, 0, 1)  # Re-clamp after inversion

    # 5. Fusion Score (Simple Mean)
    fusion_score = np.mean(df_norm_scores)

    # 6. Anomaly Flag
    is_anomaly = fusion_score > FUSION_THRESHOLD

    return {
        'AE_Error': f"{ae_error:.6f}",
        'DNN_Score': f"{dnn_score:.6f}",
        'CNN_Score': f"{cnn_score:.6f}",
        'AE_Norm': f"{df_norm_scores[0]:.4f}",
        'DNN_Norm': f"{df_norm_scores[1]:.4f}",
        'CNN_Norm': f"{df_norm_scores[2]:.4f}",
        'Fusion_Score_Normalized': f"{fusion_score:.4f}",
        'Anomaly_Flag': is_anomaly
    }


# --- Streamlit Application ---

def run_streamlit_app():
    st.set_page_config(page_title="BATADAL Multi-Model Prediction", layout="wide")
    st.title("BATADAL Multi-Model Anomaly Detection Simulation")
    st.markdown("---")
    st.markdown(f"**Configuration:** Window Size = `{WINDOW_SIZE}`. Fusion Threshold = `{FUSION_THRESHOLD:.4f}`.")

    @st.cache_data
    def load_data():
        try:
            return pd.read_csv(DATA_PATH)
        except Exception as e:
            st.error(f"Could not load data file {DATA_PATH}: {e}")
            return None

    df_data = load_data()
    models = load_all_models()

    if df_data is None or any(m is None for m in models):
        return

    # Sidebar controls
    st.sidebar.header("Simulation Controls")
    max_index = len(df_data) - WINDOW_SIZE

    if max_index < 0:
        st.error(f"Dataset is too short for the required window size ({WINDOW_SIZE}).")
        return

    start_index = st.sidebar.number_input(
        "Start Row Index",
        min_value=0,
        max_value=max_index,
        value=0,
        step=1
    )

    n_records = st.sidebar.number_input(
        "Number of Records to Process (Max 100)",
        min_value=1,
        max_value=100,
        value=50,
        step=1
    )

    sleep_time = st.sidebar.number_input(
        "Interval (seconds)",
        min_value=0.0,
        max_value=10.0,
        value=0.0,
        step=0.1
    )

    if st.sidebar.button("Start Live Simulation"):
        st.subheader("Live Prediction Stream")

        results_container = st.empty()

        results_df = pd.DataFrame(columns=[
            'Timestamp',
            'Index',
            'AE_Raw',
            'DNN_Raw',
            'CNN_Raw',
            'AE_Norm',
            'DNN_Norm',
            'CNN_Norm',
            'Fusion_Score_Normalized',
            'Prediction',
            'Ground_Truth_ATT_FLAG'
        ])

        current_index = start_index
        records_processed = 0

        status_bar = st.progress(0, text="Simulation progress...")

        for i in range(n_records):
            if current_index + WINDOW_SIZE > len(df_data):
                st.warning("End of dataset reached.")
                break

            X_cnn_input, X_scaled_window, timestamp, att_flag = preprocess_sample(df_data, current_index)

            with st.spinner(f"Processing index {current_index}..."):
                prediction_results = predict_and_fuse(models, X_cnn_input, X_scaled_window)

            # --- Update Results DataFrame ---
            new_row = {
                'Timestamp': timestamp,
                'Index': current_index,
                'AE_Raw': prediction_results['AE_Error'],
                'DNN_Raw': prediction_results['DNN_Score'],
                'CNN_Raw': prediction_results['CNN_Score'],
                'AE_Norm': prediction_results['AE_Norm'],
                'DNN_Norm': prediction_results['DNN_Norm'],
                'CNN_Norm': prediction_results['CNN_Norm'],
                'Fusion_Score_Normalized': prediction_results['Fusion_Score_Normalized'],
                'Prediction': 'ANOMALY' if prediction_results['Anomaly_Flag'] else 'NORMAL',
                'Ground_Truth_ATT_FLAG': int(att_flag)
            }
            results_df.loc[records_processed] = new_row

            # --- Display Updates ---
            with results_container.container():
                st.dataframe(results_df.sort_index(ascending=False), use_container_width=True)

            # Update status bar
            progress = (i + 1) / n_records
            status_bar.progress(progress, text=f"Processing index {current_index}...")

            current_index += 1
            records_processed += 1

            # Pause for the simulation interval
            if sleep_time > 0 and i < n_records - 1:
                time.sleep(sleep_time)

        status_bar.progress(100, text="Simulation complete.")
        st.success(f"Processed {records_processed} records.")
        st.download_button(
            label="Download Prediction Log",
            data=results_df.to_csv(index=False).encode('utf-8'),
            file_name="batadal_prediction_log.csv",
            mime="text/csv",
        )


# --- Main Execution ---
if __name__ == "__main__":
    run_streamlit_app()