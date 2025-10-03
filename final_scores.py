import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, roc_auc_score,
    roc_curve, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import numpy as np


def calculate_and_plot_metrics():
    """
    Loads, aligns, cleans, calculates, and plots all metrics for the BATADAL fusion model.
    """

    # --- 1. Load the data ---
    try:
        df_pred_full = pd.read_csv("batadal_fusion.csv")
        df_true_full = pd.read_csv("data/BATADAL/test_whitebox_attack.csv")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure both files are present.")
        return

    # --- 2. Extract and Align Data (Heuristic Slicing) ---
    y_true_labels = df_true_full['ATT_FLAG'].values.astype(int)
    true_len = len(y_true_labels)

    scores_full = df_pred_full['fusion_score'].values
    y_pred_binary_full = df_pred_full['is_anomaly'].values.astype(int)
    pred_len = len(scores_full)

    if pred_len > true_len:
        scores = scores_full[-true_len:]
        y_pred_binary = y_pred_binary_full[-true_len:]
        y_true = y_true_labels
        print(f"Alignment: Sliced prediction data from {pred_len} to {true_len} rows to match true labels.")
    elif pred_len == true_len:
        scores = scores_full
        y_pred_binary = y_pred_binary_full
        y_true = y_true_labels
        print("Alignment: Data lengths match.")
    else:  # pred_len < true_len
        scores = scores_full
        y_pred_binary = y_pred_binary_full
        y_true = y_true_labels[:pred_len]
        print(
            f"Warning: Prediction length ({pred_len}) is shorter than true label length ({true_len}). Truncating true labels.")

    # --- 3. Critical Data Cleaning Step (to resolve NumPy Inf/NaN errors) ---
    finite_mask = np.isfinite(scores)
    if not np.all(finite_mask):
        num_removed = np.sum(~finite_mask)
        scores = scores[finite_mask]
        y_true = y_true[finite_mask]
        y_pred_binary = y_pred_binary[finite_mask]
        print(f"Warning: Removed {num_removed} non-finite scores/labels for robust calculation.")

    # --- 4. Calculate Core Metrics ---
    accuracy = accuracy_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    auc_score = roc_auc_score(y_true, scores)
    cm = confusion_matrix(y_true, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()

    # --- 5. Generate Plots ---

    # 5a. ROC Curve Plot
    try:
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Recall)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig('roc_curve_final.png')
        plt.close()
        print("✅ ROC Curve plot saved as 'roc_curve_final.png'")
    except Exception as e:
        print(f"⚠️ Error plotting ROC Curve: {e}")

    # 5b. Confusion Matrix Plot
    try:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal (0)', 'Anomaly (1)'])
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        plt.title('Confusion Matrix (Fusion Model)')
        plt.savefig('confusion_matrix_final.png')
        plt.close()
        print("✅ Confusion Matrix plot saved as 'confusion_matrix_final.png'")
    except Exception as e:
        print(f"⚠️ Error plotting Confusion Matrix: {e}")

    # --- 6. Save Aligned Logs ---
    try:
        # Re-align DATETIME with the potentially cleaned data
        df_true_aligned_full = df_true_full.iloc[:true_len]
        df_true_cleaned = df_true_aligned_full[finite_mask]

        df_comparison = pd.DataFrame({
            'DATETIME': df_true_cleaned['DATETIME'].values,
            'True_ATT_FLAG': y_true,
            'Predicted_is_anomaly': y_pred_binary
        })
        df_comparison.to_csv("aligned_metrics_logs_final.csv", index=False)
        print("✅ Aligned comparison logs saved to 'aligned_metrics_logs_final.csv'")
    except Exception as e:
        print(f"⚠️ Error saving aligned logs: {e}")

    # --- 7. Print Results ---
    print("\n--- Final Metrics on Aligned and Cleaned Data ---")
    print(f"Total samples evaluated: {len(y_true)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (True Positive Rate): {recall:.4f}")
    print(f"AUC Score: {auc_score:.4f}")

    print("\n--- Confusion Matrix Breakdown ---")
    print(f"True Positives (TP): {tp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")


# --- Execution Call ---
if __name__ == "__main__":
    calculate_and_plot_metrics()