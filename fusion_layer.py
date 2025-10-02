import pandas as pd

# Load saved scores
df = pd.read_csv("batadal_scores.csv")

# Normalize each score column to [0,1] for fairness
df_norm = (df - df.min()) / (df.max() - df.min())

# Compute fusion score (simple average across models)
df_norm["fusion_score"] = df_norm.mean(axis=1)

# Compute threshold dynamically (e.g., 99th percentile of fusion scores)
threshold = df_norm["fusion_score"].quantile(0.92)

# Add anomaly flag
df_norm["is_anomaly"] = (df_norm["fusion_score"] > threshold).astype(int)

# Save results
output_file = "batadal_fusion.csv"
df_norm.to_csv(output_file, index=False)

print(f"✅ Saved fused scores with anomaly flag to {output_file}")
print(f"ℹ️ Threshold used: {threshold:.4f}")
print(f"⚠️ Detected {df_norm['is_anomaly'].sum()} anomalies out of {len(df_norm)} samples")
