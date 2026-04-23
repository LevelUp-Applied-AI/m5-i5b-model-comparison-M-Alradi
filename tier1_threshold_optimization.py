import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib
from model_comparison import load_and_preprocess

# -----------------------------
# Load model and data
# -----------------------------
MODEL_PATH = "results/best_model.joblib"

model = joblib.load(MODEL_PATH)
X_train, X_test, y_train, y_test = load_and_preprocess()

# -----------------------------
# Get predicted probabilities
# -----------------------------
probs = model.predict_proba(X_test)[:, 1]

# -----------------------------
# Threshold sweep
# -----------------------------
thresholds = np.arange(0.1, 0.91, 0.05)
results = []

for t in thresholds:
    preds = (probs >= t).astype(int)

    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)

    alerts = preds.sum()
    alerts_per_1000 = (alerts / len(preds)) * 1000

    results.append({
        "threshold": t,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "alerts_per_1000": alerts_per_1000
    })

# -----------------------------
# Convert to array for filtering
# -----------------------------
df = pd.DataFrame(results)

# Capacity constraint:
# 150 per 10,000 → 15 per 1,000
valid_df = df[df["alerts_per_1000"] <= 15]

if len(valid_df) == 0:
    print("No threshold satisfies the capacity constraint.")
    print("Try increasing threshold range (e.g., up to 0.95).")
else:
    best_row = valid_df.loc[valid_df["recall"].idxmax()]

    print("\n=== Threshold Recommendation ===")
    print(f"Threshold: {best_row['threshold']:.2f}")
    print(f"Recall: {best_row['recall']:.4f}")
    print(f"Precision: {best_row['precision']:.4f}")
    print(f"F1: {best_row['f1']:.4f}")
    print(f"Alerts per 1000: {best_row['alerts_per_1000']:.2f}")

# -----------------------------
# Save plot
# -----------------------------
os.makedirs("results", exist_ok=True)

plt.figure()
plt.plot(df["threshold"], df["precision"], label="Precision")
plt.plot(df["threshold"], df["recall"], label="Recall")
plt.plot(df["threshold"], df["f1"], label="F1")

plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Threshold Sweep")
plt.legend()

plt.savefig("results/threshold_sweep.png")
plt.close()

print("\n Plot saved to results/threshold_sweep.png")