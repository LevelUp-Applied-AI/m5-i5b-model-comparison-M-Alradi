import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from model_comparison import load_and_preprocess

# -----------------------------
# Load data
# -----------------------------
X_train, X_test, y_train, y_test = load_and_preprocess()

feature_names = X_test.columns

# -----------------------------
# Recreate top 3 models
# (adjust if your best configs differ)
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
}

# Train models
for name, model in models.items():
    model.fit(X_train, y_train)

# -----------------------------
# Permutation importance
# -----------------------------
importance_results = {}

for name, model in models.items():
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring="f1"
    )

    importance_results[name] = result.importances_mean

# -----------------------------
# DataFrame
# -----------------------------
importance_df = pd.DataFrame(importance_results, index=feature_names)

importance_df["mean"] = importance_df.mean(axis=1)
top_features = importance_df.sort_values("mean", ascending=False).head(8)

plot_df = top_features.drop(columns="mean")

# -----------------------------
# Plot
# -----------------------------
os.makedirs("results", exist_ok=True)

plot_df.plot(kind="bar")

plt.title("Permutation Importance (Top 8 Features)")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("results/permutation_importance.png")
plt.close()

print("Plot saved to results/permutation_importance.png")

# -----------------------------
# Print rankings
# -----------------------------
print("\n=== Feature Rankings ===")
for name in models:
    print(f"\n{name}:")
    print(importance_df[name].sort_values(ascending=False).head(8))