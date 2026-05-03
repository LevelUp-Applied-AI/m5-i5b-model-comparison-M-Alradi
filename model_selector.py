import json
import os
from datetime import datetime
import pandas as pd

from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from model_comparison import load_and_preprocess

# -----------------------------
# Model registry
# -----------------------------
MODEL_REGISTRY = {
    "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
}


class ModelSelector:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.output_dir = self._create_output_dir()

    def _create_output_dir(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"results/experiment_{timestamp}"
        os.makedirs(path, exist_ok=True)
        return path

    def load_data(self):
        X_train, X_test, y_train, y_test = load_and_preprocess()
        return X_train, y_train

    def run(self):
        X, y = self.load_data()
        results = []

        for model_cfg in self.config["models"]:
            name = model_cfg["name"]
            model_type = model_cfg["type"]
            params_list = model_cfg["params"]

            ModelClass = MODEL_REGISTRY[model_type]

            for params in params_list:
                model = ModelClass(**params)

                scores = cross_validate(
                    model,
                    X,
                    y,
                    cv=self.config["cv"],
                    scoring=self.config["metrics"],
                    return_train_score=False
                )

                result = {
                    "model": name,
                    "params": params
                }

                for metric in self.config["metrics"]:
                    result[f"{metric}_mean"] = scores[f"test_{metric}"].mean()

                results.append(result)

        df = pd.DataFrame(results)

        # Save results
        df.to_csv(os.path.join(self.output_dir, "results.csv"), index=False)

        print("Results saved to:", self.output_dir)

        return df