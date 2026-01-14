from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def main():
    # --- Load params ---
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)["train"]

    test_size = params["test_size"]
    random_state = params["random_state"]
    n_estimators = params["n_estimators"]
    max_depth = params["max_depth"]

    # --- Paths ---
    data_path = Path("data/iris.csv")
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "model.joblib"

    # --- Load data ---
    df = pd.read_csv(data_path)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # --- Train ---
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)

    # --- Evaluate ---
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")

    # --- MLflow ---
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("iris-mlops")

    with mlflow.start_run():
        for k, v in params.items():
            mlflow.log_param(k, v)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)

        joblib.dump(model, model_path)
        mlflow.sklearn.log_model(model, artifact_path="model")

    print("Training complete via params.yaml ✅")
    print(f"Accuracy: {acc:.4f} | F1-macro: {f1:.4f}")


if __name__ == "__main__":
    main()
