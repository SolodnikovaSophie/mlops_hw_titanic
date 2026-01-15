import os
import re

import pandas as pd
from dotenv import load_dotenv

import mlflow
import mlflow.sklearn
from pycaret.classification import setup, compare_models, finalize_model, pull

load_dotenv()


def _sanitize_metric_name(name: str) -> str:
    name = re.sub(r"[^\w\s\.\-\/]", "_", str(name))
    name = re.sub(r"\s+", "_", name).strip("_")
    return name


def main():
    data_path = os.getenv("CURRENT_PATH", "/opt/project/data/current.csv")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "mlops_hw_titanic_v2")
    out_dir = os.getenv("OUTPUT_DIR", "/tmp/mlops_hw")

    df = pd.read_csv(data_path)
    features = ["Pclass", "Sex", "Age", "Fare"]
    keep_cols = features + ["Survived"]
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    df = df[keep_cols].copy()
    target = "Survived"
    if target not in df.columns:
        raise ValueError("Column 'Survived' not found")

    os.makedirs(out_dir, exist_ok=True)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment(experiment_name)

    # закрыть возможный "зависший" run
    try:
        mlflow.end_run()
    except Exception:
        pass

    setup(
        data=df,
        target=target,
        session_id=42,
        fold=5,
        log_experiment=True,  # PyCaret создаёт run
        experiment_name=experiment_name,
        verbose=False,
    )

    best_model = compare_models(sort="Recall")
    final_model = finalize_model(best_model)

    # логируем MLflow model directory (с MLmodel), чтобы models:/... загружался в API
    with mlflow.start_run(nested=True):
        mlflow.sklearn.log_model(final_model, artifact_path="model")

        results = pull()
        results.to_csv(os.path.join(out_dir, "compare_models.csv"), index=False)

        best_row = results.iloc[0]
        for metric_name, value in best_row.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(_sanitize_metric_name(metric_name), float(value))

        mlflow.log_artifact(
            os.path.join(out_dir, "compare_models.csv"),
            artifact_path="tables",
        )

    print("Training finished. MLflow model saved to artifacts/model")


if __name__ == "__main__":
    main()
