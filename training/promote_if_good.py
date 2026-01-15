import json
import os

import mlflow


def _pick_existing_path(candidates: list[str]) -> str:
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return candidates[0] or candidates[1]


def main():
    candidates = [
        os.getenv("AB_SUMMARY_PATH"),
        "/opt/airflow/logs/ab_summary.json",
        "/opt/logs/ab_summary.json",
        "/opt/project/logs/ab_summary.json",
    ]
    summary_path = _pick_existing_path(candidates)

    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"AB summary not found: {summary_path}. Run ab_evaluate.py first."
        )

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    promote = bool(summary.get("promote", False))
    print("Loaded:", summary_path)
    print("promote:", promote)

    if not promote:
        print("Not promoting: promote=false (A/B did not prove B is better)")
        return

    model_name = os.getenv("MODEL_NAME", "titanic-survival-model")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    client = mlflow.tracking.MlflowClient()

    staging = client.get_latest_versions(model_name, stages=["Staging"])
    if not staging:
        raise RuntimeError("No Staging version found")

    v = staging[0].version

    client.transition_model_version_stage(
        name=model_name,
        version=v,
        stage="Production",
        archive_existing_versions=True,
    )

    print(f"Promoted {model_name} v{v} -> Production")


if __name__ == "__main__":
    main()
