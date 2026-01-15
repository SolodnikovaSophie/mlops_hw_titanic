import os
from dotenv import load_dotenv
import mlflow

load_dotenv()


def _run_has_mlmodel(client: mlflow.tracking.MlflowClient, run_id: str) -> bool:
    try:
        arts = client.list_artifacts(run_id, "model")
        paths = {a.path for a in arts}
        return "model/MLmodel" in paths
    except Exception:
        return False


def main():
    model_name = os.getenv("MODEL_NAME", "titanic-survival-model")
    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "mlops_hw_titanic")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        raise RuntimeError(f"Experiment not found: {exp_name}")

    # Берём последние 30, но только FINISHED
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=30,
    )
    if not runs:
        raise RuntimeError("No FINISHED runs found")

    good_run_id = None
    for r in runs:
        rid = r.info.run_id
        if _run_has_mlmodel(client, rid):
            good_run_id = rid
            break

    if good_run_id is None:
        raise RuntimeError("No FINISHED runs with artifacts/model/MLmodel found")

    model_uri = f"runs:/{good_run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)

    # новая версия -> Staging
    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage="Staging",
        archive_existing_versions=False,
    )

    # если Production пустой — сделаем initial Production
    prod = client.get_latest_versions(model_name, stages=["Production"])
    if not prod:
        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage="Production",
            archive_existing_versions=False,
        )
        print(
            f"✅ Model registered: {model_name} v{mv.version} -> Staging + Production (initial)"
        )
    else:
        print(f"✅ Model registered: {model_name} v{mv.version} -> Staging")


if __name__ == "__main__":
    main()
