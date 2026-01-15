import os
import random
from typing import Any, Dict, Optional, Tuple
import mlflow
from flask import Flask, request, jsonify
from flasgger import Swagger, swag_from
from ab_logger import log_ab_event

'''"Этот модуль реализует Flask API для A/B-тестирования моделей машинного обучения.
* подключается к MLflow Model Registry
* загружает две версии модели:
* A = Production, B = Staging
* на каждом запросе делит трафик между A и B
* делает предсказание
* логирует запрос и результат в CSV через ab_logger.py
* даёт служебные эндпоинты: health, reload моделей, смена доли трафика B'''

# Flask / Swagger
app = Flask(__name__)
app.config["SWAGGER"] = {"title": "MLOps Titanic API", "uiversion": 3}
Swagger(app)

# ENV
MODEL_NAME = os.getenv("MODEL_NAME", "titanic-survival-model") # имя модели в MLflow Model Registry
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000") # MLflow Tracking URI
AB_LOG_PATH = os.getenv("AB_LOG_PATH", "/opt/logs/ab_logs.csv") # путь к CSV логу A/B теста

# доля B (Staging)
AB_SPLIT_B = float(os.getenv("AB_SPLIT_B", "0.3")) # по умолчанию 30% трафика на B

# MLflow setup
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = mlflow.tracking.MlflowClient()

# Model cache
_model_A = None  # Production
_model_B = None  # Staging
_ver_A = "unknown"
_ver_B = "unknown"


def _get_model_version(model_name: str, stage: str) -> str:
    try:
        vers = client.get_latest_versions(model_name, stages=[stage])
        if not vers:
            return "unknown"
        # берём максимальную по номеру
        best = max(vers, key=lambda v: int(v.version))
        return str(best.version)
    except Exception:
        return "unknown"


def _load_models() -> None:
    """
    Load Staging (B) обязательно.
    Load Production (A) если есть, иначе fallback на Staging.
    """
    global _model_A, _model_B, _ver_A, _ver_B

    # B — всегда Staging
    _ver_B = _get_model_version(MODEL_NAME, "Staging")
    _model_B = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Staging")

    # A — Production, но если нет → fallback на Staging
    _ver_A = _get_model_version(MODEL_NAME, "Production")
    if _ver_A == "unknown":
        _model_A = _model_B
        _ver_A = _ver_B
        print("[WARN] Production model not found → A uses Staging")
    else:
        _model_A = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Production")


def _ensure_models_loaded() -> None:
    """
    Ensure models are loaded, raise readable error on failure.
    """
    global _model_A, _model_B
    try:
        if _model_A is None or _model_B is None:
            _load_models()
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {e}")


def _choose_variant(payload: Dict[str, Any]) -> str:
    """
    Choose A/B.
    Priority:
      1) if user_id provided -> deterministic split
      2) else -> random by AB_SPLIT_B
    """
    uid = payload.get("user_id", None)
    if uid is not None:
        try:
            return "B" if int(uid) % 2 == 1 else "A"
        except Exception:
            pass

    return "B" if random.random() < AB_SPLIT_B else "A"


def _predict_with_model(model, features: Dict[str, Any]) -> Tuple[Any, Optional[float]]:
    """
    Predict label and (if available) probability for positive class.
    """
    import pandas as pd

    X = pd.DataFrame([features])

    pred = model.predict(X)

    # normalize pred output
    if hasattr(pred, "iloc"):
        try:
            pred_val = pred.iloc[0]
        except Exception:
            pred_val = pred.values[0]
    elif hasattr(pred, "tolist"):
        pred_val = pred.tolist()[0]
    elif isinstance(pred, (list, tuple)):
        pred_val = pred[0]
    else:
        pred_val = pred

    proba_val: Optional[float] = None
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)

            if hasattr(proba, "iloc"):
                if proba.shape[1] >= 2:
                    proba_val = float(proba.iloc[0, 1])
                else:
                    proba_val = float(proba.iloc[0, 0])
            else:
                if len(proba[0]) >= 2:
                    proba_val = float(proba[0][1])
                else:
                    proba_val = float(proba[0][0])
    except Exception:
        proba_val = None

    return pred_val, proba_val


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ab/split")
@swag_from(
    {
        "tags": ["ab"],
        "consumes": ["application/json"],
        "produces": ["application/json"],
        "parameters": [
            {
                "name": "body",
                "in": "body",
                "required": True,
                "schema": {
                    "type": "object",
                    "properties": {"split_b": {"type": "number", "example": 0.3}},
                    "required": ["split_b"],
                },
            }
        ],
        "responses": {
            200: {"description": "Updated split"},
            400: {"description": "Bad request"},
        },
    }
)
def set_split():
    global AB_SPLIT_B
    payload = request.get_json(force=True)
    split_b = float(payload["split_b"])
    if split_b < 0.0 or split_b > 1.0:
        return jsonify({"ok": False, "error": "split_b must be in [0,1]"}), 400
    AB_SPLIT_B = split_b
    return jsonify({"ok": True, "split_b": AB_SPLIT_B})


@app.post("/models/reload")
@swag_from(
    {
        "tags": ["admin"],
        "responses": {
            200: {"description": "Reloaded models"},
            500: {"description": "Model reload failed (returns JSON error)"},
        },
    }
)
def reload_models():
    try:
        _load_models()
        return jsonify(
            {
                "ok": True,
                "model_name": MODEL_NAME,
                "A": {"stage": "Production", "version": _ver_A},
                "B": {"stage": "Staging", "version": _ver_B},
            }
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/predict")
@swag_from(
    {
        "tags": ["inference"],
        "consumes": ["application/json"],
        "produces": ["application/json"],
        "parameters": [
            {
                "name": "body",
                "in": "body",
                "required": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "integer", "example": 123},
                        "Pclass": {"type": "integer", "example": 3},
                        "Sex": {"type": "string", "example": "male"},
                        "Age": {"type": "number", "example": 22},
                        "Fare": {"type": "number", "example": 7.25},
                        "Survived": {"type": "integer", "example": 0},
                    },
                    "required": ["Pclass", "Sex", "Age", "Fare"],
                },
            }
        ],
        "responses": {
            200: {"description": "Prediction result"},
            500: {"description": "Inference failed (returns JSON error)"},
        },
    }
)
def predict():
    try:
        _ensure_models_loaded()

        payload = request.get_json(force=True)

        features = {
            "Pclass": payload.get("Pclass"),
            "Sex": payload.get("Sex"),
            "Age": payload.get("Age"),
            "Fare": payload.get("Fare"),
        }

        ground_truth = payload.get("Survived", None)

        variant = _choose_variant(payload)
        if variant == "B":
            model = _model_B
            stage = "Staging"
            ver = _ver_B
        else:
            model = _model_A
            stage = "Production"
            ver = _ver_A

        pred_val, proba_val = _predict_with_model(model, features)

        # log 
        try:
            log_ab_event(
                variant=variant,
                model_name=MODEL_NAME,
                model_stage=stage,
                model_version=ver,
                features=features,
                prediction=pred_val,
                proba=proba_val,
                ground_truth=ground_truth,
                log_path=AB_LOG_PATH,
            )
        except Exception as e:
            print(f"[WARN] AB logging failed: {e}")

        return jsonify(
            {
                "ok": True,
                "prediction": pred_val,
                "proba": proba_val,
                "variant": variant,
                "model_stage": stage,
                "model_version": ver,
                "input": features,
            }
        )

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
