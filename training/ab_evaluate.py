import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import chi2_contingency

"""Этот модуль выполняет оценку результатов A/B-теста моделей.
Он загружает лог A/B-теста из CSV-файла, вычисляет метрики  качества для каждой модели, 
и принимает решение о переводе модели B в Production на основе статистической значимости."""


def _pick_existing_path(candidates: list[str]) -> str:
    """Эта функция выбирает первый существующий путь из списка кандидатов.
    Если ни один из путей не существует, возвращает первый "нормальный" путь для информативной ошибки.
    """
    for p in candidates:
        if p and os.path.exists(p):
            return p
    # если ничего не нашли — вернём первый “нормальный” (чтобы в ошибке было понятно)
    return candidates[0] or candidates[1]


@dataclass
class Metrics:
    n: int
    accuracy: float
    precision: float
    recall: float
    f1: float


def compute_metrics(df: pd.DataFrame) -> Metrics:
    """Эта функция вычисляет метрики качества на основе логов предсказаний и истинных значений."""
    y_true = df["ground_truth"].astype(int).values
    y_pred = df["prediction"].astype(int).values
    return Metrics(
        n=len(df),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
    )


def pvalue_accuracy(df_a: pd.DataFrame, df_b: pd.DataFrame) -> float:
    """Эта функция вычисляет p-value для разницы в точности (accuracy) между двумя моделями
    с помощью критерия хи-квадрат."""
    # 2x2: correct/incorrect for A and B
    a_correct = int(
        (df_a["prediction"].astype(int) == df_a["ground_truth"].astype(int)).sum()
    )
    a_wrong = int(len(df_a) - a_correct)

    b_correct = int(
        (df_b["prediction"].astype(int) == df_b["ground_truth"].astype(int)).sum()
    )
    b_wrong = int(len(df_b) - b_correct)

    _, p, _, _ = chi2_contingency(
        [[a_correct, a_wrong], [b_correct, b_wrong]], correction=False
    )
    return float(p)


def main():
    candidates = [
        os.getenv("AB_LOG_PATH"),
        "/opt/airflow/logs/ab_logs.csv",
        "/opt/logs/ab_logs.csv",
        "/opt/project/logs/ab_logs.csv",
    ]
    log_path = _pick_existing_path(candidates)

    if not os.path.exists(log_path):
        raise FileNotFoundError(f"AB log file not found: {log_path}")

    df = pd.read_csv(log_path)

    required = {"variant", "prediction", "ground_truth", "model_stage", "model_version"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"Missing required columns in log CSV: {missing}")

    df = df.dropna(subset=["ground_truth"]).copy()
    df["ground_truth"] = df["ground_truth"].astype(int)

    # prediction может быть строкой/float — нормализуем
    df["prediction"] = df["prediction"].astype(float).round().astype(int)

    df_a = df[df["variant"] == "A"].copy()
    df_b = df[df["variant"] == "B"].copy()

    if len(df_a) < 20 or len(df_b) < 20:
        raise ValueError(
            f"Not enough data for A/B: A={len(df_a)}, B={len(df_b)}. Generate more /predict logs."
        )

    # возьмём “какие модели реально были”
    a_versions = sorted(df_a["model_version"].astype(str).unique().tolist())
    b_versions = sorted(df_b["model_version"].astype(str).unique().tolist())

    m_a = compute_metrics(df_a)
    m_b = compute_metrics(df_b)
    p = pvalue_accuracy(df_a, df_b)

    promote = (m_b.accuracy > m_a.accuracy) and (p < 0.05)

    print("A/B RESULTS")
    print(f"Log file: {log_path}")
    print(
        f"A (Production) versions seen: {a_versions} | n={m_a.n} | acc={m_a.accuracy:.4f} | prec={m_a.precision:.4f} | rec={m_a.recall:.4f} | f1={m_a.f1:.4f}"
    )
    print(
        f"B (Staging)    versions seen: {b_versions} | n={m_b.n} | acc={m_b.accuracy:.4f} | prec={m_b.precision:.4f} | rec={m_b.recall:.4f} | f1={m_b.f1:.4f}"
    )
    print(f"p_value_accuracy (chi-square): {p:.6f}")
    print(f"PROMOTE_B_TO_PRODUCTION: {promote}")

    summary = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "log_path": log_path,
        "A": asdict(m_a) | {"versions_seen": a_versions},
        "B": asdict(m_b) | {"versions_seen": b_versions},
        "p_value_accuracy": p,
        "promote": promote,
    }

    out_path = os.path.join(os.path.dirname(log_path), "ab_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved summary: {out_path}")


if __name__ == "__main__":
    main()
