import csv
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

"""Этот модуль отвечает за логирование A/B-теста моделей.
Он сохраняет каждый запрос к модели в CSV-лог, чтобы затем можно было:
посчитать метрики качества, сравнить модели A и B, 
принять решение о переводе модели в production."""


@dataclass
class ABLogRow:
    """Данный класс Описывает одну строку лога A/B-теста."""

    ts_utc: str
    variant: str  # "A" or "B"
    model_name: str  # имя модели
    model_stage: str  # "Production" or "Staging"
    model_version: str  # may be "unknown" if not resolved
    prediction: Any  # predicted value
    proba: Optional[float]  # predicted probability for positive class, if applicable
    features_json: str  # JSON string
    ground_truth: Optional[int]  # if provided


def _utc_now_iso() -> str:
    """Возвращает текущее время в формате ISO в UTC для корректного логирования"""
    return datetime.now(timezone.utc).isoformat()


def _ensure_parent(path: Path) -> None:
    """Проверяет, что директория для лог-файла существует,
    и создаёт её при необходимости."""
    path.parent.mkdir(parents=True, exist_ok=True)


def log_ab_event(
    *,
    variant: str,
    model_name: str,
    model_stage: str,
    model_version: str,
    features: Dict[str, Any],
    prediction: Any,
    proba: Optional[float],
    ground_truth: Optional[int] = None,
    log_path: Optional[str] = None,
) -> None:
    """
    Append one row to AB log CSV.
    """
    if log_path is None:
        log_path = os.getenv(
            "AB_LOG_PATH", "/opt/logs/ab_logs.csv"
        )  # Определяем путь к логу из переменной окружения или используем значение по умолчанию

    path = Path(log_path)
    _ensure_parent(path)

    row = ABLogRow(
        ts_utc=_utc_now_iso(),  # текущее время в UTC
        variant=str(variant),  # "A" или "B"
        model_name=str(model_name),  # имя модели
        model_stage=str(model_stage),  # "Production" или "Staging"
        model_version=str(model_version),  # версия модели
        prediction=prediction,  # предсказанное значение
        proba=float(proba) if proba is not None else None,
        features_json=json.dumps(features, ensure_ascii=False),
        ground_truth=int(ground_truth) if ground_truth is not None else None,
    )

    fieldnames = list(asdict(row).keys())
    file_exists = path.exists() and path.stat().st_size > 0

    # Открываем файл в режиме добавления и записываем строку лога
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow(asdict(row))
