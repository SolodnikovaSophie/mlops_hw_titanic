import os
from datetime import datetime

import pandas as pd
from airflow import DAG  # type: ignore
from airflow.operators.python import PythonOperator  # type: ignore

'''Этот DAG выполняет проверку дрейфа данных между тренировочным и текущим датасетом.
Он рассчитывает Population Stability Index (PSI) для выбранных признаков и выводит отчёт в лог.'''

def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    '''Данная функция рассчитывает Population Stability Index (PSI) между двумя распределениями.
    PSI используется для оценки дрейфа распределения признаков между тренировочным и текущим датасетом.
    Чем выше значение PSI, тем сильнее дрейф.'''
    import numpy as np

    expected = expected.dropna().astype(float)
    actual = actual.dropna().astype(float)

    qs = np.linspace(0, 1, bins + 1)
    breaks = np.unique(np.quantile(expected, qs))
    if len(breaks) < 3:
        return 0.0

    exp_cnt, _ = np.histogram(expected, bins=breaks)
    act_cnt, _ = np.histogram(actual, bins=breaks)

    exp_p = exp_cnt / max(exp_cnt.sum(), 1)
    act_p = act_cnt / max(act_cnt.sum(), 1)

    eps = 1e-6
    exp_p = np.clip(exp_p, eps, 1)
    act_p = np.clip(act_p, eps, 1)

    return float(((act_p - exp_p) * np.log(act_p / exp_p)).sum())


def check_drift():
    '''Эта функция загружает тренировочный и текущий датасеты,
    рассчитывает PSI для выбранных признаков и выводит отчёт в лог.'''
    train_path = os.getenv("TRAIN_PATH", "/opt/project/data/train.csv")
    current_path = os.getenv("CURRENT_PATH", "/opt/project/data/current.csv")

    train_df = pd.read_csv(train_path)
    current_df = pd.read_csv(current_path)

    features = ["Age", "Fare", "Pclass"]

    report = {f: psi(train_df[f], current_df[f]) for f in features}
    report["psi_mean"] = sum(report.values()) / len(features)

    print("DRIFT REPORT:", report)


with DAG(
    dag_id="drift_check_only",
    start_date=datetime(2026, 1, 1),
    schedule_interval=None,  # вручную
    catchup=False,
    tags=["mlops_hw"],
) as dag:
    PythonOperator(
        task_id="check_drift",
        python_callable=check_drift,
    )
