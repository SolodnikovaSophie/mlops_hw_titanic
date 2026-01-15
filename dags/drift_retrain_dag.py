import os
from datetime import datetime

import pandas as pd
import numpy as np
from airflow import DAG  # type: ignore
from airflow.operators.python import PythonOperator, BranchPythonOperator  # type: ignore
from airflow.operators.empty import EmptyOperator  # type: ignore
from airflow.utils.trigger_rule import TriggerRule  # type: ignore

'''Этот DAG выполняет проверку дрейфа данных между тренировочным и текущим датасетом.
   При обнаружении значительного дрейфа (по PSI) запускает процесс retraining модели.
   В противном случае пропускает retraining.'''

def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
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


def check_drift(**context):
    '''Эта функция загружает тренировочный и текущий датасеты,
    рассчитывает PSI для выбранных признаков и выводит отчёт в лог.'''
    train_path = os.getenv("TRAIN_PATH", "/opt/project/data/train.csv")
    current_path = os.getenv("CURRENT_PATH", "/opt/project/data/current.csv")
    threshold = float(os.getenv("DRIFT_THRESHOLD", "0.2"))

    train_df = pd.read_csv(train_path)
    current_df = pd.read_csv(current_path)

    features = ["Age", "Fare", "Pclass"]

    report = {f: psi(train_df[f], current_df[f]) for f in features}
    psi_mean = sum(report.values()) / len(features)

    print("DRIFT REPORT:", report)
    print("PSI MEAN:", psi_mean)

    # кладём в XCom, чтобы branch мог прочитать
    context["ti"].xcom_push(key="psi_mean", value=psi_mean)

    return psi_mean


def branch_on_drift(**context):
    '''Эта функция решает, запускать ли retraining модели,
    основываясь на среднем PSI, полученном из XCom или скипать его.'''
    threshold = float(os.getenv("DRIFT_THRESHOLD", "0.2"))
    psi_mean = context["ti"].xcom_pull(key="psi_mean", task_ids="check_drift")

    print(f"Branch decision: psi_mean={psi_mean}, threshold={threshold}")

    if psi_mean > threshold:
        return "run_retraining"
    else:
        return "skip_retraining"


def run_retraining_real():
    '''Эта функция запускает скрипты retraining и регистрации модели.
    после drift новая модель появляется в Registry (Staging)'''
    import subprocess
    import sys

    subprocess.check_call(
        [sys.executable, "/opt/project/training/train_with_pycaret.py"]
    )
    subprocess.check_call([sys.executable, "/opt/project/training/register_model.py"])


def skip_stub():
    '''Заглушка для пропуска retraining при отсутствии дрейфа.'''
    print("No significant drift → skipping retraining")


with DAG(
    dag_id="drift_retrain_dag",
    start_date=datetime(2026, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["mlops_hw"],
) as dag:

    check = PythonOperator(
        task_id="check_drift",
        python_callable=check_drift,
        provide_context=True,
    )

    branch = BranchPythonOperator(
        task_id="branch_on_drift",
        python_callable=branch_on_drift,
        provide_context=True,
    )

    retrain = PythonOperator(
        task_id="run_retraining",
        python_callable=run_retraining_real,
    )

    skip = PythonOperator(
        task_id="skip_retraining",
        python_callable=skip_stub,
    )

    done = EmptyOperator(
        task_id="done",
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    check >> branch
    branch >> retrain >> done
    branch >> skip >> done
