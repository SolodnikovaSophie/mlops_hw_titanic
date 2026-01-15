from datetime import datetime
from airflow import DAG  # type: ignore
from airflow.operators.python import PythonOperator  # type: ignore

import os
import subprocess
import sys


def run_ab_eval():
    '''Эта функция запускает скрипт ab_evaluate.py, который анализирует
    лог A/B-теста и сохраняет результаты в MLflow Tracking Server.'''
    subprocess.check_call([sys.executable, "/opt/project/training/ab_evaluate.py"])


def promote_if_good():
    '''"Эта функция запускает скрипт promote_if_good.py, который проверяет результаты A/B-теста
    и при выполнении условий переводит модель из Staging в Production в MLflow Model Registry.'''
    subprocess.check_call([sys.executable, "/opt/project/training/promote_if_good.py"])


with DAG(
    dag_id="ab_evaluate_and_promote",
    start_date=datetime(2026, 1, 1),
    schedule_interval=None,  # вручную
    catchup=False,
    tags=["mlops_hw"],
) as dag:

    t1 = PythonOperator(
        task_id="ab_evaluate",
        python_callable=run_ab_eval,
    )

    t2 = PythonOperator(
        task_id="promote_if_good",
        python_callable=promote_if_good,
    )

    t1 >> t2
