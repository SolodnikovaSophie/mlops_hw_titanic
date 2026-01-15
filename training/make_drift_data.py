import pandas as pd
import numpy as np
import os

"""Этот скрипт генерирует данные с дрейфом на основе тренировочного датасета.
Он изменяет распределения некоторых признаков"""

TRAIN_PATH = os.getenv("TRAIN_PATH", "./data/train.csv")
CURRENT_PATH = os.getenv("CURRENT_PATH", "./data/current.csv")


def make_drift(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Age — сдвиг распределения
    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())
        df["Age"] = df["Age"] * 1.3
        df["Age"] = df["Age"].clip(0, 90)

    # 2️ Fare — билеты стали дороже
    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())
        df["Fare"] = df["Fare"] * 1.8

    # 3️ Pclass — больше пассажиров 3 класса
    if "Pclass" in df.columns:
        np.random.seed(42)
        mask = np.random.rand(len(df)) < 0.4
        df.loc[mask, "Pclass"] = 3

    return df


if __name__ == "__main__":
    train_df = pd.read_csv(TRAIN_PATH)
    drift_df = make_drift(train_df)

    os.makedirs(os.path.dirname(CURRENT_PATH), exist_ok=True)
    drift_df.to_csv(CURRENT_PATH, index=False)

    print("Drifted current.csv generated")
