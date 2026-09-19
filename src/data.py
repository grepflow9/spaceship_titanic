"""
data.py — Загрузка и сохранение данных.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw'
OUTPUT_DIR = Path(__file__).parent.parent / 'data' / 'processed'


def load_train() -> pd.DataFrame:
    """Загружает тренировочный датасет."""
    return pd.read_csv(DATA_DIR / 'train.csv')


def load_test() -> pd.DataFrame:
    """Загружает тестовый датасет."""
    return pd.read_csv(DATA_DIR / 'test.csv')


def save_processed(train: pd.DataFrame, test: pd.DataFrame):
    """Сохраняет обработанные данные."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUTPUT_DIR / 'train_processed.csv', index=False)
    test.to_csv(OUTPUT_DIR / 'test_processed.csv', index=False)
    print(f'x сохранено в {OUTPUT_DIR}')
