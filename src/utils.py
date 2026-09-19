"""
utils.py — Вспомогательные функции.
"""

import pandas as pd


def fill_mode(x: pd.Series) -> pd.Series:
    """Заполняет пропуски модой (если есть)."""
    return x.fillna(x.mode().iloc[0]) if not x.mode().empty else x
