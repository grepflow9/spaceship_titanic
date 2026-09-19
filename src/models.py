"""
models.py — Registry моделей, build_pipeline, evaluate_model.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold

import src.config as config


def prepare_catboost(X: pd.DataFrame) -> pd.DataFrame:
    """
    Подготавливает DataFrame для CatBoost:
        - boolean -> int
        - categorical -> cat.codes
    """
    X = X.copy()
    for c in X.columns:
        if str(X[c].dtype) == 'boolean':
            X[c] = X[c].astype(int)
    for c in config.CATEGORY_COLS:
        if c in X.columns:
            X[c] = pd.Categorical(X[c].astype(str)).codes
    return X


def build_pipeline(model_name: str, params: dict = None) -> Pipeline:
    """
    Строит sklearn Pipeline для заданной модели.

    Для CatBoost: простой Pipeline с одной стадией (нет стандартизации).
    Для остальных: ColumnTransformer (num + cat) -> классификатор.
    """
    cfg = config.MODELS[model_name]
    prefix = cfg['prefix']
    cls = cfg['cls']
    base_params = cfg['params']

    # Сливаем базовые параметры с переданными (переданные приоритетнее)
    merged = {**base_params, **(params or {})}

    if cfg.get('special'):
        # CatBoost -> без ColumnTransformer
        return Pipeline([
            (prefix, cls(**merged)),
        ])
    else:
        # Остальные -> ColumnTransformer
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), config.NUMERIC_COLS),
            ('cat', OneHotEncoder(drop='if_binary', sparse_output=False,
                                   handle_unknown='ignore'), config.CATEGORY_COLS),
        ])
        return Pipeline([
            ('prep', preprocessor),
            (prefix, cls(**merged)),
        ])


def evaluate_model(df, model_name, cv=None, verbose=True):
    """
    Обучает одну модель через CV и возвращает scores.

    Параметры:
        df: DataFrame с колонкой 'Transported'
        model_name: имя модели из config.MODELS
        cv: объект кросс-валидации (по умолчанию StratifiedKFold(5))
        verbose: печатать ли результат
    """
    if cv is None:
        cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True,
                             random_state=config.RANDOM_STATE)

    if model_name not in config.MODELS:
        raise ValueError(f'Unknown model: {model_name}. '
                         f'Available: {list(config.MODELS.keys())}')

    cfg = config.MODELS[model_name]
    X = df.drop(columns=['Transported'])
    y = df['Transported'].astype(int)

    # Подготовка данных
    if cfg.get('special'):
        X = prepare_catboost(X)
        pipe = build_pipeline(model_name)
    else:
        pipe = build_pipeline(model_name)

    scores = cross_val_score(pipe, X, y, cv=cv,
                             scoring='accuracy', n_jobs=-1)

    if verbose:
        print(f'{model_name:15s}: {scores.mean():.4f} +/- {scores.std():.4f}')

    return scores
