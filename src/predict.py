"""
predict.py — Финальное обучение, ансамбль, сабмит.
"""

import pandas as pd

import src.config as config
from src.models import build_pipeline, prepare_catboost
from src.config import MODELS


def _strip_prefix(name: str, params: dict) -> dict:
    """Убирает префикс из ключей: 'cat__depth' -> 'depth'."""
    cfg = config.MODELS[name]
    prefix = cfg['prefix']
    return {k.replace(f'{prefix}__', ''): v for k, v in params.items()}


def fit_final_model(train, model_name, best_params, numeric_cols=None,
                    category_cols=None):
    """
    Обучает одну модель на полном датасете.

    Возвращает готовый pipeline.
    """
    if numeric_cols is None:
        numeric_cols = config.NUMERIC_COLS
    if category_cols is None:
        category_cols = config.CATEGORY_COLS

    X_train = train.drop(columns=['Transported'])
    y_train = train['Transported'].astype(int)

    p = _strip_prefix(model_name, best_params)
    pipe = build_pipeline(model_name, p)
    # prepare_catboost: кодирует категориальные колонки в int для CatBoost
    if MODELS[model_name].get("special"):
        X_train_prep = prepare_catboost(X_train)
    else:
        X_train_prep = X_train
    pipe.fit(X_train_prep, y_train)
    return pipe


def make_submission(pipe, test, test_ids, threshold=0.5,
                    output_path='submission.csv'):
    """
    Сохраняет предсказания модели в CSV.
    """
    probs = pipe.predict_proba(test)[:, 1]
    preds = (probs > threshold).astype(bool)
    sub = pd.DataFrame({
        'PassengerId': test_ids,
        'Transported': preds,
    })
    sub.to_csv(output_path, index=False)
    print(f'Доля True: {preds.mean():.4f}')
    print(f'x {output_path}: {sub.shape}')
    return sub


def ensemble(train, test, test_ids, best_params,
             numeric_cols=None, category_cols=None,
             weights=None, threshold=0.5,
             output_path='submission/submission.csv'):
    """
    Обучает модели из best_params, усредняет probs, сохраняет сабмит.
    """
    if numeric_cols is None:
        numeric_cols = config.NUMERIC_COLS
    if category_cols is None:
        category_cols = config.CATEGORY_COLS

    X_train = train.drop(columns=['Transported'])
    y_train = train['Transported'].astype(int)
    X_test = test.copy()

    all_probs = {}

    for name, params in best_params.items():
        p = _strip_prefix(name, params)
        pipe = build_pipeline(name, p)
        # prepare_catboost: кодирует категориальные колонки в int для CatBoost
        if MODELS[name].get("special"):
            X_train_prep = prepare_catboost(X_train)
        else:
            X_train_prep = X_train
        pipe.fit(X_train_prep, y_train)
        if MODELS[name].get("special"):
            all_probs[name] = pipe.predict_proba(prepare_catboost(X_test))[:, 1]
        else:
            all_probs[name] = pipe.predict_proba(X_test)[:, 1]
        print(f'{name:10s}: mean_prob={all_probs[name].mean():.4f}')

    # Отдельно результат каждой модели
    for name, probs in all_probs.items():
        pd.DataFrame({
            'PassengerId': test_ids,
            'Transported': (probs > threshold).astype(bool),
        }).to_csv(f'submission/submission_{name.lower()}.csv', index=False)

    # Усреднение
    if weights is None:
        weights = {n: 1.0 for n in all_probs}
    total = sum(weights[n] for n in all_probs)
    avg = sum(all_probs[n] * weights[n] for n in all_probs) / total

    # Сабмит
    preds = (avg > threshold).astype(bool)
    sub = pd.DataFrame({'PassengerId': test_ids, 'Transported': preds})
    sub.to_csv(output_path, index=False)

    print(f'\nДоля True: {preds.mean():.4f}')
    print(f'x {output_path}: {sub.shape}')
    return sub
