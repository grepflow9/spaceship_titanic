"""
train.py — Обучение, тюнинг, кросс-валидация.
"""

import pandas as pd
import optuna
from optuna.integration import OptunaSearchCV
from optuna.exceptions import ExperimentalWarning
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

import src.config as config
import warnings
from src.models import build_pipeline, evaluate_model, prepare_catboost

warnings.filterwarnings('ignore', category=ExperimentalWarning)
warnings.filterwarnings('ignore', category=FutureWarning, module='optuna')
optuna.logging.set_verbosity(optuna.logging.WARNING)


def evaluate_all(df, models=None, cv=None):
    """
    Прогоняет несколько моделей и возвращает DataFrame с результатами.
    """
    if models is None:
        models = list(config.MODELS.keys())
    if cv is None:
        cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True,
                             random_state=config.RANDOM_STATE)

    results = []
    for name in models:
        scores = evaluate_model(df, name, cv=cv, verbose=False)
        results.append({
            'model':    name,
            'accuracy': scores.mean(),
            'std':      scores.std(),
        })

    df_results = (pd.DataFrame(results)
                    .sort_values('accuracy', ascending=False)
                    .reset_index(drop=True))
    print(df_results.to_string(index=False))
    return df_results


def tune_model_gridsearch(df, model_name, param_grid, cv=None, verbose=True):
    """Подбирает гиперпараметры одной модели через GridSearchCV."""
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

    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X, y)

    if verbose:
        print(f'{model_name:20s}: {search.best_score_:.4f}')
        print(f'    best_params: {search.best_params_}')

    return search


def tune_model_optuna(df, model_name, param_distributions, n_trials=30,
                      cv=None, verbose=True):
    """Подбирает гиперпараметры одной модели через OptunaSearchCV."""
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

    # n_jobs=1 -> чтобы не было oversubscription с внутренним n_jobs моделей
    search = OptunaSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_trials=n_trials,
        cv=cv,
        scoring='accuracy',
        n_jobs=1,
        random_state=config.RANDOM_STATE,
        verbose=0,
    )
    search.fit(X, y)

    if verbose:
        print(f'{model_name:20s}: {search.best_score_:.4f}')
        print(f'    best_params: {search.best_params_}')

    return search


def tune_all(df, param_grids, param_optuna,
             use_gridsearch=True, use_optuna=False,
             n_trials=30, cv=None, models=None):
    """
    Перебирает модели и подбирает гиперпараметры выбранным методом.
    """
    if cv is None:
        cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True,
                             random_state=config.RANDOM_STATE)

    if models is None:
        models = list(config.MODELS.keys())

    results = []
    best_searches = {}

    for name in models:
        if use_gridsearch and name in param_grids:
            search = tune_model_gridsearch(
                df, name,
                param_grid=param_grids[name],
                cv=cv, verbose=True,
            )
            results.append({
                'model':       name,
                'method':      'GridSearch',
                'accuracy':    search.best_score_,
                'best_params': search.best_params_,
            })
            best_searches[name] = search

        if use_optuna and name in param_optuna:
            search = tune_model_optuna(
                df, name,
                param_distributions=param_optuna[name],
                n_trials=n_trials, cv=cv, verbose=True,
            )
            results.append({
                'model':       name,
                'method':      'Optuna',
                'accuracy':    search.best_score_,
                'best_params': search.best_params_,
            })
            best_searches[name] = search

    df_results = (pd.DataFrame(results)
                    .sort_values('accuracy', ascending=False)
                    .reset_index(drop=True))
    print()
    print(df_results[['model', 'method', 'accuracy']].to_string(index=False))

    return df_results, best_searches
