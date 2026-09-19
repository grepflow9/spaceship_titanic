"""
train.py — Обучение, тюнинг, кросс-валидация.
"""

import pandas as pd
import optuna
from optuna.integration import OptunaSearchCV
from optuna.exceptions import ExperimentalWarning
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

import time

import src.config as config
import warnings
from src.models import build_pipeline, evaluate_model, prepare_catboost

warnings.filterwarnings('ignore', category=ExperimentalWarning)
warnings.filterwarnings('ignore', category=FutureWarning, module='optuna')
optuna.logging.set_verbosity(optuna.logging.WARNING)


def format_time(seconds):
    """Format elapsed time into human-readable string."""
    if seconds < 60:
        return "%.1fs" % seconds
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return "%dm %.1fs" % (m, s)
    else:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return "%dh %dm %.1fs" % (h, m, s)


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
    # Тайминг
    t0 = time.time()
    search.fit(X, y)
    elapsed = time.time() - t0

    if verbose:
        print(f'{model_name:20s}: {search.best_score_:.4f}  [{format_time(elapsed)}]')
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
    # Тайминг
    t0 = time.time()
    search.fit(X, y)
    elapsed = time.time() - t0

    if verbose:
        print(f'{model_name:20s}: {search.best_score_:.4f}  [{format_time(elapsed)}]')
        print(f'    best_params: {search.best_params_}')

    return search


def tune_all(df, params, cv=None, models=None):
    """
    Перебирает модели и подбирает гиперпараметры выбранным методом.
    
    params — единый словарь config.PARAMS:
        {model_name: {'method': 'grid'|'optuna', 'n_trials': N, 'params': {...}}}
    """
    if cv is None:
        cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True,
                             random_state=config.RANDOM_STATE)

    if models is None:
        models = list(config.MODELS.keys())

    results = []
    best_searches = {}

    for name in models:
        if name not in params:
            continue

        method = params[name]['method']
        param_dict = params[name]['params']

        if method == 'grid':
            t0 = time.time()
            search = tune_model_gridsearch(
                df, name,
                param_grid=param_dict,
                cv=cv, verbose=False,
            )
            elapsed = time.time() - t0
            results.append({
                'model':       name,
                'method':      'GridSearch',
                'accuracy':    search.best_score_,
                'best_params': search.best_params_,
                'time':        elapsed,
            })
            best_searches[name] = search

        elif method == 'optuna':
            # Читаем n_trials из params[name], а не из глобального config
            n_trials = params[name].get('n_trials', 30)
            t0 = time.time()
            search = tune_model_optuna(
                df, name,
                param_distributions=param_dict,
                n_trials=n_trials, cv=cv, verbose=False,
            )
            elapsed = time.time() - t0
            results.append({
                'model':       name,
                'method':      'Optuna',
                'accuracy':    search.best_score_,
                'best_params': search.best_params_,
                'time':        elapsed,
            })
            best_searches[name] = search

    df_results = (pd.DataFrame(results)
                    .sort_values('accuracy', ascending=False)
                    .reset_index(drop=True))
    # Добавляем колонку с форматированным временем
    df_results['time_fmt'] = df_results['time'].apply(format_time)
    # Форматируем best_params для отображения
    df_results['best_params'] = df_results['best_params'].apply(
        lambda x: ', '.join(f'{k}={v}' for k, v in x.items())
    )
    print()
    print(df_results[['model', 'method', 'accuracy', 'time_fmt', 'best_params']].to_string(index=False))

    return df_results, best_searches
