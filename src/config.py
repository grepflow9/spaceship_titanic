"""
config.py — Все константы проекта: модели, сетки, колонки.
"""

import optuna
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Режимы выполнения
# ---------------------------------------------------------------------------
SHOW_TABLE_RESULT = 1     # 1 — показать таблицу с базовыми моделями
FIND_THE_BEST_PARAM = 0   # 1 — запускать тюнинг, 0 — пропустить
USE_GRIDSEARCH = 0        # 1 — использовать GridSearchCV
USE_OPTUNA = 0            # 1 — использовать OptunaSearchCV

N_TRIALS = 30             # количество trials для Optuna
N_SPLITS = 5              # количество фолдов для CV
RANDOM_STATE = 42         # seed для воспроизводимости

# ---------------------------------------------------------------------------
# Режимы финального этапа
# ---------------------------------------------------------------------------
USE_ENSEMBLE = 0          # 1 — делать ансамбль, 0 — пропустить
MAKE_SUBMISSION = 0       # 1 — делать submission, 0 — пропустить

# ---------------------------------------------------------------------------
# Список колонок
# ---------------------------------------------------------------------------
NUMERIC_COLS = [
    'Age', 'groupSize', 'expense', 'num',
    'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'is_child',
]

CATEGORY_COLS = [
    'HomePlanet', 'Destination', 'deck', 'CryoSleep', 'VIP', 'side',
]

# ---------------------------------------------------------------------------
# Registry моделей
# ---------------------------------------------------------------------------
MODELS = {
    'KNN': {
        'cls':    KNeighborsClassifier,
        'prefix': 'knn',
        'params': {'n_neighbors': 5},
    },
    'LogReg': {
        'cls':    LogisticRegression,
        'prefix': 'clf',
        'params': {'random_state': RANDOM_STATE, 'max_iter': 1000},
    },
    'LogReg_Lasso': {
        'cls':    LogisticRegression,
        'prefix': 'clf',
        'params': {'penalty': 'l1', 'solver': 'liblinear',
                   'random_state': RANDOM_STATE, 'max_iter': 1000},
    },
    'LogReg_Ridge': {
        'cls':    LogisticRegression,
        'prefix': 'clf',
        'params': {'penalty': 'l2', 'solver': 'lbfgs',
                   'random_state': RANDOM_STATE, 'max_iter': 1000},
    },
    'LogReg_ElasticNet': {
        'cls':    LogisticRegression,
        'prefix': 'clf',
        'params': {'penalty': 'elasticnet', 'solver': 'saga',
                   'l1_ratio': 0.5, 'random_state': RANDOM_STATE, 'max_iter': 5000},
    },
    'RandomForest': {
        'cls':    RandomForestClassifier,
        'prefix': 'rfc',
        'params': {'random_state': RANDOM_STATE},
    },
    'LGBM': {
        'cls':    LGBMClassifier,
        'prefix': 'lgbm',
        'params': {'random_state': RANDOM_STATE, 'verbose': -1},
    },
    'XGB': {
        'cls':    XGBClassifier,
        'prefix': 'xgb',
        'params': {'random_state': RANDOM_STATE, 'verbosity': 0},
    },
    'CatBoost': {
        'cls':    CatBoostClassifier,
        'prefix': 'cat',
        'params': {
            'iterations': 1000, 'depth': 6, 'learning_rate': 0.05,
            'l2_leaf_reg': 3, 'random_seed': RANDOM_STATE,
            'verbose': 0, 'task_type': 'CPU',
        },
        'special': True,  # CatBoost требует особой обработки
    },
}

# ---------------------------------------------------------------------------
# GridSearch — фиксированные сетки
# ---------------------------------------------------------------------------
PARAM_GRIDS = {
    'KNN': {
        'knn__n_neighbors': [3, 5, 7, 9],
        'knn__weights':     ['uniform', 'distance'],
        'knn__metric':      ['euclidean', 'manhattan'],
    },
    'LogReg': {
        'clf__C': [0.01, 0.1, 1, 10],
    },
    'LogReg_Lasso': {
        'clf__C': [0.01, 0.1, 1, 10],
    },
    'LogReg_Ridge': {
        'clf__C': [0.01, 0.1, 1, 10],
    },
    'LogReg_ElasticNet': {
        'clf__C':        [0.01, 0.1, 1, 10],
        'clf__l1_ratio': [0.2, 0.5, 0.8],
    },
    'RandomForest': {
        'rfc__n_estimators':      [200, 500],
        'rfc__max_depth':         [None, 10, 20],
        'rfc__min_samples_split': [2, 5],
        'rfc__min_samples_leaf':  [1, 2],
        'rfc__max_features':      ['sqrt', 'log2'],
    },
    'LGBM': {
        'lgbm__n_estimators':  [200, 500, 1000],
        'lgbm__num_leaves':    [15, 31, 63],
        'lgbm__learning_rate': [0.01, 0.05, 0.1],
    },
    'XGB': {
        'xgb__n_estimators':  [200, 500, 1000],
        'xgb__max_depth':     [3, 6, 10],
        'xgb__learning_rate': [0.01, 0.05, 0.1],
    },
    'CatBoost': {
        'cat__iterations':    [500, 1000],
        'cat__depth':         [4, 6, 8],
        'cat__learning_rate': [0.03, 0.05, 0.1],
    },
}

# ---------------------------------------------------------------------------
# Optuna — распределения для тюнинга
# ---------------------------------------------------------------------------
PARAM_OPTUNA = {
    'KNN': {
        'knn__n_neighbors': optuna.distributions.IntDistribution(3, 15),
        'knn__weights':     optuna.distributions.CategoricalDistribution(['uniform', 'distance']),
        'knn__metric':      optuna.distributions.CategoricalDistribution(['euclidean', 'manhattan']),
    },
    'LogReg': {
        'clf__C': optuna.distributions.FloatDistribution(1e-3, 10, log=True),
    },
    'LogReg_Lasso': {
        'clf__C': optuna.distributions.FloatDistribution(1e-3, 10, log=True),
    },
    'LogReg_Ridge': {
        'clf__C': optuna.distributions.FloatDistribution(1e-3, 10, log=True),
    },
    'LogReg_ElasticNet': {
        'clf__C':        optuna.distributions.FloatDistribution(1e-3, 10, log=True),
        'clf__l1_ratio': optuna.distributions.FloatDistribution(0, 1),
    },
    'RandomForest': {
        'rfc__n_estimators':      optuna.distributions.IntDistribution(200, 1000, step=100),
        'rfc__max_depth':         optuna.distributions.IntDistribution(5, 30),
        'rfc__min_samples_split': optuna.distributions.IntDistribution(2, 20),
        'rfc__min_samples_leaf':  optuna.distributions.IntDistribution(1, 10),
        'rfc__max_features':      optuna.distributions.CategoricalDistribution(['sqrt', 'log2']),
    },
    'LGBM': {
        'lgbm__n_estimators':      optuna.distributions.IntDistribution(200, 1500, step=100),
        'lgbm__num_leaves':        optuna.distributions.IntDistribution(15, 127),
        'lgbm__max_depth':         optuna.distributions.IntDistribution(3, 12),
        'lgbm__learning_rate':     optuna.distributions.FloatDistribution(0.01, 0.2, log=True),
        'lgbm__min_child_samples': optuna.distributions.IntDistribution(5, 50),
    },
    'XGB': {
        'xgb__n_estimators':     optuna.distributions.IntDistribution(200, 1500, step=100),
        'xgb__max_depth':        optuna.distributions.IntDistribution(3, 12),
        'xgb__learning_rate':    optuna.distributions.FloatDistribution(0.01, 0.2, log=True),
        'xgb__min_child_weight': optuna.distributions.IntDistribution(1, 20),
    },
    'CatBoost': {
        'cat__iterations':    optuna.distributions.IntDistribution(500, 1500, step=100),
        'cat__depth':         optuna.distributions.IntDistribution(4, 10),
        'cat__learning_rate': optuna.distributions.FloatDistribution(0.01, 0.2, log=True),
        'cat__l2_leaf_reg':   optuna.distributions.FloatDistribution(1, 10),
    },
}

# ---------------------------------------------------------------------------
# Лучшие гиперпараметры (результат тюнинга)
# ---------------------------------------------------------------------------
BEST_PARAMS = {
    'CatBoost': {
        'cat__iterations': 1100,
        'cat__depth': 6,
        'cat__learning_rate': 0.01457,
        'cat__l2_leaf_reg': 2.60,
    },
    'XGB': {
        'xgb__n_estimators': 800,
        'xgb__max_depth': 5,
        'xgb__learning_rate': 0.018896,
        'xgb__min_child_weight': 4,
    },
    'LGBM': {
        'lgbm__n_estimators': 1100,
        'lgbm__num_leaves': 22,
        'lgbm__max_depth': 9,
        'lgbm__learning_rate': 0.020845,
        'lgbm__min_child_samples': 33,
    },
}
