"""
config.py — Все константы проекта: модели, колонки, параметры тюнинга.
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
FIND_THE_BEST_PARAM = 1   # 1 — запускать тюнинг, 0 — пропустить

N_SPLITS = 5              # количество фолдов для CV
RANDOM_STATE = 42         # seed для воспроизводимости

# ---------------------------------------------------------------------------
# Режимы финального этапа
# ---------------------------------------------------------------------------
USE_ENSEMBLE = 1          # 1 — делать ансамбль, 0 — пропустить
MAKE_SUBMISSION = 1       # 1 — делать submission, 0 — пропустить

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
        'params': {'random_state': RANDOM_STATE, 'verbosity': 0,
                   'tree_method': 'hist', 'device': 'cpu'},
    },
    'CatBoost': {
        'cls':    CatBoostClassifier,
        'prefix': 'cat',
        'params': {
            'iterations': 1100, 'depth': 7, 'learning_rate': 0.01457,
            'l2_leaf_reg': 3, 'random_seed': RANDOM_STATE,
            'verbose': 0, 'task_type': 'CPU',
        },
        'special': True,  # CatBoost требует особой обработки
    },
}

# ---------------------------------------------------------------------------
# Единые параметры тюнинга с указанием метода
# ---------------------------------------------------------------------------
# Каждая модель:
#   {'method': 'grid' | 'optuna', 'n_trials': N, 'params': {...}}
# 'grid'  -> n_trials игнорируется (GridSearchCV работает по сетке)
# 'optuna' -> n_trials передаётся в OptunaSearchCV
# ---------------------------------------------------------------------------
PARAMS = {
    # ===== GridSearch (малые сетки, быстро) =====
    'KNN': {
        'method':   'grid',
        'n_trials': None,
        'params': {
            'knn__n_neighbors': [3, 5, 7, 9, 12, 15],
            'knn__weights':     ['uniform', 'distance'],
            'knn__metric':      ['euclidean', 'manhattan'],
        },
    },
    'LogReg': {
        'method':   'grid',
        'n_trials': None,
        'params': {
            'clf__C': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
        },
    },
    'LogReg_Lasso': {
        'method':   'grid',
        'n_trials': None,
        'params': {
            'clf__C': [0.01, 0.1, 1, 10],
        },
    },
    'LogReg_Ridge': {
        'method':   'grid',
        'n_trials': None,
        'params': {
            'clf__C': [0.01, 0.1, 1, 10],
        },
    },
    'LogReg_ElasticNet': {
        'method':   'grid',
        'n_trials': None,
        'params': {
            'clf__C':        [0.01, 0.1, 1, 10],
            'clf__l1_ratio': [0.2, 0.5, 0.8],
        },
    },
    # ===== Optuna (большие пространства, гибкий поиск) =====
    'RandomForest': {
        'method':   'optuna',
        'n_trials': 10,
        'params': {
            'rfc__n_estimators':      optuna.distributions.IntDistribution(200, 400, step=100),
            'rfc__max_depth':         optuna.distributions.IntDistribution(8, 15),
            'rfc__min_samples_split': optuna.distributions.IntDistribution(2, 10),
            'rfc__min_samples_leaf':  optuna.distributions.IntDistribution(1, 5),
            'rfc__max_features':      optuna.distributions.CategoricalDistribution(['sqrt', 'log2']),
        },
    },
    'LGBM': {
        'method':   'optuna',
        'n_trials': 20,
        'params': {
            'lgbm__n_estimators':      optuna.distributions.IntDistribution(400, 1500, step=100),
            'lgbm__num_leaves':        optuna.distributions.IntDistribution(15, 63),
            'lgbm__max_depth':         optuna.distributions.IntDistribution(3, 5),
            'lgbm__learning_rate':     optuna.distributions.FloatDistribution(0.01, 0.08, log=True),
            'lgbm__min_child_samples': optuna.distributions.IntDistribution(5, 30),
        },
    },
    'XGB': {
        'method':   'optuna',
        'n_trials': 20,
        'params': {
            'xgb__n_estimators':     optuna.distributions.IntDistribution(400, 900, step=100),
            'xgb__max_depth':        optuna.distributions.IntDistribution(3, 7),
            'xgb__learning_rate':    optuna.distributions.FloatDistribution(0.01, 0.08, log=True),
            'xgb__min_child_weight': optuna.distributions.IntDistribution(1, 10),
        },
    },
    'CatBoost': {
        'method':   'optuna',
        'n_trials': 15,
        'params': {
            'cat__iterations':    optuna.distributions.IntDistribution(300, 1500, step=100),
            'cat__depth':         optuna.distributions.IntDistribution(5, 7),
            'cat__learning_rate': optuna.distributions.FloatDistribution(0.01, 0.08, log=True),
            'cat__l2_leaf_reg':   optuna.distributions.FloatDistribution(1, 5),
        },
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
