import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from catboost import CatBoostClassifier, Pool, cv
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import optuna
from optuna.integration import OptunaSearchCV
from optuna.exceptions import ExperimentalWarning


import config
import time
import warnings


warnings.filterwarnings('ignore', category=FutureWarning,
                        module='optuna_integration')
warnings.filterwarnings('ignore', category=ExperimentalWarning)
warnings.filterwarnings('ignore', category=FutureWarning,
                        module='optuna')

optuna.logging.set_verbosity(optuna.logging.WARNING)


find_the_best_parm = 0   # 1 — запускать тюнинг, 0 — пропустить
use_gridsearch     = 0   # 1 — использовать GridSearchCV
use_optuna         = 1   # 1 — использовать OptunaSearchCV

get_table_result = 0 # 1 - показать таблицу с базовыми моделями

MODELS = config.MODELS

time_start = time.time()

def fill_mode(x):
    return x.fillna(x.mode().iloc[0]) if not x.mode().empty else x


def data_processing(df):
    df['group'] = df['PassengerId'].str.split('_').str[0]
    df['groupSize'] = df.groupby('group')['group'].transform('count')

    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['expense'] = df[spend_cols].sum(axis=1, min_count=1)
    df['expense'] = df['expense'].fillna(0)

    df.loc[df['CryoSleep'].isna() & (df['expense'] > 0), 'CryoSleep'] = False
    df.loc[df['CryoSleep'].isna() & (df['expense'] == 0), 'CryoSleep'] = True

    df['expense'] = np.where(df['CryoSleep'] == True, 0, df['expense'])
    df['expense'] = np.where(df['Age'] < 8, 0, df['expense'])

    df['Surname'] = df['Name'].str.split().str[-1]
    df['Surname'] = df.groupby('group')['Surname'].transform(
        lambda x: x.fillna(x.mode().iloc[0]) if not x.mode().empty else x)
    #df['Name'] = df['Name'].str.split().str[0]

    df['Cabin'] = df.groupby(['group'], dropna=False)['Cabin'].transform(fill_mode)
    df['deck'] = df['Cabin'].str.split('/').str[0]
    df['num'] = df['Cabin'].str.split('/').str[1]
    df['side'] = df['Cabin'].str.split('/').str[2]

    df['deck'] = df.groupby('group', dropna=False)['deck'].transform(fill_mode)
    df['deck'] = df['deck'].fillna(df['deck'].mode().iloc[0])

    df['side'] = df.groupby('group', dropna=False)['side'].transform(fill_mode)
    df['side'] = df.groupby('deck', dropna=False)['side'].transform(fill_mode)
    df['side'] = df['side'].fillna(df['side'].mode().iloc[0])

    df['num'] = pd.to_numeric(df['num'], errors='coerce')

    df['num'] = df.groupby('group', dropna=False)['num'].transform(lambda x: x.fillna(x.median()))
    df['num'] = df.groupby('deck', dropna=False)['num'].transform(lambda x: x.fillna(x.median()))
    df['num'] = df['num'].fillna(df['num'].median())

    df.loc[df['HomePlanet'].isna() & df['deck'].isin(['A', 'B', 'C', 'T']), 'HomePlanet'] = 'Europa'
    df.loc[df['HomePlanet'].isna() & (df['deck'] == 'G'), 'HomePlanet'] = 'Earth'
    df['HomePlanet'] = df.groupby('deck', dropna=False)['HomePlanet'].transform(fill_mode)
    df['HomePlanet'] = df.groupby(['group', 'Surname'], dropna=False)['HomePlanet'].transform(fill_mode)
    df['HomePlanet'] = df.groupby(['group'], dropna=False)['HomePlanet'].transform(fill_mode)
    df['HomePlanet'] = df['HomePlanet'].fillna(df['HomePlanet'].mode().iloc[0])

    df.loc[df['Destination'].isna() & (df['deck'] == 'T'), 'Destination'] = 'TRAPPIST-1e'
    df['Destination'] = df.groupby(['group', 'Surname', 'Cabin'], dropna=False)['Destination'].transform(fill_mode)
    df['Destination'] = df['Destination'].fillna(df['Destination'].mode().iloc[0])

    df['VIP'] = df['VIP'].astype('boolean').fillna(False)

    df['Age'] = df.groupby('group', dropna=False)['Age'].transform(lambda x: x.fillna(x.median()))
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['is_child'] = (df['Age'] < 13).astype(int)

    df['RoomService'] = df['RoomService'].fillna(0)
    df['FoodCourt'] = df['FoodCourt'].fillna(0)
    df['ShoppingMall'] = df['ShoppingMall'].fillna(0)
    df['Spa'] = df['Spa'].fillna(0)
    df['VRDeck'] = df['VRDeck'].fillna(0)

    df = df.drop(columns=['PassengerId', 'Name', 'Cabin', 'Surname', 'group'])

    return df


def evaluate_model(df, numeric_cols, category_cols, model_name,
                   cv=None, verbose=True):
    """
    Обучает одну модель через CV и возвращает scores.

    Параметры:
    - df: DataFrame с колонкой 'Transported'
    - numeric_cols: список числовых колонок
    - category_cols: список категориальных колонок
    - model_name: имя модели из MODELS
    - cv: объект кросс-валидации (по умолчанию StratifiedKFold(5))
    - verbose: печатать ли результат
    """
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    if model_name not in MODELS:
        raise ValueError(f'Unknown model: {model_name}. '
                         f'Available: {list(MODELS.keys())}')

    config = MODELS[model_name]
    X = df.drop(columns=['Transported'])
    y = df['Transported'].astype(int)

    # --- Специальная обработка для CatBoost ---
    if config.get('special'):
        X_model = X.copy()
        for c in X_model.columns:
            if str(X_model[c].dtype) == 'boolean':
                X_model[c] = X_model[c].astype(int)

        for c in category_cols:
            X_model[c] = X_model[c].astype(str).astype('category').cat.codes

        pipe = Pipeline([
            (config['prefix'], config['cls'](**config['params'])),
        ])

    # --- Обычные модели (KNN, LogReg, RF, LGBM, XGB) ---
    else:
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(drop='if_binary', sparse_output=False,
                                  handle_unknown='ignore'), category_cols),
        ])
        X_model = X
        pipe = Pipeline([
            ('prep', preprocessor),
            (config['prefix'], config['cls'](**config['params'])),
        ])

    scores = cross_val_score(pipe, X_model, y, cv=cv,
                             scoring='accuracy', n_jobs=-1)

    if verbose:
        print(f'{model_name:15s}: {scores.mean():.4f} ± {scores.std():.4f}')

    return scores


def evaluate_all(df, numeric_cols, category_cols, models=None, cv=None):
    """
    Прогоняет несколько моделей и возвращает DataFrame с результатами.
    """
    if models is None:
        models = list(MODELS.keys())
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    for name in models:
        scores = evaluate_model(df, numeric_cols, category_cols, name,
                                cv=cv, verbose=False)
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


def tune_model_gridsearch(df, numeric_cols, category_cols, model_name,
                          param_grid, cv=None, verbose=True):
    """Подбирает гиперпараметры одной модели через GridSearchCV."""
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    if model_name not in MODELS:
        raise ValueError(f'Unknown model: {model_name}. '
                         f'Available: {list(MODELS.keys())}')

    config = MODELS[model_name]
    X = df.drop(columns=['Transported'])
    y = df['Transported'].astype(int)

    # --- CatBoost: своя обработка категориальных ---
    if config.get('special'):
        X_model = X.copy()
        for c in X_model.columns:
            if str(X_model[c].dtype) == 'boolean':
                X_model[c] = X_model[c].astype(int)
        for c in category_cols:
            X_model[c] = pd.Categorical(X_model[c].astype(str)).codes

        pipe = Pipeline([
            (config['prefix'], config['cls'](**config['params'])),
        ])
    # --- Обычные модели ---
    else:
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(drop='if_binary', sparse_output=False,
                                  handle_unknown='ignore'), category_cols),
        ])
        X_model = X
        pipe = Pipeline([
            ('prep', preprocessor),
            (config['prefix'], config['cls'](**config['params'])),
        ])

    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_model, y)

    if verbose:
        print(f'{model_name:20s}: {search.best_score_:.4f}')
        print(f'    best_params: {search.best_params_}')

    return search


def tune_model_optuna(df, numeric_cols, category_cols, model_name,
                      param_distributions, n_trials=30,
                      cv=None, verbose=True):
    """Подбирает гиперпараметры одной модели через OptunaSearchCV."""
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    if model_name not in MODELS:
        raise ValueError(f'Unknown model: {model_name}. '
                         f'Available: {list(MODELS.keys())}')

    config = MODELS[model_name]
    X = df.drop(columns=['Transported'])
    y = df['Transported'].astype(int)

    if config.get('special'):
        X_model = X.copy()
        for c in X_model.columns:
            if str(X_model[c].dtype) == 'boolean':
                X_model[c] = X_model[c].astype(int)
        for c in category_cols:
            X_model[c] = pd.Categorical(X_model[c].astype(str)).codes

        pipe = Pipeline([
            (config['prefix'], config['cls'](**config['params'])),
        ])
    else:
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(drop='if_binary', sparse_output=False,
                                  handle_unknown='ignore'), category_cols),
        ])
        X_model = X
        pipe = Pipeline([
            ('prep', preprocessor),
            (config['prefix'], config['cls'](**config['params'])),
        ])

    # n_jobs=1 — чтобы не было oversubscription с внутренним n_jobs моделей
    search = OptunaSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_trials=n_trials,
        cv=cv,
        scoring='accuracy',
        n_jobs=1,
        random_state=42,
        verbose=0,
    )
    search.fit(X_model, y)

    if verbose:
        print(f'{model_name:20s}: {search.best_score_:.4f}')
        print(f'    best_params: {search.best_params_}')

    return search


def tune_all(df, numeric_cols, category_cols,
             param_grids, param_optuna,
             use_gridsearch=True, use_optuna=False,
             n_trials=30, cv=None, models=None):
    """
    Перебирает модели и подбирает гиперпараметры выбранным методом.
    """
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    if models is None:
        models = list(MODELS.keys())

    results = []
    best_searches = {}

    for name in models:
        if use_gridsearch and name in param_grids:
            search = tune_model_gridsearch(
                df, numeric_cols, category_cols, name,
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
                df, numeric_cols, category_cols, name,
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

def predict_ensemble(train, test, test_ids, best_params,
                     numeric_cols, category_cols,
                     weights=None, threshold=0.5,
                     output_path='submission.csv'):
    """Обучает модели из best_params, усредняет probs, сохраняет сабмит."""
    X_train = train.drop(columns=['Transported'])
    y_train = train['Transported'].astype(int)
    X_test = test.copy()

    all_probs = {}

    for name, params in best_params.items():
        cfg = MODELS[name]
        prefix = cfg['prefix']
        # Убираем префикс из ключей: 'cat__depth' → 'depth'
        p = {k.replace(f'{prefix}__', ''): v for k, v in params.items()}

        # Данные под модель (CatBoost — cat.codes)
        if cfg.get('special'):
            Xtr = X_train.copy()
            Xte = X_test.copy()
            for c in Xtr.columns:
                if str(Xtr[c].dtype) == 'boolean':
                    Xtr[c] = Xtr[c].astype(int)
                    Xte[c] = Xte[c].astype(int)
            for c in category_cols:
                Xtr[c] = pd.Categorical(Xtr[c].astype(str)).codes
                Xte[c] = pd.Categorical(Xte[c].astype(str)).codes
            pipe = Pipeline([(prefix, cfg['cls'](**p))])
        else:
            pre = ColumnTransformer([
                ('num', StandardScaler(), numeric_cols),
                ('cat', OneHotEncoder(drop='if_binary', sparse_output=False,
                                      handle_unknown='ignore'), category_cols),
            ])
            Xtr, Xte = X_train, X_test
            pipe = Pipeline([('prep', pre), (prefix, cfg['cls'](**p))])

        pipe.fit(Xtr, y_train)
        all_probs[name] = pipe.predict_proba(Xte)[:, 1]
        print(f'{name:10s}: mean_prob={all_probs[name].mean():.4f}')

    # Отдельно результат каждого
    for name, probs in all_probs.items():
        pd.DataFrame({
            'PassengerId': test_ids,
            'Transported': (probs > threshold).astype(bool),
        }).to_csv(f'submission_{name.lower()}.csv', index=False)

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
    print(f'💾 {output_path}: {sub.shape}')
    return sub


train_raw = pd.read_csv('train.csv')
test_raw = pd.read_csv('test.csv')

test_passenger_ids = test_raw['PassengerId'].copy()

train = data_processing(train_raw)
test = data_processing(test_raw)

X_train = train.drop(columns=['Transported'])
y_train = train['Transported'].astype(int)

X_test = test.copy()

numeric_cols = ['Age', 'groupSize', 'expense', 'num',
                'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'is_child']
category_cols = ['HomePlanet', 'Destination', 'deck', 'CryoSleep', 'VIP', 'side']

if get_table_result == 1:
    results = evaluate_all(train, numeric_cols, category_cols)

# для поиска лучших гиперпараметров у всех моделей
if find_the_best_parm == 1:
    PARAM_GRIDS = config.PARAM_GRIDS,
    PARAM_OPTUNA = config.PARAM_OPTUNA

    tuned_results, tuned_searches = tune_all(
        train, numeric_cols, category_cols,
        param_grids=PARAM_GRIDS,
        param_optuna=PARAM_OPTUNA,
        use_gridsearch=use_gridsearch,
        use_optuna=use_optuna,
        n_trials=30,
    )

predict_ensemble(
    train=train,
    test=test,
    test_ids=test_passenger_ids,
    best_params=config.BEST_PARAMS,
    numeric_cols=numeric_cols,
    category_cols=category_cols,
    output_path='submission.csv',
)

elapsed = time.time() - time_start
print(f'Время: {elapsed:.2f} сек')