"""
features.py — Feature engineering.
"""

import numpy as np
import pandas as pd


def fe_age(df: pd.DataFrame) -> pd.DataFrame:
    """Создаёт признак is_child (возраст < 13)."""
    df = df.copy()
    df['is_child'] = (df['Age'] < 13).astype(int)
    return df


def fe_expense(df: pd.DataFrame) -> pd.DataFrame:
    """Создаёт признак expense — сумма всех трат."""
    df = df.copy()
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['expense'] = df[spend_cols].sum(axis=1, min_count=1)
    df['expense'] = df['expense'].fillna(0)
    return df


def _compute_group_mode(df, group_col, value_col):
    """Вычисляет моду (наиболее частое значение) для каждой группы."""
    mode_map = (
        df.groupby(group_col)[value_col]
        .apply(lambda x: x.value_counts().index[0] if not x.mode().empty else None)
        .to_dict()
    )
    return mode_map


def data_processing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Основной пайплайн обработки данных.

    Шаги:
        1. Группа + размер группы
        2. Признак expense (сумма трат)
        3. CryoSleep — инферим по expense (если пропуск)
        4. Expense = 0 если CryoSleep или Age < 8
        5. Surname — извлекаем из Name, заполняем по группе
        6. Cabin → deck / num / side — заполняем пропуски по группе
        7. HomePlanet — инферим по deck, заполняем по группе
        8. Destination — заполняем по группе
        9. VIP — boolean
        10. Age — заполняем медианой по группе
        11. is_child — бинарный признак
        12. Траты — заполняем нулями
        13. Удаляем вспомогательные колонки
    """
    df = df.copy()

    # 1. Группа + размер
    df['group'] = df['PassengerId'].str.split('_').str[0]
    df['groupSize'] = df.groupby('group')['group'].transform('count')

    # 2. Expense
    df = fe_expense(df)

    # 3. CryoSleep — инферим по expense
    df.loc[df['CryoSleep'].isna() & (df['expense'] > 0), 'CryoSleep'] = False
    df.loc[df['CryoSleep'].isna() & (df['expense'] == 0), 'CryoSleep'] = True

    # 4. Expense = 0 если CryoSleep или Age < 8
    df['expense'] = np.where(df['CryoSleep'] == True, 0, df['expense'])
    df['expense'] = np.where(df['Age'] < 8, 0, df['expense'])

    # 5. Surname — сортируем по group, берём первое ненулевое значение
    df = df.sort_values('group')
    df['Surname'] = df['Name'].str.split().str[-1]
    df['Surname'] = df.groupby('group', dropna=False)['Surname'].transform('first')

    # 6. Cabin → deck / num / side
    # Вычисляем mode Cabin по группе ОДИН раз и используем для всех колонок
    cabin_mode_map = _compute_group_mode(df, 'group', 'Cabin')
    df['Cabin'] = df['group'].map(cabin_mode_map).fillna(df['Cabin'])

    df['deck'] = df['Cabin'].str.split('/').str[0]
    df['num'] = df['Cabin'].str.split('/').str[1]
    df['side'] = df['Cabin'].str.split('/').str[2]

    # Заполняем пропуски в deck / side по группе (первое ненулевое)
    df['deck'] = df.groupby('group', dropna=False)['deck'].transform('first')
    df['deck'] = df['deck'].fillna(df['deck'].mode().iloc[0])

    df['side'] = df.groupby('group', dropna=False)['side'].transform('first')
    df['side'] = df.groupby('deck', dropna=False)['side'].transform('first')
    df['side'] = df['side'].fillna(df['side'].mode().iloc[0])

    # num — числовой, заполняем медианой (встроенный метод, быстро)
    df['num'] = pd.to_numeric(df['num'], errors='coerce')
    df['num'] = df.groupby('group', dropna=False)['num'].transform('median')
    df['num'] = df.groupby('deck', dropna=False)['num'].transform('median')
    df['num'] = df['num'].fillna(df['num'].median())

    # 7. HomePlanet — инферим по deck, заполняем по группе (transform('first') вместо fill_mode)
    df.loc[df['HomePlanet'].isna() & df['deck'].isin(['A', 'B', 'C', 'T']),
           'HomePlanet'] = 'Europa'
    df.loc[df['HomePlanet'].isna() & (df['deck'] == 'G'),
           'HomePlanet'] = 'Earth'
    df['HomePlanet'] = df.groupby('deck', dropna=False)['HomePlanet'].transform('first')
    df['HomePlanet'] = df.groupby(['group', 'Surname'], dropna=False)['HomePlanet'].transform('first')
    df['HomePlanet'] = df.groupby(['group'], dropna=False)['HomePlanet'].transform('first')
    df['HomePlanet'] = df['HomePlanet'].fillna(df['HomePlanet'].mode().iloc[0])

    # 8. Destination
    df.loc[df['Destination'].isna() & (df['deck'] == 'T'),
           'Destination'] = 'TRAPPIST-1e'
    df['Destination'] = df.groupby(['group', 'Surname', 'Cabin'], dropna=False)['Destination'].transform('first')
    df['Destination'] = df['Destination'].fillna(df['Destination'].mode().iloc[0])

    # 9. VIP
    df['VIP'] = df['VIP'].astype('boolean').fillna(False)

    # 10. Age — заполняем медианой по группе (встроенный метод)
    df['Age'] = df.groupby('group', dropna=False)['Age'].transform('median')
    df['Age'] = df['Age'].fillna(df['Age'].median())

    # 11. is_child
    df = fe_age(df)

    # 12. Траты — заполняем нулями
    for col in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
        df[col] = df[col].fillna(0)

    # 13. Удаляем вспомогательные колонки
    df = df.drop(columns=['PassengerId', 'Name', 'Cabin', 'Surname', 'group'])

    return df
