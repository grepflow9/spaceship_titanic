
## Spaceship Titanic - [link](https://www.kaggle.com/competitions/spaceship-titanic)

---

## 📂 Описание файлов

### `EDA.ipynb`

Разведочный анализ данных и итеративный поиск признаков. Ноутбук содержит как анализ, так и эксперименты с фичами.

**1. Первичный анализ:**
- **Обзор датасета**: 8693 строки, 14 колонок. Типы: 6 `float64`, 7 `object`, 1 `bool`.
- **Пропуски**: ~180–220 на колонку. `PassengerId` и `Transported` без пропусков. Наибольшие пропуски — `CryoSleep` (217), `ShoppingMall` (208), `Name`/`Cabin` (~200).
- **`describe()` по числовым признакам**:
  - `Age`: 0–79, медиана 27.
  - `expense`-компоненты: медиана 0, максимумы 14k–30k → сильная правосторонняя асимметрия.

**2. Анализ трат по возрастным бинам:**
- Гистограмма + линия `Transported rate` по возрастам.
- **Ключевой вывод**: до 8 лет трат нет вообще → `expense = 0` для `Age < 8`.
- Пожилые (65+) — ~107 человек (1.23% по IQR), но это **реальные пассажиры**, а не ошибки. Удалять не нужно.

**3. Анализ `Cabin` → `deck`/`num`/`side`:**
- **`Transported rate by Deck`** (barh): палубы `B`, `C` — выше среднего (~0.65+), `E`, `G` — ниже. Сильный сигнал.
- **Heatmap `Deck × Side`**: внутри палубы `side` даёт дополнительный сигнал.
- **`HomePlanet × Deck`** (crosstab):
  - `A`, `B`, `C`, `T` → почти всегда `Europa`.
  - `G` → только `Earth`.
  - `F`, `E`, `D` → смешанные.
- **`Destination × Deck`**: `T` → всегда `TRAPPIST-1e`.
- **Вывод**: эти зависимости позволяют восстановить пропуски в `HomePlanet` и `Destination` по `deck`.

**4. Анализ `VIP`:**
- 8291 `False` vs 199 `True` → заполняем пропуски `False`.

**5. Проверка на выбросы:**
- **`Age`**: гистограмма + boxplot. Выбросов нет, распределение естественное. **Удалять не нужно** — 107 человек старше 64 это реальные пожилые, потеря сигнала.
- **`expense`**: гистограмма + boxplot. Тяжёлый хвост (до 35k+), но это **естественный разброс**, не ошибки. Логарифмирование в FE не применялось, удаление не дало бы прироста.

**6. Эксперименты с фичами (ablation study):**

| Фича | Идея | Итог | Вывод |
|---|---|---|---|
| `is_alone` = `groupSize == 1` | Одиночка или в семье | Все модели −0.002…−0.004 | ❌ Избыточна с `groupSize` |
| `idPersonGroup` (номер в группе) | Порядок пассажира в группе | Не помогла | ❌ Шум |
| `is_child` = `Age < 13` | Дети тратят/ведут себя иначе | LogReg +0.009, KNN +0.002, бустинги ~0 | ✅ Для линейных моделей |
| `expensePerson = expense / groupSize` | Траты на человека | Бустинги −0.007 | ❌ Коллинеарность |
| `CryoSleep` через `group`/`Surname` | Уточнение заполнения | Не помогла | ❌ Текущее правило (`expense > 0`) уже оптимально |
| **Отдельные колонки трат** (`RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck` вместо суммы) | Каждая услуга — отдельный сигнал | **+0.05 ко всем моделям** | ✅ **Главное улучшение** |

**Главный вывод EDA:**
- Разбиение `Cabin` на `deck/num/side` + отдельные колонки трат — **дают основной прирост** (+0.05).
- Заполнение `HomePlanet`/`Destination` через `group`/`Surname`/`deck` — восстанавливает пропуски точно.
- `is_alone`, `expensePerson`, `idPersonGroup` — **не работают**, не использовать.
- `is_child` — **оставить** (полезно для линейных моделей).
---

### `config.py`

Конфигурация проекта:

- **`MODELS`** — реестр моделей: класс, префикс, параметры по умолчанию.
  - KNN, LogReg (L1/L2/ElasticNet), RandomForest, LGBM, XGB, CatBoost.
- **`PARAM_GRIDS`** — сетки гиперпараметров для `GridSearchCV`.
- **`PARAM_OPTUNA`** — распределения для `OptunaSearchCV`.
- **`BEST_PARAMS`** — лучшие гиперпараметры после тюнинга (CatBoost, XGB, LGBM).


---

### `main.py`

Основной пайплайн:

1. **Загрузка данных** — `train.csv`, `test.csv`.
2. **`data_processing(df)`** — Feature Engineering:
   - Группа и `groupSize` из `PassengerId`.
   - `expense` = сумма трат, заполнение пропусков через `CryoSleep`.
   - `Cabin` → `deck` / `num` / `side`.
   - Заполнение `HomePlanet`, `Destination`, `Age` через группы и `Surname`.
   - `is_child` = `Age < 13`.
   - Заполнение пропусков в тратах нулями.
3. **`evaluate_all`** — сравнение базовых моделей через `cross_val_score`.
4. **`tune_all`** — подбор гиперпараметров (`GridSearchCV` / `OptunaSearchCV`).
5. **`predict_ensemble`** — финальное обучение, ансамбль, сабмит.

---

## 📊 Результаты

### До Feature Engineering (сырые данные + базовый FE)

| Модель | CV Accuracy | Лучшие параметры |
|---|---|---|
| KNN | 0.7194 ± 0.0045 | — |
| KNN (tuned) | **0.7429** | `n_neighbors=14`, `weights='uniform'`, `metric='manhattan'` |
| LogReg | 0.7312 ± 0.0091 | — |
| LogReg L1 (Lasso) | 0.7307 | `C=1.0` |
| LogReg L2 (Ridge) | **0.7353** | `C=0.01` |
| LogReg ElasticNet | 0.7306 | `l1_ratio=0.5` |
| DecisionTreeClassifier | 0.6758 ± 0.0120 | — |
| RandomForest | 0.7435 ± 0.0085 | — |
| RandomForest (tuned) | **0.7608** | `n_estimators=800`, `max_depth=30`, `min_samples_split=6`, `min_samples_leaf=7`, `max_features='sqrt'` |
| CatBoost | **0.7638 ± 0.0108** | `iterations=1000`, `depth=6`, `lr=0.05`, `l2=3` |
| LGBM | 0.7573 ± 0.0048 | — |
| XGB | 0.7451 ± 0.0069 | — |


---

### После Feature Engineering

Ключевое изменение: **траты разбиты на 5 отдельных колонок** (`RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`) вместо суммы единой суммы `expense` + новый признак `is_child`.

**Сравнение baseline vs tuned (Optuna):**

| Модель | CV (baseline) | CV (tuned) | Δ | Лучшие параметры |
|---|---|---|---|---|
| KNN | 0.7790 ± 0.0054 | **0.7897** | +0.0107 | `n_neighbors=12`, `weights='uniform'`, `metric='manhattan'` |
| LogReg | 0.7932 ± 0.0044 | **0.7936** | +0.0004 | `C=9.48` |
| LogReg Lasso | 0.7933 | **0.7936** | +0.0003 | `C=0.733` |
| LogReg Ridge | 0.7932 | **0.7936** | +0.0004 | `C=9.48` |
| LogReg ElasticNet | 0.7933 | **0.7935** | +0.0002 | `C=9.57`, `l1_ratio=0.87` |
| RandomForest | 0.8017 ± 0.0092 | **0.8067** | +0.0050 | `n_estimators=700`, `max_depth=25`, `min_samples_split=2`, `min_samples_leaf=4`, `max_features='log2'` |
| LGBM | 0.8118 ± 0.0082 | **0.8131** | +0.0013 | `n_estimators=1100`, `num_leaves=22`, `max_depth=9`, `lr=0.0208`, `min_child_samples=33` |
| XGB | 0.8038 ± 0.0061 | **0.8134** | +0.0096 | `n_estimators=800`, `max_depth=5`, `lr=0.0189`, `min_child_weight=4` |
| **CatBoost** | 0.8108 ± 0.0051 | **0.8153** | **+0.0045** | `iterations=1100`, `depth=6`, `lr=0.0146`, `l2_leaf_reg=2.60` |

**Итоговая таблица после тюнинга (Optuna):**

| Модель | Method | CV Accuracy |
|---|---|---|
| **CatBoost** | Optuna | **0.8153** |
| **XGB** | Optuna | **0.8134** |
| **LGBM** | Optuna | **0.8131** |
| RandomForest | Optuna | 0.8067 |
| LogReg Lasso / LogReg / Ridge | Optuna | 0.7936 |
| LogReg ElasticNet | Optuna | 0.7935 |
| KNN | Optuna | 0.7897 |

---

### После тюнинга гиперпараметров (Optuna)

| Модель | CV Accuracy | Δ |
|---|---|---|
| RandomForest | 0.8067 | +0.0050 |
| LGBM | 0.8131 | +0.0013 |
| XGB | 0.8134 | +0.0096 |
| **CatBoost** | **0.8153** | **+0.0045** |

**Лучшие параметры (CatBoost):**
```python
{
    'iterations':    1100,
    'depth':         6,
    'learning_rate': 0.01457,
    'l2_leaf_reg':   2.60,
}
```

### Финальный сабмит

| Вариант | Public LB   |
|---|-------------|
| LGBM (baseline FE) | **0.79354** |
| Ensemble (CatBoost + XGB + LGBM) | 0.80453     |
| **CatBoost alone (после FE)** | **0.81084** |
