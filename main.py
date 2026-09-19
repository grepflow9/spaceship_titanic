"""
main.py — Точка входа. Загружает данные, обучает, делает сабмит.
"""

import time

import src.config as config
from src.data import load_train, load_test
from src.features import data_processing
from src.train import evaluate_all, tune_all
from src.predict import ensemble


def main():
    time_start = time.time()

    # 1. Загрузка
    print("=== Загрузка данных ===")
    t_load_start = time.time()
    train_raw = load_train()
    test_raw = load_test()
    test_ids = test_raw['PassengerId'].copy()
    t_load = time.time() - t_load_start
    print(f" [TIME] Время загрузки: {t_load:.2f} сек")

    # 2. Обработка
    print("=== Обработка данных ===")
    t_proc_start = time.time()
    train = data_processing(train_raw)
    test = data_processing(test_raw)
    t_proc = time.time() - t_proc_start
    print(f" [TIME] Время обработки: {t_proc:.2f} сек")
    print("=== Обработка данных завершена ===")

    # 3. Оценка базовых моделей
    if config.SHOW_TABLE_RESULT:
        print("=== Базовые модели ===")
        evaluate_all(train)

    # 4. Тюнинг
    if config.FIND_THE_BEST_PARAM:
        print("\n=== Тюнинг ===")
        tuned_results, tuned_searches = tune_all(
            train,
            params=config.PARAMS,
            n_trials=config.N_TRIALS,
        )

    # 5. Финальный сабмит
    if config.USE_ENSEMBLE:
        print("\n=== Ансамбли ===")
        ensemble(
            train=train,
            test=test,
            test_ids=test_ids,
            best_params=config.BEST_PARAMS,
            output_path='submissions/submission.csv',
        )


if __name__ == '__main__':
    main()
