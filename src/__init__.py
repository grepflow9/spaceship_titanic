"""
src — Пакет для Space Titanic.
"""

from src import config, data, features, models, train, predict
from src.data import load_train, load_test
from src.features import data_processing
from src.models import build_pipeline, evaluate_model
from src.train import evaluate_all, tune_all
from src.predict import ensemble
