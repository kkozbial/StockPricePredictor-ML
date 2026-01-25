"""
Moduł predykcji bankructwa spółek.

Minimalistyczny, skoncentrowany pipeline do:
1. Ładowania danych z staging.bankruptcy_dataset
2. Feature engineering (Altman Z-score, wskaźniki stresu)
3. Treningu modeli klasyfikacyjnych (XGBoost, RF, Logistic)
4. Walidacji i ewaluacji

Przykładowe użycie:
    >>> from src.bankruptcy import train_model, compare_models

    # Trenuj pojedynczy model
    >>> result = train_model(model_name="xgboost")
    >>> print(f"Test F1: {result.metrics['test']['f1']:.4f}")

    # Porównaj wszystkie modele
    >>> comparison = compare_models()
    >>> print(comparison)

    # Cross-validation
    >>> from src.bankruptcy import cross_validate_model
    >>> cv_results = cross_validate_model(model_name="xgboost", n_folds=5)

Struktura modułu:
    - data_loader.py: Ładowanie i przygotowanie danych
    - features.py: Feature engineering
    - model.py: Definicje modeli
    - train.py: Trening i ewaluacja
"""
from __future__ import annotations

# Data loading
from .data_loader import (
    load_bankruptcy_data,
    prepare_features,
    get_train_test_split,
    load_and_prepare,
)

# Feature engineering
from .features import (
    engineer_bankruptcy_features,
    create_altman_z_features,
    create_distress_indicators,
    select_top_features,
    BANKRUPTCY_KEY_FEATURES,
)

# Models
from .model import (
    get_model,
    build_random_forest,
    build_xgboost,
    build_logistic,
    calibrate_model,
    get_feature_importance,
    MODEL_REGISTRY,
)

# Training & Evaluation
from .train import (
    train_model,
    cross_validate_model,
    compare_models,
    calculate_metrics,
    find_optimal_threshold,
    generate_report,
    TrainingResult,
)

__all__ = [
    # Data
    "load_bankruptcy_data",
    "prepare_features",
    "get_train_test_split",
    "load_and_prepare",
    # Features
    "engineer_bankruptcy_features",
    "create_altman_z_features",
    "create_distress_indicators",
    "select_top_features",
    "BANKRUPTCY_KEY_FEATURES",
    # Models
    "get_model",
    "build_random_forest",
    "build_xgboost",
    "build_logistic",
    "calibrate_model",
    "get_feature_importance",
    "MODEL_REGISTRY",
    # Training
    "train_model",
    "cross_validate_model",
    "compare_models",
    "calculate_metrics",
    "find_optimal_threshold",
    "generate_report",
    "TrainingResult",
]
