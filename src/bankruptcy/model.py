"""Modele klasyfikacyjne do predykcji bankructwa."""
from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

LOGGER = logging.getLogger("bankruptcy.model")

# Dostępne modele
MODEL_REGISTRY = {
    "random_forest": {
        "builder": "build_random_forest",
        "requires_scaling": False,
    },
    "xgboost": {
        "builder": "build_xgboost",
        "requires_scaling": False,
    },
    "logistic": {
        "builder": "build_logistic",
        "requires_scaling": True,
    },
}


def build_random_forest(
    n_estimators: int = 300,
    max_depth: int = 10,
    class_weight: str = "balanced",
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Buduje Random Forest Classifier.

    Args:
        n_estimators: Liczba drzew.
        max_depth: Maksymalna głębokość drzew.
        class_weight: Wagi klas ('balanced' dla niezbalansowanych danych).
        random_state: Ziarno losowości.

    Returns:
        Skonfigurowany RandomForestClassifier.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features="sqrt",
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )


def build_xgboost(
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    scale_pos_weight: float | None = None,
    random_state: int = 42,
):
    """
    Buduje XGBoost Classifier.

    Args:
        n_estimators: Liczba drzew.
        max_depth: Maksymalna głębokość.
        learning_rate: Szybkość uczenia.
        scale_pos_weight: Waga klasy pozytywnej (obliczana automatycznie jeśli None).
        random_state: Ziarno losowości.

    Returns:
        Skonfigurowany XGBClassifier.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        LOGGER.error("XGBoost not installed. Install with: pip install xgboost")
        raise

    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight or 1,  # Domyślnie 1 (bez dodatkowej wagi)
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
    )


def build_logistic(
    C: float = 1.0,
    class_weight: str = "balanced",
    max_iter: int = 1000,
    random_state: int = 42,
) -> LogisticRegression:
    """
    Buduje Logistic Regression.

    Args:
        C: Siła regularyzacji (odwrotność).
        class_weight: Wagi klas.
        max_iter: Maksymalna liczba iteracji.
        random_state: Ziarno losowości.

    Returns:
        Skonfigurowany LogisticRegression.
    """
    return LogisticRegression(
        C=C,
        class_weight=class_weight,
        max_iter=max_iter,
        solver="lbfgs",
        random_state=random_state,
        n_jobs=-1,
    )


def get_model(
    model_name: Literal["random_forest", "xgboost", "logistic"] = "xgboost",
    **kwargs,
):
    """
    Pobiera model po nazwie.

    Args:
        model_name: Nazwa modelu.
        **kwargs: Parametry do przekazania do buildera.

    Returns:
        Skonfigurowany model.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

    builder_name = MODEL_REGISTRY[model_name]["builder"]
    builder = globals()[builder_name]

    return builder(**kwargs)


def calibrate_model(model, X, y, method: str = "sigmoid", cv: int = 3):
    """
    Kalibruje prawdopodobieństwa modelu.

    Args:
        model: Wytrenowany model.
        X: Features.
        y: Target.
        method: Metoda kalibracji ('sigmoid' lub 'isotonic').
        cv: Liczba foldów CV.

    Returns:
        Skalibrowany model.
    """
    calibrated = CalibratedClassifierCV(model, method=method, cv=cv)
    calibrated.fit(X, y)
    LOGGER.info("Model calibrated using %s method with %d-fold CV", method, cv)
    return calibrated


def get_feature_importance(
    model,
    feature_names: list[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Pobiera ważność cech z modelu.

    Args:
        model: Wytrenowany model.
        feature_names: Nazwy features.
        top_n: Liczba top features do zwrócenia.

    Returns:
        DataFrame z ważnością cech.
    """
    # Obsłuż CalibratedClassifierCV
    if hasattr(model, "estimators_"):
        # Uśrednij ważność z wielu estymatorów
        importances = np.mean([
            est.feature_importances_ if hasattr(est, "feature_importances_")
            else np.abs(est.coef_[0]) if hasattr(est, "coef_")
            else np.zeros(len(feature_names))
            for est in model.estimators_
        ], axis=0)
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        LOGGER.warning("Model does not have feature_importances_ or coef_")
        return pd.DataFrame()

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    return importance_df.head(top_n).reset_index(drop=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    print("Available models:")
    for name, info in MODEL_REGISTRY.items():
        print(f"  - {name}: requires_scaling={info['requires_scaling']}")

    # Test budowania modeli
    rf = get_model("random_forest")
    print(f"\nRandom Forest: {rf}")

    xgb = get_model("xgboost")
    print(f"XGBoost: {xgb}")

    lr = get_model("logistic")
    print(f"Logistic: {lr}")
