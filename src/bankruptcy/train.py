"""Trening, walidacja i ewaluacja modelu bankructwa."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold

from .data_loader import load_bankruptcy_data, prepare_features, get_train_test_split
from .features import engineer_bankruptcy_features
from .model import get_model, calibrate_model, get_feature_importance, MODEL_REGISTRY

LOGGER = logging.getLogger("bankruptcy.train")


@dataclass
class TrainingResult:
    """Wynik treningu modelu."""
    model: object
    feature_names: list[str]
    metrics: dict
    feature_importance: pd.DataFrame
    predictions: pd.DataFrame
    threshold: float = 0.5


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict:
    """
    Oblicza metryki klasyfikacji.

    Args:
        y_true: Prawdziwe etykiety.
        y_pred: Przewidziane etykiety.
        y_prob: Prawdopodobieństwa klasy pozytywnej.

    Returns:
        Słownik z metrykami.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["roc_auc"] = None

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["true_negatives"] = cm[0, 0]
    metrics["false_positives"] = cm[0, 1]
    metrics["false_negatives"] = cm[1, 0]
    metrics["true_positives"] = cm[1, 1]

    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
) -> tuple[float, float]:
    """
    Znajduje optymalny próg klasyfikacji.

    Args:
        y_true: Prawdziwe etykiety.
        y_prob: Prawdopodobieństwa.
        metric: Metryka do optymalizacji ('f1', 'precision', 'recall').

    Returns:
        Tuple (optymalny_próg, wartość_metryki).
    """
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_threshold = 0.5
    best_score = 0

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)

        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            score = f1_score(y_true, y_pred, zero_division=0)

        if score > best_score:
            best_score = score
            best_threshold = thresh

    return best_threshold, best_score


def train_model(
    model_name: Literal["random_forest", "xgboost", "logistic"] = "xgboost",
    test_size: float = 0.2,
    add_features: bool = True,
    calibrate: bool = True,
    optimize_threshold: bool = True,
    db_path: Path | None = None,
) -> TrainingResult:
    """
    Trenuje model bankructwa.

    Args:
        model_name: Nazwa modelu.
        test_size: Proporcja test set.
        add_features: Czy dodać feature engineering.
        calibrate: Czy kalibrować prawdopodobieństwa.
        optimize_threshold: Czy optymalizować próg.
        db_path: Ścieżka do bazy.

    Returns:
        TrainingResult z modelem i metrykami.
    """
    LOGGER.info("=" * 60)
    LOGGER.info("Training bankruptcy model: %s", model_name)
    LOGGER.info("=" * 60)

    # 1. Załaduj dane
    LOGGER.info("Step 1: Loading data...")
    df = load_bankruptcy_data(db_path)

    # 2. Feature engineering
    if add_features:
        LOGGER.info("Step 2: Feature engineering...")
        df = engineer_bankruptcy_features(df, add_altman=True, add_distress=True)

    # 3. Train/test split
    LOGGER.info("Step 3: Train/test split...")
    train_df, test_df = get_train_test_split(df, test_size=test_size)

    # 4. Prepare features
    LOGGER.info("Step 4: Preparing features...")
    X_train, y_train, feature_names = prepare_features(train_df)
    X_test, y_test, _ = prepare_features(test_df)

    # Upewnij się, że test ma te same kolumny
    for col in feature_names:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[feature_names]

    # 5. Skalowanie (jeśli wymagane)
    requires_scaling = MODEL_REGISTRY[model_name]["requires_scaling"]
    if requires_scaling:
        LOGGER.info("Step 5: Scaling features...")
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_names)
    else:
        scaler = None

    # 6. Trening
    LOGGER.info("Step 6: Training model...")
    model = get_model(model_name)
    model.fit(X_train, y_train)

    # 7. Kalibracja
    if calibrate and hasattr(model, "predict_proba"):
        LOGGER.info("Step 7: Calibrating probabilities...")
        model = calibrate_model(model, X_train, y_train)

    # 8. Predykcje
    LOGGER.info("Step 8: Making predictions...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    y_prob_train = None
    y_prob_test = None
    if hasattr(model, "predict_proba"):
        y_prob_train = model.predict_proba(X_train)[:, 1]
        y_prob_test = model.predict_proba(X_test)[:, 1]

    # 9. Optymalizacja progu
    threshold = 0.5
    if optimize_threshold and y_prob_test is not None:
        LOGGER.info("Step 9: Optimizing threshold...")
        threshold, _ = find_optimal_threshold(y_test, y_prob_test)
        y_pred_test = (y_prob_test >= threshold).astype(int)
        LOGGER.info("Optimal threshold: %.2f", threshold)

    # 10. Metryki
    LOGGER.info("Step 10: Calculating metrics...")
    train_metrics = calculate_metrics(y_train, y_pred_train, y_prob_train)
    test_metrics = calculate_metrics(y_test, y_pred_test, y_prob_test)

    metrics = {
        "train": train_metrics,
        "test": test_metrics,
        "threshold": threshold,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "model_name": model_name,
    }

    # 11. Feature importance
    LOGGER.info("Step 11: Calculating feature importance...")
    importance = get_feature_importance(model, feature_names)

    # 12. Predictions DataFrame
    predictions = test_df[["ticker", "date"]].copy()
    predictions = predictions.iloc[:len(y_test)].reset_index(drop=True)
    predictions["y_true"] = y_test.values
    predictions["y_pred"] = y_pred_test
    if y_prob_test is not None:
        predictions["y_prob"] = y_prob_test

    # Log results
    LOGGER.info("=" * 60)
    LOGGER.info("TRAINING COMPLETE")
    LOGGER.info("=" * 60)
    LOGGER.info("Train Accuracy: %.4f", train_metrics["accuracy"])
    LOGGER.info("Test Accuracy:  %.4f", test_metrics["accuracy"])
    LOGGER.info("Test Precision: %.4f", test_metrics["precision"])
    LOGGER.info("Test Recall:    %.4f", test_metrics["recall"])
    LOGGER.info("Test F1:        %.4f", test_metrics["f1"])
    if test_metrics.get("roc_auc"):
        LOGGER.info("Test ROC-AUC:   %.4f", test_metrics["roc_auc"])
    LOGGER.info("Optimal Threshold: %.2f", threshold)
    LOGGER.info("=" * 60)

    return TrainingResult(
        model=model,
        feature_names=feature_names,
        metrics=metrics,
        feature_importance=importance,
        predictions=predictions,
        threshold=threshold,
    )


def cross_validate_model(
    model_name: Literal["random_forest", "xgboost", "logistic"] = "xgboost",
    n_folds: int = 5,
    add_features: bool = True,
    db_path: Path | None = None,
) -> dict:
    """
    Wykonuje cross-validation modelu.

    Args:
        model_name: Nazwa modelu.
        n_folds: Liczba foldów.
        add_features: Czy dodać feature engineering.
        db_path: Ścieżka do bazy.

    Returns:
        Słownik z wynikami CV.
    """
    LOGGER.info("Cross-validating %s with %d folds...", model_name, n_folds)

    # Załaduj dane
    df = load_bankruptcy_data(db_path)

    if add_features:
        df = engineer_bankruptcy_features(df)

    X, y, feature_names = prepare_features(df)

    # Skalowanie jeśli potrzebne
    if MODEL_REGISTRY[model_name]["requires_scaling"]:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)

    # Model
    model = get_model(model_name)

    # Cross-validation
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    scores = {
        "accuracy": cross_val_score(model, X, y, cv=cv, scoring="accuracy"),
        "precision": cross_val_score(model, X, y, cv=cv, scoring="precision"),
        "recall": cross_val_score(model, X, y, cv=cv, scoring="recall"),
        "f1": cross_val_score(model, X, y, cv=cv, scoring="f1"),
        "roc_auc": cross_val_score(model, X, y, cv=cv, scoring="roc_auc"),
    }

    results = {
        "model_name": model_name,
        "n_folds": n_folds,
        "scores": {k: v.tolist() for k, v in scores.items()},
        "mean_scores": {k: v.mean() for k, v in scores.items()},
        "std_scores": {k: v.std() for k, v in scores.items()},
    }

    LOGGER.info("CV Results for %s:", model_name)
    for metric, values in scores.items():
        LOGGER.info("  %s: %.4f (+/- %.4f)", metric, values.mean(), values.std() * 2)

    return results


def compare_models(
    model_names: list[str] | None = None,
    test_size: float = 0.2,
    db_path: Path | None = None,
) -> pd.DataFrame:
    """
    Porównuje wiele modeli.

    Args:
        model_names: Lista nazw modeli. Jeśli None, testuje wszystkie.
        test_size: Proporcja test set.
        db_path: Ścieżka do bazy.

    Returns:
        DataFrame z porównaniem modeli.
    """
    if model_names is None:
        model_names = list(MODEL_REGISTRY.keys())

    results = []

    for name in model_names:
        LOGGER.info("\nTraining %s...", name)
        try:
            result = train_model(
                model_name=name,
                test_size=test_size,
                db_path=db_path,
            )

            results.append({
                "model": name,
                "train_accuracy": result.metrics["train"]["accuracy"],
                "test_accuracy": result.metrics["test"]["accuracy"],
                "test_precision": result.metrics["test"]["precision"],
                "test_recall": result.metrics["test"]["recall"],
                "test_f1": result.metrics["test"]["f1"],
                "test_roc_auc": result.metrics["test"].get("roc_auc"),
                "threshold": result.threshold,
            })
        except Exception as e:
            LOGGER.error("Failed to train %s: %s", name, e)

    comparison = pd.DataFrame(results)
    comparison = comparison.sort_values("test_f1", ascending=False)

    return comparison.reset_index(drop=True)


def generate_report(result: TrainingResult, output_dir: Path | None = None) -> str:
    """
    Generuje raport tekstowy z wyników.

    Args:
        result: Wynik treningu.
        output_dir: Katalog do zapisu. Jeśli None, zwraca tylko tekst.

    Returns:
        Tekst raportu.
    """
    report = []
    report.append("=" * 60)
    report.append("BANKRUPTCY PREDICTION MODEL REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 60)
    report.append("")

    # Model info
    report.append("MODEL INFORMATION")
    report.append("-" * 40)
    report.append(f"Model: {result.metrics['model_name']}")
    report.append(f"Training samples: {result.metrics['n_train']}")
    report.append(f"Test samples: {result.metrics['n_test']}")
    report.append(f"Features: {len(result.feature_names)}")
    report.append(f"Optimal threshold: {result.threshold:.2f}")
    report.append("")

    # Metrics
    report.append("TEST SET METRICS")
    report.append("-" * 40)
    test = result.metrics["test"]
    report.append(f"Accuracy:  {test['accuracy']:.4f}")
    report.append(f"Precision: {test['precision']:.4f}")
    report.append(f"Recall:    {test['recall']:.4f}")
    report.append(f"F1 Score:  {test['f1']:.4f}")
    if test.get("roc_auc"):
        report.append(f"ROC-AUC:   {test['roc_auc']:.4f}")
    report.append("")

    # Confusion matrix
    report.append("CONFUSION MATRIX")
    report.append("-" * 40)
    report.append(f"True Negatives:  {test['true_negatives']}")
    report.append(f"False Positives: {test['false_positives']}")
    report.append(f"False Negatives: {test['false_negatives']}")
    report.append(f"True Positives:  {test['true_positives']}")
    report.append("")

    # Feature importance
    report.append("TOP 15 FEATURES BY IMPORTANCE")
    report.append("-" * 40)
    for _, row in result.feature_importance.head(15).iterrows():
        report.append(f"  {row['feature']}: {row['importance']:.4f}")
    report.append("")

    report_text = "\n".join(report)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"bankruptcy_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        report_path.write_text(report_text)
        LOGGER.info("Report saved to %s", report_path)

    return report_text


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    # Trenuj model
    result = train_model(model_name="xgboost")

    # Wygeneruj raport
    report = generate_report(result)
    print(report)

    # Porównaj modele
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    comparison = compare_models()
    print(comparison.to_string())
