"""Ładowanie i przygotowanie danych do modelu bankructwa."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import numpy as np

from ..database.connection import get_connection
from ..utils.config_loader import load_config

LOGGER = logging.getLogger("bankruptcy.data_loader")

# Kolumny do wykluczenia z features (metadata, target, leakage)
EXCLUDE_COLUMNS = [
    "ticker", "date", "end_date", "filing_date", "fiscal_quarter",
    "country", "sector", "industry", "status", "delisting_date",
    "target_bankruptcy",
]

# Kolumny z wysokim % brakujących wartości do usunięcia
HIGH_MISSING_THRESHOLD = 0.90


def load_bankruptcy_data(db_path: Path | None = None) -> pd.DataFrame:
    """
    Ładuje dane z staging.bankruptcy_dataset.

    Returns:
        DataFrame z danymi do modelu bankructwa.
    """
    if db_path is None:
        cfg = load_config()
        db_path = Path(cfg["database"]["path"])

    conn = get_connection(db_path)

    LOGGER.info("Loading bankruptcy dataset from database...")
    df = conn.execute("SELECT * FROM staging.bankruptcy_dataset").fetchdf()
    LOGGER.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    return df


def prepare_features(
    df: pd.DataFrame,
    drop_high_missing: bool = True,
    fill_strategy: str = "median",
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Przygotowuje features i target do modelu.

    Args:
        df: DataFrame z danymi.
        drop_high_missing: Czy usuwać kolumny z >90% brakujących.
        fill_strategy: Strategia wypełniania braków ('median', 'zero', 'ffill').

    Returns:
        Tuple (X, y, feature_names)
    """
    df = df.copy()

    # Target
    if "target_bankruptcy" not in df.columns:
        raise ValueError("Column 'target_bankruptcy' not found")

    y = df["target_bankruptcy"].copy()

    # Usuń wiersze bez target
    valid_mask = y.notna()
    df = df[valid_mask]
    y = y[valid_mask].astype(int)

    LOGGER.info("Valid samples with target: %d", len(df))
    LOGGER.info("Class distribution: 0=%d (%.1f%%), 1=%d (%.1f%%)",
                (y == 0).sum(), (y == 0).mean() * 100,
                (y == 1).sum(), (y == 1).mean() * 100)

    # Wybierz tylko kolumny numeryczne jako features
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLUMNS]
    X = df[feature_cols].select_dtypes(include=[np.number])

    LOGGER.info("Initial features: %d", len(X.columns))

    # Usuń kolumny z wysokim % brakujących
    if drop_high_missing:
        missing_pct = X.isnull().mean()
        cols_to_keep = missing_pct[missing_pct < HIGH_MISSING_THRESHOLD].index.tolist()
        dropped = len(X.columns) - len(cols_to_keep)
        X = X[cols_to_keep]
        LOGGER.info("Dropped %d columns with >%.0f%% missing", dropped, HIGH_MISSING_THRESHOLD * 100)

    # Zamień inf na NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    # Wypełnij braki
    if fill_strategy == "median":
        X = X.fillna(X.median())
    elif fill_strategy == "zero":
        X = X.fillna(0)
    elif fill_strategy == "ffill":
        X = X.fillna(method="ffill").fillna(method="bfill").fillna(0)

    # Ostateczne czyszczenie - usuń kolumny z samymi NaN/zerem
    valid_cols = X.columns[X.std() > 0].tolist()
    X = X[valid_cols]

    LOGGER.info("Final features: %d", len(X.columns))

    return X, y, X.columns.tolist()


def get_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    time_based: bool = True,
    cutoff_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Dzieli dane na train/test.

    Args:
        df: DataFrame z danymi.
        test_size: Proporcja test set (używana gdy time_based=False).
        time_based: Czy dzielić po dacie (zalecane).
        cutoff_date: Data graniczna dla podziału. Jeśli None, oblicza automatycznie.

    Returns:
        Tuple (train_df, test_df)
    """
    if "date" not in df.columns:
        raise ValueError("Column 'date' required for splitting")

    df = df.sort_values("date")

    if time_based:
        if cutoff_date is None:
            # Użyj 80% najstarszych dat jako train
            dates_sorted = df["date"].sort_values().unique()
            cutoff_idx = int(len(dates_sorted) * (1 - test_size))
            cutoff_date = dates_sorted[cutoff_idx]

        cutoff = pd.to_datetime(cutoff_date)
        train_df = df[df["date"] < cutoff]
        test_df = df[df["date"] >= cutoff]
    else:
        # Random split (nie zalecane dla danych czasowych)
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    LOGGER.info("Train: %d rows, Test: %d rows", len(train_df), len(test_df))
    LOGGER.info("Train date range: %s to %s", train_df["date"].min(), train_df["date"].max())
    LOGGER.info("Test date range: %s to %s", test_df["date"].min(), test_df["date"].max())

    return train_df, test_df


def load_and_prepare(
    db_path: Path | None = None,
    test_size: float = 0.2,
) -> dict:
    """
    Kompletny pipeline ładowania i przygotowania danych.

    Returns:
        Dict z kluczami: X_train, X_test, y_train, y_test, feature_names, train_df, test_df
    """
    # Załaduj dane
    df = load_bankruptcy_data(db_path)

    # Podziel na train/test
    train_df, test_df = get_train_test_split(df, test_size=test_size)

    # Przygotuj features osobno dla train i test
    X_train, y_train, feature_names = prepare_features(train_df)
    X_test, y_test, _ = prepare_features(test_df)

    # Upewnij się, że test ma te same kolumny co train
    missing_cols = set(feature_names) - set(X_test.columns)
    for col in missing_cols:
        X_test[col] = 0

    X_test = X_test[feature_names]

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "train_df": train_df,
        "test_df": test_df,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    data = load_and_prepare()
    print(f"\nX_train shape: {data['X_train'].shape}")
    print(f"X_test shape: {data['X_test'].shape}")
    print(f"Features: {len(data['feature_names'])}")
