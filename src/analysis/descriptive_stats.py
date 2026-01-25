"""Rozszerzone statystyki opisowe dla datasetu z bazy DuckDB."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..database.connection import get_connection
from ..utils.config_loader import load_config

LOGGER = logging.getLogger("analysis.descriptive_stats")

# Domyślne kolumny do analizy statystycznej
DEFAULT_NUMERIC_COLS = [
    # Wskaźniki wyceny
    "pe_ratio", "price_to_book", "price_to_sales", "price_to_earnings",
    # Wskaźniki rentowności
    "roe", "roa", "ros", "gross_margin", "operating_margin",
    # Wskaźniki zadłużenia
    "debt_ratio", "debt_to_equity",
    # Wskaźniki płynności
    "current_ratio", "quick_ratio",
    # Wskaźniki efektywności
    "asset_turnover", "inventory_turnover",
    # Cash flow
    "ocf_ratio", "capex_ratio", "free_cash_flow",
    # Wzrost
    "revenue_growth", "net_income_growth",
    # Dywidendy
    "dividend_yield", "payout_ratio",
    # Dane rynkowe
    "close", "high", "low",
    # Makro
    "gdp_yoy", "cpi_yoy", "unemp_level", "rate_level",
]


def load_master_dataset(
    db_path: Path | None = None,
    table: str = "staging.master_dataset",
    columns: list[str] | None = None,
    where: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Ładuje dane z tabeli master_dataset (lub innej) z bazy DuckDB.

    Args:
        db_path: Ścieżka do bazy. Jeśli None, używa konfiguracji.
        table: Nazwa tabeli (domyślnie staging.master_dataset).
        columns: Lista kolumn do pobrania. Jeśli None, pobiera wszystkie.
        where: Opcjonalna klauzula WHERE (bez słowa WHERE).
        limit: Opcjonalny limit rekordów.

    Returns:
        DataFrame z danymi.
    """
    if db_path is None:
        cfg = load_config()
        db_path = Path(cfg["database"]["path"])

    conn = get_connection(db_path)

    cols_sql = ", ".join(columns) if columns else "*"
    query = f"SELECT {cols_sql} FROM {table}"

    if where:
        query += f" WHERE {where}"
    if limit:
        query += f" LIMIT {limit}"

    LOGGER.info("Loading data from %s", table)
    df = conn.execute(query).fetchdf()
    LOGGER.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    return df


def describe_dataset(
    df: pd.DataFrame | None = None,
    db_path: Path | None = None,
    columns: list[str] | None = None,
    percentiles: list[float] | None = None,
) -> pd.DataFrame:
    """
    Zwraca rozszerzone statystyki opisowe dla kolumn numerycznych.

    Args:
        df: DataFrame do analizy. Jeśli None, ładuje z bazy.
        db_path: Ścieżka do bazy (używana gdy df=None).
        columns: Lista kolumn do analizy. Jeśli None, używa domyślnych.
        percentiles: Lista percentyli do obliczenia.

    Returns:
        DataFrame ze statystykami (transponowany).
    """
    if df is None:
        df = load_master_dataset(db_path)

    if columns is None:
        columns = [c for c in DEFAULT_NUMERIC_COLS if c in df.columns]

    if percentiles is None:
        percentiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]

    numeric_df = df[columns].select_dtypes(include="number")

    # Podstawowe statystyki
    stats = numeric_df.describe(percentiles=percentiles).transpose()

    # Dodatkowe statystyki
    stats["missing"] = numeric_df.isnull().sum()
    stats["missing_pct"] = (numeric_df.isnull().sum() / len(df) * 100).round(2)
    stats["zeros"] = (numeric_df == 0).sum()
    stats["negative"] = (numeric_df < 0).sum()
    stats["skewness"] = numeric_df.skew()
    stats["kurtosis"] = numeric_df.kurtosis()
    stats["iqr"] = stats["75%"] - stats["25%"]

    # Zmień kolejność kolumn
    col_order = [
        "count", "missing", "missing_pct", "mean", "std", "min",
        "1%", "5%", "25%", "50%", "75%", "95%", "99%", "max",
        "skewness", "kurtosis", "iqr", "zeros", "negative"
    ]
    stats = stats[[c for c in col_order if c in stats.columns]]

    return stats


def describe_by_group(
    df: pd.DataFrame | None = None,
    db_path: Path | None = None,
    group_col: str = "sector",
    value_col: str = "pe_ratio",
) -> pd.DataFrame:
    """
    Zwraca statystyki opisowe pogrupowane wg kategorii.

    Args:
        df: DataFrame do analizy. Jeśli None, ładuje z bazy.
        db_path: Ścieżka do bazy (używana gdy df=None).
        group_col: Kolumna do grupowania (np. sector, industry).
        value_col: Kolumna z wartościami do analizy.

    Returns:
        DataFrame ze statystykami per grupa.
    """
    if df is None:
        df = load_master_dataset(db_path)

    if group_col not in df.columns:
        raise ValueError(f"Column '{group_col}' not found in dataset")
    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not found in dataset")

    grouped = df.groupby(group_col, dropna=False)[value_col].agg([
        ("count", "count"),
        ("mean", "mean"),
        ("std", "std"),
        ("min", "min"),
        ("q25", lambda x: x.quantile(0.25)),
        ("median", "median"),
        ("q75", lambda x: x.quantile(0.75)),
        ("max", "max"),
        ("skewness", "skew"),
    ]).round(4)

    return grouped.sort_values("count", ascending=False)


def missing_values_report(
    df: pd.DataFrame | None = None,
    db_path: Path | None = None,
    threshold_pct: float = 0.0,
) -> pd.DataFrame:
    """
    Generuje raport o brakujących wartościach.

    Args:
        df: DataFrame do analizy. Jeśli None, ładuje z bazy.
        db_path: Ścieżka do bazy (używana gdy df=None).
        threshold_pct: Minimalny % brakujących wartości do wyświetlenia.

    Returns:
        DataFrame z informacją o brakujących wartościach.
    """
    if df is None:
        df = load_master_dataset(db_path)

    missing = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.values,
        "missing_count": df.isnull().sum().values,
        "missing_pct": (df.isnull().sum() / len(df) * 100).round(2).values,
        "non_null_count": df.notnull().sum().values,
    })

    missing = missing[missing["missing_pct"] >= threshold_pct]
    missing = missing.sort_values("missing_pct", ascending=False)

    return missing.reset_index(drop=True)


def data_quality_summary(
    df: pd.DataFrame | None = None,
    db_path: Path | None = None,
) -> dict:
    """
    Zwraca podsumowanie jakości danych.

    Args:
        df: DataFrame do analizy. Jeśli None, ładuje z bazy.
        db_path: Ścieżka do bazy (używana gdy df=None).

    Returns:
        Słownik z metrykami jakości.
    """
    if df is None:
        df = load_master_dataset(db_path)

    total_cells = df.size
    missing_cells = df.isnull().sum().sum()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    summary = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "total_cells": total_cells,
        "missing_cells": missing_cells,
        "missing_pct": round(missing_cells / total_cells * 100, 2),
        "completeness_pct": round((1 - missing_cells / total_cells) * 100, 2),
        "numeric_columns": len(numeric_cols),
        "categorical_columns": len(df.columns) - len(numeric_cols),
        "unique_tickers": df["ticker"].nunique() if "ticker" in df.columns else None,
        "date_range": {
            "min": str(df["date"].min()) if "date" in df.columns else None,
            "max": str(df["date"].max()) if "date" in df.columns else None,
        },
        "sectors": df["sector"].nunique() if "sector" in df.columns else None,
    }

    return summary


def outlier_report(
    df: pd.DataFrame | None = None,
    db_path: Path | None = None,
    columns: list[str] | None = None,
    method: str = "iqr",
    threshold: float = 1.5,
) -> pd.DataFrame:
    """
    Wykrywa wartości odstające w danych.

    Args:
        df: DataFrame do analizy. Jeśli None, ładuje z bazy.
        db_path: Ścieżka do bazy (używana gdy df=None).
        columns: Lista kolumn do analizy. Jeśli None, używa domyślnych.
        method: Metoda wykrywania ('iqr' lub 'zscore').
        threshold: Próg dla metody (1.5 dla IQR, 3 dla z-score).

    Returns:
        DataFrame z informacją o outlierach.
    """
    if df is None:
        df = load_master_dataset(db_path)

    if columns is None:
        columns = [c for c in DEFAULT_NUMERIC_COLS if c in df.columns]

    results = []

    for col in columns:
        if col not in df.columns:
            continue

        series = df[col].dropna()
        if len(series) == 0:
            continue

        if method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outliers_mask = (series < lower_bound) | (series > upper_bound)
        elif method == "zscore":
            from scipy import stats
            z_scores = stats.zscore(series)
            outliers_mask = abs(z_scores) > threshold
        else:
            raise ValueError(f"Unknown method: {method}")

        outliers_count = outliers_mask.sum()
        outliers_pct = round(outliers_count / len(series) * 100, 2)

        results.append({
            "column": col,
            "total_values": len(series),
            "outliers_count": outliers_count,
            "outliers_pct": outliers_pct,
            "lower_bound": round(lower_bound, 4) if method == "iqr" else None,
            "upper_bound": round(upper_bound, 4) if method == "iqr" else None,
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    print("\n=== Data Quality Summary ===")
    quality = data_quality_summary()
    for key, value in quality.items():
        print(f"  {key}: {value}")

    print("\n=== Descriptive Statistics (sample) ===")
    stats = describe_dataset(columns=["pe_ratio", "roe", "debt_ratio", "current_ratio"])
    print(stats.to_string())

    print("\n=== Missing Values Report (>50% missing) ===")
    missing = missing_values_report(threshold_pct=50.0)
    print(missing.to_string())
