"""Feature engineering dla modelu bankructwa."""
from __future__ import annotations

import logging

import pandas as pd
import numpy as np

LOGGER = logging.getLogger("bankruptcy.features")

# Kluczowe wskaźniki dla predykcji bankructwa (Altman Z-score inspired)
BANKRUPTCY_KEY_FEATURES = [
    # Płynność
    "current_ratio",
    "quick_ratio",
    # Rentowność
    "roe",
    "roa",
    "ros",
    "operating_margin",
    "gross_margin",
    # Zadłużenie
    "debt_ratio",
    "debt_to_equity",
    # Cash flow
    "ocf_ratio",
    "free_cash_flow",
    "fcf_ttm",
    # Efektywność
    "asset_turnover",
    # Wzrost
    "revenue_growth",
    "net_income_growth",
    # Wycena
    "price_to_book",
    # Fundamentals
    "Assets",
    "Liabilities",
    "StockholdersEquity",
    "Revenues",
    "NetIncomeLoss",
    "OperatingCashFlow",
]


def create_altman_z_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tworzy cechy inspirowane Altman Z-score.

    Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
    gdzie:
    - X1 = Working Capital / Total Assets
    - X2 = Retained Earnings / Total Assets
    - X3 = EBIT / Total Assets
    - X4 = Market Value of Equity / Total Liabilities
    - X5 = Sales / Total Assets
    """
    df = df.copy()

    # X1: Working Capital / Total Assets
    if "AssetsCurrent" in df.columns and "LiabilitiesCurrent" in df.columns and "Assets" in df.columns:
        df["altman_x1"] = (df["AssetsCurrent"] - df["LiabilitiesCurrent"]) / df["Assets"].replace(0, np.nan)

    # X2: Retained Earnings / Total Assets
    if "RetainedEarnings" in df.columns and "Assets" in df.columns:
        df["altman_x2"] = df["RetainedEarnings"] / df["Assets"].replace(0, np.nan)

    # X3: EBIT / Total Assets (używamy OperatingIncomeLoss jako proxy)
    if "OperatingIncomeLoss" in df.columns and "Assets" in df.columns:
        df["altman_x3"] = df["OperatingIncomeLoss"] / df["Assets"].replace(0, np.nan)

    # X4: Market Cap / Total Liabilities
    if "close" in df.columns and "shares_outstanding" in df.columns and "Liabilities" in df.columns:
        market_cap = df["close"] * df["shares_outstanding"]
        df["altman_x4"] = market_cap / df["Liabilities"].replace(0, np.nan)

    # X5: Sales / Total Assets
    if "Revenues" in df.columns and "Assets" in df.columns:
        df["altman_x5"] = df["Revenues"] / df["Assets"].replace(0, np.nan)

    # Oblicz uproszczone Z-score
    altman_cols = ["altman_x1", "altman_x2", "altman_x3", "altman_x4", "altman_x5"]
    existing_cols = [c for c in altman_cols if c in df.columns]

    if len(existing_cols) == 5:
        df["altman_z_approx"] = (
            1.2 * df["altman_x1"] +
            1.4 * df["altman_x2"] +
            3.3 * df["altman_x3"] +
            0.6 * df["altman_x4"] +
            1.0 * df["altman_x5"]
        )
        LOGGER.info("Created Altman Z-score approximation")

    return df


def create_distress_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tworzy binarne wskaźniki stresu finansowego.
    """
    df = df.copy()

    # Ujemny kapitał własny
    if "StockholdersEquity" in df.columns:
        df["negative_equity"] = (df["StockholdersEquity"] < 0).astype(int)

    # Ujemny zysk netto
    if "NetIncomeLoss" in df.columns:
        df["negative_income"] = (df["NetIncomeLoss"] < 0).astype(int)

    # Ujemny cash flow operacyjny
    if "OperatingCashFlow" in df.columns:
        df["negative_ocf"] = (df["OperatingCashFlow"] < 0).astype(int)

    # Spadek przychodów YoY
    if "revenue_growth" in df.columns:
        df["revenue_decline"] = (df["revenue_growth"] < 0).astype(int)

    # Wysoki dług
    if "debt_to_equity" in df.columns:
        df["high_leverage"] = (df["debt_to_equity"] > 2.0).astype(int)

    # Niska płynność
    if "current_ratio" in df.columns:
        df["low_liquidity"] = (df["current_ratio"] < 1.0).astype(int)

    # Suma wskaźników stresu
    distress_cols = [
        "negative_equity", "negative_income", "negative_ocf",
        "revenue_decline", "high_leverage", "low_liquidity"
    ]
    existing = [c for c in distress_cols if c in df.columns]
    if existing:
        df["distress_score"] = df[existing].sum(axis=1)
        LOGGER.info("Created distress score from %d indicators", len(existing))

    return df


def create_trend_features(df: pd.DataFrame, windows: list[int] = [3, 6, 12]) -> pd.DataFrame:
    """
    Tworzy cechy trendów (rolling mean, std) dla key metrics.

    UWAGA: Wymaga sortowania po ticker i date.
    """
    df = df.copy()

    if "ticker" not in df.columns or "date" not in df.columns:
        LOGGER.warning("Cannot create trend features without ticker and date columns")
        return df

    df = df.sort_values(["ticker", "date"])

    trend_cols = ["roe", "roa", "debt_ratio", "current_ratio", "revenue_growth"]
    trend_cols = [c for c in trend_cols if c in df.columns]

    for col in trend_cols:
        for window in windows:
            # Rolling mean
            df[f"{col}_ma{window}"] = df.groupby("ticker")[col].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            # Rolling std (volatility)
            df[f"{col}_std{window}"] = df.groupby("ticker")[col].transform(
                lambda x: x.rolling(window, min_periods=2).std()
            )

    LOGGER.info("Created trend features for %d columns with windows %s", len(trend_cols), windows)

    return df


def create_lag_features(df: pd.DataFrame, lags: list[int] = [1, 3, 6]) -> pd.DataFrame:
    """
    Tworzy lag features dla kluczowych wskaźników.
    """
    df = df.copy()

    if "ticker" not in df.columns:
        LOGGER.warning("Cannot create lag features without ticker column")
        return df

    df = df.sort_values(["ticker", "date"])

    lag_cols = ["roe", "debt_ratio", "current_ratio", "close"]
    lag_cols = [c for c in lag_cols if c in df.columns]

    for col in lag_cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df.groupby("ticker")[col].shift(lag)

    LOGGER.info("Created lag features for %d columns with lags %s", len(lag_cols), lags)

    return df


def engineer_bankruptcy_features(
    df: pd.DataFrame,
    add_altman: bool = True,
    add_distress: bool = True,
    add_trends: bool = False,
    add_lags: bool = False,
) -> pd.DataFrame:
    """
    Kompletny pipeline feature engineering dla bankructwa.

    Args:
        df: DataFrame z danymi.
        add_altman: Czy dodać cechy Altman Z-score.
        add_distress: Czy dodać wskaźniki stresu.
        add_trends: Czy dodać cechy trendów (wolniejsze).
        add_lags: Czy dodać lag features.

    Returns:
        DataFrame z nowymi cechami.
    """
    LOGGER.info("Starting bankruptcy feature engineering...")
    initial_cols = len(df.columns)

    if add_altman:
        df = create_altman_z_features(df)

    if add_distress:
        df = create_distress_indicators(df)

    if add_trends:
        df = create_trend_features(df)

    if add_lags:
        df = create_lag_features(df)

    new_cols = len(df.columns) - initial_cols
    LOGGER.info("Feature engineering complete. Added %d new features.", new_cols)

    return df


def select_top_features(
    X: pd.DataFrame,
    y: pd.Series,
    n_features: int = 30,
    method: str = "mutual_info",
) -> list[str]:
    """
    Wybiera top N najważniejszych features.

    Args:
        X: Features DataFrame.
        y: Target Series.
        n_features: Liczba features do wybrania.
        method: Metoda selekcji ('mutual_info', 'f_classif', 'chi2').

    Returns:
        Lista nazw wybranych features.
    """
    from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif, chi2

    # Usuń NaN
    valid_mask = X.notna().all(axis=1) & y.notna()
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]

    if method == "mutual_info":
        selector = SelectKBest(mutual_info_classif, k=min(n_features, len(X.columns)))
    elif method == "f_classif":
        selector = SelectKBest(f_classif, k=min(n_features, len(X.columns)))
    elif method == "chi2":
        # Chi2 wymaga wartości nieujemnych
        X_clean = X_clean - X_clean.min() + 1e-10
        selector = SelectKBest(chi2, k=min(n_features, len(X.columns)))
    else:
        raise ValueError(f"Unknown method: {method}")

    selector.fit(X_clean, y_clean)

    # Pobierz nazwy wybranych features
    selected_mask = selector.get_support()
    selected_features = X.columns[selected_mask].tolist()

    LOGGER.info("Selected %d features using %s", len(selected_features), method)

    return selected_features


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    from .data_loader import load_bankruptcy_data

    df = load_bankruptcy_data()
    df = engineer_bankruptcy_features(df, add_altman=True, add_distress=True)

    print(f"\nColumns after feature engineering: {len(df.columns)}")
    print("\nNew features:")
    new_cols = [c for c in df.columns if c.startswith(("altman_", "negative_", "distress", "high_", "low_", "revenue_decline"))]
    for col in new_cols:
        print(f"  - {col}")
