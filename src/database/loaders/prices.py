"""Loader danych cenowych (prices)."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger("database")


def load_prices_from_raw(raw_dir: Path) -> pd.DataFrame:
    """
    Wczytuje dane cenowe z folderu raw.

    Ładuje zarówno dane US (prices_TIMESTAMP.csv) jak i PL (prices_pl_TIMESTAMP.csv).

    Args:
        raw_dir: Ścieżka do katalogu data/raw

    Returns:
        DataFrame z cenami.
    """
    prices_dir = raw_dir / "prices"
    if not prices_dir.exists():
        LOGGER.warning("Katalog z cenami nie istnieje: %s", prices_dir)
        return pd.DataFrame()

    # Znajdujemy najnowsze pliki z cenami dla każdego kraju
    # US: prices_YYYYMMDD_HHMMSS.csv (bez _pl)
    # PL: prices_pl_YYYYMMDD_HHMMSS.csv
    us_files = sorted([f for f in prices_dir.glob("prices_*.csv") if "_pl_" not in f.name and "_stooq_" not in f.name])
    pl_files = sorted([f for f in prices_dir.glob("prices_pl_*.csv") if "_stooq_" not in f.name])

    all_dfs = []

    # Wczytaj najnowszy plik US
    if us_files:
        latest_us = us_files[-1]
        LOGGER.info("Wczytywanie cen USA z pliku: %s", latest_us.name)
        df_us = pd.read_csv(latest_us)
        df_us["Date"] = pd.to_datetime(df_us["Date"])
        # Dodaj Country jeśli nie istnieje
        if "Country" not in df_us.columns and "country" not in df_us.columns:
            df_us["Country"] = "US"
        all_dfs.append(df_us)

    # Wczytaj najnowszy plik PL
    if pl_files:
        latest_pl = pl_files[-1]
        LOGGER.info("Wczytywanie cen Polski z pliku: %s", latest_pl.name)
        df_pl = pd.read_csv(latest_pl)
        df_pl["Date"] = pd.to_datetime(df_pl["Date"])
        all_dfs.append(df_pl)

    if not all_dfs:
        LOGGER.warning("Brak plików z cenami w %s", prices_dir)
        return pd.DataFrame()

    # Połącz wszystkie DataFrame'y
    df = pd.concat(all_dfs, ignore_index=True)

    # Mapowanie kolumn do schematu bazy (tylko surowe dane ze źródła)
    rename_map = {
        "Ticker": "ticker",
        "Date": "date",
        "Close": "close",
    }

    # Dodaj opcjonalne kolumny jeśli istnieją
    optional_cols = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Volume": "volume",
    }

    for old_name, new_name in optional_cols.items():
        if old_name in df.columns:
            rename_map[old_name] = new_name

    df = df.rename(columns=rename_map)

    # Dodaj brakujące kolumny z wartościami NULL
    # (potrzebne dla schematu bazy, ale mogą być puste)
    required_schema_cols = ["ticker", "date", "open", "high", "low", "close", "volume"]
    for col in required_schema_cols:
        if col not in df.columns:
            df[col] = None
            LOGGER.debug("Dodano brakującą kolumnę '%s' z wartościami NULL", col)

    # Dodaj kolumnę country jeśli nie istnieje
    # Sprawdź czy w danych jest już kolumna Country (z Stooq)
    if "Country" in df.columns:
        df = df.rename(columns={"Country": "country"})
    elif "country" not in df.columns:
        # Domyślnie US dla danych z yfinance
        df["country"] = "US"
        LOGGER.debug("Dodano kolumnę 'country' z wartością domyślną 'US'")

    required_cols = ["ticker", "date", "close"]
    if not all(col in df.columns for col in required_cols):
        LOGGER.error("Brak wymaganych kolumn w danych cenowych: %s", required_cols)
        return pd.DataFrame()

    # Usuń wiersze z NULL w kluczowych kolumnach (ticker, date)
    initial_count = len(df)
    df = df.dropna(subset=["ticker", "date"])
    removed_count = initial_count - len(df)
    if removed_count > 0:
        LOGGER.warning("Usunięto %d wierszy z NULL ticker/date", removed_count)

    # KRYTYCZNE: Zwróć kolumny w DOKŁADNEJ kolejności schematu tabeli prices
    # Schema: ticker, date, country, open, high, low, close, volume
    schema_order = [
        "ticker", "date", "country", "open", "high", "low", "close", "volume"
    ]

    LOGGER.debug("Zwracanie DataFrame z kolumnami w kolejności schematu: %s", schema_order)
    return df[schema_order]
