"""Loader danych o dywidendach (dividends)."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger("database")


def load_dividends_from_raw(raw_dir: Path) -> pd.DataFrame:
    """
    Wczytuje dywidendy z folderu raw.

    Obsługuje dwa formaty:
    1. CSV z katalogu others/ (nowy format: dividends_TIMESTAMP.csv)
    2. JSON z katalogu dividends/ (stary format: TICKER_dividends.json)

    Args:
        raw_dir: Ścieżka do katalogu data/raw

    Returns:
        DataFrame z dywidendami.
    """
    # NOWY FORMAT: Szukaj CSV w others/
    others_dir = raw_dir / "others"
    csv_files = sorted(others_dir.glob("dividends_*.csv")) if others_dir.exists() else []

    if csv_files:
        # Użyj najnowszego pliku CSV
        latest_csv = csv_files[-1]
        LOGGER.info("Wczytywanie dywidend z pliku CSV: %s", latest_csv.name)
        df = pd.read_csv(latest_csv)

        # Mapowanie kolumn CSV do schematu bazy
        column_mapping = {
            "Ticker": "ticker",
            "Date": "dividend_date",
            "Dividend": "dividend_amount",
        }
        df = df.rename(columns=column_mapping)

        df["dividend_date"] = pd.to_datetime(df["dividend_date"])
        df["dividend_amount"] = pd.to_numeric(df["dividend_amount"], errors="coerce")

        # Dodaj kolumnę country
        if "country" not in df.columns:
            df["country"] = df["ticker"].apply(lambda t: "PL" if isinstance(t, str) and t.endswith(".PL") else "US")

        # Deduplikacja
        df = df.drop_duplicates(subset=["ticker", "dividend_date", "country"], keep="last")

        LOGGER.info("Załadowano %d unikalnych rekordów dywidend z CSV", len(df))
        return df[["ticker", "dividend_date", "country", "dividend_amount"]].dropna()

    # STARY FORMAT: Fallback do JSON w dividends/
    dividends_dir = raw_dir / "dividends"
    if not dividends_dir.exists():
        LOGGER.warning("Brak danych dywidend: katalog 'dividends' i 'others/dividends_*.csv' nie istnieją")
        return pd.DataFrame()

    json_files = list(dividends_dir.glob("*_dividends.json"))
    if not json_files:
        LOGGER.warning("Brak plików JSON z dywidendami w %s", dividends_dir)
        return pd.DataFrame()

    all_records = []

    for json_file in json_files:
        ticker = json_file.stem.replace("_dividends", "")

        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)

            # Obsługa struktury: {"ticker": "...", "dividends": [...]}
            if isinstance(data, dict) and "dividends" in data:
                records = data["dividends"]
            elif isinstance(data, list):
                records = data
            else:
                LOGGER.warning("Nieoczekiwana struktura JSON w %s", json_file.name)
                continue

            # Dodaj ticker do każdego rekordu
            for record in records:
                record["ticker"] = ticker
                all_records.append(record)

        except Exception as exc:
            LOGGER.warning("Błąd wczytywania %s: %s", json_file.name, exc)
            continue

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    # Mapowanie kolumn do schematu bazy
    column_mapping = {
        "date": "dividend_date",
        "dividend": "dividend_amount",
    }
    df = df.rename(columns=column_mapping)

    df["dividend_date"] = pd.to_datetime(df["dividend_date"])
    df["dividend_amount"] = pd.to_numeric(df["dividend_amount"], errors="coerce")

    # Dodaj kolumnę country (domyślnie US)
    if "country" not in df.columns:
        df["country"] = df["ticker"].apply(lambda t: "PL" if isinstance(t, str) and t.endswith(".PL") else "US")

    # Deduplikacja - zachowaj ostatnią wartość dla (ticker, dividend_date, country)
    df = df.drop_duplicates(subset=["ticker", "dividend_date", "country"], keep="last")

    LOGGER.info("Załadowano %d unikalnych rekordów dywidend z JSON", len(df))
    return df[["ticker", "dividend_date", "country", "dividend_amount"]].dropna()
