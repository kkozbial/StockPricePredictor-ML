"""Loader metadanych spółek (sectors, company_metadata, company_status, shares_outstanding)."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger("database")


def load_sectors_from_raw(raw_dir: Path) -> pd.DataFrame:
    """
    Wczytuje sektory z folderu raw.

    UWAGA: Wczytuje WSZYSTKIE pliki sectors_*.csv i łączy je,
    aby zachować zarówno dane USA jak i PL.

    Args:
        raw_dir: Ścieżka do katalogu data/raw

    Returns:
        DataFrame z sektorami.
    """
    others_dir = raw_dir / "others"
    if not others_dir.exists():
        return pd.DataFrame()

    sector_files = sorted(others_dir.glob("sectors_*.csv"))
    if not sector_files:
        LOGGER.warning("Brak plików z sektorami w %s", others_dir)
        return pd.DataFrame()

    # Wczytaj WSZYSTKIE pliki z sektorami i połącz je
    all_dfs = []
    for sector_file in sector_files:
        LOGGER.debug("Wczytywanie sektorów z pliku: %s", sector_file.name)
        df = pd.read_csv(sector_file)

        df = df.rename(columns={
            "Ticker": "ticker",
            "Sector": "sector",
            "Industry": "industry",
        })

        # Dodaj kolumnę country jeśli nie istnieje
        # Wykryj kraj na podstawie tickera (.PL -> PL, inaczej US)
        if "country" not in df.columns:
            df["country"] = df["ticker"].apply(lambda t: "PL" if isinstance(t, str) and t.endswith(".PL") else "US")

        all_dfs.append(df)

    # Połącz wszystkie DataFrame'y
    if not all_dfs:
        return pd.DataFrame()

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df["last_updated"] = datetime.now()

    # Usuń wiersze z NULL ticker
    initial_count = len(combined_df)
    combined_df = combined_df.dropna(subset=["ticker"])
    removed_count = initial_count - len(combined_df)
    if removed_count > 0:
        LOGGER.warning("Usunięto %d wierszy z NULL ticker w sectors", removed_count)

    # Usuń duplikaty - zachowaj ostatni wpis dla (ticker, country)
    # Sortuj według czasu modyfikacji pliku (ostatni plik = najnowsze dane)
    combined_df = combined_df.drop_duplicates(["ticker", "country"], keep="last")

    return combined_df[["ticker", "country", "sector", "industry", "last_updated"]]


def load_shares_from_raw(raw_dir: Path) -> pd.DataFrame:
    """
    Wczytuje liczbę akcji w obiegu z folderu raw (JSON).

    Args:
        raw_dir: Ścieżka do katalogu data/raw

    Returns:
        DataFrame z liczbą akcji.
    """
    shares_dir = raw_dir / "shares"
    if not shares_dir.exists():
        return pd.DataFrame()

    json_files = list(shares_dir.glob("*_shares.json"))
    if not json_files:
        LOGGER.warning("Brak plików JSON z danymi o akcjach w %s", shares_dir)
        return pd.DataFrame()

    all_records = []

    for json_file in json_files:
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)

            # Obsługa struktury: {"ticker": "...", "shares_history": [...]}
            if isinstance(data, dict) and "shares_history" in data:
                ticker = data["ticker"]
                records = data["shares_history"]
            elif isinstance(data, list):
                ticker = json_file.stem.replace("_shares", "")
                records = data
            else:
                LOGGER.warning("Nieznany format JSON w %s", json_file.name)
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
        "filed": "filing_date",
        "value": "shares",
        "end": "end_date",
        "fy": "fiscal_year",
        "fp": "fiscal_period",
    }
    df = df.rename(columns=column_mapping)

    # Konwersja typów
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce")
    df["fiscal_year"] = pd.to_numeric(df["fiscal_year"], errors="coerce").astype("Int64")

    # Dodaj kolumnę country (domyślnie US)
    if "country" not in df.columns:
        df["country"] = "US"

    # Deduplikacja - zachowaj ostatnią wartość dla (ticker, filing_date, country)
    df = df.drop_duplicates(subset=["ticker", "filing_date", "country"], keep="last")

    # Zachowaj wszystkie istotne kolumny zgodnie ze schematem
    result_columns = ["ticker", "filing_date", "country", "shares", "end_date", "fiscal_year", "fiscal_period", "source_type"]

    # Dodaj brakujące kolumny z NULL jeśli nie istnieją
    for col in result_columns:
        if col not in df.columns:
            df[col] = None

    return df[result_columns].dropna(subset=["ticker", "filing_date", "shares"])


def load_company_metadata_from_raw(raw_dir: Path) -> pd.DataFrame:
    """
    Wczytuje metadane firm z folderu raw/metadata (JSON).

    Args:
        raw_dir: Ścieżka do katalogu data/raw

    Returns:
        DataFrame z metadanymi firm.
    """
    metadata_dir = raw_dir / "metadata"
    if not metadata_dir.exists():
        LOGGER.warning("Katalog z metadanymi firm nie istnieje: %s", metadata_dir)
        return pd.DataFrame()

    metadata_file = metadata_dir / "company_metadata.json"
    if not metadata_file.exists():
        LOGGER.warning("Plik z metadanymi firm nie istnieje: %s", metadata_file)
        return pd.DataFrame()

    try:
        with open(metadata_file, encoding="utf-8") as f:
            data = json.load(f)

        companies = data.get("companies", [])
        if not companies:
            LOGGER.warning("Brak firm w pliku metadanych")
            return pd.DataFrame()

        all_records = []

        # Każda firma może mieć wiele tickerów
        for company in companies:
            cik = company.get("cik")
            name = company.get("name")
            sic = company.get("sic")
            sic_description = company.get("sic_description")
            exchanges = company.get("exchanges", [])
            all_tickers = company.get("all_tickers", [])

            # Dla każdego tickera firmy tworzymy osobny rekord
            for ticker in all_tickers:
                # Przypisz giełdę (jeśli jest wiele, bierzemy pierwszą)
                exchange = exchanges[0] if exchanges else None

                all_records.append({
                    "cik": cik,
                    "ticker": ticker,
                    "name": name,
                    "sic": sic,
                    "sic_description": sic_description,
                    "exchange": exchange,
                    "is_excluded": False,  # Plik zawiera tylko niewyłączone firmy
                })

        if not all_records:
            return pd.DataFrame()

        df = pd.DataFrame(all_records)

        # Dodaj kolumnę country (domyślnie US)
        if "country" not in df.columns:
            df["country"] = "US"

        # Konwertuj puste stringi na None dla kolumny sic (która jest INT)
        if "sic" in df.columns:
            df["sic"] = df["sic"].replace("", None)
            df["sic"] = pd.to_numeric(df["sic"], errors="coerce")

        df["last_updated"] = datetime.now()

        # Deduplikacja - zachowaj ostatnią wartość dla (cik, ticker, country)
        df = df.drop_duplicates(subset=["cik", "ticker", "country"], keep="last")

        LOGGER.info("Załadowano metadane dla %d tickerów (%d unikalnych firm)",
                   len(df), df["cik"].nunique())

        return df[["cik", "ticker", "country", "name", "sic", "sic_description", "exchange", "is_excluded", "last_updated"]]

    except Exception as exc:
        LOGGER.error("Błąd wczytywania metadanych firm: %s", exc)
        return pd.DataFrame()


def load_company_status_from_raw(raw_dir: Path) -> pd.DataFrame:
    """
    Wczytuje statusy spółek z folderu raw/sec_metadata (CSV).

    Args:
        raw_dir: Ścieżka do katalogu data/raw

    Returns:
        DataFrame ze statusami spółek (ACTIVE/DELISTED/DEREGISTERED/ZOMBIE).
    """
    sec_metadata_dir = raw_dir / "sec_metadata"
    if not sec_metadata_dir.exists():
        LOGGER.warning("Katalog z metadanymi SEC nie istnieje: %s", sec_metadata_dir)
        return pd.DataFrame()

    status_file = sec_metadata_dir / "sec_status.csv"
    if not status_file.exists():
        LOGGER.warning("Plik z statusami spółek nie istnieje: %s", status_file)
        return pd.DataFrame()

    try:
        df = pd.read_csv(status_file)

        # Konwersja dat
        date_columns = ["delisting_date", "last_filing_date", "last_financial_report_date"]
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Dodaj kolumnę country (domyślnie US)
        if "country" not in df.columns:
            df["country"] = "US"

        df["last_updated"] = datetime.now()

        # Usuń rekordy z NULL ticker (mogą wystąpić jeśli spółka nie ma tickera)
        initial_count = len(df)
        df = df.dropna(subset=["ticker"])
        if len(df) < initial_count:
            LOGGER.warning("Usunięto %d rekordów z NULL ticker", initial_count - len(df))

        # Deduplikacja - zachowaj ostatnią wartość dla (ticker, country)
        df = df.drop_duplicates(subset=["ticker", "country"], keep="last")

        LOGGER.info("Załadowano statusy dla %d tickerów", len(df))

        # Mapuj kolumny do schematu bazy
        schema_columns = [
            "ticker", "cik", "country", "status", "delisting_date", "delisting_form",
            "last_filing_date", "last_financial_report_date", "reason", "last_updated"
        ]

        # Dodaj brakujące kolumny
        for col in schema_columns:
            if col not in df.columns:
                df[col] = None

        return df[schema_columns]

    except Exception as exc:
        LOGGER.error("Błąd wczytywania statusów spółek: %s", exc)
        return pd.DataFrame()
