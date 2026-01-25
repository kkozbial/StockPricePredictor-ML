"""Loader danych finansowych (financials)."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger("database")


def load_financials_from_raw(raw_dir: Path) -> pd.DataFrame:
    """
    Wczytuje dane finansowe z folderu raw (JSON files).

    Args:
        raw_dir: Ścieżka do katalogu data/raw

    Returns:
        DataFrame z danymi finansowymi.
    """
    financials_dir = raw_dir / "financials"
    if not financials_dir.exists():
        LOGGER.warning("Katalog z finansami nie istnieje: %s", financials_dir)
        return pd.DataFrame()

    json_files = list(financials_dir.glob("*_financials.json"))
    if not json_files:
        LOGGER.warning("Brak plików JSON z danymi finansowymi w %s", financials_dir)
        return pd.DataFrame()

    all_records = []

    for json_file in json_files:
        ticker = json_file.stem.replace("_financials", "")

        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)

            # Obsługa struktury z SEC API: {"ticker": "...", "financials": [...]}
            if isinstance(data, dict) and "financials" in data:
                records = data["financials"]
            elif isinstance(data, list):
                records = data
            else:
                LOGGER.warning("Nieoczekiwana struktura JSON w %s", json_file.name)
                continue

            # Przekształć surowe dane SEC (concept, end, value) na tabelę
            # Grupujemy po dacie końcowej (end) i concept
            for record in records:
                if "concept" in record and "end" in record and "value" in record:
                    # To są surowe dane z SEC - dodajemy ticker i zostawiamy surową strukturę
                    record["ticker"] = ticker
                    all_records.append(record)
                else:
                    # Stara struktura - dodaj ticker
                    record["ticker"] = ticker
                    all_records.append(record)

        except Exception as exc:
            LOGGER.warning("Błąd wczytywania %s: %s", json_file.name, exc)
            continue

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    # Mapowanie surowych kolumn SEC API do schematu bazy
    # Struktura SEC: ticker, concept, unit, start, end, value
    column_mapping = {
        "start": "start_date",
        "end": "end_date",
    }

    df = df.rename(columns=column_mapping)

    # Konwersja dat
    if "end_date" in df.columns:
        df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    if "start_date" in df.columns:
        df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    if "filing_date" in df.columns:
        df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")

    # Dodaj kolumnę country jeśli nie istnieje
    # Wykryj kraj na podstawie tickera (.PL -> PL, inaczej US)
    if "country" not in df.columns:
        df["country"] = df["ticker"].apply(lambda t: "PL" if t.endswith(".PL") else "US")
        LOGGER.debug("Dodano kolumnę 'country' na podstawie tickera")

    # Wybierz tylko kolumny zgodne ze schematem
    required_cols = ["ticker", "concept", "end_date", "value"]
    if not all(col in df.columns for col in required_cols):
        LOGGER.warning("Brak wymaganych kolumn w danych finansowych: %s", required_cols)
        return pd.DataFrame()

    # Zwróć w kolejności schematu: ticker, concept, country, unit, start_date, end_date, value, form, filing_date
    # UWAGA: filing_date i form są teraz pobierane bezpośrednio z companyfacts API
    schema_order = ["ticker", "concept", "country", "unit", "start_date", "end_date", "value", "form", "filing_date"]

    # Dodaj brakujące kolumny opcjonalne
    for col in schema_order:
        if col not in df.columns:
            df[col] = None

    # Usuń duplikaty - NOWA STRATEGIA:
    # Klucz deduplikacji: (ticker, concept, end_date) - BEZ start_date
    # Dla tej samej kombinacji mogą być różne wartości (restatements, różne formy)
    # Strategia wyboru:
    #   1. Jeśli są rekordy Z filing_date → zachowaj najnowszy filing_date (najbardziej aktualne dane)
    #   2. Jeśli wszystkie BEZ filing_date → zachowaj ostatni (keep="last")
    df = df[schema_order].dropna(subset=["end_date", "value"])

    # Sortuj:
    # - filing_date rosnąco (NaT/None na początku, najnowsze na końcu)
    # - potem value rosnąco (zapewnia deterministyczne zachowanie)
    # Keep="last" zachowa rekordy z końca (najnowszy filing_date lub ostatnią wartość)
    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    df = df.sort_values(
        ["ticker", "concept", "end_date", "filing_date", "value"],
        na_position="first"  # NaT na początku, więc keep="last" wybierze rekordy z filing_date
    )

    # Deduplikacja: klucz = (ticker, concept, end_date), zachowaj ostatni
    # Dzięki sortowaniu z na_position="first", ostatni rekord to:
    # - jeśli są filing_date → ten z najnowszym filing_date
    # - jeśli wszystkie bez filing_date → ostatni według value
    df = df.drop_duplicates(subset=["ticker", "concept", "end_date"], keep="last")

    # FALLBACK: Uzupełnij brakujące filing_date z SEC metadata (dla starych danych lub gdy API ich nie zwraca)
    # Nowe dane z fetch_financials.py już mają filing_date z companyfacts API
    missing_filing_dates = df["filing_date"].isna().sum()
    if missing_filing_dates > 0:
        filing_dates_df = load_sec_filing_dates(raw_dir)
        if not filing_dates_df.empty:
            # Merge na ticker i end_date - tylko dla rekordów bez filing_date
            df = df.merge(
                filing_dates_df[["ticker", "end_date", "filing_date"]],
                on=["ticker", "end_date"],
                how="left",
                suffixes=("", "_sec")
            )
            # Wypełnij brakujące filing_date z SEC metadata
            df["filing_date"] = df["filing_date"].fillna(df["filing_date_sec"])
            df = df.drop(columns=["filing_date_sec"], errors="ignore")
            filled_count = missing_filing_dates - df["filing_date"].isna().sum()
            if filled_count > 0:
                LOGGER.info("Uzupełniono %d brakujących filing_date z SEC metadata (fallback)", filled_count)

        # Raportuj jeśli nadal są braki
        remaining_missing = df["filing_date"].isna().sum()
        if remaining_missing > 0:
            LOGGER.warning("%d rekordów nadal bez filing_date (brak w obu źródłach)", remaining_missing)

    LOGGER.info("Załadowano %s unikalnych rekordów finansowych", len(df))
    return df


def load_sec_filing_dates(raw_dir: Path) -> pd.DataFrame:
    """
    Wczytuje daty publikacji raportów SEC (filing_date) z folderu raw/sec_metadata.

    To rozwiązuje Look-ahead Bias przez mapowanie end_date -> filing_date.

    Args:
        raw_dir: Ścieżka do katalogu data/raw

    Returns:
        DataFrame z kolumnami: ticker, end_date, filing_date
    """
    sec_dir = raw_dir / "sec_metadata"
    if not sec_dir.exists():
        LOGGER.debug("Katalog SEC metadata nie istnieje: %s", sec_dir)
        return pd.DataFrame()

    filing_dates_file = sec_dir / "sec_filing_dates.csv"
    if not filing_dates_file.exists():
        LOGGER.debug("Plik SEC filing_dates nie istnieje: %s", filing_dates_file)
        return pd.DataFrame()

    try:
        df = pd.read_csv(filing_dates_file)

        # Konwersja dat
        df["end_date"] = pd.to_datetime(df["end_date"])
        df["filing_date"] = pd.to_datetime(df["filing_date"])

        # Deduplikacja: jeśli jest wiele raportów dla tej samej (ticker, end_date),
        # zachowaj najwcześniejszą filing_date (pierwsze ogłoszenie)
        df = df.sort_values(["ticker", "end_date", "filing_date"])
        df = df.drop_duplicates(subset=["ticker", "end_date"], keep="first")

        LOGGER.info("Załadowano %d rekordów SEC filing_dates", len(df))
        return df[["ticker", "end_date", "filing_date"]]

    except Exception as exc:
        LOGGER.error("Błąd wczytywania SEC filing_dates: %s", exc)
        return pd.DataFrame()
