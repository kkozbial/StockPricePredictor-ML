"""Moduł ładowania danych do bazy DuckDB z logiką BUILD i UPDATE."""
from __future__ import annotations

import logging
from pathlib import Path

from src.database.connection import table_exists
from src.database.loaders.common import upsert_dataframe
from src.database.loaders.dividends import load_dividends_from_raw
from src.database.loaders.financials import load_financials_from_raw
from src.database.loaders.macro import load_macro_from_raw
from src.database.loaders.metadata import (
    load_company_metadata_from_raw,
    load_company_status_from_raw,
    load_sectors_from_raw,
    load_shares_from_raw,
)
from src.database.loaders.prices import load_prices_from_raw
from src.database.schema_raw import create_all_tables, get_date_range

LOGGER = logging.getLogger("database")


def build_all_tables(raw_dir: Path) -> dict[str, int]:
    """
    BUILD: Tworzy tabele i ładuje wszystkie dane z raw (początkowe załadowanie).

    Args:
        raw_dir: Ścieżka do katalogu data/raw

    Returns:
        Słownik z liczbą załadowanych rekordów dla każdej tabeli.
    """
    LOGGER.info("BUILD: Tworzenie tabel i ładowanie danych...")

    # Tworzymy tabele
    create_all_tables()

    results = {}

    # Prices
    LOGGER.info("Ładowanie danych cenowych...")
    prices_df = load_prices_from_raw(raw_dir)
    if not prices_df.empty:
        results["prices"] = upsert_dataframe(prices_df, "prices", ["ticker", "date", "country"])
        LOGGER.info("Załadowano %s rekordów cen", results["prices"])
    else:
        results["prices"] = 0

    # Financials
    LOGGER.info("Ładowanie danych finansowych...")
    financials_df = load_financials_from_raw(raw_dir)
    if not financials_df.empty:
        results["financials"] = upsert_dataframe(financials_df, "financials", ["ticker", "concept", "start_date", "end_date"])
        LOGGER.info("Załadowano %s rekordów finansowych", results["financials"])
    else:
        results["financials"] = 0

    # Macro
    LOGGER.info("Ładowanie danych makroekonomicznych...")
    macro_df = load_macro_from_raw(raw_dir)
    if not macro_df.empty:
        results["macro"] = upsert_dataframe(macro_df, "macro", ["date"])
        LOGGER.info("Załadowano %s rekordów makro", results["macro"])
    else:
        results["macro"] = 0

    # Dividends
    LOGGER.info("Ładowanie dywidend...")
    dividends_df = load_dividends_from_raw(raw_dir)
    if not dividends_df.empty:
        results["dividends"] = upsert_dataframe(dividends_df, "dividends", ["ticker", "dividend_date", "country"])
        LOGGER.info("Załadowano %s rekordów dywidend", results["dividends"])
    else:
        results["dividends"] = 0

    # Sectors
    LOGGER.info("Ładowanie sektorów...")
    sectors_df = load_sectors_from_raw(raw_dir)
    if not sectors_df.empty:
        results["sectors"] = upsert_dataframe(sectors_df, "sectors", ["ticker", "country"])
        LOGGER.info("Załadowano %s rekordów sektorów", results["sectors"])
    else:
        results["sectors"] = 0

    # Shares
    LOGGER.info("Ładowanie danych o akcjach w obiegu...")
    shares_df = load_shares_from_raw(raw_dir)
    if not shares_df.empty:
        results["shares_outstanding"] = upsert_dataframe(shares_df, "shares_outstanding", ["ticker", "filing_date", "country"])
        LOGGER.info("Załadowano %s rekordów akcji", results["shares_outstanding"])
    else:
        results["shares_outstanding"] = 0

    # Company Metadata
    LOGGER.info("Ładowanie metadanych firm...")
    metadata_df = load_company_metadata_from_raw(raw_dir)
    if not metadata_df.empty:
        results["company_metadata"] = upsert_dataframe(metadata_df, "company_metadata", ["cik", "ticker", "country"])
        LOGGER.info("Załadowano %s rekordów metadanych firm", results["company_metadata"])
    else:
        results["company_metadata"] = 0

    # Company Status
    LOGGER.info("Ładowanie statusów spółek...")
    status_df = load_company_status_from_raw(raw_dir)
    if not status_df.empty:
        results["company_status"] = upsert_dataframe(status_df, "company_status", ["ticker", "country"])
        LOGGER.info("Załadowano %s rekordów statusów spółek", results["company_status"])
    else:
        results["company_status"] = 0

    LOGGER.info("BUILD zakończony. Podsumowanie: %s", results)
    return results


def update_all_tables(raw_dir: Path) -> dict[str, int]:
    """
    UPDATE: Aktualizuje istniejące tabele nowymi danymi z raw (przyrostowa aktualizacja).

    Args:
        raw_dir: Ścieżka do katalogu data/raw

    Returns:
        Słownik z liczbą zaktualizowanych rekordów dla każdej tabeli.
    """
    LOGGER.info("UPDATE: Aktualizacja danych w istniejących tabelach...")

    results = {}

    # Prices
    if table_exists("prices"):
        min_date, max_date = get_date_range("prices", "date")
        LOGGER.info("Aktualizacja cen (obecny zakres: %s - %s)", min_date, max_date)

        prices_df = load_prices_from_raw(raw_dir)
        if not prices_df.empty:
            # Filtrujemy tylko nowe dane (opcjonalnie)
            # W przypadku UPSERT to nie jest konieczne, ale może przyspieszyć
            results["prices"] = upsert_dataframe(prices_df, "prices", ["ticker", "date", "country"])
            LOGGER.info("Zaktualizowano %s rekordów cen", results["prices"])
        else:
            results["prices"] = 0
    else:
        LOGGER.warning("Tabela 'prices' nie istnieje. Uruchom BUILD najpierw.")
        results["prices"] = 0

    # Financials
    if table_exists("financials"):
        financials_df = load_financials_from_raw(raw_dir)
        if not financials_df.empty:
            results["financials"] = upsert_dataframe(financials_df, "financials", ["ticker", "concept", "start_date", "end_date"])
            LOGGER.info("Zaktualizowano %s rekordów finansowych", results["financials"])
        else:
            results["financials"] = 0
    else:
        results["financials"] = 0

    # Macro
    if table_exists("macro"):
        macro_df = load_macro_from_raw(raw_dir)
        if not macro_df.empty:
            results["macro"] = upsert_dataframe(macro_df, "macro", ["date"])
            LOGGER.info("Zaktualizowano %s rekordów makro", results["macro"])
        else:
            results["macro"] = 0
    else:
        results["macro"] = 0

    # Dividends
    if table_exists("dividends"):
        dividends_df = load_dividends_from_raw(raw_dir)
        if not dividends_df.empty:
            results["dividends"] = upsert_dataframe(dividends_df, "dividends", ["ticker", "dividend_date", "country"])
            LOGGER.info("Zaktualizowano %s rekordów dywidend", results["dividends"])
        else:
            results["dividends"] = 0
    else:
        results["dividends"] = 0

    # Sectors
    if table_exists("sectors"):
        sectors_df = load_sectors_from_raw(raw_dir)
        if not sectors_df.empty:
            results["sectors"] = upsert_dataframe(sectors_df, "sectors", ["ticker", "country"])
            LOGGER.info("Zaktualizowano %s rekordów sektorów", results["sectors"])
        else:
            results["sectors"] = 0
    else:
        results["sectors"] = 0

    # Shares
    if table_exists("shares_outstanding"):
        shares_df = load_shares_from_raw(raw_dir)
        if not shares_df.empty:
            results["shares_outstanding"] = upsert_dataframe(shares_df, "shares_outstanding", ["ticker", "filing_date", "country"])
            LOGGER.info("Zaktualizowano %s rekordów akcji", results["shares_outstanding"])
        else:
            results["shares_outstanding"] = 0
    else:
        results["shares_outstanding"] = 0

    # Company Metadata
    if table_exists("company_metadata"):
        metadata_df = load_company_metadata_from_raw(raw_dir)
        if not metadata_df.empty:
            results["company_metadata"] = upsert_dataframe(metadata_df, "company_metadata", ["cik", "ticker", "country"])
            LOGGER.info("Zaktualizowano %s rekordów metadanych firm", results["company_metadata"])
        else:
            results["company_metadata"] = 0
    else:
        results["company_metadata"] = 0

    # Company Status
    if table_exists("company_status"):
        status_df = load_company_status_from_raw(raw_dir)
        if not status_df.empty:
            results["company_status"] = upsert_dataframe(status_df, "company_status", ["ticker", "country"])
            LOGGER.info("Zaktualizowano %s rekordów statusów spółek", results["company_status"])
        else:
            results["company_status"] = 0
    else:
        results["company_status"] = 0

    LOGGER.info("UPDATE zakończony. Podsumowanie: %s", results)
    return results


__all__ = [
    "build_all_tables",
    "update_all_tables",
    "upsert_dataframe",
    "load_prices_from_raw",
    "load_financials_from_raw",
    "load_macro_from_raw",
    "load_dividends_from_raw",
    "load_sectors_from_raw",
    "load_shares_from_raw",
    "load_company_metadata_from_raw",
    "load_company_status_from_raw",
]
