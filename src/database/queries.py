"""Zapytania SQL do ekstrakcji danych analitycznych z bazy DuckDB."""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.database.connection import get_connection

LOGGER = logging.getLogger("database")


def get_prices_cleaned(
    ticker: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    country: str | None = None,
) -> pd.DataFrame:
    """
    Zwraca wyczyszczone dane cenowe z cleaned.prices_cleaned.

    Args:
        ticker: Ticker (np. 'AAPL'). Jeśli None - wszystkie.
        start_date: Data początkowa.
        end_date: Data końcowa.
        country: Kraj ('US', 'PL'). Jeśli None - wszystkie.

    Returns:
        DataFrame z cenami.
    """
    conn = get_connection()

    query = "SELECT * FROM cleaned.prices_cleaned WHERE 1=1"
    params = []

    if ticker:
        query += " AND ticker = ?"
        params.append(ticker)

    if start_date:
        query += " AND date >= ?"
        params.append(start_date)

    if end_date:
        query += " AND date <= ?"
        params.append(end_date)

    if country:
        query += " AND country = ?"
        params.append(country)

    query += " ORDER BY ticker, date"

    return conn.execute(query, params).df()


def get_financials_wide(
    ticker: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    country: str | None = None,
) -> pd.DataFrame:
    """
    Zwraca dane finansowe w formacie wide z cleaned.financials_wide.

    Args:
        ticker: Ticker. Jeśli None - wszystkie.
        start_date: Data początkowa (end_date).
        end_date: Data końcowa (end_date).
        country: Kraj ('US', 'PL'). Jeśli None - wszystkie.

    Returns:
        DataFrame z danymi finansowymi (pivot format).
    """
    conn = get_connection()

    query = "SELECT * FROM cleaned.financials_wide WHERE 1=1"
    params = []

    if ticker:
        query += " AND ticker = ?"
        params.append(ticker)

    if start_date:
        query += " AND end_date >= ?"
        params.append(start_date)

    if end_date:
        query += " AND end_date <= ?"
        params.append(end_date)

    if country:
        query += " AND country = ?"
        params.append(country)

    query += " ORDER BY ticker, end_date"

    return conn.execute(query, params).df()


def get_macro_normalized(
    start_date: str | None = None,
    end_date: str | None = None,
    country: str | None = None,
) -> pd.DataFrame:
    """
    Zwraca znormalizowane dane makro z cleaned.macro_normalized.

    Args:
        start_date: Data początkowa (period_date).
        end_date: Data końcowa (period_date).
        country: Kraj ('US', 'PL'). Jeśli None - wszystkie.

    Returns:
        DataFrame z danymi makro.
    """
    conn = get_connection()

    query = "SELECT * FROM cleaned.macro_normalized WHERE 1=1"
    params = []

    if start_date:
        query += " AND period_date >= ?"
        params.append(start_date)

    if end_date:
        query += " AND period_date <= ?"
        params.append(end_date)

    if country:
        query += " AND country = ?"
        params.append(country)

    query += " ORDER BY period_date, country"

    return conn.execute(query, params).df()


def get_table_stats() -> dict[str, dict[str, Any]]:
    """
    Zwraca statystyki dla wszystkich tabel.

    Returns:
        Słownik ze statystykami: count, date_range itp.
    """
    from src.database.schema_raw import get_date_range, get_table_count

    tables = ["prices", "financials", "macro", "dividends", "sectors", "shares_outstanding"]
    stats = {}

    for table in tables:
        count = get_table_count(table)
        stats[table] = {"count": count}

        # Date range dla tabel z datami
        if table in ["prices", "macro"]:
            date_col = "date"
            min_date, max_date = get_date_range(table, date_col)
            stats[table]["date_range"] = (min_date, max_date)
        elif table == "shares_outstanding":
            min_date, max_date = get_date_range(table, "filing_date")
            stats[table]["date_range"] = (min_date, max_date)
        elif table == "financials":
            min_date, max_date = get_date_range(table, "end_date")
            stats[table]["date_range"] = (min_date, max_date)
        elif table == "dividends":
            min_date, max_date = get_date_range(table, "dividend_date")
            stats[table]["date_range"] = (min_date, max_date)

    return stats


def execute_custom_query(query: str, params: list | None = None) -> pd.DataFrame:
    """
    Wykonuje dowolne zapytanie SQL i zwraca DataFrame.

    Args:
        query: Zapytanie SQL.
        params: Parametry zapytania.

    Returns:
        DataFrame z wynikami.
    """
    conn = get_connection()

    try:
        if params:
            return conn.execute(query, params).df()
        return conn.execute(query).df()
    except Exception as exc:
        LOGGER.error("Błąd podczas wykonywania zapytania: %s", exc)
        raise
