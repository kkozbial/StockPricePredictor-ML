"""Definicje schematów tabel w bazie DuckDB."""
from __future__ import annotations

import logging

from src.database.connection import get_connection, table_exists

LOGGER = logging.getLogger("database")


# Definicje SQL dla tabel
TABLES = {
    "prices": """
        CREATE TABLE IF NOT EXISTS prices (
            ticker VARCHAR NOT NULL,
            date DATE NOT NULL,
            country VARCHAR DEFAULT 'US',
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume BIGINT,
            PRIMARY KEY (ticker, date, country)
        )
    """,
    "financials": """
        CREATE TABLE IF NOT EXISTS financials (
            ticker VARCHAR NOT NULL,
            concept VARCHAR NOT NULL,
            country VARCHAR DEFAULT 'US',
            unit VARCHAR,
            start_date DATE,
            end_date DATE NOT NULL,
            value DOUBLE,
            form VARCHAR,
            filing_date DATE
        )
    """,
    "macro": """
        CREATE TABLE IF NOT EXISTS macro (
            date DATE NOT NULL PRIMARY KEY,
            -- Merged indicators (same measure, different countries)
            inflation_usa DOUBLE,                -- CPI Index (CPIAUCSL)
            inflation_pl DOUBLE,                 -- HICP % YoY (Eurostat)
            unemployment_rate_usa DOUBLE,        -- Unemployment % (UNRATE)
            unemployment_rate_pl DOUBLE,         -- Unemployment % (Eurostat)
            interest_rate_usa DOUBLE,            -- Fed Funds Rate % (FEDFUNDS)
            interest_rate_pl DOUBLE,             -- 3M rate % (Eurostat)
            -- USA-specific indicators
            gdp_real_usa DOUBLE,                 -- Real GDP w mld USD (GDPC1)
            retail_sales_usa DOUBLE,             -- Retail Sales w mln USD (RSXFS)
            -- Poland-specific indicators
            gdp_real_pl DOUBLE,                  -- Real GDP w mln PLN chain-linked (CLV_MNAC)
            retail_sales_index_pl DOUBLE         -- Indeks sprzedaży 2021=100 (I21)
        )
    """,
    "dividends": """
        CREATE TABLE IF NOT EXISTS dividends (
            ticker VARCHAR NOT NULL,
            dividend_date DATE NOT NULL,
            country VARCHAR DEFAULT 'US',
            dividend_amount DOUBLE,
            PRIMARY KEY (ticker, dividend_date, country)
        )
    """,
    "sectors": """
        CREATE TABLE IF NOT EXISTS sectors (
            ticker VARCHAR NOT NULL,
            country VARCHAR DEFAULT 'US',
            sector VARCHAR,
            industry VARCHAR,
            last_updated TIMESTAMP,
            PRIMARY KEY (ticker, country)
        )
    """,
    "shares_outstanding": """
        CREATE TABLE IF NOT EXISTS shares_outstanding (
            ticker VARCHAR NOT NULL,
            filing_date DATE NOT NULL,
            country VARCHAR DEFAULT 'US',
            shares BIGINT,
            end_date DATE,
            fiscal_year INTEGER,
            fiscal_period VARCHAR,
            source_type VARCHAR,
            PRIMARY KEY (ticker, filing_date, country)
        )
    """,
    "company_metadata": """
        CREATE TABLE IF NOT EXISTS company_metadata (
            cik VARCHAR NOT NULL,
            ticker VARCHAR NOT NULL,
            country VARCHAR DEFAULT 'US',
            name VARCHAR,
            sic INTEGER,
            sic_description VARCHAR,
            exchange VARCHAR,
            is_excluded BOOLEAN DEFAULT FALSE,
            last_updated TIMESTAMP,
            PRIMARY KEY (cik, ticker, country)
        )
    """,
    "company_status": """
        CREATE TABLE IF NOT EXISTS company_status (
            ticker VARCHAR NOT NULL,
            cik VARCHAR NOT NULL,
            country VARCHAR DEFAULT 'US',
            status VARCHAR NOT NULL,
            delisting_date DATE,
            delisting_form VARCHAR,
            last_filing_date DATE,
            last_financial_report_date DATE,
            reason VARCHAR,
            last_updated TIMESTAMP,
            PRIMARY KEY (ticker, country)
        )
    """,
}


def create_all_tables() -> None:
    """Tworzy wszystkie tabele w bazie danych jeśli nie istnieją."""
    conn = get_connection()

    for table_name, create_sql in TABLES.items():
        try:
            conn.execute(create_sql)
            if table_exists(table_name):
                LOGGER.info("Tabela '%s' utworzona lub już istnieje", table_name)
            else:
                LOGGER.warning("Nie udało się utworzyć tabeli '%s'", table_name)
        except Exception as exc:
            LOGGER.error("Błąd podczas tworzenia tabeli '%s': %s", table_name, exc)
            raise


def drop_all_tables() -> None:
    """Usuwa wszystkie tabele z bazy danych."""
    conn = get_connection()

    for table_name in TABLES.keys():
        try:
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            LOGGER.info("Tabela '%s' usunięta", table_name)
        except Exception as exc:
            LOGGER.error("Błąd podczas usuwania tabeli '%s': %s", table_name, exc)
            raise


def get_table_info(table_name: str) -> list[tuple]:
    """
    Zwraca informacje o strukturze tabeli.

    Args:
        table_name: Nazwa tabeli.

    Returns:
        Lista krotek z informacjami o kolumnach.
    """
    conn = get_connection()
    return conn.execute(f"PRAGMA table_info({table_name})").fetchall()


def get_table_count(table_name: str) -> int:
    """
    Zwraca liczbę rekordów w tabeli.

    Args:
        table_name: Nazwa tabeli.

    Returns:
        Liczba rekordów.
    """
    if not table_exists(table_name):
        return 0

    conn = get_connection()
    result = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
    return result[0] if result else 0


def get_date_range(table_name: str, date_column: str = "date", country: str | None = None) -> tuple[str | None, str | None]:
    """
    Zwraca zakres dat w tabeli.

    Args:
        table_name: Nazwa tabeli.
        date_column: Nazwa kolumny z datą.
        country: Opcjonalnie filtruj po kraju (dla tabel z kolumną country).

    Returns:
        Krotka (min_date, max_date) lub (None, None) jeśli tabela pusta.
    """
    if not table_exists(table_name):
        return None, None

    conn = get_connection()

    # Dla tabeli macro (bez kolumny country) lub innych tabel bez filtru
    if country is None or table_name == "macro":
        result = conn.execute(
            f"SELECT MIN({date_column}), MAX({date_column}) FROM {table_name}"
        ).fetchone()
    else:
        # Dla tabel z kolumną country (prices, financials, etc.)
        result = conn.execute(
            f"SELECT MIN({date_column}), MAX({date_column}) FROM {table_name} WHERE country = ?",
            [country]
        ).fetchone()

    if result and result[0]:
        return str(result[0]), str(result[1])
    return None, None
