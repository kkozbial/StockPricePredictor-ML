"""Funkcje pomocnicze do pobierania przyrostowego danych."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

from ..database.connection import get_connection, table_exists
from ..database.schema_raw import get_date_range

LOGGER = logging.getLogger("utils.incremental")


def get_last_date_for_ticker(table_name: str, ticker: str, date_column: str = "date", country: Optional[str] = None) -> Optional[str]:
    """
    Zwraca ostatnią datę dla konkretnego tickera w tabeli.

    Args:
        table_name: Nazwa tabeli (np. 'prices', 'financials').
        ticker: Ticker akcji (np. 'AAPL').
        date_column: Nazwa kolumny z datą.
        country: Kod kraju (np. 'US', 'PL'). Jeśli None, nie filtruje po kraju.

    Returns:
        Ostatnia data w formacie YYYY-MM-DD lub None jeśli brak danych.
    """
    try:
        if not table_exists(table_name):
            LOGGER.debug("[incremental] Tabela '%s' nie istnieje - brak danych dla %s", table_name, ticker)
            return None

        conn = get_connection()

        # Buduj zapytanie z opcjonalnym filtrowaniem po kraju
        if country:
            query = f"SELECT MAX({date_column}) FROM {table_name} WHERE ticker = ? AND country = ?"
            result = conn.execute(query, [ticker, country]).fetchone()
        else:
            query = f"SELECT MAX({date_column}) FROM {table_name} WHERE ticker = ?"
            result = conn.execute(query, [ticker]).fetchone()

        if result and result[0]:
            last_date = str(result[0])
            country_info = f" (country={country})" if country else ""
            LOGGER.debug("[incremental] Ostatnia data dla %s%s w %s: %s", ticker, country_info, table_name, last_date)
            return last_date

        LOGGER.debug("[incremental] Brak danych dla %s w tabeli %s", ticker, table_name)
        return None
    except ValueError as exc:
        # Baza nie jest jeszcze zainicjalizowana
        if "Database path must be provided" in str(exc):
            LOGGER.debug("[incremental] Baza danych nie jest jeszcze zainicjalizowana - zwracam None dla %s", ticker)
            return None
        raise
    except Exception as exc:
        LOGGER.warning("[incremental] Błąd podczas pobierania ostatniej daty dla %s: %s", ticker, exc)
        return None


def get_last_date_for_series(table_name: str, series_id: str) -> Optional[str]:
    """
    Zwraca ostatnią datę dla konkretnej serii makro w tabeli.

    Args:
        table_name: Nazwa tabeli (zwykle 'macro').
        series_id: ID serii (np. 'CPIAUCSL').

    Returns:
        Ostatnia data w formacie YYYY-MM-DD lub None jeśli brak danych.
    """
    try:
        if not table_exists(table_name):
            LOGGER.debug("[incremental] Tabela '%s' nie istnieje", table_name)
            return None

        # Dla danych makro sprawdzamy ostatnią datę globalnie
        # (wszystkie serie są w jednej tabeli z różnymi kolumnami)
        _, max_date = get_date_range(table_name, date_column="date")

        if max_date:
            LOGGER.debug("[incremental] Ostatnia data w tabeli macro: %s", max_date)
            return max_date

        LOGGER.debug("[incremental] Brak danych w tabeli macro")
        return None
    except ValueError as exc:
        # Baza nie jest jeszcze zainicjalizowana
        if "Database path must be provided" in str(exc):
            LOGGER.debug("[incremental] Baza danych nie jest jeszcze zainicjalizowana - zwracam None")
            return None
        raise
    except Exception as exc:
        LOGGER.warning("[incremental] Błąd podczas pobierania ostatniej daty dla serii makro: %s", exc)
        return None


def calculate_incremental_start_date(
    last_date: Optional[str],
    default_start: str,
    overlap_days: int = 1,
    ticker: Optional[str] = None
) -> str:
    """
    Oblicza datę początkową dla pobierania przyrostowego.

    Args:
        last_date: Ostatnia data w bazie (YYYY-MM-DD) lub None.
        default_start: Domyślna data początkowa jeśli brak danych w bazie.
        overlap_days: Ile dni nakładki (dla pewności, domyślnie 1 dzień).
        ticker: Opcjonalny ticker do wyświetlenia w logach.

    Returns:
        Data początkowa dla fetch w formacie YYYY-MM-DD.
    """
    ticker_info = f" [{ticker}]" if ticker else ""

    if last_date is None:
        LOGGER.info("[incremental]%s Brak danych w bazie - pobieranie pełnej historii od %s", ticker_info, default_start)
        return default_start

    try:
        # Parsuj ostatnią datę i dodaj overlap
        last_dt = datetime.strptime(last_date, "%Y-%m-%d")
        # Odejmij overlap_days, żeby złapać ewentualne luki
        start_dt = last_dt - timedelta(days=overlap_days)
        start_date = start_dt.strftime("%Y-%m-%d")

        LOGGER.debug(
            "[incremental]%s Ostatnia data: %s, pobieranie od: %s (overlap: %d dni)",
            ticker_info, last_date, start_date, overlap_days
        )
        return start_date
    except ValueError:
        LOGGER.warning("[incremental]%s Nieprawidłowy format daty %s, używam domyślnej", ticker_info, last_date)
        return default_start


def should_use_incremental(config: dict, data_type: str) -> bool:
    """
    Sprawdza czy dla danego typu danych powinno być użyte pobieranie przyrostowe.

    Args:
        config: Słownik konfiguracji z settings.yaml.
        data_type: Typ danych ('prices', 'financials', 'macro').

    Returns:
        True jeśli pobieranie przyrostowe jest włączone, False w przeciwnym razie.
    """
    try:
        incremental_config = config.get("fetch", {}).get("incremental", {})
        is_incremental = incremental_config.get(data_type, False)

        if is_incremental:
            LOGGER.info("[incremental] Pobieranie przyrostowe WŁĄCZONE dla %s", data_type)
        else:
            LOGGER.info("[incremental] Pobieranie przyrostowe WYŁĄCZONE dla %s - pełne pobieranie", data_type)

        return is_incremental
    except Exception as exc:
        LOGGER.warning("[incremental] Błąd odczytu konfiguracji dla %s: %s, używam pełnego pobierania", data_type, exc)
        return False


def get_delisted_tickers() -> set[str]:
    """
    Zwraca zestaw tickerów ze statusem DELISTED lub DEREGISTERED.

    Returns:
        Set tickerów (uppercase), dla których dalsze pobieranie danych jest zbędne.
    """
    try:
        if not table_exists("company_status"):
            return set()

        conn = get_connection()
        rows = conn.execute("""
            SELECT ticker FROM company_status
            WHERE status IN ('DELISTED', 'DEREGISTERED')
        """).fetchall()

        return {row[0].upper() for row in rows}
    except ValueError as exc:
        if "Database path must be provided" in str(exc):
            return set()
        raise
    except Exception as exc:
        LOGGER.warning("[incremental] Błąd podczas pobierania delisted tickerów: %s", exc)
        return set()


def get_existing_financial_records(ticker: str) -> set[tuple[str, str, float]]:
    """
    Zwraca zestaw istniejących rekordów finansowych dla tickera.

    Args:
        ticker: Ticker akcji.

    Returns:
        Set krotek (ticker, date, value) reprezentujących istniejące wpisy.
    """
    try:
        if not table_exists("financials"):
            return set()

        conn = get_connection()
        # Pobieramy unikalne kombinacje ticker + financial_date + revenues (jako identyfikator)
        query = """
            SELECT ticker, financial_date, revenues
            FROM financials
            WHERE ticker = ?
        """
        result = conn.execute(query, [ticker]).fetchall()

        records = set()
        for row in result:
            # Tworzymy tuple jako fingerprint wpisu
            if row[2] is not None:  # revenues może być None
                records.add((row[0], str(row[1]), float(row[2])))

        LOGGER.debug("[incremental] Znaleziono %d istniejących rekordów finansowych dla %s", len(records), ticker)
        return records
    except ValueError as exc:
        # Baza nie jest jeszcze zainicjalizowana
        if "Database path must be provided" in str(exc):
            LOGGER.debug("[incremental] Baza danych nie jest jeszcze zainicjalizowana - zwracam pusty set")
            return set()
        raise
    except Exception as exc:
        LOGGER.warning("[incremental] Błąd podczas pobierania istniejących rekordów dla %s: %s", ticker, exc)
        return set()
