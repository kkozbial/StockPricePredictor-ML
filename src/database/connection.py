"""Zarządzanie połączeniem z bazą danych DuckDB."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import duckdb

LOGGER = logging.getLogger("database")


class DatabaseConnection:
    """Singleton zarządzający połączeniem z DuckDB."""

    _instance: duckdb.DuckDBPyConnection | None = None
    _db_path: Path | None = None

    @classmethod
    def get_connection(cls, db_path: str | Path | None = None) -> duckdb.DuckDBPyConnection:
        """
        Zwraca połączenie z bazą DuckDB (singleton).

        Args:
            db_path: Ścieżka do pliku bazy danych. Jeśli None, używa wcześniej ustalonej.

        Returns:
            Połączenie DuckDB.
        """
        if cls._instance is None or (db_path is not None and Path(db_path) != cls._db_path):
            if db_path is None:
                raise ValueError("Database path must be provided on first connection")

            cls._db_path = Path(db_path)
            cls._db_path.parent.mkdir(parents=True, exist_ok=True)

            cls._instance = duckdb.connect(str(cls._db_path))
            LOGGER.info("Połączono z bazą DuckDB: %s", cls._db_path)

        return cls._instance

    @classmethod
    def close(cls) -> None:
        """Zamyka połączenie z bazą danych."""
        if cls._instance is not None:
            cls._instance.close()
            LOGGER.info("Zamknięto połączenie z bazą DuckDB")
            cls._instance = None
            cls._db_path = None


def get_connection(db_path: str | Path | None = None) -> duckdb.DuckDBPyConnection:
    """
    Zwraca połączenie z bazą DuckDB.

    Args:
        db_path: Ścieżka do pliku bazy danych.

    Returns:
        Połączenie DuckDB.
    """
    return DatabaseConnection.get_connection(db_path)


def init_database(db_path: str | Path) -> duckdb.DuckDBPyConnection:
    """
    Inicjalizuje bazę danych i zwraca połączenie.

    Args:
        db_path: Ścieżka do pliku bazy danych.

    Returns:
        Połączenie DuckDB.
    """
    conn = get_connection(db_path)
    LOGGER.info("Zainicjalizowano bazę danych: %s", db_path)
    return conn


def execute_query(query: str, params: dict[str, Any] | None = None) -> Any:
    """
    Wykonuje zapytanie SQL na bieżącym połączeniu.

    Args:
        query: Zapytanie SQL.
        params: Parametry zapytania.

    Returns:
        Wynik zapytania.
    """
    conn = get_connection()
    if params:
        return conn.execute(query, params).fetchall()
    return conn.execute(query).fetchall()


def table_exists(table_name: str) -> bool:
    """
    Sprawdza czy tabela istnieje w bazie danych.

    Args:
        table_name: Nazwa tabeli.

    Returns:
        True jeśli tabela istnieje, False w przeciwnym razie.
    """
    conn = get_connection()
    result = conn.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
        [table_name]
    ).fetchone()
    return result[0] > 0 if result else False
