"""Moduł zarządzania bazą danych DuckDB dla projektu stock market."""
from __future__ import annotations

from src.database.connection import get_connection, init_database
from src.database.loaders import build_all_tables, update_all_tables
from src.database.schema_raw import create_all_tables, drop_all_tables

__all__ = [
    "get_connection",
    "init_database",
    "create_all_tables",
    "drop_all_tables",
    "build_all_tables",
    "update_all_tables",
]
