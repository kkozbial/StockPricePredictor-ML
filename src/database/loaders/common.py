"""Wspólne funkcje dla loaderów danych."""
from __future__ import annotations

import logging

import pandas as pd

from src.database.connection import get_connection

LOGGER = logging.getLogger("database")


def upsert_dataframe(df: pd.DataFrame, table_name: str, primary_keys: list[str]) -> int:
    """
    Wstawia lub aktualizuje dane w tabeli (UPSERT).

    Args:
        df: DataFrame z danymi.
        table_name: Nazwa tabeli.
        primary_keys: Lista kolumn będących kluczem głównym.

    Returns:
        Liczba wstawionych/zaktualizowanych rekordów.
    """
    if df.empty:
        return 0

    conn = get_connection()

    # DuckDB wspiera INSERT OR REPLACE
    # Najpierw tworzymy tymczasową tabelę
    temp_table = f"temp_{table_name}"

    try:
        conn.execute(f"DROP TABLE IF EXISTS {temp_table}")
        conn.execute(f"CREATE TEMP TABLE {temp_table} AS SELECT * FROM {table_name} WHERE 1=0")

        # Pobierz kolejność kolumn z schematu tabeli
        schema_cols = [row[0] for row in conn.execute(f"DESCRIBE {table_name}").fetchall()]

        # Uporządkuj DataFrame według schematu tabeli
        df_ordered = df[schema_cols]

        # Wstawiamy dane do temp
        conn.register("temp_df", df_ordered)
        conn.execute(f"INSERT INTO {temp_table} SELECT * FROM temp_df")

        # DELETE old records matching primary keys, then INSERT new
        # Obsługa NULL wartości: używamy IS NOT DISTINCT FROM zamiast =
        pk_conditions = " AND ".join([
            f"{table_name}.{pk} IS NOT DISTINCT FROM {temp_table}.{pk}"
            for pk in primary_keys
        ])

        conn.execute(f"""
            DELETE FROM {table_name}
            WHERE EXISTS (
                SELECT 1 FROM {temp_table}
                WHERE {pk_conditions}
            )
        """)

        # Insert all from temp
        conn.execute(f"INSERT INTO {table_name} SELECT * FROM {temp_table}")

        count = len(df)
        conn.execute(f"DROP TABLE {temp_table}")

        return count

    except Exception as exc:
        LOGGER.error("Błąd podczas UPSERT do tabeli %s: %s", table_name, exc)
        raise
