"""Moduł normalizacji danych makroekonomicznych dla ML - integracja z pipeline (NOWY SCHEMAT 2026-01-04)."""
from __future__ import annotations

import logging
from datetime import timedelta

import pandas as pd

from src.database.connection import get_connection

LOGGER = logging.getLogger("preprocessing")

# Lag publikacji dla danych makro (w dniach)
# Dane makro publikowane są zazwyczaj z miesięcznym opóźnieniem
PUBLICATION_LAG_DAYS = 30


def _compute_yoy(series: pd.Series) -> pd.Series:
    """
    Oblicza zmianę % rok-do-roku (YoY).

    Formuła: (Obecny - RokTemu) / RokTemu * 100
    Używa pct_change z periods=12 (zakładamy dane miesięczne).
    """
    if series.empty:
        return series
    # pct_change(12) zwraca ułamek dziesiętny, mnożymy przez 100 dla %
    return series.pct_change(periods=12, fill_method=None) * 100


def _compute_diff(series: pd.Series) -> pd.Series:
    """
    Oblicza różnicę miesiąc-do-miesiąca (M/M).

    Formuła: Obecny - Poprzedni
    """
    if series.empty:
        return series
    return series.diff(periods=1)


def process_macro_to_staging() -> int:
    """
    Pobiera dane z main.macro, transformuje je zgodnie z wytycznymi ekonomicznymi
    i zapisuje do staging.macro_normalized.

    NOWA STRUKTURA (2026-01-04):
    - Transformacje:
      * Retail Sales: YoY (zmiana % rok-do-roku)
      * GDP: YoY (zmiana % rok-do-roku)
      * CPI: YoY (inflacja)
      * Bezrobocie: poziom + różnica M/M
      * Stopa %: poziom + różnica M/M
    - Daty:
      * period_date: data okresu pomiaru
      * publication_date: period_date + lag (~1 miesiąc)
    - Format: długi (USA i PL w osobnych wierszach)

    Returns:
        Liczba przetworzonych rekordów
    """
    conn = get_connection()

    # Sprawdź czy tabela main.macro istnieje
    check_query = """
        SELECT COUNT(*) FROM information_schema.tables
        WHERE table_schema = 'main' AND table_name = 'macro'
    """
    table_exists = conn.execute(check_query).fetchone()[0] > 0

    if not table_exists:
        LOGGER.warning("Tabela main.macro nie istnieje. Najpierw uruchom BUILD/UPDATE.")
        return 0

    # Pobierz dane z main.macro
    df = conn.execute("SELECT * FROM main.macro").df()

    if df.empty:
        LOGGER.warning("Brak danych w main.macro - preprocessing pominięty")
        return 0

    LOGGER.info("Pobrano %d rekordów z main.macro", len(df))

    # Konwertuj datę
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # ========== PRZETWARZANIE USA ==========
    LOGGER.info("Transformacja danych makro USA...")
    df_usa = pd.DataFrame()
    df_usa["period_date"] = df["date"].copy()

    # Retail Sales: YoY - tylko jeśli kolumna istnieje i ma jakieś dane
    if "retail_sales_usa" in df.columns and df["retail_sales_usa"].notna().any():
        df_usa["retail_sales_yoy"] = _compute_yoy(df["retail_sales_usa"])
    else:
        df_usa["retail_sales_yoy"] = None

    # GDP: YoY - tylko jeśli kolumna istnieje i ma jakieś dane
    if "gdp_real_usa" in df.columns and df["gdp_real_usa"].notna().any():
        df_usa["gdp_yoy"] = _compute_yoy(df["gdp_real_usa"])
    else:
        df_usa["gdp_yoy"] = None

    # CPI: YoY (inflacja - oblicz % YoY z surowego indeksu CPI)
    if "inflation_usa" in df.columns and df["inflation_usa"].notna().any():
        df_usa["cpi_yoy"] = _compute_yoy(df["inflation_usa"])
    else:
        df_usa["cpi_yoy"] = None

    # Bezrobocie: poziom + różnica M/M
    if "unemployment_rate_usa" in df.columns:
        df_usa["unemp_level"] = df["unemployment_rate_usa"].copy()
        if df["unemployment_rate_usa"].notna().any():
            df_usa["unemp_diff"] = _compute_diff(df["unemployment_rate_usa"])
        else:
            df_usa["unemp_diff"] = None
    else:
        df_usa["unemp_level"] = None
        df_usa["unemp_diff"] = None

    # Stopa %: poziom + różnica M/M
    if "interest_rate_usa" in df.columns:
        df_usa["rate_level"] = df["interest_rate_usa"].copy()
        if df["interest_rate_usa"].notna().any():
            df_usa["rate_diff"] = _compute_diff(df["interest_rate_usa"])
        else:
            df_usa["rate_diff"] = None
    else:
        df_usa["rate_level"] = None
        df_usa["rate_diff"] = None

    df_usa["country"] = "US"

    # ========== PRZETWARZANIE PL ==========
    LOGGER.info("Transformacja danych makro Polski...")
    df_pl = pd.DataFrame()
    df_pl["period_date"] = df["date"].copy()

    # Retail Sales: YoY - tylko jeśli kolumna istnieje i ma jakieś dane
    if "retail_sales_index_pl" in df.columns and df["retail_sales_index_pl"].notna().any():
        df_pl["retail_sales_yoy"] = _compute_yoy(df["retail_sales_index_pl"])
    else:
        df_pl["retail_sales_yoy"] = None

    # GDP: YoY - tylko jeśli kolumna istnieje i ma jakieś dane
    # ZMIANA (2026-01-04): Używamy Real GDP (CLV) zamiast Nominal (CP)
    if "gdp_real_pl" in df.columns and df["gdp_real_pl"].notna().any():
        df_pl["gdp_yoy"] = _compute_yoy(df["gdp_real_pl"])
    else:
        df_pl["gdp_yoy"] = None

    # CPI: YoY (inflacja - dla PL już jest % YoY z Eurostat, więc kopiujemy)
    # UWAGA: inflation_pl to już HICP % YoY, NIE indeks!
    if "inflation_pl" in df.columns:
        df_pl["cpi_yoy"] = df["inflation_pl"].copy()
    else:
        df_pl["cpi_yoy"] = None

    # Bezrobocie: poziom + różnica M/M
    if "unemployment_rate_pl" in df.columns:
        df_pl["unemp_level"] = df["unemployment_rate_pl"].copy()
        if df["unemployment_rate_pl"].notna().any():
            df_pl["unemp_diff"] = _compute_diff(df["unemployment_rate_pl"])
        else:
            df_pl["unemp_diff"] = None
    else:
        df_pl["unemp_level"] = None
        df_pl["unemp_diff"] = None

    # Stopa %: poziom + różnica M/M
    if "interest_rate_pl" in df.columns:
        df_pl["rate_level"] = df["interest_rate_pl"].copy()
        if df["interest_rate_pl"].notna().any():
            df_pl["rate_diff"] = _compute_diff(df["interest_rate_pl"])
        else:
            df_pl["rate_diff"] = None
    else:
        df_pl["rate_level"] = None
        df_pl["rate_diff"] = None

    df_pl["country"] = "PL"

    # ========== ŁĄCZENIE USA + PL ==========
    df_combined = pd.concat([df_usa, df_pl], ignore_index=True)

    # Dodaj datę publikacji - pierwszy dzień następnego miesiąca
    # Np. okres 2023-12-01 → publikacja 2024-01-01 (lag ~1 miesiąc)
    df_combined["publication_date"] = (
        df_combined["period_date"] + pd.offsets.MonthBegin(1)
    )

    # Usuń wiersze gdzie wszystkie wskaźniki są NULL (okres przed danymi)
    indicator_cols = ["retail_sales_yoy", "gdp_yoy", "cpi_yoy", "unemp_level", "unemp_diff", "rate_level", "rate_diff"]
    df_combined = df_combined.dropna(subset=indicator_cols, how="all")

    LOGGER.info("Utworzono %d znormalizowanych rekordów (%d USA + %d PL)",
                len(df_combined), len(df_usa), len(df_pl))

    # Utwórz tabelę staging.macro_normalized jeśli nie istnieje
    _create_staging_macro_table(conn)

    # Zapisz do staging.macro_normalized
    _upsert_to_staging(conn, df_combined)

    LOGGER.info("Zapisano %d znormalizowanych rekordów do staging.macro_normalized", len(df_combined))
    return len(df_combined)


def _create_staging_macro_table(conn) -> None:
    """
    Tworzy tabelę staging.macro_normalized jeśli nie istnieje.

    NOWA STRUKTURA (2026-01-04):
    - Transformacje wskaźników zgodnie z wytycznymi ekonomicznymi
    - Dwie kolumny dat: period_date (okres pomiaru) i publication_date (data publikacji z lagiem)
    - Osobne wiersze dla USA i PL (kolumna country)
    """

    # Najpierw utwórz schemat staging jeśli nie istnieje
    conn.execute("CREATE SCHEMA IF NOT EXISTS staging")

    # NOWA STRUKTURA: Transformacje ekonomiczne + daty publikacji
    create_table_sql = """
        CREATE TABLE IF NOT EXISTS staging.macro_normalized (
            period_date DATE NOT NULL,
            publication_date DATE NOT NULL,
            country VARCHAR NOT NULL,
            -- Wzrost / Popyt
            retail_sales_yoy DOUBLE,           -- Retail Sales: zmiana % YoY
            -- Gospodarka
            gdp_yoy DOUBLE,                    -- GDP: zmiana % YoY
            -- Ceny
            cpi_yoy DOUBLE,                    -- CPI: zmiana % YoY (inflacja)
            -- Rynek Pracy
            unemp_level DOUBLE,                -- Bezrobocie: poziom %
            unemp_diff DOUBLE,                 -- Bezrobocie: różnica M/M
            -- Pieniądz
            rate_level DOUBLE,                 -- Stopa %: poziom
            rate_diff DOUBLE,                  -- Stopa %: różnica M/M
            PRIMARY KEY (period_date, country)
        )
    """

    conn.execute(create_table_sql)
    LOGGER.debug("Tabela staging.macro_normalized utworzona lub już istnieje (NOWA STRUKTURA)")


def _upsert_to_staging(conn, df: pd.DataFrame) -> None:
    """
    Wstawia lub aktualizuje dane w staging.macro_normalized (UPSERT).

    NOWA STRUKTURA: df ma kolumny period_date, publication_date, country + wskaźniki.
    """

    if df.empty:
        return

    # Upewnij się że mamy wszystkie wymagane kolumny (wypełnij NULL jeśli brakuje)
    required_cols = [
        "period_date",
        "publication_date",
        "country",
        "retail_sales_yoy",
        "gdp_yoy",
        "cpi_yoy",
        "unemp_level",
        "unemp_diff",
        "rate_level",
        "rate_diff"
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    # Wybierz tylko kolumny ze schematu
    df_to_insert = df[required_cols].copy()

    # DuckDB UPSERT: DELETE matching rows, then INSERT
    temp_table = "temp_macro_normalized"

    try:
        conn.execute(f"DROP TABLE IF EXISTS {temp_table}")
        conn.execute(f"CREATE TEMP TABLE {temp_table} AS SELECT * FROM staging.macro_normalized WHERE 1=0")

        # Wstaw dane do temp
        conn.register("temp_df", df_to_insert)
        conn.execute(f"INSERT INTO {temp_table} SELECT * FROM temp_df")

        # DELETE old records, INSERT new (po period_date i country)
        conn.execute(f"""
            DELETE FROM staging.macro_normalized
            WHERE EXISTS (
                SELECT 1 FROM {temp_table}
                WHERE staging.macro_normalized.period_date = {temp_table}.period_date
                  AND staging.macro_normalized.country = {temp_table}.country
            )
        """)

        conn.execute(f"INSERT INTO staging.macro_normalized SELECT * FROM {temp_table}")
        conn.execute(f"DROP TABLE {temp_table}")

    except Exception as exc:
        LOGGER.error("Błąd podczas UPSERT do staging.macro_normalized: %s", exc)
        raise


def get_normalized_macro(
    start_date: str | None = None,
    end_date: str | None = None,
    country: str | None = None
) -> pd.DataFrame:
    """
    Pobiera znormalizowane dane makro ze staging.macro_normalized.

    NOWA STRUKTURA (2026-01-04):
    - Kolumny: period_date, publication_date, country + wskaźniki ekonomiczne
    - USA i PL w osobnych wierszach (long format)

    Args:
        start_date: Opcjonalnie filtruj od daty okresu (format: 'YYYY-MM-DD')
        end_date: Opcjonalnie filtruj do daty okresu (format: 'YYYY-MM-DD')
        country: Opcjonalnie filtruj po kraju ('US' lub 'PL')

    Returns:
        DataFrame ze znormalizowanymi danymi makro
    """
    conn = get_connection()

    query = "SELECT * FROM staging.macro_normalized WHERE 1=1"
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

    if params:
        df = conn.execute(query, params).df()
    else:
        df = conn.execute(query).df()

    return df
