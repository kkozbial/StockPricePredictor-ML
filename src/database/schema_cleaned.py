"""
Moduł definiujący schemat 'cleaned' z wyczyszczonymi danymi.

Schema cleaned zawiera tabele po wstępnym czyszczeniu danych z main:
- Usunięte outliers
- Wypełnione braki
- Normalizacja wartości
- Gotowe do feature engineering i ML
"""
from __future__ import annotations

import logging
from pathlib import Path

import duckdb

from src.database.connection import get_connection

LOGGER = logging.getLogger("database.cleaned")


def create_cleaned_schema() -> None:
    """
    Tworzy schemat 'cleaned' jeśli nie istnieje.
    """
    conn = get_connection()
    conn.execute("CREATE SCHEMA IF NOT EXISTS cleaned")
    LOGGER.info("Schema 'cleaned' utworzony lub już istnieje")


def drop_cleaned_schema() -> None:
    """
    Usuwa cały schemat 'cleaned' wraz z wszystkimi tabelami.
    """
    conn = get_connection()
    conn.execute("DROP SCHEMA IF EXISTS cleaned CASCADE")
    LOGGER.info("Schema 'cleaned' usunięty")


def create_prices_cleaned() -> int:
    """
    Tworzy tabelę cleaned.prices_cleaned z wyczyszczonymi danymi cenowymi.

    Operacje czyszczenia:
    - Usunięcie rekordów z NULL w close
    - Usunięcie outliers (cena <= 0)
    - Sortowanie po ticker, date

    Returns:
        Liczba rekordów w tabeli cleaned.prices_cleaned
    """
    conn = get_connection()

    conn.execute("""
        CREATE OR REPLACE TABLE cleaned.prices_cleaned AS
        SELECT
            ticker,
            date,
            country,
            open,
            high,
            low,
            close,
            volume
        FROM prices
        WHERE close IS NOT NULL
          AND close > 0
        ORDER BY ticker, date
    """)

    count = conn.execute("SELECT COUNT(*) FROM cleaned.prices_cleaned").fetchone()[0]
    LOGGER.info("Utworzono cleaned.prices_cleaned: %d rekordów", count)
    return count


def create_financials_cleaned() -> int:
    """
    Tworzy tabelę cleaned.financials_cleaned z wyczyszczonymi danymi finansowymi.

    KLUCZOWE ZMIANY:
    - Dodanie kolumny period_type: 'SNAPSHOT' (Balance Sheet) vs 'FLOW' (Income/Cash Flow)
    - Dodanie kolumny period_length_months: liczba miesięcy okresu (1, 3, 6, 9, 12)
    - Dodanie kolumny is_cumulative: czy dane są kumulatywne od początku roku fiskalnego (TRUE/FALSE)
    - Dodanie kolumny fiscal_quarter: kwartał fiskalny (Q1, Q2, Q3, Q4) lub NULL dla snapshots
    - Uzupełnienie filing_date: najbliższa data po end_date lub end_date + 1 miesiąc z form='forecasted'

    UWAGA: Dane FLOW z SEC są KUMULATYWNE (YTD - Year To Date):
    - Q1: 3 miesiące (kumulacja od początku roku)
    - Q2: 6 miesięcy (kumulacja od początku roku)
    - Q3: 9 miesięcy (kumulacja od początku roku)
    - FY: 12 miesięcy (cały rok)

    Returns:
        Liczba rekordów w tabeli cleaned.financials_cleaned
    """
    conn = get_connection()

    conn.execute("""
        CREATE OR REPLACE TABLE cleaned.financials_cleaned AS
        WITH
        -- Krok 1: Znajdź najbliższą filing_date dla każdej grupy (ticker, end_date)
        ranked_filings AS (
            SELECT
                ticker,
                end_date,
                filing_date,
                form,
                ROW_NUMBER() OVER (
                    PARTITION BY ticker, end_date
                    ORDER BY filing_date NULLS LAST
                ) AS rn
            FROM financials
            WHERE end_date IS NOT NULL
              AND filing_date IS NOT NULL
              AND filing_date > end_date
        ),
        filing_dates_per_group AS (
            SELECT
                ticker,
                end_date,
                filing_date AS nearest_filing_date,
                form AS nearest_form
            FROM ranked_filings
            WHERE rn = 1
        ),
        -- Krok 2: Dołącz filling_date do wszystkich rekordów
        base_data AS (
            SELECT
                f.ticker,
                f.concept,
                f.country,
                f.unit,
                f.start_date,
                f.end_date,
                f.value,
                -- Uzupełnij form: z oryginalnego wiersza, lub z grupy, lub 'forecasted'
                COALESCE(
                    f.form,
                    fd.nearest_form,
                    'forecasted'
                ) AS form,
                -- Uzupełnij filing_date: z oryginalnego, lub najbliższa z grupy, lub end_date + 1 miesiąc
                COALESCE(
                    f.filing_date,
                    fd.nearest_filing_date,
                    f.end_date + INTERVAL 1 MONTH
                ) AS filing_date
            FROM financials f
            LEFT JOIN filing_dates_per_group fd
                ON f.ticker = fd.ticker
                AND f.end_date = fd.end_date
            WHERE f.value IS NOT NULL
              AND f.end_date IS NOT NULL
        ),
        -- Krok 3: Oblicz value_quarterly (dane dyskretne/kwartalne) z danych kumulatywnych
        base_with_lags AS (
            SELECT
                *,
                -- Pobierz poprzednie wartości kumulatywne dla tego samego concept i roku fiskalnego
                -- KLUCZOWE: Rok fiskalny = EXTRACT(YEAR FROM start_date) dla FLOW
                -- Dzięki temu Apple FY2023 (start=2022-10-01, end=2023-09-30) będzie w jednej grupie
                LAG(value, 1) OVER (
                    PARTITION BY ticker, concept, EXTRACT(YEAR FROM start_date)
                    ORDER BY end_date
                ) AS prev_cumulative_value,
                LAG(CAST(ROUND(DATEDIFF('day', start_date, end_date) / 30.44) AS INTEGER), 1) OVER (
                    PARTITION BY ticker, concept, EXTRACT(YEAR FROM start_date)
                    ORDER BY end_date
                ) AS prev_period_length,
                -- Policz ile raportów jest w tym samym roku fiskalnym
                COUNT(*) OVER (
                    PARTITION BY ticker, concept, EXTRACT(YEAR FROM start_date)
                ) AS reports_in_fiscal_year,
                -- Numer raportu w sekwencji (1, 2, 3, 4 dla Q1-Q4)
                ROW_NUMBER() OVER (
                    PARTITION BY ticker, concept, EXTRACT(YEAR FROM start_date)
                    ORDER BY end_date
                ) AS report_sequence_number,
                -- Czy to OSTATNI raport 12-miesięczny w roku fiskalnym?
                -- (dla przypadków gdy jest kilka raportów 12m - bierzemy ostatni jako Q4)
                CASE
                    WHEN CAST(ROUND(DATEDIFF('day', start_date, end_date) / 30.44) AS INTEGER) >= 11
                    THEN ROW_NUMBER() OVER (
                        PARTITION BY ticker, concept, EXTRACT(YEAR FROM start_date),
                                     CASE WHEN CAST(ROUND(DATEDIFF('day', start_date, end_date) / 30.44) AS INTEGER) >= 11 THEN 1 ELSE 0 END
                        ORDER BY end_date DESC
                    )
                    ELSE NULL
                END AS rank_among_12m_reports
            FROM base_data
        )
        SELECT
            ticker,
            concept,
            country,
            unit,
            start_date,
            end_date,
            value,
            form,
            filing_date,

            -- PERIOD_TYPE: SNAPSHOT vs FLOW
            CASE
                WHEN start_date IS NULL THEN 'SNAPSHOT'  -- Balance Sheet (stan na dzień)
                ELSE 'FLOW'  -- Income Statement, Cash Flow (przepływ w okresie)
            END AS period_type,

            -- PERIOD_LENGTH_MONTHS: długość okresu w miesiącach
            CASE
                WHEN start_date IS NULL THEN NULL  -- Snapshot nie ma długości
                ELSE CAST(ROUND(DATEDIFF('day', start_date, end_date) / 30.44) AS INTEGER)  -- ~30.44 dni/miesiąc
            END AS period_length_months,

            -- IS_CUMULATIVE: czy dane są kumulatywne (YTD)
            -- Dla FLOW: TRUE (SEC raportuje kumulatywnie od początku roku fiskalnego)
            -- Dla SNAPSHOT: FALSE (stan na dzień)
            CASE
                WHEN start_date IS NULL THEN FALSE  -- SNAPSHOT
                ELSE TRUE  -- FLOW
            END AS is_cumulative,

            -- FISCAL_QUARTER: określenie kwartału na podstawie period_length_months i end_date
            -- NOWA LOGIKA (v6 - based on actual period characteristics):
            -- 1. Raporty 3-miesięczne → kwartał określony przez miesiąc end_date
            -- 2. Raporty 6-miesięczne → '2' (półroczne = Q2)
            -- 3. Raporty 9-miesięczne → '3' (3 kwartały = Q3)
            -- 4. Raporty 12-miesięczne → 'FY' jeśli jedyny w roku, '4' w przeciwnym razie
            -- Format: 1, 2, 3, 4, FY (bez prefiksu 'Q')
            CASE
                WHEN start_date IS NULL THEN NULL

                -- Raporty 3-miesięczne: kwartał na podstawie miesiąca end_date
                WHEN CAST(ROUND(DATEDIFF('day', start_date, end_date) / 30.44) AS INTEGER) IN (2, 3, 4) THEN
                    CASE
                        WHEN EXTRACT(MONTH FROM end_date) IN (1, 2, 3) THEN '1'
                        WHEN EXTRACT(MONTH FROM end_date) IN (4, 5, 6) THEN '2'
                        WHEN EXTRACT(MONTH FROM end_date) IN (7, 8, 9) THEN '3'
                        WHEN EXTRACT(MONTH FROM end_date) IN (10, 11, 12) THEN '4'
                    END

                -- Raporty 5-7 miesięczne (półroczne)
                WHEN CAST(ROUND(DATEDIFF('day', start_date, end_date) / 30.44) AS INTEGER) IN (5, 6, 7) THEN '2'

                -- Raporty 8-10 miesięczne (9-miesięczne)
                WHEN CAST(ROUND(DATEDIFF('day', start_date, end_date) / 30.44) AS INTEGER) IN (8, 9, 10) THEN '3'

                -- Raporty 11+ miesięczne (roczne)
                WHEN CAST(ROUND(DATEDIFF('day', start_date, end_date) / 30.44) AS INTEGER) >= 11 THEN
                    CASE
                        WHEN reports_in_fiscal_year = 1 THEN 'FY'  -- Jedyny raport w roku → FY
                        ELSE '4'  -- Nie jedyny → Q4
                    END

                -- Inne (bardzo rzadkie)
                ELSE 'OTHER'
            END AS fiscal_quarter,

            -- VALUE_QUARTERLY: Przekształcenie z kumulatywnego na dyskretne (kwartalne)
            -- Dla FLOW (kumulatywne): odejmij poprzedni okres tego samego roku
            -- Dla SNAPSHOT: value (stan bilansowy - kopiuj value)
            -- Q1 (3m): value (bez zmian, to już czysty Q1)
            -- Q2 (6m): value - value_Q1 (czysty Q2)
            -- Q3 (9m): value - value_Q2 (czysty Q3)
            -- Q4/FY (12m): value - value_Q3 (czysty Q4) lub value (jeśli FY = tylko roczne)
            CASE
                WHEN start_date IS NULL THEN value  -- SNAPSHOT - kopiuj value
                WHEN prev_cumulative_value IS NULL THEN value  -- Pierwszy okres roku (Q1 lub FY) - bez zmian
                ELSE value - prev_cumulative_value  -- Odejmij poprzedni kumulatywny
            END AS value_quarterly

        FROM base_with_lags
        ORDER BY ticker, end_date, concept
    """)

    count = conn.execute("SELECT COUNT(*) FROM cleaned.financials_cleaned").fetchone()[0]

    # Statystyki okresów
    stats = conn.execute("""
        SELECT
            period_type,
            fiscal_quarter,
            COUNT(*) as count
        FROM cleaned.financials_cleaned
        GROUP BY period_type, fiscal_quarter
        ORDER BY period_type, fiscal_quarter
    """).fetchdf()

    LOGGER.info("Utworzono cleaned.financials_cleaned: %d rekordów", count)
    LOGGER.info("Rozkład typów okresów:\n%s", stats.to_string())

    return count


def create_financials_wide() -> int:
    """
    Tworzy tabelę cleaned.financials_wide - format szeroki (pivot) z danymi finansowymi.

    Struktura:
    - Klucz: (ticker, end_date, fiscal_quarter, country)
    - Każdy concept staje się kolumną
    - Wartości z value_quarterly
    - Nazwy kolumn: concept (np. Revenue, Assets, NetIncomeLoss - BEZ suffiksu waluty)
    - Dane USD i PLN są w osobnych wierszach (country = 'US' lub 'PL')

    WALIDACJA: Sprawdza czy dla każdego (ticker, end_date) wszystkie rekordy
    mają ten sam fiscal_quarter (nie można mieszać Q1 z Q2 na tej samej dacie).

    Returns:
        Liczba rekordów w tabeli cleaned.financials_wide
    """
    conn = get_connection()

    # Krok 1: WALIDACJA - sprawdź czy są mieszane fiscal_quarters dla tego samego (ticker, end_date)
    LOGGER.info("Walidacja: sprawdzanie czy fiscal_quarter jest spójny dla każdego (ticker, end_date)...")

    validation = conn.execute("""
        SELECT
            ticker,
            end_date,
            COUNT(DISTINCT fiscal_quarter) as distinct_quarters,
            STRING_AGG(DISTINCT fiscal_quarter::VARCHAR, ', ') as quarters
        FROM cleaned.financials_cleaned
        WHERE fiscal_quarter IS NOT NULL
        GROUP BY ticker, end_date
        HAVING COUNT(DISTINCT fiscal_quarter) > 1
        LIMIT 10
    """).fetchdf()

    if not validation.empty:
        LOGGER.warning("UWAGA: Znaleziono %d przypadków mieszanych fiscal_quarter dla tej samej daty!", len(validation))
        LOGGER.warning("Przykłady:\n%s", validation.to_string())
        LOGGER.warning("Będą one pomijane w tabeli wide (zachowamy tylko pierwszy quarter)")

    # Krok 2: Pobierz listę unikalnych concepts (bez unit)
    concepts = conn.execute("""
        SELECT DISTINCT concept
        FROM cleaned.financials_cleaned
        WHERE concept IS NOT NULL
        ORDER BY concept
    """).fetchdf()

    LOGGER.info("Znaleziono %d unikalnych concepts", len(concepts))

    # Krok 3: Przygotuj listę agregacji (jeden MAX dla każdego concept)
    # Kolumny: Assets, Revenue, NetIncomeLoss itd. (BEZ _USD, _PLN)
    aggregations = []
    for concept in concepts['concept']:
        # Escapuj wartości dla SQL
        concept_escaped = concept.replace("'", "''")

        agg = f"""
            MAX(CASE
                WHEN concept = '{concept_escaped}'
                THEN value_quarterly
                ELSE NULL
            END) AS "{concept}"
        """
        aggregations.append(agg)

    aggregations_sql = ",\n            ".join(aggregations)

    # Krok 5: Utwórz zapytanie SQL
    # NOWA STRATEGIA:
    # 1. FLOW z SNAPSHOT na tej samej dacie → łączymy bezpośrednio
    # 2. SNAPSHOT z datą różną od FLOW → przypisz do najbliższego późniejszego FLOW (max 365 dni)
    # 3. SNAPSHOT >365 dni od najbliższego FLOW → USUŃ
    # 4. SNAPSHOT bez żadnego FLOW dla tickera → USUŃ (tickery z tylko bilansami, bez przychodów/zysków)
    query = f"""
        CREATE OR REPLACE TABLE cleaned.financials_wide AS
        WITH
        -- Krok 1: Znajdź fiscal_quarter dla każdego (ticker, end_date) z danych FLOW
        fiscal_quarters_flow AS (
            SELECT
                ticker,
                end_date,
                country,
                FIRST(fiscal_quarter ORDER BY fiscal_quarter NULLS LAST) as fiscal_quarter
            FROM cleaned.financials_cleaned
            WHERE fiscal_quarter IS NOT NULL  -- Tylko FLOW (mają fiscal_quarter)
            GROUP BY ticker, end_date, country
        ),
        -- Krok 2: Dla SNAPSHOT - znajdź najbliższy późniejszy FLOW
        snapshot_to_flow_mapping AS (
            SELECT DISTINCT
                s.ticker,
                s.end_date as snapshot_end_date,
                s.country,
                -- Znajdź najbliższy FLOW po dacie SNAPSHOT (dla tego samego tickera i kraju)
                (SELECT f.end_date
                 FROM fiscal_quarters_flow f
                 WHERE f.ticker = s.ticker
                 AND f.country = s.country
                 AND f.end_date >= s.end_date
                 ORDER BY f.end_date ASC
                 LIMIT 1
                ) as matched_flow_end_date,
                -- Oblicz odległość w dniach
                DATEDIFF('day', s.end_date,
                    (SELECT f.end_date
                     FROM fiscal_quarters_flow f
                     WHERE f.ticker = s.ticker
                     AND f.country = s.country
                     AND f.end_date >= s.end_date
                     ORDER BY f.end_date ASC
                     LIMIT 1)
                ) as days_to_flow
            FROM cleaned.financials_cleaned s
            WHERE s.fiscal_quarter IS NULL  -- Tylko SNAPSHOT
        ),
        -- Krok 3: Filtruj SNAPSHOT - zachowaj tylko te w odległości <=365 dni od FLOW
        valid_snapshot_mappings AS (
            SELECT
                ticker,
                snapshot_end_date,
                country,
                matched_flow_end_date,
                days_to_flow
            FROM snapshot_to_flow_mapping
            WHERE matched_flow_end_date IS NOT NULL  -- Ma odpowiadający FLOW
            AND days_to_flow <= 365  -- Max 1 rok odległości
        ),
        -- Krok 4: Utwórz końcowe mapowanie end_date -> fiscal_quarter
        -- Przypadek A: SNAPSHOT z FLOW na tej samej dacie
        flow_snapshot_same_date AS (
            SELECT
                f.ticker,
                f.end_date,
                f.country,
                f.fiscal_quarter
            FROM fiscal_quarters_flow f
        ),
        -- Przypadek B: SNAPSHOT przypisany do najbliższego FLOW (zmiana daty!)
        snapshot_reassigned AS (
            SELECT
                v.ticker,
                v.matched_flow_end_date as end_date,  -- ZMIENIONA DATA!
                v.country,
                f.fiscal_quarter
            FROM valid_snapshot_mappings v
            INNER JOIN fiscal_quarters_flow f
                ON v.ticker = f.ticker
                AND v.matched_flow_end_date = f.end_date
                AND v.country = f.country
            WHERE v.snapshot_end_date != v.matched_flow_end_date  -- Tylko różne daty
        ),
        -- Przypadek C: SNAPSHOT bez FLOW dla tickera - USUŃ (nie dodawaj do all_quarters)
        -- Te tickery mają TYLKO bilanse, bez danych FLOW - są niepełne i nieprzydatne
        -- Krok 5: Połącz przypadki A i B (BEZ snapshot_orphans)
        all_quarters AS (
            SELECT * FROM flow_snapshot_same_date
            UNION ALL
            SELECT * FROM snapshot_reassigned
            -- snapshot_orphans są POMIJANE - usuwamy tickery z tylko SNAPSHOT
        ),
        -- Krok 6: Przypisz fiscal_quarter do WSZYSTKICH danych
        -- WAŻNE: SNAPSHOT z datą różną od FLOW będzie przypisany do daty FLOW (zmiana end_date!)
        cleaned_with_quarter AS (
            SELECT
                fc.ticker,
                fc.concept,
                fc.country,
                fc.unit,
                fc.value_quarterly,
                fc.filing_date,
                -- Dla SNAPSHOT z inną datą, użyj daty z mapowania (matched_flow_end_date)
                COALESCE(
                    (SELECT matched_flow_end_date
                     FROM valid_snapshot_mappings v
                     WHERE v.ticker = fc.ticker
                     AND v.snapshot_end_date = fc.end_date
                     AND v.country = fc.country
                     AND v.snapshot_end_date != v.matched_flow_end_date
                    ),
                    fc.end_date  -- Dla reszty - bez zmian
                ) as end_date,
                aq.fiscal_quarter
            FROM cleaned.financials_cleaned fc
            INNER JOIN all_quarters aq
                ON fc.ticker = aq.ticker
                AND fc.country = aq.country
                AND (
                    -- Przypadek 1: Ta sama data (FLOW + SNAPSHOT na tej samej dacie)
                    fc.end_date = aq.end_date
                    OR
                    -- Przypadek 2: SNAPSHOT z inną datą → dopasuj po zmienionej dacie
                    (fc.fiscal_quarter IS NULL AND EXISTS (
                        SELECT 1 FROM valid_snapshot_mappings v
                        WHERE v.ticker = fc.ticker
                        AND v.snapshot_end_date = fc.end_date
                        AND v.matched_flow_end_date = aq.end_date
                        AND v.country = fc.country
                    ))
                )
        )
        SELECT
            ticker,
            end_date,
            fiscal_quarter,
            country,
            MAX(filing_date) as filing_date,
            {aggregations_sql}
        FROM cleaned_with_quarter
        GROUP BY ticker, end_date, fiscal_quarter, country
        ORDER BY ticker, end_date
    """

    LOGGER.info("Tworzenie tabeli financials_wide...")
    conn.execute(query)

    count = conn.execute("SELECT COUNT(*) FROM cleaned.financials_wide").fetchone()[0]
    LOGGER.info("Utworzono cleaned.financials_wide: %d rekordów", count)

    # Statystyki
    stats = conn.execute("""
        SELECT
            fiscal_quarter,
            COUNT(*) as count
        FROM cleaned.financials_wide
        GROUP BY fiscal_quarter
        ORDER BY fiscal_quarter
    """).fetchdf()

    LOGGER.info("Rozkład fiscal_quarter w tabeli wide:\n%s", stats.to_string())

    return count


def create_macro_normalized() -> int:
    """
    Kopiuje tabelę staging.macro_normalized do cleaned.macro_normalized.

    Tabela staging.macro_normalized jest tworzona przez normalize_macro.py i zawiera:
    - Transformacje ekonomiczne (YoY dla GDP, Retail Sales, CPI)
    - Poziomy i różnice M/M dla bezrobocia i stóp procentowych
    - Daty: period_date (okres pomiaru) i publication_date (data publikacji z lagiem)

    Returns:
        Liczba rekordów w tabeli cleaned.macro_normalized
    """
    conn = get_connection()

    # Sprawdź czy staging.macro_normalized istnieje
    staging_exists = conn.execute("""
        SELECT COUNT(*) FROM information_schema.tables
        WHERE table_schema = 'staging' AND table_name = 'macro_normalized'
    """).fetchone()[0] > 0

    if not staging_exists:
        LOGGER.warning("Tabela staging.macro_normalized nie istnieje - najpierw uruchom preprocessing")
        return 0

    # Kopiuj z staging do cleaned (już przetworzone dane)
    conn.execute("""
        CREATE OR REPLACE TABLE cleaned.macro_normalized AS
        SELECT * FROM staging.macro_normalized
        ORDER BY period_date, country
    """)

    count = conn.execute("SELECT COUNT(*) FROM cleaned.macro_normalized").fetchone()[0]
    LOGGER.info("Skopiowano cleaned.macro_normalized ze staging: %d rekordów", count)
    return count


def create_dividends_cleaned() -> int:
    """
    Tworzy tabelę cleaned.dividends_cleaned z wyczyszczonymi dywidendami.

    Operacje:
    - Usunięcie rekordów z NULL lub <= 0 w dividend_amount

    Returns:
        Liczba rekordów
    """
    conn = get_connection()

    conn.execute("""
        CREATE OR REPLACE TABLE cleaned.dividends_cleaned AS
        SELECT
            ticker,
            dividend_date,
            country,
            dividend_amount
        FROM dividends
        WHERE dividend_amount IS NOT NULL
          AND dividend_amount > 0
        ORDER BY ticker, dividend_date
    """)

    count = conn.execute("SELECT COUNT(*) FROM cleaned.dividends_cleaned").fetchone()[0]
    LOGGER.info("Utworzono cleaned.dividends_cleaned: %d rekordów", count)
    return count


def create_sectors_cleaned() -> int:
    """
    Tworzy tabelę cleaned.sectors_cleaned.

    Operacje:
    - Usunięcie rekordów z NULL w sector

    Returns:
        Liczba rekordów
    """
    conn = get_connection()

    conn.execute("""
        CREATE OR REPLACE TABLE cleaned.sectors_cleaned AS
        SELECT
            ticker,
            country,
            sector,
            industry,
            last_updated
        FROM sectors
        WHERE sector IS NOT NULL
        ORDER BY ticker, country
    """)

    count = conn.execute("SELECT COUNT(*) FROM cleaned.sectors_cleaned").fetchone()[0]
    LOGGER.info("Utworzono cleaned.sectors_cleaned: %d rekordów", count)
    return count


def create_shares_cleaned() -> int:
    """
    Tworzy tabelę cleaned.shares_cleaned z wyczyszczonymi danymi o akcjach.

    Operacje:
    - Usunięcie rekordów z NULL lub <= 0 w shares
    - Dodanie end_date jako ostatni dzień kwartału (zgodnie z formatem financials_wide)
    - Forward fill NULL shares w ramach każdego tickera
    - Deduplikacja (już jest w main, ale dla pewności)

    Returns:
        Liczba rekordów
    """
    conn = get_connection()

    conn.execute("""
        CREATE OR REPLACE TABLE cleaned.shares_cleaned AS
        WITH base_data AS (
            SELECT
                ticker,
                filing_date,
                country,
                shares,
                end_date AS original_end_date,
                fiscal_year,
                fiscal_period,
                source_type,
                -- Oblicz end_date jako ostatni dzień kwartału
                -- Wykorzystujemy fiscal_year i fiscal_period do wyznaczenia daty
                CASE
                    WHEN fiscal_period = 'Q1' THEN MAKE_DATE(fiscal_year, 3, 31)
                    WHEN fiscal_period = 'Q2' THEN MAKE_DATE(fiscal_year, 6, 30)
                    WHEN fiscal_period = 'Q3' THEN MAKE_DATE(fiscal_year, 9, 30)
                    WHEN fiscal_period IN ('Q4', 'FY') THEN MAKE_DATE(fiscal_year, 12, 31)
                    ELSE end_date  -- Fallback dla innych przypadków
                END AS end_date
            FROM shares_outstanding
            WHERE shares IS NOT NULL
              AND shares > 0
        ),
        -- Forward fill: dla każdego tickera, wypełnij NULL shares wartością z poprzedniego okresu
        filled_data AS (
            SELECT
                ticker,
                filing_date,
                country,
                -- Forward fill shares: jeśli NULL, weź ostatnią znaną wartość dla tego tickera
                COALESCE(
                    shares,
                    LAST_VALUE(shares IGNORE NULLS) OVER (
                        PARTITION BY ticker, country
                        ORDER BY filing_date
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    )
                ) AS shares,
                end_date,
                fiscal_year,
                fiscal_period,
                source_type
            FROM base_data
        )
        SELECT
            ticker,
            filing_date,
            country,
            shares,
            end_date,
            fiscal_year,
            fiscal_period,
            source_type
        FROM filled_data
        WHERE shares IS NOT NULL  -- Usuń rekordy gdzie forward fill się nie powiódł (brak poprzedniej wartości)
        ORDER BY ticker, filing_date
    """)

    count = conn.execute("SELECT COUNT(*) FROM cleaned.shares_cleaned").fetchone()[0]
    LOGGER.info("Utworzono cleaned.shares_cleaned: %d rekordów", count)
    return count


def build_all_cleaned_tables() -> dict[str, int]:
    """
    Tworzy wszystkie tabele w schemacie cleaned.

    UWAGA: Wymaga aby staging.macro_normalized już istniał (utworzony przez normalize_macro.py)

    Returns:
        Słownik z liczbą rekordów dla każdej tabeli
    """
    LOGGER.info("=== TWORZENIE SCHEMA CLEANED ===")

    # Utwórz schemat cleaned (NIE usuwamy staging - jest używany przez preprocessing!)
    create_cleaned_schema()

    results = {}

    # Twórz wszystkie tabele
    results["prices_cleaned"] = create_prices_cleaned()
    results["financials_cleaned"] = create_financials_cleaned()
    results["financials_wide"] = create_financials_wide()  # Format szeroki (pivot)
    results["macro_normalized"] = create_macro_normalized()  # Kopiuje ze staging
    results["dividends_cleaned"] = create_dividends_cleaned()
    results["sectors_cleaned"] = create_sectors_cleaned()
    results["shares_cleaned"] = create_shares_cleaned()

    LOGGER.info("=== CLEANED SCHEMA GOTOWY ===")
    for table, count in results.items():
        LOGGER.info("  %s: %d rekordów", table, count)

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Inicjalizuj połączenie (get_connection jest już zaimportowane na górze pliku)
    conn = get_connection(Path("data/stock_market.db"))

    # Zbuduj wszystkie tabele cleaned
    build_all_cleaned_tables()
