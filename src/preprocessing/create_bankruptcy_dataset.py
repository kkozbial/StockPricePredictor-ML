"""
Tworzy tabelę staging.bankruptcy_dataset do przewidywania bankructwa.

Różnice vs staging.master_dataset:
1. Dane tylko do delisting_date dla spółek delisted/deregistered
2. Dodatkowa kolumna: target_bankruptcy (0/1)
3. Filtruje dane po delisting_date aby uniknąć look-ahead bias
"""
from __future__ import annotations

import logging
from pathlib import Path

import duckdb

from ..utils.config_loader import load_config

LOGGER = logging.getLogger("preprocessing.bankruptcy_dataset")


def create_bankruptcy_dataset(db_path: Path | None = None) -> int:
    """
    Tworzy tabelę staging.bankruptcy_dataset do przewidywania bankructwa.

    Args:
        db_path: Ścieżka do bazy DuckDB. Jeśli None, używa ścieżki z konfiguracji.

    Returns:
        Liczba rekordów w tabeli staging.bankruptcy_dataset

    Strategia:
        1. Bazuje na staging.master_dataset
        2. Dołącza company_status (status, delisting_date)
        3. Filtruje date <= delisting_date dla delisted/deregistered
        4. Dodaje target_bankruptcy:
           - 1: DELISTED lub DEREGISTERED
           - 0: ACTIVE
    """
    if db_path is None:
        cfg = load_config()
        db_path = Path(cfg["database"]["path"])

    LOGGER.info("Connecting to database: %s", db_path)
    conn = duckdb.connect(str(db_path))

    try:
        # Krok 1: Utwórz schemat staging jeśli nie istnieje
        conn.execute("CREATE SCHEMA IF NOT EXISTS staging")
        LOGGER.info("Schema 'staging' ready")

        # Optymalizacja pamięci
        conn.execute("SET preserve_insertion_order=false")
        conn.execute("SET memory_limit='4GB'")
        LOGGER.info("Memory settings configured")

        # Krok 2: Sprawdź czy staging.master_dataset istnieje
        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'staging'"
        ).fetchall()
        if ("master_dataset",) not in tables:
            LOGGER.error("staging.master_dataset does not exist! Run create_master_dataset() first.")
            return 0

        # Sprawdź czy company_status istnieje
        tables_main = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
        if ("company_status",) not in tables_main:
            LOGGER.warning("company_status does not exist - creating bankruptcy_dataset without status filtering")
            has_status = False
        else:
            has_status = True

        LOGGER.info("Creating staging.bankruptcy_dataset...")

        if has_status:
            # Z company_status - filtrujemy po delisting_date
            # UWAGA: Target = 1 tylko dla spółek z oznakami FINANSOWEGO STRESU przed delistingiem
            # (nie wszystkich delisted - wiele to fuzje, przejęcia, privatyzacje)
            query = """
            CREATE OR REPLACE TABLE staging.bankruptcy_dataset AS
            WITH
            -- Krok 1: Dołącz status spółki
            with_status AS (
                SELECT
                    md.*,
                    cs.status,
                    cs.delisting_date
                FROM staging.master_dataset md
                LEFT JOIN main.company_status cs
                    ON md.ticker = cs.ticker
            ),
            -- Krok 2: Filtruj date <= delisting_date dla delisted/deregistered
            filtered AS (
                SELECT
                    *
                FROM with_status
                WHERE
                    -- Dla ACTIVE: wszystkie daty
                    (status = 'ACTIVE' OR status IS NULL)
                    OR
                    -- Dla DELISTED/DEREGISTERED: tylko date <= delisting_date
                    (status IN ('DELISTED', 'DEREGISTERED') AND date <= delisting_date)
            ),
            -- Krok 2b: Oblicz wskaźniki stresu finansowego na poziomie spółki
            -- Spółka jest uznana za bankruta jeśli przed delistingiem miała oznaki stresu
            company_distress AS (
                SELECT
                    ticker,
                    -- Wskaźniki stresu w ostatnim roku przed delistingiem
                    MIN(CASE WHEN StockholdersEquity < 0 THEN 1 ELSE 0 END) AS ever_negative_equity,
                    AVG(CASE WHEN StockholdersEquity < 0 THEN 1.0 ELSE 0.0 END) AS pct_negative_equity,
                    AVG(CASE WHEN NetIncomeLoss < 0 THEN 1.0 ELSE 0.0 END) AS pct_negative_income,
                    AVG(CASE WHEN OperatingCashFlow < 0 THEN 1.0 ELSE 0.0 END) AS pct_negative_ocf,
                    AVG(COALESCE(debt_ratio, 0)) AS avg_debt_ratio,
                    AVG(COALESCE(current_ratio, 999)) AS avg_current_ratio,
                    -- Altman Z-score proxy (uproszczony)
                    AVG(CASE
                        WHEN Assets > 0 AND Liabilities > 0 THEN
                            1.2 * COALESCE((AssetsCurrent - LiabilitiesCurrent) / NULLIF(Assets, 0), 0) +
                            1.4 * COALESCE(RetainedEarnings / NULLIF(Assets, 0), 0) +
                            3.3 * COALESCE(OperatingIncomeLoss / NULLIF(Assets, 0), 0) +
                            1.0 * COALESCE(Revenues / NULLIF(Assets, 0), 0)
                        ELSE NULL
                    END) AS avg_altman_partial
                FROM filtered
                WHERE status IN ('DELISTED', 'DEREGISTERED')
                GROUP BY ticker
            ),
            -- Krok 2c: Oznacz spółki jako bankruci na podstawie wskaźników stresu
            -- Kryteria: spółka DELISTED + co najmniej 2 z poniższych:
            --   1. Ujemny kapitał własny kiedykolwiek
            --   2. >50% kwartałów z ujemnym zyskiem netto
            --   3. >50% kwartałów z ujemnym cash flow operacyjnym
            --   4. Średni debt_ratio > 0.8 (wysoki dług)
            --   5. Średni current_ratio < 1.0 (niska płynność)
            --   6. Altman Z-score < 1.8 (strefa zagrożenia)
            distressed_tickers AS (
                SELECT
                    ticker,
                    (CASE WHEN ever_negative_equity = 1 THEN 1 ELSE 0 END +
                     CASE WHEN pct_negative_income > 0.5 THEN 1 ELSE 0 END +
                     CASE WHEN pct_negative_ocf > 0.5 THEN 1 ELSE 0 END +
                     CASE WHEN avg_debt_ratio > 0.8 THEN 1 ELSE 0 END +
                     CASE WHEN avg_current_ratio < 1.0 THEN 1 ELSE 0 END +
                     CASE WHEN avg_altman_partial < 1.8 THEN 1 ELSE 0 END
                    ) AS distress_signals
                FROM company_distress
            ),
            -- Krok 2d: Przypisz target - bankructwo tylko dla spółek ze stresem finansowym
            with_target AS (
                SELECT
                    f.*,
                    -- Target: 1 = prawdopodobne bankructwo (DELISTED + stres finansowy)
                    -- 0 = aktywna LUB delisted bez oznak stresu (fuzja/przejęcie/privatyzacja)
                    CASE
                        WHEN f.status IN ('DELISTED', 'DEREGISTERED')
                             AND dt.distress_signals >= 2 THEN 1
                        ELSE 0
                    END AS target_bankruptcy
                FROM filtered f
                LEFT JOIN distressed_tickers dt ON f.ticker = dt.ticker
            ),
            -- Krok 3: Deduplikacja po (ticker, date) - zachowaj najnowszy end_date
            deduplicated AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY ticker, date
                        ORDER BY end_date DESC, filing_date DESC NULLS LAST
                    ) AS rn
                FROM with_target
            )
            SELECT
                -- Podstawowe kolumny
                ticker,
                date,
                end_date,
                filing_date,
                fiscal_quarter,
                country,
                status,
                delisting_date,
                target_bankruptcy,
                -- Wszystkie pozostałe kolumny z master_dataset
                Assets,
                AssetsCurrent,
                AssetsNoncurrent,
                CapitalExpenditures,
                DiscontinuedOperationsNetIncome,
                DividendIncome,
                EBITDA,
                EarningsPerShareDiluted,
                EquityInEarningsOfSubsidiaries,
                FeeExpense,
                FeeIncome,
                GeneralAndAdministrativeExpense,
                GrossProfit,
                IncomeLossFromContinuingOperationsBeforeIncomeTaxes,
                InterestExpense,
                InterestIncome,
                Inventory,
                Liabilities,
                LiabilitiesCurrent,
                LiabilitiesNoncurrent,
                NetFeeIncome,
                NetIncomeLoss,
                NetInterestIncome,
                OperatingCashFlow,
                OperatingIncomeLoss,
                OtherFinancialInstrumentsResult,
                OtherOperatingExpense,
                OtherOperatingIncome,
                ProvisionForLoanLosses,
                Receivables,
                ResearchAndDevelopment,
                RetainedEarnings,
                Revenues,
                StockholdersEquity,
                TradingAndRevaluationResult,
                high,
                low,
                close,
                sector,
                industry,
                shares_outstanding,
                dividend_amount,
                retail_sales_yoy,
                gdp_yoy,
                cpi_yoy,
                unemp_level,
                unemp_diff,
                rate_level,
                rate_diff,
                -- Engineered features (per-share)
                revenue_per_share,
                operating_income_per_share,
                assets_per_share,
                rnd_per_share,
                sga_per_share,
                net_income_per_share,
                -- Engineered features (profitability)
                roe,
                roa,
                ros,
                gross_margin,
                operating_margin,
                -- Engineered features (leverage)
                debt_ratio,
                debt_to_equity,
                -- Engineered features (cash flow)
                free_cash_flow,
                ocf_ratio,
                capex_ratio,
                -- Engineered features (liquidity)
                current_ratio,
                quick_ratio,
                -- Engineered features (valuation)
                price_to_book,
                price_to_sales,
                price_to_earnings,
                pe_ratio,
                -- Engineered features (dividends)
                dividend_yield,
                payout_ratio,
                -- Engineered features (efficiency)
                asset_turnover,
                inventory_turnover,
                -- TTM metrics
                revenue_ttm,
                net_income_ttm,
                ocf_ttm,
                capex_ttm,
                fcf_ttm,
                -- Growth rates
                revenue_growth,
                net_income_growth
            FROM deduplicated
            WHERE rn = 1  -- Tylko pierwszy rekord (najnowszy end_date)
            ORDER BY ticker, date
            """
        else:
            # Bez company_status - kopiuj master_dataset bez filtrowania
            query = """
            CREATE OR REPLACE TABLE staging.bankruptcy_dataset AS
            SELECT
                *,
                NULL AS status,
                NULL AS delisting_date,
                0 AS target_bankruptcy  -- Domyślnie wszystkie jako ACTIVE
            FROM staging.master_dataset
            ORDER BY ticker, date
            """

        conn.execute(query)

        # Sprawdź wynik
        count = conn.execute("SELECT COUNT(*) FROM staging.bankruptcy_dataset").fetchone()[0]
        LOGGER.info("Created staging.bankruptcy_dataset: %d rows", count)

        # Statystyki
        stats = conn.execute("""
            SELECT
                COUNT(DISTINCT ticker) AS unique_tickers,
                COUNT(DISTINCT date) AS unique_dates,
                MIN(date) AS min_date,
                MAX(date) AS max_date,
                SUM(target_bankruptcy) AS bankrupt_rows,
                SUM(CASE WHEN target_bankruptcy = 0 THEN 1 ELSE 0 END) AS active_rows,
                COUNT(*) AS total_rows
            FROM staging.bankruptcy_dataset
        """).fetchdf()

        LOGGER.info("Statistics:\n%s", stats.to_string())

        # Statystyki per status
        if has_status:
            status_stats = conn.execute("""
                SELECT
                    status,
                    COUNT(DISTINCT ticker) AS unique_tickers,
                    COUNT(*) AS rows,
                    MIN(date) AS first_date,
                    MAX(date) AS last_date
                FROM staging.bankruptcy_dataset
                WHERE status IS NOT NULL
                GROUP BY status
                ORDER BY status
            """).fetchdf()

            LOGGER.info("Status breakdown:\n%s", status_stats.to_string())

        # Przykładowe delisted spółki
        if has_status:
            sample_delisted = conn.execute("""
                SELECT
                    ticker,
                    status,
                    delisting_date,
                    MAX(date) AS last_data_date,
                    COUNT(*) AS total_rows
                FROM staging.bankruptcy_dataset
                WHERE status IN ('DELISTED', 'DEREGISTERED')
                GROUP BY ticker, status, delisting_date
                ORDER BY delisting_date DESC
                LIMIT 10
            """).fetchdf()

            LOGGER.info("Sample delisted companies (verification):\n%s", sample_delisted.to_string())

        return count

    except Exception as exc:
        LOGGER.error("Failed to create bankruptcy_dataset: %s", exc)
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(name)s - %(message)s"
    )

    LOGGER.info("=" * 60)
    LOGGER.info("Creating staging.bankruptcy_dataset")
    LOGGER.info("=" * 60)

    rows = create_bankruptcy_dataset()

    LOGGER.info("=" * 60)
    LOGGER.info("COMPLETED: %d rows in staging.bankruptcy_dataset", rows)
    LOGGER.info("=" * 60)
