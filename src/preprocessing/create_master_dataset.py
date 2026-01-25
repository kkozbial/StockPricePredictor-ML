"""
Tworzy tabelę staging.master_dataset łączącą wszystkie dane z cleaned dla modelu ML.

GRANULACJA: MIESIĘCZNA (01.10.2015, 01.11.2015, itd.)

Strategia łączenia:
1. Generuj serie miesięczne dla każdego tickera (od pierwszego do ostatniego filing_date)
2. ASOF JOIN financials_wide - forward-fill danych kwartalnych na miesiące
   - filing_date -> date (początek miesiąca po filing_date, max różnica 31 dni)
3. prices_cleaned - LEFT JOIN po (ticker, date)
   - Pomiń kolumny: open, volume, country
4. sectors_cleaned - LEFT JOIN po ticker
   - Dodaj: sector, industry
5. shares_cleaned - ASOF JOIN po (ticker, end_date <= date, max 31 dni różnicy)
   - Pomiń kolumny: country, filing_date, fiscal_year, fiscal_period
6. dividends_cleaned - LEFT JOIN po (ticker, dividend_date <= date)
   - Suma dywidend w danym miesiącu
7. macro_normalized - ASOF JOIN po (publication_date <= date, country)
"""
from __future__ import annotations

import logging
from pathlib import Path

import duckdb

from ..utils.config_loader import load_config

LOGGER = logging.getLogger("preprocessing.master_dataset")


def create_master_dataset(db_path: Path | None = None) -> int:
    """
    Tworzy tabelę staging.master_dataset łączącą dane z cleaned.

    Args:
        db_path: Ścieżka do bazy DuckDB. Jeśli None, używa ścieżki z konfiguracji.

    Returns:
        Liczba rekordów w tabeli staging.master_dataset
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

        # Krok 2: Przygotuj bazę z financials_wide + date (początek miesiąca po filing_date)
        LOGGER.info("Step 1: Preparing financials base with date...")

        query = """
        CREATE OR REPLACE TABLE staging.master_dataset AS
        WITH
        -- Krok 1: Przygotuj dane finansowe z datą publikacji (filing_date -> date)
        -- Deduplikuj po (ticker, publication_date) - zachowaj najstarszy end_date (najpierw opublikowany raport)
        financials_with_date_raw AS (
            SELECT
                ticker,
                end_date,
                fiscal_quarter,
                country,
                filing_date,
                -- Data = początek następnego miesiąca po filing_date
                DATE_TRUNC('month', COALESCE(filing_date, end_date) + INTERVAL '1 day') + INTERVAL '1 month' AS publication_date,
                -- Wszystkie kolumny finansowe
                Assets, AssetsCurrent, AssetsNoncurrent, CapitalExpenditures,
                DiscontinuedOperationsNetIncome, DividendIncome, EBITDA, EarningsPerShareDiluted,
                EquityInEarningsOfSubsidiaries, FeeExpense, FeeIncome, GeneralAndAdministrativeExpense,
                GrossProfit, IncomeLossFromContinuingOperationsBeforeIncomeTaxes, InterestExpense,
                InterestIncome, Inventory, Liabilities, LiabilitiesCurrent, LiabilitiesNoncurrent,
                NetFeeIncome, NetIncomeLoss, NetInterestIncome, OperatingCashFlow, OperatingIncomeLoss,
                OtherFinancialInstrumentsResult, OtherOperatingExpense, OtherOperatingIncome,
                ProvisionForLoanLosses, Receivables, ResearchAndDevelopment, RetainedEarnings,
                Revenues, StockholdersEquity, TradingAndRevaluationResult,
                ROW_NUMBER() OVER (
                    PARTITION BY ticker, country, DATE_TRUNC('month', COALESCE(filing_date, end_date) + INTERVAL '1 day') + INTERVAL '1 month'
                    ORDER BY end_date ASC  -- Starszy end_date = starszy raport = priorytet
                ) AS rn
            FROM cleaned.financials_wide
            WHERE
                -- Filtruj nieprawidłowe daty z przyszłości (max: dzisiaj + 1 rok)
                (filing_date IS NULL OR filing_date <= CURRENT_DATE + INTERVAL '1 year')
                AND (end_date IS NULL OR end_date <= CURRENT_DATE + INTERVAL '1 year')
                -- Tylko rekordy gdzie różnica między filing_date a date <= 31 dni
                AND (filing_date IS NULL
                    OR DATEDIFF('day', filing_date, DATE_TRUNC('month', filing_date + INTERVAL '1 day') + INTERVAL '1 month') <= 31)
        ),
        financials_with_date AS (
            SELECT
                ticker, end_date, fiscal_quarter, country, filing_date, publication_date,
                Assets, AssetsCurrent, AssetsNoncurrent, CapitalExpenditures,
                DiscontinuedOperationsNetIncome, DividendIncome, EBITDA, EarningsPerShareDiluted,
                EquityInEarningsOfSubsidiaries, FeeExpense, FeeIncome, GeneralAndAdministrativeExpense,
                GrossProfit, IncomeLossFromContinuingOperationsBeforeIncomeTaxes, InterestExpense,
                InterestIncome, Inventory, Liabilities, LiabilitiesCurrent, LiabilitiesNoncurrent,
                NetFeeIncome, NetIncomeLoss, NetInterestIncome, OperatingCashFlow, OperatingIncomeLoss,
                OtherFinancialInstrumentsResult, OtherOperatingExpense, OtherOperatingIncome,
                ProvisionForLoanLosses, Receivables, ResearchAndDevelopment, RetainedEarnings,
                Revenues, StockholdersEquity, TradingAndRevaluationResult
            FROM financials_with_date_raw
            WHERE rn = 1  -- Tylko najstarszy end_date dla każdej publication_date
        ),

        -- Krok 2: Znajdź zakres dat dla każdego tickera (od pierwszej do ostatniej publikacji)
        ticker_date_ranges AS (
            SELECT
                ticker,
                country,
                MIN(publication_date) AS first_date,
                MAX(publication_date) AS last_date
            FROM financials_with_date
            GROUP BY ticker, country
        ),

        -- Krok 3: Generuj miesięczne serie dla każdego tickera
        monthly_series AS (
            SELECT
                tdr.ticker,
                tdr.country,
                -- Generuj serie miesięczne od first_date do last_date
                UNNEST(
                    generate_series(
                        tdr.first_date,
                        tdr.last_date,
                        INTERVAL '1 month'
                    )
                )::DATE AS date
            FROM ticker_date_ranges tdr
        ),

        -- Krok 4: ASOF JOIN finansów na miesięczne serie (forward-fill max 5 miesięcy)
        monthly_with_financials AS (
            SELECT
                ms.ticker,
                ms.date,
                ms.country,
                -- end_date, filing_date, fiscal_quarter tylko gdy dane są aktualne (<=150 dni)
                CASE WHEN DATEDIFF('day', fwd.publication_date, ms.date) <= 150 THEN fwd.end_date END AS end_date,
                CASE WHEN DATEDIFF('day', fwd.publication_date, ms.date) <= 150 THEN fwd.filing_date END AS filing_date,
                CASE WHEN DATEDIFF('day', fwd.publication_date, ms.date) <= 150 THEN fwd.fiscal_quarter END AS fiscal_quarter,
                -- Tylko jeśli różnica <= 5 miesięcy (150 dni), inaczej NULL
                CASE WHEN DATEDIFF('day', fwd.publication_date, ms.date) <= 150 THEN fwd.Assets END AS Assets,
                CASE WHEN DATEDIFF('day', fwd.publication_date, ms.date) <= 150 THEN fwd.AssetsCurrent END AS AssetsCurrent,
                CASE WHEN DATEDIFF('day', fwd.publication_date, ms.date) <= 150 THEN fwd.AssetsNoncurrent END AS AssetsNoncurrent,
                CASE WHEN DATEDIFF('day', fwd.publication_date, ms.date) <= 150 THEN fwd.CapitalExpenditures END AS CapitalExpenditures,
                CASE WHEN DATEDIFF('day', fwd.publication_date, ms.date) <= 150 THEN fwd.DiscontinuedOperationsNetIncome END AS DiscontinuedOperationsNetIncome,
                CASE WHEN DATEDIFF('day', fwd.publication_date, ms.date) <= 150 THEN fwd.DividendIncome END AS DividendIncome,
                CASE WHEN DATEDIFF('day', fwd.publication_date, ms.date) <= 150 THEN fwd.EBITDA END AS EBITDA,
                CASE WHEN DATEDIFF('day', fwd.publication_date, ms.date) <= 150 THEN fwd.EarningsPerShareDiluted END AS EarningsPerShareDiluted,
                CASE WHEN DATEDIFF('day', fwd.publication_date, ms.date) <= 150 THEN fwd.EquityInEarningsOfSubsidiaries END AS EquityInEarningsOfSubsidiaries,
                CASE WHEN DATEDIFF('day', fwd.publication_date, ms.date) <= 150 THEN fwd.FeeExpense END AS FeeExpense,
                CASE WHEN DATEDIFF('day', fwd.publication_date, ms.date) <= 150 THEN fwd.FeeIncome END AS FeeIncome,
                CASE WHEN DATEDIFF('day', fwd.publication_date, ms.date) <= 150 THEN fwd.GeneralAndAdministrativeExpense END AS GeneralAndAdministrativeExpense,
                CASE WHEN DATEDIFF('day', fwd.publication_date, ms.date) <= 150 THEN fwd.GrossProfit END AS GrossProfit,
                CASE WHEN DATEDIFF('day', fwd.publication_date, ms.date) <= 150 THEN fwd.IncomeLossFromContinuingOperationsBeforeIncomeTaxes END AS IncomeLossFromContinuingOperationsBeforeIncomeTaxes,
                CASE WHEN DATEDIFF('day', fwd.publication_date, ms.date) <= 150 THEN fwd.InterestExpense END AS InterestExpense,
                CASE WHEN DATEDIFF('day', fwd.publication_date, ms.date) <= 150 THEN fwd.InterestIncome END AS InterestIncome,
                CASE WHEN DATEDIFF('day', fwd.publication_date, ms.date) <= 150 THEN fwd.Inventory END AS Inventory,
                CASE WHEN DATEDIFF('day', fwd.publication_date, ms.date) <= 150 THEN fwd.Liabilities END AS Liabilities,
                CASE WHEN DATEDIFF('day', fwd.publication_date, ms.date) <= 150 THEN fwd.LiabilitiesCurrent END AS LiabilitiesCurrent,
                CASE WHEN DATEDIFF('day', fwd.publication_date, ms.date) <= 150 THEN fwd.LiabilitiesNoncurrent END AS LiabilitiesNoncurrent,
                CASE WHEN DATEDIFF('day', fwd.publication_date, ms.date) <= 150 THEN fwd.NetFeeIncome END AS NetFeeIncome,
                CASE WHEN DATEDIFF('day', fwd.publication_date, ms.date) <= 150 THEN fwd.NetIncomeLoss END AS NetIncomeLoss,
                CASE WHEN DATEDIFF('day', fwd.publication_date, ms.date) <= 150 THEN fwd.NetInterestIncome END AS NetInterestIncome,
                CASE WHEN DATEDIFF('day', fwd.publication_date, ms.date) <= 150 THEN fwd.OperatingCashFlow END AS OperatingCashFlow,
                CASE WHEN DATEDIFF('day', fwd.publication_date, ms.date) <= 150 THEN fwd.OperatingIncomeLoss END AS OperatingIncomeLoss,
                CASE WHEN DATEDIFF('day', fwd.publication_date, ms.date) <= 150 THEN fwd.OtherFinancialInstrumentsResult END AS OtherFinancialInstrumentsResult,
                CASE WHEN DATEDIFF('day', fwd.publication_date, ms.date) <= 150 THEN fwd.OtherOperatingExpense END AS OtherOperatingExpense,
                CASE WHEN DATEDIFF('day', fwd.publication_date, ms.date) <= 150 THEN fwd.OtherOperatingIncome END AS OtherOperatingIncome,
                CASE WHEN DATEDIFF('day', fwd.publication_date, ms.date) <= 150 THEN fwd.ProvisionForLoanLosses END AS ProvisionForLoanLosses,
                CASE WHEN DATEDIFF('day', fwd.publication_date, ms.date) <= 150 THEN fwd.Receivables END AS Receivables,
                CASE WHEN DATEDIFF('day', fwd.publication_date, ms.date) <= 150 THEN fwd.ResearchAndDevelopment END AS ResearchAndDevelopment,
                CASE WHEN DATEDIFF('day', fwd.publication_date, ms.date) <= 150 THEN fwd.RetainedEarnings END AS RetainedEarnings,
                CASE WHEN DATEDIFF('day', fwd.publication_date, ms.date) <= 150 THEN fwd.Revenues END AS Revenues,
                CASE WHEN DATEDIFF('day', fwd.publication_date, ms.date) <= 150 THEN fwd.StockholdersEquity END AS StockholdersEquity,
                CASE WHEN DATEDIFF('day', fwd.publication_date, ms.date) <= 150 THEN fwd.TradingAndRevaluationResult END AS TradingAndRevaluationResult
            FROM monthly_series ms
            ASOF LEFT JOIN financials_with_date fwd
                ON ms.ticker = fwd.ticker
                AND ms.country = fwd.country
                AND ms.date >= fwd.publication_date
        ),

        -- Krok 5: Dołącz prices (bez open, volume, country)
        with_prices AS (
            SELECT
                mwf.*,
                p.high,
                p.low,
                p.close
            FROM monthly_with_financials mwf
            LEFT JOIN cleaned.prices_cleaned p
                ON mwf.ticker = p.ticker
                AND mwf.date = p.date
                AND mwf.country = p.country
        ),

        -- Krok 6: Dołącz sectors (sector, industry)
        with_sectors AS (
            SELECT
                wp.*,
                s.sector,
                s.industry
            FROM with_prices wp
            LEFT JOIN cleaned.sectors_cleaned s
                ON wp.ticker = s.ticker
                AND wp.country = s.country
        ),

        -- Krok 7: Dołącz shares (ASOF JOIN - najbliższa data <= date)
        with_shares AS (
            SELECT
                ws.*,
                sc.shares AS shares_outstanding
            FROM with_sectors ws
            ASOF LEFT JOIN cleaned.shares_cleaned sc
                ON ws.ticker = sc.ticker
                AND ws.country = sc.country
                AND ws.date >= sc.end_date
        ),

        -- Krok 8: Dołącz dividends (suma dywidend w danym miesiącu)
        dividends_monthly AS (
            SELECT
                ticker,
                country,
                DATE_TRUNC('month', dividend_date) AS month_start,
                SUM(dividend_amount) AS dividend_amount
            FROM cleaned.dividends_cleaned
            GROUP BY ticker, country, DATE_TRUNC('month', dividend_date)
        ),
        with_dividends AS (
            SELECT
                wsh.*,
                dm.dividend_amount
            FROM with_shares wsh
            LEFT JOIN dividends_monthly dm
                ON wsh.ticker = dm.ticker
                AND wsh.country = dm.country
                AND wsh.date = dm.month_start
        ),

        -- Krok 9: Dołącz macro (ASOF JOIN - publication_date <= date, ten sam country)
        with_macro AS (
            SELECT
                wd.*,
                mn.retail_sales_yoy,
                mn.gdp_yoy,
                mn.cpi_yoy,
                mn.unemp_level,
                mn.unemp_diff,
                mn.rate_level,
                mn.rate_diff
            FROM with_dividends wd
            ASOF LEFT JOIN cleaned.macro_normalized mn
                ON wd.country = mn.country
                AND wd.date >= mn.publication_date
        ),

        -- Krok 10: Feature Engineering - podstawowe wskaźniki finansowe
        with_features_basic AS (
            SELECT
                *,
                -- Per-share metrics
                CASE WHEN shares_outstanding > 0 THEN Revenues / shares_outstanding END AS revenue_per_share,
                CASE WHEN shares_outstanding > 0 THEN OperatingIncomeLoss / shares_outstanding END AS operating_income_per_share,
                CASE WHEN shares_outstanding > 0 THEN Assets / shares_outstanding END AS assets_per_share,
                CASE WHEN shares_outstanding > 0 THEN ResearchAndDevelopment / shares_outstanding END AS rnd_per_share,
                CASE WHEN shares_outstanding > 0 THEN GeneralAndAdministrativeExpense / shares_outstanding END AS sga_per_share,
                CASE WHEN shares_outstanding > 0 THEN NetIncomeLoss / shares_outstanding END AS net_income_per_share,

                -- Profitability ratios
                CASE WHEN StockholdersEquity > 0 THEN NetIncomeLoss / StockholdersEquity END AS roe,
                CASE WHEN Assets > 0 THEN NetIncomeLoss / Assets END AS roa,
                CASE WHEN Revenues > 0 THEN NetIncomeLoss / Revenues END AS ros,
                CASE WHEN Revenues > 0 THEN GrossProfit / Revenues END AS gross_margin,
                CASE WHEN Revenues > 0 THEN OperatingIncomeLoss / Revenues END AS operating_margin,

                -- Leverage ratios
                CASE WHEN Assets > 0 THEN Liabilities / Assets END AS debt_ratio,
                CASE WHEN StockholdersEquity > 0 THEN Liabilities / StockholdersEquity END AS debt_to_equity,

                -- Cash flow metrics
                CASE WHEN OperatingCashFlow IS NOT NULL AND CapitalExpenditures IS NOT NULL
                    THEN OperatingCashFlow - CapitalExpenditures END AS free_cash_flow,
                CASE WHEN Revenues > 0 AND OperatingCashFlow IS NOT NULL
                    THEN OperatingCashFlow / Revenues END AS ocf_ratio,
                CASE WHEN Assets > 0 AND CapitalExpenditures IS NOT NULL
                    THEN CapitalExpenditures / Assets END AS capex_ratio,

                -- Liquidity ratios
                CASE WHEN LiabilitiesCurrent > 0 THEN AssetsCurrent / LiabilitiesCurrent END AS current_ratio,
                CASE WHEN LiabilitiesCurrent > 0 AND Inventory IS NOT NULL
                    THEN (AssetsCurrent - Inventory) / LiabilitiesCurrent END AS quick_ratio,

                -- Valuation ratios (price-based)
                CASE WHEN close > 0 AND shares_outstanding > 0 AND StockholdersEquity > 0
                    THEN (close * shares_outstanding) / StockholdersEquity END AS price_to_book,
                CASE WHEN close > 0 AND shares_outstanding > 0 AND Revenues > 0
                    THEN (close * shares_outstanding) / Revenues END AS price_to_sales,
                CASE WHEN close > 0 AND shares_outstanding > 0 AND NetIncomeLoss > 0
                    THEN (close * shares_outstanding) / NetIncomeLoss END AS price_to_earnings,
                CASE WHEN close > 0 AND EarningsPerShareDiluted > 0
                    THEN close / EarningsPerShareDiluted END AS pe_ratio,

                -- Dividend metrics
                CASE WHEN close > 0 AND dividend_amount IS NOT NULL
                    THEN dividend_amount / close END AS dividend_yield,
                CASE WHEN NetIncomeLoss > 0 AND dividend_amount IS NOT NULL
                    THEN (dividend_amount * shares_outstanding) / NetIncomeLoss END AS payout_ratio,

                -- Efficiency ratios
                CASE WHEN Assets > 0 THEN Revenues / Assets END AS asset_turnover,
                CASE WHEN Inventory > 0 AND Revenues > 0
                    THEN Revenues / Inventory END AS inventory_turnover

            FROM with_macro
        ),

        -- Krok 10b: TTM (Trailing Twelve Months) metrics + Growth rates
        with_features AS (
            SELECT
                fb.*,
                -- TTM metrics (sum over last 12 months for flow items)
                SUM(fb.Revenues) OVER (
                    PARTITION BY fb.ticker
                    ORDER BY fb.date
                    ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
                ) AS revenue_ttm,
                SUM(fb.NetIncomeLoss) OVER (
                    PARTITION BY fb.ticker
                    ORDER BY fb.date
                    ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
                ) AS net_income_ttm,
                SUM(fb.OperatingCashFlow) OVER (
                    PARTITION BY fb.ticker
                    ORDER BY fb.date
                    ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
                ) AS ocf_ttm,
                SUM(fb.CapitalExpenditures) OVER (
                    PARTITION BY fb.ticker
                    ORDER BY fb.date
                    ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
                ) AS capex_ttm,
                SUM(fb.free_cash_flow) OVER (
                    PARTITION BY fb.ticker
                    ORDER BY fb.date
                    ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
                ) AS fcf_ttm,

                -- Growth rates (YoY - 12 months ago)
                LAG(fb.Revenues, 12) OVER (
                    PARTITION BY fb.ticker
                    ORDER BY fb.date
                ) AS revenues_12m_ago,
                LAG(fb.NetIncomeLoss, 12) OVER (
                    PARTITION BY fb.ticker
                    ORDER BY fb.date
                ) AS net_income_12m_ago

            FROM with_features_basic fb
        ),

        -- Krok 10c: Calculate growth rates
        with_growth AS (
            SELECT
                *,
                -- Revenue growth (YoY)
                CASE WHEN revenues_12m_ago IS NOT NULL AND revenues_12m_ago != 0
                    THEN (Revenues - revenues_12m_ago) / ABS(revenues_12m_ago)
                END AS revenue_growth,
                -- Net income growth (YoY)
                CASE WHEN net_income_12m_ago IS NOT NULL AND net_income_12m_ago != 0
                    THEN (NetIncomeLoss - net_income_12m_ago) / ABS(net_income_12m_ago)
                END AS net_income_growth

            FROM with_features
        ),

        -- Krok 11: Deduplikacja po (ticker, date) - zachowaj rekord z najmniejszą liczbą NULL
        deduplicated AS (
            SELECT
                *,
                (CASE WHEN Assets IS NULL THEN 1 ELSE 0 END +
                 CASE WHEN Liabilities IS NULL THEN 1 ELSE 0 END +
                 CASE WHEN Revenues IS NULL THEN 1 ELSE 0 END +
                 CASE WHEN NetIncomeLoss IS NULL THEN 1 ELSE 0 END +
                 CASE WHEN close IS NULL THEN 1 ELSE 0 END) AS null_count,
                ROW_NUMBER() OVER (
                    PARTITION BY ticker, date
                    ORDER BY
                        (CASE WHEN Assets IS NULL THEN 1 ELSE 0 END +
                         CASE WHEN Liabilities IS NULL THEN 1 ELSE 0 END +
                         CASE WHEN Revenues IS NULL THEN 1 ELSE 0 END +
                         CASE WHEN NetIncomeLoss IS NULL THEN 1 ELSE 0 END +
                         CASE WHEN close IS NULL THEN 1 ELSE 0 END) ASC,
                        end_date DESC NULLS LAST,
                        filing_date DESC NULLS LAST
                ) AS rn
            FROM with_growth
        )

        SELECT
            -- Base columns
            ticker, date, end_date, filing_date, fiscal_quarter, country,
            -- Financial statement items
            Assets, AssetsCurrent, AssetsNoncurrent, CapitalExpenditures,
            DiscontinuedOperationsNetIncome, DividendIncome, EBITDA, EarningsPerShareDiluted,
            EquityInEarningsOfSubsidiaries, FeeExpense, FeeIncome, GeneralAndAdministrativeExpense,
            GrossProfit, IncomeLossFromContinuingOperationsBeforeIncomeTaxes, InterestExpense,
            InterestIncome, Inventory, Liabilities, LiabilitiesCurrent, LiabilitiesNoncurrent,
            NetFeeIncome, NetIncomeLoss, NetInterestIncome, OperatingCashFlow, OperatingIncomeLoss,
            OtherFinancialInstrumentsResult, OtherOperatingExpense, OtherOperatingIncome,
            ProvisionForLoanLosses, Receivables, ResearchAndDevelopment, RetainedEarnings,
            Revenues, StockholdersEquity, TradingAndRevaluationResult,
            -- Market data
            high, low, close,
            -- Reference data
            sector, industry, shares_outstanding, dividend_amount,
            -- Macro data
            retail_sales_yoy, gdp_yoy, cpi_yoy, unemp_level, unemp_diff, rate_level, rate_diff,
            -- Engineered features (per-share)
            revenue_per_share, operating_income_per_share, assets_per_share,
            rnd_per_share, sga_per_share, net_income_per_share,
            -- Engineered features (profitability ratios)
            roe, roa, ros, gross_margin, operating_margin,
            -- Engineered features (leverage ratios)
            debt_ratio, debt_to_equity,
            -- Engineered features (cash flow)
            free_cash_flow, ocf_ratio, capex_ratio,
            -- Engineered features (liquidity)
            current_ratio, quick_ratio,
            -- Engineered features (valuation)
            price_to_book, price_to_sales, price_to_earnings, pe_ratio,
            -- Engineered features (dividends)
            dividend_yield, payout_ratio,
            -- Engineered features (efficiency)
            asset_turnover, inventory_turnover,
            -- TTM metrics
            revenue_ttm, net_income_ttm, ocf_ttm, capex_ttm, fcf_ttm,
            -- Growth rates
            revenue_growth, net_income_growth
        FROM deduplicated
        WHERE rn = 1
        ORDER BY ticker, date
        """

        LOGGER.info("Creating staging.master_dataset...")
        conn.execute(query)

        # Sprawdź wynik
        count = conn.execute("SELECT COUNT(*) FROM staging.master_dataset").fetchone()[0]
        LOGGER.info("Created staging.master_dataset: %d rows", count)

        # Statystyki
        stats = conn.execute("""
            SELECT
                COUNT(DISTINCT ticker) AS unique_tickers,
                COUNT(DISTINCT date) AS unique_dates,
                MIN(date) AS min_date,
                MAX(date) AS max_date,
                COUNT(*) AS total_rows
            FROM staging.master_dataset
        """).fetchdf()

        LOGGER.info("Statistics:\n%s", stats.to_string())

        # Sprawdź pokrycie
        coverage = conn.execute("""
            SELECT
                COUNT(*) AS total_rows,
                SUM(CASE WHEN close IS NOT NULL THEN 1 ELSE 0 END) AS has_price,
                SUM(CASE WHEN sector IS NOT NULL THEN 1 ELSE 0 END) AS has_sector,
                SUM(CASE WHEN shares_outstanding IS NOT NULL THEN 1 ELSE 0 END) AS has_shares,
                SUM(CASE WHEN dividend_amount IS NOT NULL THEN 1 ELSE 0 END) AS has_dividend,
                SUM(CASE WHEN gdp_yoy IS NOT NULL THEN 1 ELSE 0 END) AS has_macro
            FROM staging.master_dataset
        """).fetchdf()

        LOGGER.info("Data coverage:\n%s", coverage.to_string())

        return count

    except Exception as exc:
        LOGGER.error("Failed to create master_dataset: %s", exc)
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(name)s - %(message)s"
    )

    LOGGER.info("=" * 60)
    LOGGER.info("Creating staging.master_dataset")
    LOGGER.info("=" * 60)

    rows = create_master_dataset()

    LOGGER.info("=" * 60)
    LOGGER.info("COMPLETED: %d rows in staging.master_dataset", rows)
    LOGGER.info("=" * 60)
