"""Loader danych makroekonomicznych (macro)."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger("database")


def load_macro_from_raw(raw_dir: Path) -> pd.DataFrame:
    """
    Wczytuje dane makroekonomiczne z folderu raw.

    NOWY SCHEMAT (2026-01): Bez kolumny country, osobne kolumny dla USA i PL.
    - USA: cpi_usa, unemployment_rate_usa, gdp_real_usa, retail_sales_usa, interest_rate_usa
    - PL: hicp_pl, unemployment_rate_pl, gdp_real_pl, gdp_nominal_pl, retail_sales_index_pl, interest_rate_pl

    Args:
        raw_dir: Ścieżka do katalogu data/raw

    Returns:
        DataFrame z jedną tabelą szeroki format (date + kolumny USA + kolumny PL).
    """
    macro_dir = raw_dir / "macro"
    if not macro_dir.exists():
        LOGGER.warning("Katalog z danymi makro nie istnieje: %s", macro_dir)
        return pd.DataFrame()

    # Mapowanie nazw plików USA (FRED) na kolumny w bazie
    us_column_mapping = {
        "cpiaucsl": "inflation_usa",         # Consumer Price Index → inflation
        "unrate": "unemployment_rate_usa",   # Unemployment Rate
        "gdpc1": "gdp_real_usa",            # Real GDP w mld USD
        "rsxfs": "retail_sales_usa",        # Retail Sales w mln USD
        "fedfunds": "interest_rate_usa",    # Federal Funds Rate
    }

    # Mapowanie nazw plików PL (Eurostat) na kolumny w bazie
    pl_column_mapping = {
        # Nowe serie (2026-01) - ZMIENIONO nazwy dla consistency
        "prc_hicp_manr_coicop_cp00_geo_pl_unit_rch_a": "inflation_pl",  # HICP → inflation
        "ei_lmhr_m_indic_lm-un-t-tot_geo_pl_s_adj_nsa": "unemployment_rate_pl",  # Bezrobocie %
        "namq_10_gdp_na_item_b1gq_geo_pl_unit_clv15_mnac_s_adj_sca": "gdp_real_pl",  # PKB realny mln PLN (chain-linked 2015)
        "namq_10_gdp_na_item_b1gq_geo_pl_unit_cp_mnac_s_adj_sca": "gdp_nominal_pl",  # PKB nominalny (current prices) — zapisujemy osobno
        "sts_trtu_m_geo_pl_nace_r2_g47_s_adj_sca_unit_i21": "retail_sales_index_pl",  # Indeks 2021=100
        "irt_st_m_geo_pl_int_rt_irt_m3": "interest_rate_pl",  # Stopa % 3M
    }

    # Rozdziel pliki na US i PL
    us_files = [f for f in macro_dir.glob("macro_*.csv") if not f.stem.startswith("macro_pl_")]
    pl_files = list(macro_dir.glob("macro_pl_*.csv"))

    # Przetwórz dane USA
    us_data = {}
    if us_files:
        for file in us_files:
            series_name_raw = file.stem.replace("macro_", "").lower()
            column_name = us_column_mapping.get(series_name_raw)

            if not column_name:
                LOGGER.debug("Pominięto nieznany plik USA: %s", file.name)
                continue

            frame = pd.read_csv(file)
            frame = frame.rename(columns={"date": "Date", "value": column_name})
            frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
            frame[column_name] = pd.to_numeric(frame[column_name], errors="coerce")
            us_data[column_name] = frame[["Date", column_name]].dropna(subset=["Date"])

        LOGGER.info("Załadowano %d serii danych makro USA", len(us_data))

    # Przetwórz dane Polski
    pl_data = {}
    if pl_files:
        for file in pl_files:
            series_name_raw = file.stem.replace("macro_pl_", "").lower()
            column_name = pl_column_mapping.get(series_name_raw)

            if not column_name:
                LOGGER.debug("Pominięto nieznany plik PL: %s", file.name)
                continue

            frame = pd.read_csv(file)
            frame = frame.rename(columns={"date": "Date", "value": column_name})
            frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
            frame[column_name] = pd.to_numeric(frame[column_name], errors="coerce")
            pl_data[column_name] = frame[["Date", column_name]].dropna(subset=["Date"])

        LOGGER.info("Załadowano %d serii danych makro Polski", len(pl_data))

    if not us_data and not pl_data:
        LOGGER.warning("Brak plików z danymi makro w %s", macro_dir)
        return pd.DataFrame()

    # Merge wszystkich serii w jedną szeroką tabelę (outer join po dacie)
    all_series = list(us_data.values()) + list(pl_data.values())

    merged = all_series[0]
    for series_df in all_series[1:]:
        merged = merged.merge(series_df, on="Date", how="outer")

    # Sortuj i deduplikuj
    merged = merged.sort_values("Date").reset_index(drop=True)
    merged = merged.drop_duplicates("Date", keep="last")
    merged = merged.rename(columns={"Date": "date"})

    # Dodaj wszystkie wymagane kolumny ze schematu (z NULL jeśli brakuje)
    required_schema_cols = [
        "date",
        "inflation_usa", "unemployment_rate_usa", "gdp_real_usa", "retail_sales_usa", "interest_rate_usa",
        "inflation_pl", "unemployment_rate_pl", "gdp_real_pl", "gdp_nominal_pl", "retail_sales_index_pl", "interest_rate_pl"
    ]

    for col in required_schema_cols:
        if col not in merged.columns:
            merged[col] = None
            LOGGER.debug("Dodano brakującą kolumnę makro '%s' z wartościami NULL", col)

    # Zwróć w kolejności schematu
    LOGGER.info("Przygotowano tabelę makro: %d rekordów, zakres %s - %s",
                len(merged), merged["date"].min(), merged["date"].max())
    return merged[required_schema_cols]
