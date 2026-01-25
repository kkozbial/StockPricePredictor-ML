"""Zbiera funkcje uruchamiające kolejne etapy pipeline'u danych - NOWA WERSJA Z BAZĄ DANYCH."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.analysis.correlations import correlation_matrix
from src.analysis.descriptive_stats import describe_dataset
from src.analysis.visualization import plot_price_history
from src.data_fetch.fetch_biznesradar import fetch_biznesradar_financials, fetch_biznesradar_sectors
from src.data_fetch.fetch_financials import fetch_financial_reports
from src.data_fetch.fetch_macro import fetch_macro_series
from src.data_fetch.fetch_macro_pl import fetch_macro_series_pl
from src.data_fetch.fetch_others import fetch_dividends, fetch_sectors
from src.data_fetch.fetch_prices import fetch_prices
from src.data_fetch.fetch_prices_stooq import fetch_prices_stooq
from src.data_fetch.fetch_sec_metadata import fetch_all_company_metadata, get_filtered_tickers
from src.data_fetch.fetch_sec_status import SecMetadataFetcher
from src.data_fetch.fetch_shares import fetch_shares_outstanding
from src.database import build_all_tables, init_database, update_all_tables
from src.database.queries import get_prices_cleaned, get_table_stats
from src.database.schema_raw import get_table_count
from src.preprocessing.normalize_macro import process_macro_to_staging
from src.utils.config_loader import get_path_from_config
from src.utils.io_helpers import ensure_dir, write_csv
from src.utils.log_helpers import FetchResult, log_step, summarize_counts

LOGGER = logging.getLogger("pipeline")


@log_step("fetch_prices")
def _fetch_prices_step(tickers: Sequence[str], cfg: dict, raw_dir: Path) -> FetchResult:
    """Wrapper zapewniający zunifikowane logowanie dla cen."""
    return fetch_prices(
        tickers=tickers,
        start=cfg["fetch"]["price_start"],
        interval=cfg["fetch"]["price_interval"],
        output_dir=raw_dir / "prices",
    )


@log_step("fetch_sec_metadata")
def _fetch_sec_metadata_step(raw_dir: Path) -> FetchResult:
    """Wrapper zapewniający zunifikowane logowanie dla metadanych SEC."""
    return fetch_all_company_metadata(
        output_dir=raw_dir / "metadata",
    )


def _fetch_sec_status_step(tickers: Sequence[str], raw_dir: Path) -> None:
    """
    Pobiera SEC status i filing dates dla wykrywania delistingu i Look-ahead Bias.

    Generuje 2 pliki:
    - sec_status.csv: Status spółki (ACTIVE/DELISTED/DEREGISTERED/ZOMBIE)
    - sec_filing_dates.csv: Mapowanie end_date -> filing_date
    """
    import json

    metadata_file = raw_dir / "metadata" / "company_metadata.json"
    if not metadata_file.exists():
        LOGGER.warning("Brak pliku company_metadata.json - pomiń SEC status")
        return

    # Wczytaj metadane (potrzebujemy CIK dla każdego tickera)
    with open(metadata_file, encoding="utf-8") as f:
        metadata = json.load(f)

    # Przygotuj listę spółek z CIK i ticker
    companies = []
    for company in metadata.get("companies", []):
        cik = company.get("cik")
        tickers_list = company.get("all_tickers", [])
        if cik and tickers_list:
            # Dodaj pierwszego tickera (główny)
            companies.append({
                "cik": cik,
                "ticker": tickers_list[0],
            })

    if not companies:
        LOGGER.warning("Brak spółek z CIK w metadanych - pomiń SEC status")
        return

    LOGGER.info("Pobieranie SEC status i filing dates dla %d spółek...", len(companies))

    # Inicjalizuj fetcher z cache
    fetcher = SecMetadataFetcher(
        user_agent="Stock Market ML Pipeline research@example.com",
        cache_dir=raw_dir / "cache" / "sec",
    )

    # 1. Generuj tabelę statusów
    status_df = fetcher.generate_status_table(companies)

    # 2. Ekstrahuj daty publikacji raportów
    filing_dates_df = fetcher.extract_filing_dates(companies)

    # Zapisz wyniki
    output_dir = raw_dir / "sec_metadata"
    output_dir.mkdir(parents=True, exist_ok=True)

    status_file = output_dir / "sec_status.csv"
    filing_dates_file = output_dir / "sec_filing_dates.csv"

    status_df.to_csv(status_file, index=False)
    filing_dates_df.to_csv(filing_dates_file, index=False)

    LOGGER.info("Zapisano SEC status: %s (%d rekordów)", status_file, len(status_df))
    LOGGER.info("Zapisano SEC filing dates: %s (%d rekordów)", filing_dates_file, len(filing_dates_df))

    # Statystyki statusów
    status_counts = status_df["status"].value_counts()
    LOGGER.info("Statystyki statusów spółek:")
    for status, count in status_counts.items():
        LOGGER.info("  %s: %d spółek", status, count)


@log_step("fetch_macro")
def _fetch_macro_step(series_ids: Sequence[str], raw_dir: Path) -> FetchResult:
    """Wrapper zapewniający zunifikowane logowanie dla danych makro."""
    return fetch_macro_series(
        series_ids=series_ids,
        output_dir=raw_dir / "macro",
    )


@log_step("fetch_financials")
def _fetch_financials_step(tickers: Sequence[str], raw_dir: Path) -> FetchResult:
    """Wrapper zapewniający zunifikowane logowanie dla raportów finansowych."""
    return fetch_financial_reports(
        tickers=tickers,
        output_dir=raw_dir / "financials",
    )


@log_step("fetch_shares")
def _fetch_shares_step(tickers: Sequence[str], raw_dir: Path) -> FetchResult:
    """Wrapper zapewniający zunifikowane logowanie dla danych o liczbie akcji."""
    return fetch_shares_outstanding(
        tickers=tickers,
        output_dir=raw_dir / "shares",
    )


@log_step("fetch_dividends")
def _fetch_dividends_step(tickers: Sequence[str], raw_dir: Path) -> FetchResult:
    """Wrapper zapewniający zunifikowane logowanie dla dywidend."""
    return fetch_dividends(
        tickers=tickers,
        output_dir=raw_dir / "others",
    )


@log_step("fetch_sectors")
def _fetch_sectors_step(tickers: Sequence[str], raw_dir: Path) -> FetchResult:
    """Wrapper zapewniający zunifikowane logowanie dla sektorów."""
    return fetch_sectors(
        tickers=tickers,
        output_dir=raw_dir / "others",
    )


@log_step("fetch_prices_pl")
def _fetch_prices_pl_step(tickers: Sequence[str], cfg: dict, raw_dir: Path) -> FetchResult:
    """Wrapper zapewniający zunifikowane logowanie dla cen polskich akcji (Yahoo Finance)."""
    return fetch_prices_stooq(
        tickers=tickers,
        start=cfg["fetch"]["price_start"],
        output_dir=raw_dir / "prices",
    )


@log_step("fetch_macro_pl")
def _fetch_macro_pl_step(series_ids: Sequence[str], raw_dir: Path) -> FetchResult:
    """Wrapper zapewniający zunifikowane logowanie dla danych makro Polski (DBnomics)."""
    return fetch_macro_series_pl(
        series_ids=series_ids,
        output_dir=raw_dir / "macro",
    )


@log_step("fetch_financials_pl")
def _fetch_financials_pl_step(tickers: Sequence[str], raw_dir: Path) -> FetchResult:
    """Wrapper zapewniający zunifikowane logowanie dla raportów finansowych Polski (BiznesRadar)."""
    return fetch_biznesradar_financials(
        tickers=tickers,
        output_dir=raw_dir / "financials",
    )


@log_step("fetch_sectors_pl")
def _fetch_sectors_pl_step(tickers: Sequence[str], raw_dir: Path) -> FetchResult:
    """Wrapper zapewniający zunifikowane logowanie dla sektorów Polski (BiznesRadar)."""
    return fetch_biznesradar_sectors(
        tickers=tickers,
        output_dir=raw_dir / "others",
    )


def load_tickers(config_dir: Path) -> list[str]:
    """
    Wczytuje listę tickerów z metadanych SEC (przefiltrowanych).

    UWAGA: Ta funkcja została zmieniona aby używać filtrowanych tickerów z SEC.
    Stary plik config/tickers_list.csv NIE JEST JUŻ UŻYWANY dla danych USA.

    Returns:
        Lista przefiltrowanych tickerów (bez funduszy, banków, ubezpieczeń).
    """
    LOGGER.info("Wczytywanie tickerów z metadanych SEC (przefiltrowanych)...")
    tickers = get_filtered_tickers()

    if not tickers:
        LOGGER.warning("Brak przefiltrowanych tickerów. Upewnij się, że najpierw uruchomiono pobieranie metadanych SEC.")
        LOGGER.warning("Możesz to zrobić dodając '--metadata' do flagi --fetch")

    return tickers


def load_tickers_pl(config_dir: Path) -> list[str]:
    """Wczytuje listę polskich tickerów z pliku Excel."""
    tickers_path = config_dir / "tickers_pl.xlsx"
    if not tickers_path.exists():
        LOGGER.warning("Plik tickers_pl.xlsx nie istnieje, zwracam pustą listę")
        return []
    df = pd.read_excel(tickers_path)
    # Oczekujemy kolumny 'Ticker'
    if "Ticker" not in df.columns:
        LOGGER.warning("Kolumna 'Ticker' nie znaleziona w tickers_pl.xlsx")
        return []
    # Usuń białe znaki i zwróć unikalne tickery
    tickers = df["Ticker"].str.strip().dropna().unique().tolist()
    return tickers


def run_fetch(cfg: dict, tickers: list[str], fetch_metadata: bool = False, fetch_modules: list[str] | None = None) -> None:
    """
    Realizuje etap pobierania danych dla USA i Polski.

    Args:
        cfg: Konfiguracja pipeline'u.
        tickers: Lista tickerów do pobrania. Jeśli pusta i fetch_metadata=True,
                zostanie automatycznie wypełniona po pobraniu metadanych.
        fetch_metadata: Czy pobrać metadane SEC przed pobieraniem danych.
        fetch_modules: Lista modułów do pobrania. Jeśli None, pobiera wszystko oprócz sectors.
                      Dostępne: ["prices", "financials", "shares", "dividends", "sectors", "macro"]
    """
    # Jeśli nie podano modułów, użyj domyślnych (wszystko oprócz sectors)
    if fetch_modules is None:
        fetch_modules = ["prices", "financials", "shares", "dividends", "macro"]

    LOGGER.info("Etap: pobieranie danych.")
    LOGGER.info("Wybrane moduły: %s", ", ".join(fetch_modules))

    raw_dir = get_path_from_config("paths", "raw", cfg)
    ensure_dir(raw_dir)

    # Inicjalizuj połączenie z bazą danych dla pobierania przyrostowego
    # Jeśli baza nie istnieje, zostanie utworzona podczas run_build/run_update
    db_path = get_path_from_config("database", "path", cfg)
    init_database(db_path)

    # Serie makro dla USA i Polski
    series_ids = cfg["fetch"].get("macro_series") or []
    series_ids_pl = cfg["fetch"].get("macro_series_pl") or []

    # Tickery polskie - wczytaj z pliku Excel
    config_dir = Path("config")
    tickers_pl = load_tickers_pl(config_dir)

    results: dict[str, FetchResult] = {}

    # KROK 1: Pobierz metadane SEC (jeśli wymagane)
    if fetch_metadata:
        LOGGER.info("Pobieranie metadanych SEC (ticker->CIK + filtrowanie po SIC)...")
        results["sec_metadata"] = _fetch_sec_metadata_step(raw_dir)

        # Jeśli nie podano tickerów, użyj przefiltrowanych z metadanych
        if not tickers:
            LOGGER.info("Automatyczne wczytywanie tickerów z metadanych SEC...")
            tickers = get_filtered_tickers()
            LOGGER.info("Znaleziono %d przefiltrowanych tickerów", len(tickers))

        # KROK 1.1: Pobierz SEC status i filing dates (dla Look-ahead Bias i Survivorship Bias)
        LOGGER.info("Pobieranie SEC status i filing dates dla %d spółek...", len(tickers))
        _fetch_sec_status_step(tickers, raw_dir)

    # Upewnij się że mamy tickery
    if not tickers:
        LOGGER.warning("Brak tickerów do pobrania. Sprawdź czy pobrano metadane SEC.")
        LOGGER.warning("Użyj flagi --fetch-metadata przy pobieraniu lub uruchom pipeline z --build")

    # USA data - warunkowe uruchamianie modułów
    if "prices" in fetch_modules:
        results["prices_us"] = _fetch_prices_step(tickers, cfg, raw_dir)
    else:
        LOGGER.info("Pominięto moduł: prices")

    if "macro" in fetch_modules:
        results["macro_us"] = _fetch_macro_step(series_ids, raw_dir)
    else:
        LOGGER.info("Pominięto moduł: macro")

    if "financials" in fetch_modules:
        results["financials_us"] = _fetch_financials_step(tickers, raw_dir)
    else:
        LOGGER.info("Pominięto moduł: financials")

    if "shares" in fetch_modules:
        results["shares_us"] = _fetch_shares_step(tickers, raw_dir)
    else:
        LOGGER.info("Pominięto moduł: shares")

    if "dividends" in fetch_modules:
        results["dividends_us"] = _fetch_dividends_step(tickers, raw_dir)
    else:
        LOGGER.info("Pominięto moduł: dividends")

    if "sectors" in fetch_modules:
        results["sectors_us"] = _fetch_sectors_step(tickers, raw_dir)
    else:
        LOGGER.info("Pominięto moduł: sectors (opcjonalny)")

    # Poland data
    if tickers_pl:
        LOGGER.info("Pobieranie danych dla %d polskich tickerów z tickers_pl.xlsx", len(tickers_pl))

        if "prices" in fetch_modules:
            results["prices_pl"] = _fetch_prices_pl_step(tickers_pl, cfg, raw_dir)

        if "financials" in fetch_modules:
            results["financials_pl"] = _fetch_financials_pl_step(tickers_pl, raw_dir)

        if "sectors" in fetch_modules:
            results["sectors_pl"] = _fetch_sectors_pl_step(tickers_pl, raw_dir)

        LOGGER.info("Pobrano dane dla %d polskich tickerów", len(tickers_pl))

    if series_ids_pl and "macro" in fetch_modules:
        results["macro_pl"] = _fetch_macro_pl_step(series_ids_pl, raw_dir)
        LOGGER.info("Pobrano %d serii makro dla Polski", len(series_ids_pl))

    summary = ", ".join(f"{name}={summarize_counts(res)}" for name, res in results.items())
    LOGGER.info("[fetch] Summary: %s", summary)


def run_build(cfg: dict, auto_confirm: bool = False) -> None:
    """
    BUILD: Tworzy bazę danych i ładuje wszystkie dane z raw (initial load).

    Wykonuje się gdy baza nie istnieje lub jest pusta.

    Args:
        cfg: Konfiguracja.
        auto_confirm: Jeśli True, pomija pytanie o potwierdzenie (dla automatycznego wywołania).
    """
    LOGGER.info("Etap: BUILD - tworzenie bazy danych i początkowe załadowanie")

    # Inicjalizacja bazy
    db_path = get_path_from_config("database", "path", cfg)
    init_database(db_path)

    # Sprawdzamy czy tabele są puste
    prices_count = get_table_count("prices")

    if prices_count > 0 and not auto_confirm:
        LOGGER.warning(
            "Tabela 'prices' zawiera już %s rekordów. "
            "Czy na pewno chcesz uruchomić BUILD? Użyj UPDATE do aktualizacji.",
            prices_count
        )
        response = input("Kontynuować BUILD (nadpisze dane)? [y/N]: ")
        if response.lower() != "y":
            LOGGER.info("BUILD anulowany przez użytkownika")
            return

    raw_dir = get_path_from_config("paths", "raw", cfg)

    # Ładowanie danych
    results = build_all_tables(raw_dir)

    # Podsumowanie
    total_records = sum(results.values())
    LOGGER.info("BUILD zakończony pomyślnie. Łącznie załadowano %s rekordów", total_records)

    # Statystyki tabel
    stats = get_table_stats()
    for table, info in stats.items():
        count = info.get("count", 0)
        date_range = info.get("date_range", (None, None))
        if date_range[0]:
            LOGGER.info("Tabela '%s': %s rekordów, zakres dat: %s - %s", table, count, date_range[0], date_range[1])
        else:
            LOGGER.info("Tabela '%s': %s rekordów", table, count)


def run_update(cfg: dict, run_preprocessing: bool = True) -> None:
    """
    UPDATE: Aktualizuje istniejącą bazę danych nowymi danymi z raw (incremental update).

    Używa logiki UPSERT - dodaje nowe rekordy i aktualizuje istniejące.
    Jeśli baza jest pusta, automatycznie uruchamia BUILD.

    Args:
        cfg: Konfiguracja.
        run_preprocessing: Czy uruchomić preprocessing po aktualizacji (domyślnie: True).
    """
    LOGGER.info("Etap: UPDATE - przyrostowa aktualizacja bazy danych")

    # Inicjalizacja połączenia
    db_path = get_path_from_config("database", "path", cfg)
    init_database(db_path)

    # Sprawdzamy czy tabele istnieją
    prices_count = get_table_count("prices")

    if prices_count == 0:
        LOGGER.warning(
            "Tabela 'prices' jest pusta. Automatyczne uruchomienie BUILD..."
        )
        # Uwaga: run_build() wywoła init_database() ponownie, ale to jest OK
        # (Singleton zapewnia że używamy tego samego połączenia)
        run_build(cfg, auto_confirm=True)
        LOGGER.info("BUILD zakończony. Kontynuowanie UPDATE...")
        # Po BUILD sprawdzamy czy dane zostały załadowane
        prices_count = get_table_count("prices")

        if prices_count == 0:
            LOGGER.error("BUILD zakończył się, ale baza nadal pusta. Sprawdź logi.")
            return

    LOGGER.info("Bieżący stan bazy - liczba rekordów w 'prices': %s", prices_count)

    raw_dir = get_path_from_config("paths", "raw", cfg)

    # Aktualizacja danych
    results = update_all_tables(raw_dir)

    # Podsumowanie
    total_updated = sum(results.values())
    LOGGER.info("UPDATE zakończony pomyślnie. Łącznie zaktualizowano %s rekordów", total_updated)

    # Nowe statystyki
    stats = get_table_stats()
    for table, info in stats.items():
        count = info.get("count", 0)
        date_range = info.get("date_range", (None, None))
        if date_range[0]:
            LOGGER.info("Tabela '%s': %s rekordów, zakres dat: %s - %s", table, count, date_range[0], date_range[1])
        else:
            LOGGER.info("Tabela '%s': %s rekordów", table, count)

    # PREPROCESSING: normalizacja danych do staging
    if run_preprocessing:
        run_preprocess(cfg)


def run_preprocess(cfg: dict) -> None:
    """
    PREPROCESSING: Przetwarza surowe dane z main do staging (normalizacja, feature engineering).

    Obecnie obsługiwane moduły:
    - normalize_macro: Normalizuje dane makro (YoY + z-score) i zapisuje do staging.macro_normalized

    Args:
        cfg: Konfiguracja.
    """
    LOGGER.info("Etap: PREPROCESSING - przetwarzanie danych do staging")

    # Inicjalizacja połączenia
    db_path = get_path_from_config("database", "path", cfg)
    init_database(db_path)

    # Sprawdź czy tabele main istnieją
    macro_count = get_table_count("macro")

    if macro_count == 0:
        LOGGER.warning("Tabela main.macro jest pusta - preprocessing pominięty")
        LOGGER.warning("Najpierw uruchom BUILD lub UPDATE aby załadować dane")
        return

    # KROK 1: Normalizacja danych makro do staging
    LOGGER.info("Rozpoczynam normalizację danych makro (%d rekordów w main.macro)", macro_count)
    try:
        processed_count = process_macro_to_staging()
        LOGGER.info("Normalizacja danych makro zakończona: %d rekordów w staging.macro_normalized", processed_count)
    except Exception as exc:
        LOGGER.error("Błąd podczas normalizacji danych makro: %s", exc)
        raise

    # KROK 2: Tworzenie schema CLEANED z wyczyszczonymi danymi
    LOGGER.info("Rozpoczynam budowanie schema CLEANED")
    try:
        from src.database.schema_cleaned import build_all_cleaned_tables
        results = build_all_cleaned_tables()
        total_cleaned = sum(results.values())
        LOGGER.info("Schema CLEANED utworzony pomyślnie: %d rekordów w %d tabelach", total_cleaned, len(results))
    except Exception as exc:
        LOGGER.error("Błąd podczas tworzenia schema cleaned: %s", exc)
        raise

    # KROK 3: Tworzenie staging.master_dataset do trenowania modelu ML (ogólny)
    LOGGER.info("Rozpoczynam budowanie staging.master_dataset (zbiór treningowy ML - ogólny)")
    try:
        from src.preprocessing.create_master_dataset import create_master_dataset
        master_count = create_master_dataset()
        LOGGER.info("staging.master_dataset utworzony pomyślnie: %d rekordów", master_count)
    except Exception as exc:
        LOGGER.error("Błąd podczas tworzenia master_dataset: %s", exc)
        raise

    # KROK 4: Tworzenie staging.bankruptcy_dataset do przewidywania bankructwa
    LOGGER.info("Rozpoczynam budowanie staging.bankruptcy_dataset (model przewidywania bankructwa)")
    try:
        from src.preprocessing.create_bankruptcy_dataset import create_bankruptcy_dataset
        bankruptcy_count = create_bankruptcy_dataset()
        LOGGER.info("staging.bankruptcy_dataset utworzony pomyślnie: %d rekordów", bankruptcy_count)
    except Exception as exc:
        LOGGER.error("Błąd podczas tworzenia bankruptcy_dataset: %s", exc)
        raise

    LOGGER.info("PREPROCESSING zakończony pomyślnie")


def run_analysis(cfg: dict, tickers: list[str]) -> None:
    """
    Przygotowuje podstawowe tabele i wykresy analityczne z bazy danych.

    Używa tabel cleaned do analizy.
    """
    LOGGER.info("Etap: analiza wynikowa (z tabel cleaned)")

    # Inicjalizacja połączenia
    db_path = get_path_from_config("database", "path", cfg)
    init_database(db_path)

    reports_dir = get_path_from_config("paths", "reports", cfg)
    tables_dir = reports_dir / "tables"
    figures_dir = reports_dir / "figures"
    ensure_dir(tables_dir)
    ensure_dir(figures_dir)

    # Pobieramy dane cenowe z cleaned.prices_cleaned
    try:
        df = get_prices_cleaned()

        if df.empty:
            LOGGER.warning("Tabela cleaned.prices_cleaned jest pusta - analiza pominięta")
            LOGGER.warning("Uruchom najpierw preprocessing (--preprocess) aby utworzyć tabele cleaned")
            return

        LOGGER.info("Pobrano %s rekordów cen z cleaned.prices_cleaned", len(df))

        # Statystyki opisowe dla cen
        stats = df[["open", "high", "low", "close", "volume"]].describe().reset_index().rename(columns={"index": "metric"})
        write_csv(stats, tables_dir / "descriptive_stats.csv", index=False)
        LOGGER.info("Zapisano statystyki opisowe")

        # Macierz korelacji dla danych cenowych
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            corr = df[numeric_cols].corr().reset_index().rename(columns={"index": "feature"})
            write_csv(corr, tables_dir / "correlations.csv", index=False)
            LOGGER.info("Zapisano macierz korelacji")

        # Wykres cen dla wybranych tickerów
        if {"ticker", "date", "close"}.issubset(df.columns):
            top_tickers = tickers[:3] if tickers else df["ticker"].dropna().unique()[:3]
            if len(top_tickers) > 0:
                try:
                    # Przygotowujemy dane dla plot_price_history
                    plot_df = df[["ticker", "date", "close"]].rename(columns={
                        "ticker": "Ticker",
                        "date": "Date",
                        "close": "Close"
                    })
                    plot_price_history(plot_df, top_tickers, figures_dir / "price_history.png")
                    LOGGER.info("Zapisano wykres historii cen dla %d tickerów", len(top_tickers))
                except Exception as exc:
                    LOGGER.warning("Nie udało się utworzyć wykresu: %s", exc)

    except Exception as exc:
        LOGGER.error("Błąd podczas analizy: %s", exc)
        raise
