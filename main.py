"""Glowny punkt wejsciowy pipeline'u projekt_dane."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from src.pipeline.runner import load_tickers, run_analysis, run_build, run_fetch, run_preprocess, run_update
from src.utils.config_loader import load_config
from src.utils.logging_utils import setup_logging

LOGGER = logging.getLogger("pipeline")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parsuje argumenty linii polecen."""
    parser = argparse.ArgumentParser(
        description="Stock Market ML Pipeline - Modulowe uruchamianie pipeline'u danych z bazą DuckDB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady użycia:
  python main.py                              # Tryb interaktywny (domyślnie: fetch + update)
  python main.py --steps fetch update         # Pobierz dane i zaktualizuj bazę
  python main.py --steps preprocess           # Tylko preprocessing (cleaned + staging.master_dataset)
  python main.py --steps fetch --fetch-modules prices financials  # Pobierz tylko ceny i finanse
  python main.py --fetch-metadata --steps fetch  # Pobierz metadane SEC (pierwsze uruchomienie)

Schemat przepływu danych:
  FETCH → data/raw/ → BUILD/UPDATE → main.* → PREPROCESS → cleaned.* + staging.* → ANALYSIS
        """
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["fetch", "build", "update", "preprocess", "analysis"],
        default=None,
        help="Lista etapów: fetch=pobierz dane, build=zbuduj bazę od zera, update=zaktualizuj bazę (auto preprocessing), preprocess=przetwórz do cleaned/staging, analysis=raporty.",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        help="Nadpisuje liste tickerow z pliku config/tickers_list.csv.",
    )
    parser.add_argument(
        "--fetch-metadata",
        action="store_true",
        help="Pobierz metadane SEC przed pobieraniem danych finansowych (zajmuje ~30-40 min przy pierwszym uruchomieniu).",
    )
    parser.add_argument(
        "--fetch-modules",
        nargs="+",
        choices=["prices", "financials", "shares", "dividends", "sectors", "macro", "all"],
        default=None,
        help="Lista modułów do pobrania podczas etapu fetch. Domyślnie: pytaj użytkownika. Użyj 'all' aby pobrać wszystko.",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Wyłącz automatyczne uruchamianie preprocessing po UPDATE (domyślnie: uruchamia się automatycznie).",
    )
    return parser.parse_args(argv)


def load_default_tickers() -> list[str]:
    """Wczytuje liste tickerow z pliku konfiguracyjnego."""
    config_dir = Path(__file__).resolve().parent / "config"
    return load_tickers(config_dir)


def prompt_user_for_steps() -> list[str]:
    """Pyta użytkownika które etapy pipeline uruchomić."""
    print("\n" + "=" * 60)
    print("=== WYBÓR ETAPÓW PIPELINE ===")
    print("=" * 60)
    print("\nDostępne etapy:")
    print("\n  1. fetch      - Pobierz dane z API (prices, financials, shares, dividends, macro)")
    print("                  └─> zapisuje do data/raw/")
    print("\n  2. build      - Zbuduj bazę danych od zera")
    print("                  ├─> tworzy schema: main (surowe dane)")
    print("                  └─> ładuje dane z data/raw/ do DuckDB")
    print("\n  3. update     - Zaktualizuj istniejącą bazę")
    print("                  ├─> UPSERT danych z data/raw/ do main")
    print("                  └─> automatycznie uruchamia PREPROCESSING")
    print("\n  4. preprocess - Przetwórz dane (normalizacja + feature engineering)")
    print("                  ├─> tworzy schema: staging (dane znormalizowane)")
    print("                  ├─> tworzy schema: cleaned (dane wyczyszczone)")
    print("                  └─> staging.master_dataset (gotowe do ML)")
    print("\n  5. analysis   - Uruchom analizy i generuj raporty")
    print("                  └─> używa tabel cleaned.*")
    print("\n" + "-" * 60)
    print("UWAGA: UPDATE automatycznie uruchamia PREPROCESSING")
    print("Domyślnie: fetch + update")
    print("=" * 60)

    response = input("\nWybierz etapy (oddziel spacją, np. 'fetch update') lub naciśnij Enter dla domyślnych: ").strip()

    if not response:
        return ["fetch", "update"]

    steps = response.split()
    valid_steps = ["fetch", "build", "update", "preprocess", "analysis"]
    selected = [s for s in steps if s in valid_steps]

    if not selected:
        LOGGER.warning("Nie wybrano prawidłowych etapów, używam domyślnych: fetch update")
        return ["fetch", "update"]

    return selected


def prompt_user_for_fetch_modules() -> list[str]:
    """Pyta użytkownika które moduły fetch uruchomić."""
    print("\n" + "=" * 60)
    print("=== WYBÓR MODUŁÓW DO POBRANIA ===")
    print("=" * 60)
    print("\nDostępne moduły:")
    print("\n  1. prices      - Ceny akcji (Yahoo Finance)")
    print("                   ├─> OHLCV dzienne")
    print("                   └─> Czas: ~60 min dla ~9000 tickerów")
    print("\n  2. financials  - Dane finansowe (SEC EDGAR + BiznesRadar)")
    print("                   ├─> Bilanse, rachunki zysków/strat, przepływy")
    print("                   └─> Czas: ~20 min")
    print("\n  3. shares      - Liczba akcji (SEC EDGAR)")
    print("                   ├─> Shares outstanding dla obliczenia market cap")
    print("                   └─> Czas: ~20 min")
    print("\n  4. dividends   - Dywidendy (Yahoo Finance)")
    print("                   ├─> Historia wypłat dywidend")
    print("                   └─> Czas: ~60 min")
    print("\n  5. sectors     - Sektory i branże (Yahoo Finance, OPCJONALNE)")
    print("                   ├─> Klasyfikacja sektorowa")
    print("                   └─> Czas: ~40 min")
    print("\n  6. macro       - Dane makroekonomiczne (FRED + DBnomics)")
    print("                   ├─> GDP, CPI, stopy procentowe, bezrobocie")
    print("                   └─> Czas: ~1 min")
    print("\n  7. all         - Wszystkie moduły powyżej")
    print("\n" + "-" * 60)
    print("UWAGA: 'metadata' pobierane przez flagę --fetch-metadata")
    print("Domyślnie: prices financials shares dividends macro (bez sectors)")
    print("=" * 60)

    response = input("\nWybierz moduły (oddziel spacją) lub naciśnij Enter dla domyślnych: ").strip()

    if not response:
        return ["prices", "financials", "shares", "dividends", "macro"]

    if response == "all":
        return ["prices", "financials", "shares", "dividends", "sectors", "macro"]

    modules = response.split()
    valid_modules = ["prices", "financials", "shares", "dividends", "sectors", "macro"]
    selected = [m for m in modules if m in valid_modules]

    if not selected:
        LOGGER.warning("Nie wybrano prawidłowych modułów, używam domyślnych")
        return ["prices", "financials", "shares", "dividends", "macro"]

    return selected


def main(argv: Sequence[str] | None = None) -> None:
    """Uruchamia wybrane etapy pipeline'u."""
    args = parse_args(argv)
    setup_logging()
    cfg = load_config()

    # Jeśli nie podano steps, zapytaj użytkownika
    if args.steps is None:
        steps = prompt_user_for_steps()
    else:
        steps = args.steps

    # Jeśli fetch w steps i nie podano fetch_modules, zapytaj użytkownika
    fetch_modules = None
    if "fetch" in steps:
        if args.fetch_modules is None:
            fetch_modules = prompt_user_for_fetch_modules()
        else:
            if "all" in args.fetch_modules:
                fetch_modules = ["prices", "financials", "shares", "dividends", "sectors", "macro"]
            else:
                fetch_modules = args.fetch_modules

    # Jeśli użytkownik chce pobrać metadane, tickers będą wczytane automatycznie w run_fetch
    if args.fetch_metadata:
        tickers = args.tickers if args.tickers else []
        LOGGER.info("Wybrane etapy: %s", ", ".join(steps))
        LOGGER.info("Tryb pobierania metadanych SEC - tickery zostaną automatycznie przefiltrowane")
    else:
        tickers = args.tickers if args.tickers else load_default_tickers()
        LOGGER.info("Wybrane etapy: %s", ", ".join(steps))
        LOGGER.info("Liczba tickerow w uzyciu: %s", len(tickers))

    if fetch_modules:
        LOGGER.info("Wybrane moduły fetch: %s", ", ".join(fetch_modules))

    if "fetch" in steps:
        run_fetch(cfg, tickers, fetch_metadata=args.fetch_metadata, fetch_modules=fetch_modules)
    if "build" in steps:
        run_build(cfg)
    if "update" in steps:
        # UPDATE automatycznie uruchamia preprocessing, chyba że użytkownik użył --no-preprocess
        run_preprocessing = not args.no_preprocess
        run_update(cfg, run_preprocessing=run_preprocessing)
    if "preprocess" in steps:
        # Możliwość ręcznego uruchomienia preprocessing
        run_preprocess(cfg)
    if "analysis" in steps:
        run_analysis(cfg, tickers)


if __name__ == "__main__":
    main()
