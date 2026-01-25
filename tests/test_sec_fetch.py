"""Diagnostyka pobierania danych z SEC EDGAR.

Skrypt uruchamia istniejace funkcje fetch_* oraz dodatkowe testy,
aby zidentyfikowac potencjalne problemy (np. brak naglowka User-Agent
lub niezgodna sygnatura safe_get)."""
from __future__ import annotations

import argparse
import inspect
import logging
import os
from pathlib import Path
from typing import Iterable

import requests

# Zapewniamy importy pakietu src niezaleznie od katalogu roboczego.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from src.data_fetch.fetch_financials import (  # type: ignore  # pylint: disable=wrong-import-position
    COMPANYFACTS_URL,
    HEADERS,
    fetch_financial_reports,
    get_cik_map,
)
from src.data_fetch.fetch_shares import fetch_shares_outstanding  # type: ignore  # pylint: disable=wrong-import-position
from src.utils.api_helpers import safe_get  # type: ignore  # pylint: disable=wrong-import-position
from src.utils.log_helpers import FetchResult, summarize_counts  # type: ignore  # pylint: disable=wrong-import-position
from src.utils.logging_utils import setup_logging  # type: ignore  # pylint: disable=wrong-import-position

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnostyka pobierania danych z SEC EDGAR.")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=DEFAULT_TICKERS,
        help="Lista tickerow do przetestowania (domyslnie: AAPL MSFT GOOGL).",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Pomin rzeczywiste wywolania fetch_* i wykonaj tylko diagnostyke polaczenia.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Timeout w sekundach dla bezposrednich zapytan do SEC.",
    )
    return parser.parse_args()


def run_fetch_checks(tickers: Iterable[str]) -> None:
    """Uruchamia fetch_financial_reports oraz fetch_shares_outstanding i wypisuje wyniki."""
    logger = logging.getLogger("pipeline")

    results: dict[str, FetchResult] = {
        "financials": fetch_financial_reports(tickers),
        "shares": fetch_shares_outstanding(tickers),
    }

    for name, result in results.items():
        logger.info("[test_sec] %s -> %s", name, summarize_counts(result))
        if result.saved_paths:
            logger.info("[test_sec] %s zapisane pliki: %s", name, result.saved_paths)
        if result.note:
            logger.info("[test_sec] %s note: %s", name, result.note)


def diagnose_ticker(ticker: str, cik_map: dict[str, str], timeout: float) -> None:
    """Sprawdza dostepnosc danych EDGAR dla konkretnego tickera."""
    logger = logging.getLogger("pipeline")
    normalized = ticker.upper()
    cik = cik_map.get(normalized)
    if not cik:
        logger.warning("[test_sec] %s -> CIK nieznany w mappingu SEC.", normalized)
        return

    url = COMPANYFACTS_URL.format(cik=cik)
    accepts_headers = "headers" in inspect.signature(safe_get).parameters

    if not accepts_headers:
        logger.warning(
            "[test_sec] safe_get nie przyjmuje parametru headers. Wywolania w fetch_* zakoncza sie TypeError."
        )
        try:
            safe_get(url, headers=HEADERS)  # type: ignore[arg-type]
        except TypeError as exc:
            logger.error("[test_sec] Przyklad bledu safe_get dla %s: %s", normalized, exc)
        except Exception as exc:  # pragma: no cover
            logger.error("[test_sec] Inny blad safe_get dla %s: %s", normalized, exc)
    else:
        try:
            safe_get(url, headers=HEADERS)  # type: ignore[arg-type]
            logger.info("[test_sec] safe_get z naglowkiem dziala dla %s", normalized)
        except Exception as exc:  # pragma: no cover
            logger.error("[test_sec] safe_get zgasil blad dla %s: %s", normalized, exc)

    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout)
        response.raise_for_status()
        sample = response.text[:200].replace("\n", " ")
        logger.info(
            "[test_sec] bezposrednie zapytanie OK (%s) dla %s, pierwsze znaki odpowiedzi: %s",
            response.status_code,
            normalized,
            sample,
        )
    except Exception as exc:  # pragma: no cover
        logger.error("[test_sec] Blad bezposredniego zapytania dla %s: %s", normalized, exc)


def main() -> None:
    args = parse_args()
    setup_logging()
    logger = logging.getLogger("pipeline")
    logger.info("[test_sec] Diagnostyka rozpoczęta. Tickers=%s", ", ".join(args.tickers))

    if not args.skip_fetch:
        run_fetch_checks(args.tickers)
    else:
        logger.info("[test_sec] Pomijam wywolania fetch_* (tryb --skip-fetch).")

    cik_map = get_cik_map()
    if not cik_map:
        logger.error("[test_sec] Mapowanie CIK jest puste. Sprawdz polaczenie z https://www.sec.gov/files/company_tickers.json")
        return

    logger.info("[test_sec] Mapowanie CIK zawiera %s wpisów.", len(cik_map))
    for ticker in args.tickers:
        diagnose_ticker(ticker, cik_map, timeout=args.timeout)


if __name__ == "__main__":  # pragma: no cover
    main()
