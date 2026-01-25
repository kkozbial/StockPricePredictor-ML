"""Pobiera dodatkowe informacje o spolkach z yfinance."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import yfinance as yf

from ..database.connection import get_connection, table_exists
from ..utils.config_loader import get_path_from_config, load_config
from ..utils.io_helpers import ensure_dir, write_csv
from ..utils.log_helpers import FetchResult, ProgressTracker

LOGGER = logging.getLogger("data_fetch.others")


def _get_last_dividend_date(ticker: str) -> Optional[str]:
    """
    Pobiera ostatnią datę dywidendy z bazy danych dla danego tickera.

    Args:
        ticker: Symbol tickera.

    Returns:
        Data w formacie 'YYYY-MM-DD' lub None jeśli brak danych.
    """
    if not table_exists("dividends"):
        return None

    try:
        conn = get_connection()
        result = conn.execute(
            "SELECT MAX(dividend_date) FROM dividends WHERE ticker = ?",
            [ticker.upper()]
        ).fetchone()

        if result and result[0]:
            return str(result[0])
    except Exception as exc:
        LOGGER.warning("[_get_last_dividend_date] Error querying database for %s: %s", ticker, exc)

    return None


def _merge_dividend_records(existing: list[dict], new: list[dict]) -> list[dict]:
    """
    Łączy istniejące i nowe rekordy dywidend, usuwając duplikaty po dacie.

    Args:
        existing: Lista istniejących rekordów.
        new: Lista nowych rekordów.

    Returns:
        Połączona lista bez duplikatów.
    """
    # Tworzymy set z unikalnych krotek (date, dividend) dla istniejących rekordów
    existing_set = set()
    for record in existing:
        key = (
            record.get("date"),
            record.get("dividend")
        )
        existing_set.add(key)

    # Dodajemy tylko nowe rekordy, których nie ma w existing_set
    combined = list(existing)
    for record in new:
        key = (
            record.get("date"),
            record.get("dividend")
        )
        if key not in existing_set:
            combined.append(record)
            existing_set.add(key)

    return combined


def fetch_dividends(
    tickers: Iterable[str],
    output_dir: Optional[Path] = None,
) -> FetchResult:
    """
    Pobiera dywidendy dla podanych tickerow i zapisuje każdy ticker jako plik JSON.

    Obsługuje tryb przyrostowy w dwóch wersjach:
    1. Jeśli ticker istnieje w bazie danych - pobiera tylko od ostatniej daty (SZYBKIE ~10 min dla 10k tickerów)
    2. Jeśli ticker nie ma danych w bazie - pobiera pełną historię (WOLNE ~90 min dla 10k tickerów)

    Merge z istniejącymi plikami JSON zapewnia deduplikację.
    """
    result = FetchResult()
    cfg = load_config()
    base_dir = Path(output_dir) if output_dir else get_path_from_config("paths", "raw", cfg) / "dividends"
    ensure_dir(base_dir)
    tickers_list = [ticker for ticker in tickers if ticker]

    if not tickers_list:
        LOGGER.warning("[fetch_dividends] No tickers provided.")
        result.note = "no tickers"
        return result

    LOGGER.info("[fetch_dividends] Starting dividends download for %s tickers", len(tickers_list))
    progress = ProgressTracker(len(tickers_list), "dividends")

    for ticker in tickers_list:
        normalized = ticker.upper()

        # Wczytaj istniejące dane jeśli plik istnieje (tryb przyrostowy)
        existing_records = []
        output_path = base_dir / f"{normalized}_dividends.json"
        if output_path.exists():
            try:
                existing_data = json.loads(output_path.read_text(encoding="utf-8"))
                existing_records = existing_data.get("dividends", [])
            except Exception:
                existing_records = []

        # Sprawdź ostatnią datę w bazie danych (optymalizacja)
        last_date_str = _get_last_dividend_date(normalized)

        if last_date_str:
            # Pobierz tylko nowe dywidendy od ostatniej daty (z małym buforem)
            last_date = datetime.strptime(last_date_str, "%Y-%m-%d")
            start_date = (last_date - timedelta(days=7)).strftime("%Y-%m-%d")

            try:
                # Użyj history() z parametrem start dla przyrostowego pobierania
                hist = yf.Ticker(ticker).history(start=start_date, actions=True)
                dividends = hist["Dividends"] if "Dividends" in hist.columns else pd.Series(dtype=float)
            except Exception:  # pragma: no cover - sieciowe
                result.errors += 1
                progress.update(error=True)
                continue
        else:
            # Brak danych w bazie - pobierz pełną historię
            try:
                dividends = yf.Ticker(ticker).dividends
            except Exception:  # pragma: no cover - sieciowe
                result.errors += 1
                progress.update(error=True)
                continue

        if dividends is None or dividends.empty:
            result.skipped += 1
            progress.update(skipped=True)
            continue

        # Konwertuj Series na listę słowników
        new_records = []
        for date, value in dividends.items():
            new_records.append({
                "date": date.strftime("%Y-%m-%d"),
                "dividend": float(value)
            })

        if not new_records:
            result.skipped += 1
            progress.update(skipped=True)
            continue

        # W trybie przyrostowym: merge istniejących i nowych danych
        if existing_records:
            combined_records = _merge_dividend_records(existing_records, new_records)
            added_count = len(combined_records) - len(existing_records)

            if added_count == 0:
                result.skipped += 1
                progress.update(skipped=True)
                continue

            final_records = combined_records
        else:
            # Tryb pełny lub brak istniejących danych
            final_records = new_records

        # Zapisz jako JSON
        output_payload = {"ticker": normalized, "dividends": final_records}
        output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
        result.updated += 1
        result.add_path(output_path)
        progress.update(updated=True)

    progress.finish()

    if result.updated == 0 and result.errors == 0 and result.skipped > 0:
        result.note = "no updates"
    return result


def fetch_sectors(
    tickers: Iterable[str],
    output_dir: Optional[Path] = None,
) -> FetchResult:
    """Pobiera informacje o sektorach i zapisuje je w CSV."""
    result = FetchResult()
    cfg = load_config()
    base_dir = Path(output_dir) if output_dir else get_path_from_config("paths", "raw", cfg) / "others"
    ensure_dir(base_dir)
    tickers_list = [ticker for ticker in tickers if ticker]

    if not tickers_list:
        LOGGER.warning("[fetch_sectors] No tickers provided.")
        result.note = "no tickers"
        return result

    sector_rows: list[dict[str, str]] = []
    for ticker in tickers_list:
        LOGGER.info("[fetch_sectors] Fetching sector info for %s", ticker)
        try:
            info = yf.Ticker(ticker).get_info()
        except Exception as exc:  # pragma: no cover - sieciowe
            LOGGER.warning("[fetch_sectors] Error for %s: %s", ticker, exc)
            result.errors += 1
            continue

        sector = info.get("sector")
        industry = info.get("industry")
        if sector or industry:
            sector_rows.append({"Ticker": ticker, "Sector": sector or "", "Industry": industry or ""})
            result.updated += 1
        else:
            LOGGER.info("[fetch_sectors] No updates for %s", ticker)
            result.skipped += 1

    if sector_rows:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        df_sector = pd.DataFrame(sector_rows)
        sectors_path = base_dir / f"sectors_{timestamp}.csv"
        write_csv(df_sector, sectors_path, index=False)
        result.add_path(sectors_path)
        LOGGER.info("[fetch_sectors] Saved %s", sectors_path)
    else:
        LOGGER.info("[fetch_sectors] No updates (tickers=%s)", len(tickers_list))

    return result


def fetch_dividends_and_sector(
    tickers: Iterable[str],
    output_dir: Optional[Path] = None,
) -> dict[str, Path]:
    """Zachowuje wsteczna kompatybilnosc, uruchamiajac oba pobrania."""
    outputs: dict[str, Path] = {}
    dividends_result = fetch_dividends(tickers, output_dir)
    if dividends_result.saved_paths:
        outputs["dividends"] = Path(dividends_result.saved_paths[-1])

    sectors_result = fetch_sectors(tickers, output_dir)
    if sectors_result.saved_paths:
        outputs["sectors"] = Path(sectors_result.saved_paths[-1])

    return outputs

