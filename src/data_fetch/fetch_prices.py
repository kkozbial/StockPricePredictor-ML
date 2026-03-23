"""Pobiera dane cenowe akcji i zapisuje je w data/raw/prices."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import yfinance as yf

from ..utils.config_loader import get_path_from_config, load_config
from ..utils.io_helpers import ensure_dir, write_csv
from ..utils.log_helpers import FetchResult
from ..utils.incremental_helpers import (
    calculate_incremental_start_date,
    get_last_date_for_ticker,
    should_use_incremental,
)

LOGGER = logging.getLogger("data_fetch.prices")


def fetch_prices(
    tickers: Iterable[str],
    start: str,
    interval: str,
    output_dir: Optional[Path] = None,
) -> FetchResult:
    """
    Pobiera dane OHLCV skorygowane o splity i zapisuje do pliku CSV.

    Obsługuje tryb przyrostowy (incremental) - jeśli włączony w konfiguracji,
    pobiera tylko nowe dane od ostatniej daty w bazie dla każdego tickera.

    Parametry
    ---------
    tickers: lista tickerow do pobrania.
    start: data poczatkowa w formacie ISO (używana gdy brak danych lub tryb pełny).
    interval: interwal yfinance (np. 1mo).
    output_dir: katalog docelowy; domyslnie z konfiguracji.
    """
    result = FetchResult()
    cfg = load_config()
    target_dir = Path(output_dir) if output_dir else get_path_from_config("paths", "raw", cfg) / "prices"
    ensure_dir(target_dir)

    tickers_list = [ticker for ticker in tickers if ticker]
    if not tickers_list:
        LOGGER.warning("[fetch_prices] No tickers provided.")
        result.note = "no tickers"
        return result

    # Pomiń delisted tickery, dla których mamy już dane do daty delistingu
    tickers_list = _filter_out_covered_delisted(tickers_list)

    # Sprawdź czy używać pobierania przyrostowego
    use_incremental = should_use_incremental(cfg, "prices")

    # Dla trybu przyrostowego, grupujemy tickery po dacie początkowej
    if use_incremental:
        # Grupujemy tickery według ich ostatniej daty w bazie
        ticker_groups = _group_tickers_by_start_date(tickers_list, start)
        LOGGER.info(
            "[fetch_prices] Tryb przyrostowy: %d grup tickerów z różnymi datami początkowymi",
            len(ticker_groups)
        )
    else:
        # Tryb pełny - wszystkie tickery z tą samą datą
        ticker_groups = [(start, tickers_list)]
        LOGGER.info("[fetch_prices] Tryb pełny: pobieranie od %s dla %d tickerów", start, len(tickers_list))

    # Pobierz dane dla każdej grupy
    all_data_frames = []
    for group_start, group_tickers in ticker_groups:
        LOGGER.info("[fetch_prices] Pobieranie %d tickerów od %s: %s",
                   len(group_tickers), group_start, ", ".join(group_tickers[:10]) +
                   (f" ... (+{len(group_tickers)-10} więcej)" if len(group_tickers) > 10 else ""))

        try:
            data = yf.download(
                tickers=group_tickers,
                start=group_start,
                interval=interval,
                auto_adjust=False,
                group_by="ticker",
                progress=False,
            )
        except Exception as exc:  # pragma: no cover - network call
            LOGGER.exception("[fetch_prices] Error downloading prices: %s", exc)
            result.errors += len(group_tickers)
            continue

        if data.empty:
            LOGGER.warning("[fetch_prices] No data returned for group starting %s", group_start)
            result.skipped += len(group_tickers)
            continue

        df_long = _flatten_columns(data, group_tickers)
        if not df_long.empty:
            all_data_frames.append(df_long)

    # Połącz wszystkie dataframe'y
    if not all_data_frames:
        LOGGER.warning("[fetch_prices] No data downloaded for any ticker group")
        result.skipped += len(tickers_list)
        return result

    df_combined = pd.concat(all_data_frames, ignore_index=True)

    if df_combined.empty:
        LOGGER.warning("[fetch_prices] Combined dataset is empty.")
        result.skipped += len(tickers_list)
        return result

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_file = target_dir / f"prices_{timestamp}.csv"
    write_csv(df_combined, out_file, index=False)
    unique_tickers = df_combined.get("Ticker")
    result.updated += len(unique_tickers.dropna().unique()) if unique_tickers is not None else len(tickers_list)
    result.add_path(out_file)
    LOGGER.info("[fetch_prices] Saved %s with %d rows", out_file, len(df_combined))
    return result


def _group_tickers_by_start_date(tickers: list[str], default_start: str) -> list[tuple[str, list[str]]]:
    """
    Grupuje tickery według ich daty początkowej dla pobierania przyrostowego.

    Args:
        tickers: Lista tickerów.
        default_start: Domyślna data początkowa dla tickerów bez danych.

    Returns:
        Lista krotek (start_date, [tickers]) zgrupowanych po dacie początkowej.
    """
    from collections import defaultdict

    # Słownik: data -> lista tickerów
    date_groups = defaultdict(list)

    for ticker in tickers:
        # Pobierz ostatnią datę dla tickera
        last_date = get_last_date_for_ticker("prices", ticker, date_column="date")

        # Oblicz datę początkową
        start_date = calculate_incremental_start_date(
            last_date=last_date,
            default_start=default_start,
            overlap_days=1,  # 1 dzień nakładki dla pewności
            ticker=ticker
        )

        date_groups[start_date].append(ticker)

    # Konwertuj na listę krotek
    result = [(date, tickers_list) for date, tickers_list in date_groups.items()]

    return result


def _filter_out_covered_delisted(tickers: list[str]) -> list[str]:
    """
    Pomija tickery delisted, dla których mamy już ceny do daty delistingu.

    Logika: jeśli ticker ma status DELISTED/DEREGISTERED w company_status
    i ostatnia data cen w bazie >= delisting_date → nie ma czego pobierać.
    """
    try:
        from ..database.connection import get_connection, table_exists
        if not table_exists("company_status") or not table_exists("prices"):
            return tickers

        conn = get_connection()

        # Pobierz delisted tickery z datą delistingu
        delisted = conn.execute("""
            SELECT cs.ticker, cs.delisting_date
            FROM company_status cs
            WHERE cs.status IN ('DELISTED', 'DEREGISTERED')
              AND cs.delisting_date IS NOT NULL
        """).fetchall()

        if not delisted:
            return tickers

        # Sprawdź które z nich mamy już pokryte w prices
        delisted_covered = set()
        for ticker, delisting_date in delisted:
            last_price_date = conn.execute(
                "SELECT MAX(date) FROM prices WHERE ticker = ?", [ticker]
            ).fetchone()[0]
            if last_price_date is not None and last_price_date >= delisting_date:
                delisted_covered.add(ticker)

        if delisted_covered:
            LOGGER.info(
                "[fetch_prices] Pomijam %d delisted tickerów z pełnymi danymi (np. %s)",
                len(delisted_covered),
                ", ".join(list(delisted_covered)[:5])
            )

        return [t for t in tickers if t not in delisted_covered]

    except Exception as exc:
        LOGGER.warning("[fetch_prices] Nie udało się odfiltrować delisted tickerów: %s", exc)
        return tickers


def _flatten_columns(data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Konwertuje wielopoziomowe kolumny yfinance i zwraca wszystkie kolumny OHLCV + Volume."""

    # 1. Logika spłaszczania (bez zmian)
    if isinstance(data.columns, pd.MultiIndex):
        # Dla wielu tickerów: Ticker jest w kolumnach, przenosimy go do wierszy (stack)
        flattened = data.stack(0).rename_axis(["Date", "Ticker"]).reset_index()
    else:
        # Dla pojedynczego tickera
        frame = data.reset_index()
        if "Date" not in frame.columns:
            frame = frame.rename(columns={frame.columns[0]: "Date"})
        frame.insert(1, "Ticker", tickers[0])
        flattened = frame

    # 2. FILTROWANIE KOLUMN
    # Wybieramy wszystkie kolumny OHLCV + Volume
    wanted_cols = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]
    available_cols = [c for c in wanted_cols if c in flattened.columns]

    return flattened[available_cols]
