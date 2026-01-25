"""Pobiera dane cenowe polskich akcji (GPW) z Yahoo Finance i zapisuje je w data/raw/prices."""
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

LOGGER = logging.getLogger("data_fetch.prices_pl")


def fetch_prices_stooq(
    tickers: Iterable[str],
    start: str,
    output_dir: Optional[Path] = None,
) -> FetchResult:
    """
    Pobiera dane OHLCV dla polskich akcji z Yahoo Finance i zapisuje do pliku CSV.

    Obsługuje tryb przyrostowy (incremental) - jeśli włączony w konfiguracji,
    pobiera tylko nowe dane od ostatniej daty w bazie dla każdego tickera.

    Parametry
    ---------
    tickers: lista tickerów GPW (np. PKO.PL, PZU.PL) - konwertowane na format YF (PKO.WA, PZU.WA).
    start: data początkowa w formacie ISO (używana gdy brak danych lub tryb pełny).
    output_dir: katalog docelowy; domyślnie z konfiguracji.
    """
    result = FetchResult()
    cfg = load_config()
    target_dir = Path(output_dir) if output_dir else get_path_from_config("paths", "raw", cfg) / "prices"
    ensure_dir(target_dir)

    tickers_list = [ticker for ticker in tickers if ticker]
    if not tickers_list:
        LOGGER.warning("[fetch_prices_pl] No tickers provided.")
        result.note = "no tickers"
        return result

    # Sprawdź czy używać pobierania przyrostowego
    use_incremental = should_use_incremental(cfg, "prices")

    # Konwertuj tickery do formatu Yahoo Finance (PKO.PL -> PKO.WA)
    yf_tickers = [_convert_ticker_to_yfinance(t) for t in tickers_list]

    # Dla trybu przyrostowego, grupujemy tickery po dacie początkowej
    if use_incremental:
        ticker_groups = _group_tickers_by_start_date_pl(tickers_list, yf_tickers, start)
        LOGGER.info("[fetch_prices_pl] Tryb przyrostowy: %d grup tickerów", len(ticker_groups))
    else:
        # Tryb pełny - wszystkie tickery z tą samą datą
        ticker_groups = [(start, list(zip(tickers_list, yf_tickers)))]
        LOGGER.info("[fetch_prices_pl] Tryb pełny: pobieranie od %s dla %d tickerów", start, len(tickers_list))

    # Pobierz dane dla każdej grupy
    all_data_frames = []
    for group_start, group_ticker_pairs in ticker_groups:
        yf_tickers_group = [yf_t for _, yf_t in group_ticker_pairs]
        original_tickers = [orig_t for orig_t, _ in group_ticker_pairs]

        LOGGER.info("[fetch_prices_pl] Pobieranie %d tickerów od %s: %s",
                   len(yf_tickers_group), group_start, ", ".join(original_tickers[:10]) +
                   (f" ... (+{len(original_tickers)-10} więcej)" if len(original_tickers) > 10 else ""))

        try:
            data = yf.download(
                tickers=yf_tickers_group,
                start=group_start,
                interval="1mo",
                auto_adjust=False,
                group_by="ticker",
                progress=False,
            )
        except Exception as exc:
            LOGGER.exception("[fetch_prices_pl] Error downloading prices: %s", exc)
            result.errors += len(yf_tickers_group)
            continue

        if data.empty:
            LOGGER.warning("[fetch_prices_pl] No data returned for group starting %s", group_start)
            result.skipped += len(yf_tickers_group)
            continue

        # Konwertuj z formatu yfinance na nasz format
        df_long = _flatten_columns_pl(data, original_tickers, yf_tickers_group)
        if not df_long.empty:
            all_data_frames.append(df_long)

    # Połącz wszystkie dataframe'y
    if not all_data_frames:
        LOGGER.warning("[fetch_prices_pl] No data downloaded for any ticker group")
        result.skipped += len(tickers_list)
        return result

    df_combined = pd.concat(all_data_frames, ignore_index=True)

    if df_combined.empty:
        LOGGER.warning("[fetch_prices_pl] Combined dataset is empty.")
        result.skipped += len(tickers_list)
        return result

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_file = target_dir / f"prices_pl_{timestamp}.csv"
    write_csv(df_combined, out_file, index=False)
    unique_tickers = df_combined.get("Ticker")
    result.updated += len(unique_tickers.dropna().unique()) if unique_tickers is not None else len(tickers_list)
    result.add_path(out_file)
    LOGGER.info("[fetch_prices_pl] Saved %s with %d rows", out_file, len(df_combined))
    return result


def _convert_ticker_to_yfinance(ticker: str) -> str:
    """
    Konwertuje ticker GPW do formatu Yahoo Finance.

    Przykłady:
    - PKO.PL -> PKO.WA
    - PZU.PL -> PZU.WA
    - CDR.PL -> CDR.WA
    """
    return ticker.replace(".PL", ".WA")


def _group_tickers_by_start_date_pl(
    original_tickers: list[str],
    yf_tickers: list[str],
    default_start: str
) -> list[tuple[str, list[tuple[str, str]]]]:
    """
    Grupuje tickery według ich daty początkowej dla pobierania przyrostowego.

    Returns:
        Lista krotek (start_date, [(original_ticker, yf_ticker), ...])
    """
    from collections import defaultdict

    date_groups = defaultdict(list)

    for orig_ticker, yf_ticker in zip(original_tickers, yf_tickers):
        last_date = get_last_date_for_ticker("prices", orig_ticker, date_column="date", country="PL")
        start_date = calculate_incremental_start_date(
            last_date=last_date,
            default_start=default_start,
            overlap_days=1,
            ticker=orig_ticker
        )
        date_groups[start_date].append((orig_ticker, yf_ticker))

    return [(date, pairs) for date, pairs in date_groups.items()]


def _flatten_columns_pl(data: pd.DataFrame, original_tickers: list[str], yf_tickers: list[str]) -> pd.DataFrame:
    """
    Konwertuje wielopoziomowe kolumny yfinance dla polskich akcji.

    Mapuje tickery Yahoo Finance (.WA) z powrotem na oryginalne (.PL) i dodaje kolumnę Country='PL'.

    Args:
        data: DataFrame z yfinance (multi-index lub single-index)
        original_tickers: Lista oryginalnych tickerów (PKO.PL, PZU.PL, etc.)
        yf_tickers: Lista tickerów Yahoo Finance (PKO.WA, PZU.WA, etc.)

    Returns:
        DataFrame z kolumnami: Date, Ticker, Country, Open, High, Low, Close, Volume
    """
    # Mapowanie yf_ticker -> original_ticker
    ticker_map = dict(zip(yf_tickers, original_tickers))

    # Logika spłaszczania (podobnie jak w fetch_prices.py)
    if isinstance(data.columns, pd.MultiIndex):
        # Dla wielu tickerów: Ticker jest w kolumnach, przenosimy go do wierszy (stack)
        flattened = data.stack(0).rename_axis(["Date", "Ticker"]).reset_index()
        # Zamień tickery YF na oryginalne
        flattened["Ticker"] = flattened["Ticker"].map(ticker_map)
    else:
        # Dla pojedynczego tickera
        frame = data.reset_index()
        if "Date" not in frame.columns:
            frame = frame.rename(columns={frame.columns[0]: "Date"})
        # Użyj oryginalnego tickera
        frame.insert(1, "Ticker", original_tickers[0])
        flattened = frame

    # Dodaj kolumnę Country
    flattened.insert(2, "Country", "PL")

    # Wybierz i uporządkuj kolumny
    wanted_cols = ["Date", "Ticker", "Country", "Open", "High", "Low", "Close", "Volume"]
    available_cols = [c for c in wanted_cols if c in flattened.columns]

    return flattened[available_cols]
