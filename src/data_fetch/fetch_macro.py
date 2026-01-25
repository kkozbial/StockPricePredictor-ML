"""Pobiera szeregi czasowe danych makroekonomicznych."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from ..utils.api_helpers import safe_get
from ..utils.config_loader import get_path_from_config, load_config
from ..utils.io_helpers import ensure_dir, write_csv
from ..utils.log_helpers import FetchResult
from ..utils.incremental_helpers import (
    calculate_incremental_start_date,
    get_last_date_for_series,
    should_use_incremental,
)

LOGGER = logging.getLogger("data_fetch.macro")

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"


def fetch_macro_series(series_ids: Optional[Iterable[str]] = None, output_dir: Optional[Path] = None) -> FetchResult:
    """
    Pobiera wskaźniki makroekonomiczne z API FRED.

    Obsługuje tryb przyrostowy (incremental) - jeśli włączony w konfiguracji,
    pobiera tylko nowe dane od ostatniej daty w bazie i łączy z istniejącymi.
    """
    result = FetchResult()
    cfg = load_config()
    fred_key = cfg["api"]["fred_api_key"]
    if fred_key.startswith("YOUR_"):
        LOGGER.warning("[fetch_macro] Skipped: missing FRED API key.")
        result.note = "missing api key"
        result.skipped = 1
        return result

    series = list(series_ids) if series_ids else cfg["fetch"].get("macro_series", [])
    target_dir = Path(output_dir) if output_dir else get_path_from_config("paths", "raw", cfg) / "macro"
    ensure_dir(target_dir)

    if not series:
        LOGGER.info("[fetch_macro] No macro series configured.")
        result.note = "no series"
        return result

    # Sprawdź czy używać pobierania przyrostowego
    use_incremental = should_use_incremental(cfg, "macro")

    for series_id in series:
        out_file = target_dir / f"macro_{series_id}.csv"

        # Dla trybu przyrostowego, sprawdź ostatnią datę
        observation_start = None
        existing_df = None

        if use_incremental and out_file.exists():
            try:
                existing_df = pd.read_csv(out_file)
                if not existing_df.empty and "date" in existing_df.columns:
                    # Znajdź ostatnią datę w pliku
                    last_date = existing_df["date"].max()
                    # Oblicz datę początkową dla pobierania
                    observation_start = calculate_incremental_start_date(
                        last_date=last_date,
                        default_start="1900-01-01",  # bardzo stara data dla FRED
                        overlap_days=7  # 7 dni nakładki dla pewności
                    )
                    LOGGER.info(
                        "[fetch_macro] Tryb przyrostowy dla %s: ostatnia data %s, pobieranie od %s",
                        series_id, last_date, observation_start
                    )
            except Exception as exc:
                LOGGER.warning("[fetch_macro] Błąd wczytywania istniejących danych dla %s: %s", series_id, exc)
                existing_df = None
                observation_start = None

        # Parametry dla FRED API
        params = {"series_id": series_id, "api_key": fred_key, "file_type": "json"}
        if observation_start:
            params["observation_start"] = observation_start

        try:
            if observation_start:
                LOGGER.info("[fetch_macro] Fetching series %s from %s", series_id, observation_start)
            else:
                LOGGER.info("[fetch_macro] Fetching full series %s", series_id)

            response = safe_get(FRED_URL, params=params)
            data = response.json().get("observations", [])
        except Exception as exc:  # pragma: no cover - sieciowe
            LOGGER.exception("[fetch_macro] Error for series %s: %s", series_id, exc)
            result.errors += 1
            continue

        new_df = pd.DataFrame(data)
        if new_df.empty:
            LOGGER.info("[fetch_macro] No new data for %s", series_id)
            result.skipped += 1
            continue

        # W trybie przyrostowym: połącz z istniejącymi danymi
        if use_incremental and existing_df is not None and not existing_df.empty:
            # Usuń duplikaty po dacie (nowe dane nadpisują stare)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            # Usuń duplikaty zachowując ostatnie wpisy
            combined_df = combined_df.drop_duplicates(subset=["date"], keep="last")
            # Sortuj po dacie
            combined_df = combined_df.sort_values("date").reset_index(drop=True)

            new_rows = len(combined_df) - len(existing_df)
            if new_rows <= 0:
                LOGGER.info("[fetch_macro] No new records for %s after merge", series_id)
                result.skipped += 1
                continue

            LOGGER.info("[fetch_macro] Added %d new rows to %s (total: %d)", new_rows, series_id, len(combined_df))
            final_df = combined_df
        else:
            final_df = new_df

        write_csv(final_df, out_file, index=False)
        LOGGER.info("[fetch_macro] Saved %s with %d rows", out_file, len(final_df))
        result.updated += 1
        result.add_path(out_file)

    if result.updated == 0 and result.errors == 0 and result.skipped == 0:
        result.note = "no series processed"
    return result
