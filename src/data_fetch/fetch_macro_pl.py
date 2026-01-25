"""Pobiera dane makroekonomiczne dla Polski z Eurostat API."""
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
    should_use_incremental,
)

LOGGER = logging.getLogger("data_fetch.macro_pl")

EUROSTAT_API_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"


def fetch_macro_series_pl(series_ids: Optional[Iterable[str]] = None, output_dir: Optional[Path] = None) -> FetchResult:
    """
    Pobiera wskaźniki makroekonomiczne dla Polski z API Eurostat.

    Obsługuje tryb przyrostowy (incremental) - jeśli włączony w konfiguracji,
    pobiera tylko nowe dane od ostatniej daty w bazie i łączy z istniejącymi.

    Format series_id: dataset_code:dimension1=value1&dimension2=value2
    (np. prc_hicp_manr:coicop=CP00&geo=PL&unit=RCH_A)
    """
    result = FetchResult()
    cfg = load_config()

    series = list(series_ids) if series_ids else cfg["fetch"].get("macro_series_pl", [])
    target_dir = Path(output_dir) if output_dir else get_path_from_config("paths", "raw", cfg) / "macro"
    ensure_dir(target_dir)

    if not series:
        LOGGER.info("[fetch_macro_pl] No macro series configured for Poland.")
        result.note = "no series"
        return result

    # Sprawdź czy używać pobierania przyrostowego
    use_incremental = should_use_incremental(cfg, "macro")

    for series_full_id in series:
        # Parsuj ID serii (format: dataset_code:dimension_filters)
        parts = series_full_id.split(":", 1)
        if len(parts) < 2:
            LOGGER.warning("[fetch_macro_pl] Invalid series format: %s (expected dataset_code:filters)", series_full_id)
            result.errors += 1
            continue

        dataset_code = parts[0]
        dimension_filters = parts[1] if len(parts) > 1 else ""

        # Nazwa pliku - używamy bezpiecznej nazwy z kodu datasetu i filtrów
        series_short_name = f"{dataset_code}_{dimension_filters}".replace("=", "_").replace("&", "_").replace(":", "_")
        out_file = target_dir / f"macro_pl_{series_short_name}.csv"

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
                        default_start="1990-01-01",  # Domyślna data dla Polski
                        overlap_days=30  # 30 dni nakładki dla pewności
                    )
                    LOGGER.info(
                        "[fetch_macro_pl] Tryb przyrostowy dla %s: ostatnia data %s, pobieranie od %s",
                        series_short_name, last_date, observation_start
                    )
            except Exception as exc:
                LOGGER.warning("[fetch_macro_pl] Błąd wczytywania istniejących danych dla %s: %s", series_short_name, exc)
                existing_df = None
                observation_start = None

        try:
            # Pobierz dane z Eurostat
            if observation_start:
                LOGGER.info("[fetch_macro_pl] Fetching series %s from %s", series_full_id, observation_start)
            else:
                LOGGER.info("[fetch_macro_pl] Fetching full series %s", series_full_id)

            df = _fetch_eurostat_series(dataset_code, dimension_filters, observation_start)

            if df is None or df.empty:
                LOGGER.info("[fetch_macro_pl] No new data for %s", series_short_name)
                result.skipped += 1
                continue

        except Exception as exc:
            LOGGER.exception("[fetch_macro_pl] Error for series %s: %s", series_full_id, exc)
            result.errors += 1
            continue

        # W trybie przyrostowym: połącz z istniejącymi danymi
        if use_incremental and existing_df is not None and not existing_df.empty:
            # Upewnij się że kolumna date w istniejących danych jest typu datetime
            existing_df["date"] = pd.to_datetime(existing_df["date"])
            # Usuń duplikaty po dacie (nowe dane nadpisują stare)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            # Usuń duplikaty zachowując ostatnie wpisy
            combined_df = combined_df.drop_duplicates(subset=["date"], keep="last")
            # Sortuj po dacie
            combined_df = combined_df.sort_values("date").reset_index(drop=True)

            new_rows = len(combined_df) - len(existing_df)
            if new_rows <= 0:
                LOGGER.info("[fetch_macro_pl] No new records for %s after merge", series_short_name)
                result.skipped += 1
                continue

            LOGGER.info("[fetch_macro_pl] Added %d new rows to %s (total: %d)", new_rows, series_short_name, len(combined_df))
            final_df = combined_df
        else:
            final_df = df

        write_csv(final_df, out_file, index=False)
        LOGGER.info("[fetch_macro_pl] Saved %s with %d rows", out_file, len(final_df))
        result.updated += 1
        result.add_path(out_file)

    if result.updated == 0 and result.errors == 0 and result.skipped == 0:
        result.note = "no series processed"
    return result


def _fetch_eurostat_series(
    dataset_code: str,
    dimension_filters: str,
    start_date: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    Pobiera pojedynczą serię z Eurostat API.

    Args:
        dataset_code: Kod datasetu Eurostat (np. 'prc_hicp_manr', 'namq_10_gdp').
        dimension_filters: Filtry wymiarów (np. 'coicop=CP00&geo=PL&unit=RCH_A').
        start_date: Opcjonalna data początkowa (YYYY-MM-DD lub YYYY-MM lub YYYY).

    Returns:
        DataFrame z kolumnami: date, value lub None jeśli błąd.
    """
    # Buduj URL dla datasetu
    # API endpoint: https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{dataset_code}
    url = f"{EUROSTAT_API_URL}/{dataset_code}"

    # Parsuj filtry wymiarów jeśli są podane
    params = {"format": "JSON", "lang": "EN"}

    # Dodaj filtry wymiarów do parametrów
    if dimension_filters:
        for filter_pair in dimension_filters.split("&"):
            if "=" in filter_pair:
                key, value = filter_pair.split("=", 1)
                params[key] = value

    # Dodaj filtr daty jeśli podano
    if start_date:
        # Eurostat używa sinceTimePeriod dla filtrowania od daty
        params["sinceTimePeriod"] = start_date

    try:
        response = safe_get(url, params=params)
        data = response.json()

        # Eurostat API zwraca JSON-stat format
        # Struktura: {"version": "2.0", "class": "dataset", "value": {...}, "dimension": {...}}

        if "value" not in data or "dimension" not in data:
            LOGGER.warning("[fetch_macro_pl] Invalid JSON-stat response for %s", dataset_code)
            return None

        # Pobierz wartości
        values = data["value"]

        if not values:
            LOGGER.info("[fetch_macro_pl] No values in response for %s", dataset_code)
            return None

        # Pobierz wymiary
        dimensions = data["dimension"]

        # Znajdź wymiar czasu (zwykle 'time')
        time_dim = None
        for dim_id, dim_data in dimensions.items():
            if dim_id.lower() == "time" or "category" in dim_data and "label" in dim_data["category"]:
                if dim_id.lower() == "time":
                    time_dim = dim_id
                    break

        if not time_dim:
            LOGGER.warning("[fetch_macro_pl] Could not find time dimension for %s", dataset_code)
            return None

        # Pobierz kategorie czasu
        time_categories = dimensions[time_dim]["category"]["index"]

        if not time_categories:
            LOGGER.info("[fetch_macro_pl] No time categories for %s", dataset_code)
            return None

        # Konwertuj wartości słownikowe na listy
        # W JSON-stat 2.0, values to obiekt {index: value}
        # Indeksy odpowiadają kombinacjom wymiarów

        # Wyodrębnij okresy i wartości
        periods = []
        vals = []

        # Sortuj kategorie czasu według indeksów
        time_items = sorted(time_categories.items(), key=lambda x: x[1])

        for time_label, time_idx in time_items:
            # Dla pojedynczych wartości wymiarów (przefiltrowane geo=PL, unit=X, etc.)
            # indeks czasu = indeks wartości
            idx = time_idx

            if str(idx) in values:
                periods.append(time_label)
                vals.append(values[str(idx)])

        if not periods or not vals:
            LOGGER.info("[fetch_macro_pl] No data extracted for %s", dataset_code)
            return None

        # Konwertuj na DataFrame
        df = pd.DataFrame({
            "date": periods,
            "value": vals
        })

        # Konwertuj typy
        # Okresy mogą być w różnych formatach: YYYY, YYYY-MM, YYYY-QX, YYYY-MM-DD
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        # Usuń wiersze z brakującymi wartościami
        df = df.dropna(subset=["date", "value"])

        # Sortuj po dacie
        df = df.sort_values("date").reset_index(drop=True)

        return df

    except Exception as exc:
        LOGGER.error("[fetch_macro_pl] Error fetching %s: %s", dataset_code, exc)
        return None


