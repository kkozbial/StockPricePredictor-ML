"""Udostepnia funkcje do pobierania liczby akcji w obrocie z SEC EDGAR."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Optional

from ..utils.api_helpers import safe_get
from ..utils.config_loader import get_path_from_config, load_config
from ..utils.io_helpers import ensure_dir
from ..utils.log_helpers import FetchResult, ProgressTracker

LOGGER = logging.getLogger("data_fetch.shares")

HEADERS = {"User-Agent": "Kamil Kozbial <kamilkozbial1@gmail.com>"}
CIK_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
# Tags dla liczby akcji w różnych taksonomach
SHARE_TAGS_DEI = ("EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding")
SHARE_TAGS_USGAAP = (
    "WeightedAverageNumberOfSharesOutstandingBasic",
    "WeightedAverageNumberOfDilutedSharesOutstanding",
    "WeightedAverageNumberOfSharesOutstandingDiluted",
)

# Tags dla fallback calculation
NET_INCOME_TAGS = ("NetIncomeLoss", "ProfitLoss")
EPS_TAGS = ("EarningsPerShareDiluted", "EarningsPerShareBasic")

CIK_MAP: Optional[dict[str, str]] = None


def load_tickers_from_metadata(metadata_path: Optional[Path] = None) -> list[str]:
    """
    Wczytuje listę tickerów z pliku company_metadata.json.

    Args:
        metadata_path: Opcjonalna ścieżka do pliku metadata. Jeśli None, używa domyślnej ścieżki.

    Returns:
        Lista tickerów ze wszystkich spółek w metadanych.
    """
    if metadata_path is None:
        cfg = load_config()
        metadata_path = get_path_from_config("paths", "raw", cfg) / "metadata" / "company_metadata.json"

    if not metadata_path.exists():
        LOGGER.error("[fetch_shares] Metadata file not found: %s", metadata_path)
        return []

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        companies = metadata.get("companies", [])

        tickers = []
        for company in companies:
            # Preferuj tickers, jeśli brak to użyj all_tickers
            company_tickers = company.get("tickers") or company.get("all_tickers", [])
            for ticker in company_tickers:
                if ticker and ticker not in tickers:
                    tickers.append(ticker)

        LOGGER.info("[fetch_shares] Loaded %s unique tickers from metadata", len(tickers))
        return tickers

    except Exception as exc:
        LOGGER.error("[fetch_shares] Failed to load metadata from %s: %s", metadata_path, exc)
        return []


def get_cik_map() -> dict[str, str]:
    """Pobiera mapowanie ticker -> CIK i keszuje wynik w module."""
    global CIK_MAP
    if CIK_MAP is not None:
        return CIK_MAP

    try:
        response = safe_get(CIK_URL, headers=HEADERS)
        payload = response.json()
    except Exception as exc:  # pragma: no cover - sieciowe
        LOGGER.error("[fetch_shares] Failed to download SEC ticker mapping: %s", exc)
        CIK_MAP = {}
        return CIK_MAP

    mapping: dict[str, str] = {}
    entries = payload.values() if isinstance(payload, dict) else payload
    for entry in entries:
        ticker = (entry.get("ticker") or "").upper()
        cik_str = entry.get("cik_str")
        if ticker and cik_str:
            mapping[ticker] = str(cik_str).zfill(10)

    if not mapping:
        LOGGER.error("[fetch_shares] SEC ticker mapping returned no entries.")

    CIK_MAP = mapping
    return CIK_MAP


def _extract_share_history(units: dict, source_type: str = "SEC_DEI") -> list[dict]:
    """Builds full historical share counts from EDGAR units.

    Args:
        units: Dictionary of units from EDGAR API
        source_type: Type of source ('SEC_DEI', 'SEC_USGAAP', or 'Calculated')

    Returns:
        List of share history records with source_type flag
    """
    history: list[dict] = []
    for unit_name, entries in units.items():
        if unit_name.lower() != "shares":
            continue
        for entry in entries or []:
            value = entry.get("val")
            if value is None:
                continue
            history.append(
                {
                    "end": entry.get("end"),
                    "filed": entry.get("filed"),
                    "fy": entry.get("fy"),
                    "fp": entry.get("fp"),
                    "value": value,
                    "source_type": source_type,
                }
            )
    return history


def _extract_calculated_shares(facts: dict) -> list[dict]:
    """Calculates share count as NetIncomeLoss / EarningsPerShareDiluted.

    Args:
        facts: Full facts dictionary from EDGAR API

    Returns:
        List of calculated share records with source_type='Calculated'
    """
    usgaap_facts = facts.get("us-gaap", {})

    # Find net income data
    net_income_data = None
    for tag in NET_INCOME_TAGS:
        if tag in usgaap_facts:
            net_income_data = usgaap_facts[tag]
            break

    # Find EPS data
    eps_data = None
    for tag in EPS_TAGS:
        if tag in usgaap_facts:
            eps_data = usgaap_facts[tag]
            break

    if not net_income_data or not eps_data:
        return []

    # Extract USD values
    net_income_units = net_income_data.get("units", {}).get("USD", [])
    eps_units = eps_data.get("units", {}).get("USD/shares", [])

    if not net_income_units or not eps_units:
        return []

    # Create mapping by (end, fy, fp) for matching
    eps_map = {}
    for eps_entry in eps_units:
        key = (eps_entry.get("end"), eps_entry.get("fy"), eps_entry.get("fp"))
        eps_map[key] = eps_entry

    calculated_shares = []
    for ni_entry in net_income_units:
        key = (ni_entry.get("end"), ni_entry.get("fy"), ni_entry.get("fp"))
        eps_entry = eps_map.get(key)

        if not eps_entry:
            continue

        net_income = ni_entry.get("val")
        eps = eps_entry.get("val")

        if net_income is None or eps is None or eps == 0:
            continue

        # Calculate shares = Net Income / EPS
        calculated_value = abs(net_income / eps)

        calculated_shares.append(
            {
                "end": ni_entry.get("end"),
                "filed": ni_entry.get("filed"),
                "fy": ni_entry.get("fy"),
                "fp": ni_entry.get("fp"),
                "value": calculated_value,
                "source_type": "Calculated",
            }
        )

    return calculated_shares


def _merge_share_records(existing: list[dict], new: list[dict]) -> list[dict]:
    """
    Łączy istniejące i nowe rekordy shares, usuwając duplikaty po dacie filed.

    Args:
        existing: Lista istniejących rekordów.
        new: Lista nowych rekordów z API.

    Returns:
        Połączona lista bez duplikatów.
    """
    # Tworzymy set z unikalnych krotek (filed, end, value, source_type) dla istniejących rekordów
    existing_set = set()
    for record in existing:
        key = (
            record.get("filed"),
            record.get("end"),
            record.get("value"),
            record.get("source_type", "SEC_tag")
        )
        existing_set.add(key)

    # Dodajemy tylko nowe rekordy, których nie ma w existing_set
    combined = list(existing)
    for record in new:
        key = (
            record.get("filed"),
            record.get("end"),
            record.get("value"),
            record.get("source_type", "SEC_tag")
        )
        if key not in existing_set:
            combined.append(record)
            existing_set.add(key)

    return combined


def fetch_shares_outstanding(tickers: Iterable[str], output_dir: Optional[Path] = None) -> FetchResult:
    """
    Pobiera dane o liczbie akcji w obrocie i zapisuje kazdy ticker jako plik JSON.

    Obsługuje tryb przyrostowy - jeśli plik dla tickera już istnieje,
    porównuje nowe dane z istniejącymi i dołącza tylko nowe wpisy.
    """
    result = FetchResult()
    cfg = load_config()
    target_dir = Path(output_dir) if output_dir else get_path_from_config("paths", "raw", cfg) / "shares"
    ensure_dir(target_dir)
    cik_map = get_cik_map()
    tickers_list = [ticker for ticker in tickers if ticker]
    if not tickers_list:
        LOGGER.warning("[fetch_shares] No tickers provided.")
        result.note = "no tickers"
        return result

    LOGGER.info("[fetch_shares] Starting EDGAR download for %s tickers", len(tickers_list))
    progress = ProgressTracker(len(tickers_list), "shares")

    for ticker in tickers_list:
        normalized = ticker.upper()
        cik = cik_map.get(normalized)
        if not cik:
            result.skipped += 1
            progress.update(skipped=True)
            continue

        # Wczytaj istniejące dane jeśli plik istnieje (tryb przyrostowy)
        existing_records = []
        output_path = target_dir / f"{normalized}_shares.json"
        if output_path.exists():
            try:
                existing_data = json.loads(output_path.read_text(encoding="utf-8"))
                existing_records = existing_data.get("shares_history", [])
            except Exception:
                existing_records = []

        try:
            response = safe_get(COMPANYFACTS_URL.format(cik=cik), headers=HEADERS)
            data = response.json()
        except Exception:  # pragma: no cover - sieciowe
            result.errors += 1
            progress.update(error=True)
            continue

        facts = data.get("facts", {})
        new_records: list[dict] = []

        # 1. Próba pobrania z taksonomii DEI
        dei_facts = facts.get("dei", {})
        for tag in SHARE_TAGS_DEI:
            concept = dei_facts.get(tag)
            if not concept:
                continue
            units = concept.get("units", {})
            new_records = _extract_share_history(units, source_type="SEC_DEI")
            if new_records:
                break

        # 2. Jeśli brak w DEI, próba pobrania z taksonomii US-GAAP
        if not new_records:
            usgaap_facts = facts.get("us-gaap", {})
            for tag in SHARE_TAGS_USGAAP:
                concept = usgaap_facts.get(tag)
                if not concept:
                    continue
                units = concept.get("units", {})
                new_records = _extract_share_history(units, source_type="SEC_USGAAP")
                if new_records:
                    break

        # 3. Fallback: oblicz jako NetIncomeLoss / EarningsPerShareDiluted
        if not new_records:
            new_records = _extract_calculated_shares(facts)

        if not new_records:
            result.skipped += 1
            progress.update(skipped=True)
            continue

        # W trybie przyrostowym: merge istniejących i nowych danych
        if existing_records:
            combined_records = _merge_share_records(existing_records, new_records)
            added_count = len(combined_records) - len(existing_records)

            if added_count == 0:
                result.skipped += 1
                progress.update(skipped=True)
                continue

            final_records = combined_records
        else:
            # Tryb pełny lub brak istniejących danych
            final_records = new_records

        output_payload = {"ticker": normalized, "shares_history": final_records}
        output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
        result.updated += 1
        result.add_path(output_path)
        progress.update(updated=True)

    progress.finish()

    if result.updated == 0 and result.errors == 0 and result.skipped > 0:
        result.note = "no updates"
    return result


if __name__ == "__main__":
    # Pobierz tickery z metadanych
    tickers = load_tickers_from_metadata()
    if not tickers:
        LOGGER.warning("No tickers loaded from metadata. Using sample tickers.")
        tickers = ["AAPL", "MSFT", "GOOGL"]

    LOGGER.info("Starting shares download for %s tickers from metadata", len(tickers))
    result = fetch_shares_outstanding(tickers)
    LOGGER.info(
        "Shares download complete: updated=%s, skipped=%s, errors=%s",
        result.updated,
        result.skipped,
        result.errors,
    )
