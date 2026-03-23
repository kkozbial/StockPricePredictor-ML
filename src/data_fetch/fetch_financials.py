"""
Moduł pobierania danych finansowych (Financials) z API SEC EDGAR.

Ten moduł służy do automatycznego pobierania, przetwarzania i zapisywania
wybranych wskaźników finansowych (GAAP) dla spółek notowanych na giełdach w USA.
Dane są pobierane w formacie XBRL JSON, filtrowane pod kątem kluczowych metryk
(np. Przychody, Zysk Netto) i zapisywane lokalnie.

Wymagane zależności zewnętrzne (utils):
    - api_helpers: do bezpiecznych zapytań HTTP.
    - config_loader: do ładowania konfiguracji ścieżek.
    - log_helpers: klasa FetchResult do raportowania statusu.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Optional

from ..utils.api_helpers import safe_get
from ..utils.config_loader import get_path_from_config, load_config
from ..utils.io_helpers import ensure_dir
from ..utils.log_helpers import FetchResult, ProgressTracker
from ..utils.incremental_helpers import should_use_incremental, get_delisted_tickers

LOGGER = logging.getLogger("data_fetch.financials")

HEADERS = {"User-Agent": "Kamil Kozbial <kamilkozbial1@gmail.com>"}
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
# Mapowanie alternatywnych nazw tagów GAAP do naszej kanonicznej nazwy kolumny.
# Rozszerzona lista aliasów - łata dziury historyczne
METRIC_ALIASES: dict[str, list[str]] = {
    "Assets": ["Assets", "AssetsNet"],
    "Liabilities": ["Liabilities", "LiabilitiesCurrent"],
    "StockholdersEquity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "Equity",
    ],
    "GrossProfit": ["GrossProfit"],
    "Revenues": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueServicesGross",
        "RevenuesNetOfInterestExpense",
        "SalesRevenueGoodsNet",
        "OilAndGasRevenue",
    ],
    "NetIncomeLoss": [
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
        "NetIncomeLossAvailableToCommonStockholdersDiluted",
    ],
    "EarningsPerShareDiluted": ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"],
    "OperatingIncomeLoss": ["OperatingIncomeLoss"],
    "OperatingCashFlow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "CapitalExpenditures": [
        "CapitalExpenditures",
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ],
    "EBITDA": [
        "EarningsBeforeInterestTaxesDepreciationAmortization",
        "EarningsBeforeInterestAndTaxesDepreciationAndAmortization",
        "EBITDA",
    ],
    "ResearchAndDevelopment": [
        "ResearchAndDevelopment",
        "ResearchAndDevelopmentExpense",
        "ResearchAndDevelopmentExpenditures",
    ],
    # === NOWE METRYKI (2026-01-06) - Płynność, Zadłużenie, Efektywność ===
    "AssetsCurrent": [
        "AssetsCurrent",
        "AssetsCurrentAbstract",
    ],
    "LiabilitiesCurrent": [
        "LiabilitiesCurrent",
        "LiabilitiesCurrentAbstract",
    ],
    "InterestExpense": [
        "InterestExpense",
        "InterestExpenseDebt",
        "InterestExpenseRelatedParty",
        "InterestPaidNet",
    ],
    "RetainedEarnings": [
        "RetainedEarningsAccumulatedDeficit",
        "RetainedEarnings",
    ],
    "Inventory": [
        "InventoryNet",
        "Inventory",
        "InventoryGross",
    ],
    "Receivables": [
        "ReceivablesNetCurrent",
        "AccountsReceivableNetCurrent",
        "ReceivablesNet",
        "AccountsReceivableNet",
    ],
}
# Dopuszczalne jednostki dla poszczególnych metryk. Domyślnie USD, ale np. EPS
# jest raportowane w USD/shares.
METRIC_UNITS: dict[str, set[str]] = {
    "EarningsPerShareDiluted": {"USD/shares", "USD/share"},
    "Assets": {"USD"},
    "Liabilities": {"USD"},
    "StockholdersEquity": {"USD"},
    "Revenues": {"USD"},
    "NetIncomeLoss": {"USD"},
    "OperatingIncomeLoss": {"USD"},
    "OperatingCashFlow": {"USD"},
    "CapitalExpenditures": {"USD"},
    "EBITDA": {"USD"},
    "GrossProfit": {"USD"},
    "ResearchAndDevelopment": {"USD"},
    # Nowe metryki (2026-01-06)
    "AssetsCurrent": {"USD"},
    "LiabilitiesCurrent": {"USD"},
    "InterestExpense": {"USD"},
    "RetainedEarnings": {"USD"},
    "Inventory": {"USD"},
    "Receivables": {"USD"},
}


def get_cik_map_from_metadata(metadata_file: Optional[Path] = None) -> Dict[str, str]:
    """
    Pobiera mapowanie ticker -> CIK z pliku metadanych firm.

    Args:
        metadata_file: Opcjonalna ścieżka do pliku z metadanymi.
                      Domyślnie data/raw/metadata/company_metadata.json.

    Returns
    -------
    Dict[str, str]
        Słownik mapujący Ticker na CIK, np. {'AAPL': '0000320193'}.
        Zwraca pusty słownik jeśli plik nie istnieje.

    Notes
    -----
    Ta funkcja zastąpiła starą get_cik_map(), która pobierała wszystkie tickery z SEC.
    Teraz używamy tylko przefiltrowanych tickerów (bez funduszy, banków, ubezpieczeń).
    """
    if metadata_file is None:
        cfg = load_config()
        metadata_file = get_path_from_config("paths", "raw", cfg) / "metadata" / "company_metadata.json"

    if not metadata_file.exists():
        LOGGER.warning("[fetch_financials] Metadata file not found: %s", metadata_file)
        LOGGER.warning("[fetch_financials] Please run fetch_sec_metadata first to generate metadata.")
        return {}

    try:
        data = json.loads(metadata_file.read_text())
        companies = data.get("companies", [])

        mapping: Dict[str, str] = {}
        for company in companies:
            cik = company.get("cik")
            all_tickers = company.get("all_tickers", [])

            # Dla każdego tickera przypisz CIK
            for ticker in all_tickers:
                mapping[ticker.upper()] = cik

        LOGGER.info("[fetch_financials] Loaded %d ticker-CIK mappings from metadata", len(mapping))
        return mapping

    except Exception as exc:
        LOGGER.error("[fetch_financials] Failed to load ticker-CIK mapping from metadata: %s", exc)
        return {}


def fetch_financial_reports(tickers: Iterable[str], output_dir: Optional[Path] = None) -> FetchResult:
    """
    Pobiera podstawowe raporty finansowe ze źródła SEC EDGAR.

    Obsługuje tryb przyrostowy (incremental) - jeśli włączony w konfiguracji,
    porównuje nowe dane z istniejącymi i dołącza tylko nowe wpisy.
    """
    result = FetchResult()
    cfg = load_config()
    target_dir = Path(output_dir) if output_dir else get_path_from_config("paths", "raw", cfg) / "financials"
    ensure_dir(target_dir)
    cik_map = get_cik_map_from_metadata()

    if not cik_map:
        LOGGER.error("[fetch_financials] No CIK mappings found. Please run fetch_sec_metadata first.")
        result.note = "no cik mappings - run fetch_sec_metadata first"
        return result

    tickers_list = [ticker for ticker in tickers if ticker]
    if not tickers_list:
        LOGGER.warning("[fetch_financials] No tickers provided.")
        result.note = "no tickers"
        return result

    # Sprawdź czy używać pobierania przyrostowego
    use_incremental = should_use_incremental(cfg, "financials")

    # Pobierz delisted tickery raz przed pętlą (tylko w trybie przyrostowym)
    delisted_tickers = get_delisted_tickers() if use_incremental else set()
    if delisted_tickers:
        LOGGER.info("[fetch_financials] Znaleziono %d delisted tickerów - zostaną pominięte jeśli mają dane", len(delisted_tickers))

    LOGGER.info("[fetch_financials] Starting EDGAR download for %s tickers", len(tickers_list))
    progress = ProgressTracker(len(tickers_list), "financials")

    for ticker in tickers_list:
        normalized = ticker.upper()
        cik = cik_map.get(normalized)
        if not cik:
            result.skipped += 1
            progress.update(skipped=True)
            continue

        output_path = target_dir / f"{normalized}_financials.json"

        # W trybie przyrostowym pomijamy delisted spółki, które mają już dane
        if use_incremental and normalized in delisted_tickers and output_path.exists():
            result.skipped += 1
            progress.update(skipped=True)
            continue

        # Wczytaj istniejące dane jeśli tryb przyrostowy
        existing_records = []
        if use_incremental and output_path.exists():
            try:
                existing_data = json.loads(output_path.read_text())
                existing_records = existing_data.get("financials", [])
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
        gaap_facts = facts.get("us-gaap", {})
        new_records = []
        alias_lookup = {alias: canonical for canonical, aliases in METRIC_ALIASES.items() for alias in aliases}

        for concept, payload in gaap_facts.items():
            canonical = alias_lookup.get(concept)
            if not canonical:
                continue  # pomijamy wszystkie nieużywane tagi
            units = payload.get("units", {})
            allowed_units = METRIC_UNITS.get(canonical, {"USD"})
            for unit_name in allowed_units:
                entries = units.get(unit_name)
                if not entries:
                    continue
                for entry in entries:
                    new_records.append(
                        {
                            "concept": canonical,
                            "unit": unit_name,
                            "start": entry.get("start"),
                            "end": entry.get("end"),
                            "value": entry.get("val"),
                            "form": entry.get("form"),  # Typ formularza (10-K, 10-Q, etc.) - może być None
                            "filing_date": entry.get("filed"),  # Data publikacji raportu - może być None
                        }
                    )

        # W trybie przyrostowym: merge istniejących i nowych danych
        if use_incremental and existing_records:
            combined_records = _merge_financial_records(existing_records, new_records)
            added_count = len(combined_records) - len(existing_records)

            if added_count == 0:
                result.skipped += 1
                progress.update(skipped=True)
                continue

            final_records = combined_records
        else:
            # Tryb pełny lub brak istniejących danych
            if not new_records:
                result.skipped += 1
                progress.update(skipped=True)
                continue
            final_records = new_records

        output_payload = {"ticker": normalized, "financials": final_records}
        output_path.write_text(json.dumps(output_payload, indent=2))
        result.updated += 1
        result.add_path(output_path)
        progress.update(updated=True)

    progress.finish()

    if result.updated == 0 and result.skipped > 0 and result.errors == 0:
        result.note = "no updates"
    return result


def _merge_financial_records(existing: list[dict], new: list[dict]) -> list[dict]:
    """
    Łączy istniejące i nowe rekordy finansowe, usuwając duplikaty.

    Args:
        existing: Lista istniejących rekordów.
        new: Lista nowych rekordów z API.

    Returns:
        Połączona lista bez duplikatów.

    Notes:
        - Stare rekordy mogą nie mieć pól 'form' i 'filing_date'
        - Nowe rekordy zawsze mają te pola (lub None)
        - Klucz unikalności: (concept, end, value) - bez form/filing_date
        - Jeśli stary rekord nie ma filing_date, a nowy ma → nowy wygrywa
    """
    # Indeks istniejących rekordów: (concept, end, value) -> pozycja na liście
    existing_index: dict[tuple, int] = {}
    combined = list(existing)
    for i, record in enumerate(combined):
        key = (record.get("concept"), record.get("end"), record.get("value"))
        existing_index[key] = i

    for record in new:
        key = (record.get("concept"), record.get("end"), record.get("value"))
        if key not in existing_index:
            # Nowy rekord — dodaj
            existing_index[key] = len(combined)
            combined.append(record)
        else:
            # Rekord już istnieje — zaktualizuj filing_date i form jeśli stary ich nie ma
            idx = existing_index[key]
            old = combined[idx]
            if not old.get("filing_date") and record.get("filing_date"):
                old["filing_date"] = record["filing_date"]
            if not old.get("form") and record.get("form"):
                old["form"] = record["form"]

    return combined


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL"]
    fetch_financial_reports(tickers)
