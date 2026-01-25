"""
Moduł pobierania metadanych spółek z SEC EDGAR.

Ten moduł pobiera:
1. Mapowanie ticker -> CIK z https://www.sec.gov/include/ticker.txt
2. Metadane firm (w tym kody SIC) z SEC API
3. Filtruje spółki wykluczając fundusze, banki i ubezpieczenia

Kody SIC do wykluczenia:
- 6722: Fundusze inwestycyjne (99% ETF-ów)
- 6726: Inne fundusze/powiernictwa
- 6021, 6022: Banki (specyficzne sprawozdania)
- 6311: Firmy ubezpieczeniowe
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

from ..utils.api_helpers import safe_get
from ..utils.config_loader import get_path_from_config, load_config
from ..utils.io_helpers import ensure_dir
from ..utils.log_helpers import FetchResult

LOGGER = logging.getLogger("data_fetch.sec_metadata")

HEADERS = {"User-Agent": "Kamil Kozbial <kamilkozbial1@gmail.com>"}

# URLs SEC
TICKER_TXT_URL = "https://www.sec.gov/include/ticker.txt"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

# Kody SIC do wykluczenia (finanse, fundusze, banki, ubezpieczenia)
EXCLUDED_SIC_CODES = {
    6722,  # Fundusze inwestycyjne (ETF-y)
    6726,  # Inne fundusze/powiernictwa
    6021,  # Banki krajowe
    6022,  # Banki stanowe
    6311,  # Firmy ubezpieczeniowe
}


def fetch_ticker_cik_mapping() -> Dict[str, str]:
    """
    Pobiera mapowanie ticker -> CIK z pliku SEC ticker.txt.

    Returns
    -------
    Dict[str, str]
        Słownik {ticker: cik_10_digits}, np. {'aapl': '0000320193'}.
        Zwraca pusty słownik w przypadku błędu.

    Notes
    -----
    Format pliku ticker.txt:
    ticker\tcik
    aapl\t320193
    msft\t789019
    """
    try:
        response = safe_get(TICKER_TXT_URL, headers=HEADERS)
        text = response.text
    except Exception as exc:
        LOGGER.error("[fetch_sec_metadata] Failed to download ticker.txt: %s", exc)
        return {}

    mapping = {}
    for line in text.strip().split('\n'):
        parts = line.split('\t')
        if len(parts) == 2:
            ticker, cik = parts
            # Normalizuj: CIK do 10 cyfr, ticker do uppercase
            mapping[ticker.strip().upper()] = str(cik).strip().zfill(10)

    LOGGER.info("[fetch_sec_metadata] Loaded %d ticker-CIK mappings from SEC", len(mapping))
    return mapping


def fetch_company_metadata(cik: str) -> Optional[Dict]:
    """
    Pobiera metadane firmy z SEC Submissions API.

    Args:
        cik: 10-cyfrowy numer CIK (z zerami z przodu).

    Returns
    -------
    Optional[Dict]
        Słownik z metadanymi firmy lub None w przypadku błędu.

    Notes
    -----
    Zwraca dane z głównego poziomu JSON submissions:
    - name: Pełna nazwa firmy
    - sic: Kod SIC (Standard Industrial Classification)
    - sicDescription: Opis kodu SIC
    - ticker: Ticker główny (często lista tickerów)
    - exchanges: Lista giełd
    """
    try:
        url = SUBMISSIONS_URL.format(cik=cik)
        response = safe_get(url, headers=HEADERS)
        data = response.json()

        return {
            "cik": cik,
            "name": data.get("name"),
            "sic": data.get("sic"),
            "sic_description": data.get("sicDescription"),
            "tickers": data.get("tickers", []),  # może być lista
            "exchanges": data.get("exchanges", []),
        }
    except Exception as exc:
        LOGGER.warning("[fetch_sec_metadata] Failed to fetch metadata for CIK %s: %s", cik, exc)
        return None


def is_excluded_company(sic_code: Optional[int], company_name: Optional[str] = None) -> bool:
    """
    Sprawdza czy firma powinna być wykluczona na podstawie kodu SIC lub nazwy.

    Args:
        sic_code: Kod SIC firmy.
        company_name: Nazwa firmy (opcjonalne - używane do wykrywania ETF-ów bez SIC).

    Returns
    -------
    bool
        True jeśli firma powinna być wykluczona, False w przeciwnym razie.

    Notes
    -----
    Niektóre ETF-y nie mają przypisanego kodu SIC w SEC, więc wykrywamy je po nazwie.
    """
    # Wykluczenie na podstawie SIC
    if sic_code is not None and sic_code in EXCLUDED_SIC_CODES:
        return True

    # Dodatkowe wykluczenie na podstawie nazwy (dla ETF-ów bez SIC)
    if company_name:
        name_lower = company_name.lower()
        # Wykrywanie ETF-ów i funduszy po nazwie
        etf_keywords = [
            " etf", "etf ", " trust", "fund", "index", "series",
            "portfolio", "investment company", "closed fund"
        ]
        for keyword in etf_keywords:
            if keyword in name_lower:
                return True

    return False


def fetch_all_company_metadata(output_dir: Optional[Path] = None) -> FetchResult:
    """
    Pobiera metadane dla wszystkich firm z SEC i zapisuje do JSON.

    Pipeline:
    1. Pobiera mapowanie ticker -> CIK z ticker.txt
    2. Dla każdego CIK pobiera metadane z Submissions API
    3. Filtruje firmy wykluczając fundusze, banki i ubezpieczenia
    4. Zapisuje kompletne metadane do pliku JSON

    Args:
        output_dir: Opcjonalny katalog docelowy. Domyślnie data/raw/metadata.

    Returns
    -------
    FetchResult
        Status operacji z liczbą pobranych i wykluczonych firm.
    """
    result = FetchResult()
    cfg = load_config()
    target_dir = Path(output_dir) if output_dir else get_path_from_config("paths", "raw", cfg) / "metadata"
    ensure_dir(target_dir)

    # Krok 1: Pobierz mapowanie ticker -> CIK
    LOGGER.info("[fetch_sec_metadata] Fetching ticker-CIK mapping from SEC...")
    ticker_cik_map = fetch_ticker_cik_mapping()

    if not ticker_cik_map:
        LOGGER.error("[fetch_sec_metadata] No ticker-CIK mappings found")
        result.note = "no mappings"
        return result

    # Krok 2: Pobierz metadane dla każdego CIK
    LOGGER.info("[fetch_sec_metadata] Fetching metadata for %d companies...", len(ticker_cik_map))

    all_metadata = []
    excluded_count = 0

    # Aby uniknąć duplikatów CIK (niektóre firmy mają wiele tickerów), zbieramy unikalne CIK
    unique_ciks = set(ticker_cik_map.values())
    cik_to_tickers = {}
    for ticker, cik in ticker_cik_map.items():
        if cik not in cik_to_tickers:
            cik_to_tickers[cik] = []
        cik_to_tickers[cik].append(ticker)

    for i, cik in enumerate(unique_ciks, 1):
        if i % 100 == 0:
            LOGGER.info("[fetch_sec_metadata] Progress: %d/%d companies processed", i, len(unique_ciks))

        metadata = fetch_company_metadata(cik)
        if metadata is None:
            result.errors += 1
            continue

        # Dodaj wszystkie tickery związane z tym CIK
        metadata["all_tickers"] = cik_to_tickers[cik]

        # Filtrowanie: wykluczamy firmy z zakazanych SIC lub na podstawie nazwy
        sic = metadata.get("sic")
        company_name = metadata.get("name")

        # Konwertuj SIC na int jeśli istnieje
        sic_int = int(sic) if sic else None

        if is_excluded_company(sic_int, company_name):
            reason = f"SIC: {sic}" if sic else "nazwa zawiera słowa kluczowe (ETF/Fund/Trust)"
            LOGGER.debug(
                "[fetch_sec_metadata] Excluded %s (%s - %s)",
                company_name,
                reason,
                metadata.get("sic_description") or "N/A"
            )
            excluded_count += 1
            result.skipped += 1
            continue

        all_metadata.append(metadata)
        result.updated += 1

    # Krok 3: Zapisz metadane
    output_file = target_dir / "company_metadata.json"
    output_data = {
        "total_companies": len(all_metadata),
        "excluded_count": excluded_count,
        "excluded_sic_codes": list(EXCLUDED_SIC_CODES),
        "companies": all_metadata,
    }

    output_file.write_text(json.dumps(output_data, indent=2))
    LOGGER.info(
        "[fetch_sec_metadata] Saved metadata for %d companies (excluded %d) to %s",
        len(all_metadata),
        excluded_count,
        output_file
    )

    result.add_path(output_file)
    result.note = f"excluded {excluded_count} companies by SIC"
    return result


def get_filtered_tickers(metadata_file: Optional[Path] = None) -> list[str]:
    """
    Zwraca listę tickerów po filtrowaniu (wykluczając fundusze, banki, ubezpieczenia).

    Args:
        metadata_file: Opcjonalna ścieżka do pliku z metadanymi.
                      Domyślnie data/raw/metadata/company_metadata.json.

    Returns
    -------
    list[str]
        Lista tickerów do pobierania danych finansowych.

    Notes
    -----
    Jeśli plik nie istnieje, zwraca pustą listę.
    Dla firm z wieloma tickerami, zwraca wszystkie tickery danej firmy.
    """
    if metadata_file is None:
        cfg = load_config()
        metadata_file = get_path_from_config("paths", "raw", cfg) / "metadata" / "company_metadata.json"

    if not metadata_file.exists():
        LOGGER.warning("[fetch_sec_metadata] Metadata file not found: %s", metadata_file)
        return []

    try:
        data = json.loads(metadata_file.read_text())
        companies = data.get("companies", [])

        # Zbierz wszystkie tickery z firm (może być wiele tickerów na firmę)
        all_tickers = []
        for company in companies:
            tickers = company.get("all_tickers", [])
            all_tickers.extend(tickers)

        # Usuń duplikaty i posortuj
        unique_tickers = sorted(set(all_tickers))

        LOGGER.info(
            "[fetch_sec_metadata] Loaded %d filtered tickers from %d companies",
            len(unique_tickers),
            len(companies)
        )
        return unique_tickers

    except Exception as exc:
        LOGGER.error("[fetch_sec_metadata] Failed to load filtered tickers: %s", exc)
        return []


if __name__ == "__main__":
    # Test pobierania metadanych
    fetch_all_company_metadata()
