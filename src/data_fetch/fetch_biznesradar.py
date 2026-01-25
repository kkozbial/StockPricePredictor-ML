"""
Moduł pobierania danych finansowych i sektorowych dla polskich spółek z BiznesRadar.pl.

Ten moduł służy do automatycznego pobierania (web scraping), przetwarzania i zapisywania
danych finansowych oraz informacji o sektorach dla spółek notowanych na GPW.
Dane są pobierane ze stron biznesradar.pl i zapisywane lokalnie.

Wymagane zależności zewnętrzne:
    - requests, beautifulsoup4: do scrapingu HTML.
    - pandas: do przetwarzania danych.
    - utils: config_loader, io_helpers, log_helpers.
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

from ..utils.config_loader import get_path_from_config, load_config
from ..utils.io_helpers import ensure_dir
from ..utils.log_helpers import FetchResult

LOGGER = logging.getLogger("data_fetch.biznesradar")

# URL patterns dla BiznesRadar
BIZNESRADAR_BASE = "https://www.biznesradar.pl"
PROFILE_URL = f"{BIZNESRADAR_BASE}/notowania/{{ticker}}"
INCOME_STATEMENT_URL = f"{BIZNESRADAR_BASE}/raporty-finansowe-rachunek-zyskow-i-strat/{{ticker}},Q"
BALANCE_SHEET_URL = f"{BIZNESRADAR_BASE}/raporty-finansowe-bilans/{{ticker}},Q"
CASHFLOW_URL = f"{BIZNESRADAR_BASE}/raporty-finansowe-przeplywy-pieniezne/{{ticker}},Q"

# Headers dla requestów (symulacja przeglądarki)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "pl-PL,pl;q=0.9,en-US;q=0.8,en;q=0.7",
}

# Mapowanie polskich nazw wskaźników na angielskie (zgodnie z schematem SEC)
INCOME_STATEMENT_MAPPING = {
    # Przychody
    "Przychody ze sprzedaży": "Revenues",
    "Przychody netto ze sprzedaży": "Revenues",
    "Przychody odsetkowe": "InterestIncome",
    "Przychody z działalności operacyjnej": "OperatingRevenue",
    "Przychody prowizyjne": "FeeIncome",
    "Przychody z tytułu dywidend": "DividendIncome",
    "Pozostałe przychody operacyjne": "OtherOperatingIncome",
    # Koszty
    "Koszty odsetkowe": "InterestExpense",
    "Koszty prowizyjne": "FeeExpense",
    "Ogólne koszty administracyjne": "GeneralAndAdministrativeExpense",
    "Pozostałe koszty operacyjne": "OtherOperatingExpense",
    "Odpisy netto z tytułu utraty wartości kredytów": "ProvisionForLoanLosses",
    # Wyniki
    "Wynik z tytułu odsetek": "NetInterestIncome",
    "Wynik z tytułu prowizji": "NetFeeIncome",
    "Wynik handlowy i rewaluacja": "TradingAndRevaluationResult",
    "Wynik na pozostałych instrumentach finansowych": "OtherFinancialInstrumentsResult",
    "Zysk (strata) brutto ze sprzedaży": "GrossProfit",
    "Zysk (strata) na działalności operacyjnej": "OperatingIncomeLoss",
    "Zysk przed opodatkowaniem": "IncomeLossFromContinuingOperationsBeforeIncomeTaxes",
    "Zysk (strata) przed opodatkowaniem": "IncomeLossFromContinuingOperationsBeforeIncomeTaxes",
    "Zysk (strata) netto": "NetIncomeLoss",
    "Zysk netto": "NetIncomeLoss",
    "Zysk netto akcjonariuszy jednostki dominującej": "NetIncomeLoss",
    "Wynik operacyjny": "OperatingIncomeLoss",
    "Wynik netto": "NetIncomeLoss",
    "Udział w zyskach jednostek podporządkowanych": "EquityInEarningsOfSubsidiaries",
    "Zysk (strata) netto z działalności zaniechanej": "DiscontinuedOperationsNetIncome",
    # Wskaźniki
    "EBITDA": "EBITDA",
}

BALANCE_SHEET_MAPPING = {
    "Aktywa": "Assets",
    "Aktywa razem": "Assets",
    "Aktywa trwałe": "AssetsNoncurrent",
    "Aktywa obrotowe": "AssetsCurrent",
    "Zobowiązania": "Liabilities",
    "Zobowiązania i rezerwy na zobowiązania": "Liabilities",
    "Zobowiązania długoterminowe": "LiabilitiesNoncurrent",
    "Zobowiązania krótkoterminowe": "LiabilitiesCurrent",
    "Kapitał (fundusz) własny": "StockholdersEquity",
    "Kapitał własny": "StockholdersEquity",
    "Kapitały własne": "StockholdersEquity",
}

CASHFLOW_MAPPING = {
    "Przepływy pieniężne netto z działalności operacyjnej": "NetCashProvidedByUsedInOperatingActivities",
    "Przepływy środków pieniężnych z działalności operacyjnej": "NetCashProvidedByUsedInOperatingActivities",
    "Przepływy pieniężne netto z działalności inwestycyjnej": "NetCashProvidedByUsedInInvestingActivities",
    "Przepływy środków pieniężnych z działalności inwestycyjnej": "NetCashProvidedByUsedInInvestingActivities",
    "Przepływy pieniężne netto z działalności finansowej": "NetCashProvidedByUsedInFinancingActivities",
    "Przepływy środków pieniężnych z działalności finansowej": "NetCashProvidedByUsedInFinancingActivities",
    "Nakłady inwestycyjne": "CapitalExpenditures",
}


def fetch_biznesradar_financials(
    tickers: Iterable[str],
    output_dir: Optional[Path] = None,
) -> FetchResult:
    """
    Pobiera dane finansowe dla polskich spółek z BiznesRadar.pl.

    Dla każdego tickera pobiera:
    - Rachunek zysków i strat (income statement)
    - Bilans (balance sheet)
    - Przepływy pieniężne (cash flow)

    Args:
        tickers: Lista tickerów GPW (np. 'PKO', 'PZU', 'CDR').
        output_dir: Katalog docelowy; domyślnie z konfiguracji.

    Returns:
        FetchResult z informacją o sukcesie/błędach.
    """
    result = FetchResult()
    cfg = load_config()
    target_dir = Path(output_dir) if output_dir else get_path_from_config("paths", "raw", cfg) / "financials"
    ensure_dir(target_dir)

    tickers_list = [ticker for ticker in tickers if ticker]
    if not tickers_list:
        LOGGER.warning("[fetch_biznesradar] No tickers provided.")
        result.note = "no tickers"
        return result

    for ticker in tickers_list:
        # Konwertuj ticker z formatu PKO.PL na PKO
        clean_ticker = ticker.replace(".PL", "")

        # Sprawdź czy plik już istnieje
        out_file = target_dir / f"{ticker}_financials.json"
        if out_file.exists():
            LOGGER.debug("[fetch_biznesradar] File already exists for %s, skipping", clean_ticker)
            result.skipped += 1
            continue

        LOGGER.info("[fetch_biznesradar] Fetching financials for %s", clean_ticker)

        try:
            # Pobierz dane z trzech źródeł
            all_records = []

            # 1. Rachunek zysków i strat
            income_records = _scrape_financial_table(
                INCOME_STATEMENT_URL.format(ticker=clean_ticker),
                INCOME_STATEMENT_MAPPING,
                clean_ticker
            )
            all_records.extend(income_records)
            LOGGER.debug("[fetch_biznesradar] Fetched %d income statement records for %s", len(income_records), clean_ticker)

            # Opóźnienie między requestami (rate limiting)
            time.sleep(1)

            # 2. Bilans
            balance_records = _scrape_financial_table(
                BALANCE_SHEET_URL.format(ticker=clean_ticker),
                BALANCE_SHEET_MAPPING,
                clean_ticker
            )
            all_records.extend(balance_records)
            LOGGER.debug("[fetch_biznesradar] Fetched %d balance sheet records for %s", len(balance_records), clean_ticker)

            time.sleep(1)

            # 3. Przepływy pieniężne
            cashflow_records = _scrape_financial_table(
                CASHFLOW_URL.format(ticker=clean_ticker),
                CASHFLOW_MAPPING,
                clean_ticker
            )
            all_records.extend(cashflow_records)
            LOGGER.debug("[fetch_biznesradar] Fetched %d cash flow records for %s", len(cashflow_records), clean_ticker)

            if not all_records:
                LOGGER.warning("[fetch_biznesradar] No financial data found for %s", clean_ticker)
                result.skipped += 1
                continue

            # Zapisz do pliku JSON (format kompatybilny z fetch_financials.py)
            _save_financials_json(ticker, all_records, out_file)

            LOGGER.info("[fetch_biznesradar] Saved %d financial records for %s to %s", len(all_records), clean_ticker, out_file)
            result.updated += 1
            result.add_path(out_file)

        except Exception as exc:
            LOGGER.exception("[fetch_biznesradar] Error fetching data for %s: %s", clean_ticker, exc)
            result.errors += 1
            continue

    return result


def fetch_biznesradar_sectors(
    tickers: Iterable[str],
    output_dir: Optional[Path] = None,
) -> FetchResult:
    """
    Pobiera informacje o sektorach dla polskich spółek z BiznesRadar.pl.

    Args:
        tickers: Lista tickerów GPW (np. 'PKO', 'PZU', 'CDR').
        output_dir: Katalog docelowy; domyślnie z konfiguracji.

    Returns:
        FetchResult z informacją o sukcesie/błędach.
    """
    result = FetchResult()
    cfg = load_config()
    target_dir = Path(output_dir) if output_dir else get_path_from_config("paths", "raw", cfg) / "others"
    ensure_dir(target_dir)

    tickers_list = [ticker for ticker in tickers if ticker]
    if not tickers_list:
        LOGGER.warning("[fetch_biznesradar_sectors] No tickers provided.")
        result.note = "no tickers"
        return result

    sector_rows = []

    for ticker in tickers_list:
        # Konwertuj ticker z formatu PKO.PL na PKO
        clean_ticker = ticker.replace(".PL", "")

        LOGGER.info("[fetch_biznesradar_sectors] Fetching sector info for %s", clean_ticker)

        try:
            url = PROFILE_URL.format(ticker=clean_ticker)
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # Szukamy linku do branży (industry)
            # Format: <a href="/gielda/branza:banki">Banki</a>
            # UWAGA: W BiznesRadar "Branża" = Industry, Sector zostawiamy pusty
            industry = None
            industry_link = soup.find("a", href=re.compile(r"/gielda/branza:"))
            if industry_link:
                industry = industry_link.get_text(strip=True)

            # Na BiznesRadar jest tylko "Branża" (Industry), nie ma osobnego Sector
            if industry:
                sector_rows.append({
                    "Ticker": ticker,
                    "Sector": None,  # None dla polskich spółek (będzie NULL w bazie)
                    "Industry": industry,  # Branża z BiznesRadar
                })
                result.updated += 1
                LOGGER.info("[fetch_biznesradar_sectors] Found industry '%s' for %s", industry, clean_ticker)
            else:
                LOGGER.warning("[fetch_biznesradar_sectors] No sector found for %s", clean_ticker)
                result.skipped += 1

            # Rate limiting
            time.sleep(1)

        except Exception as exc:
            LOGGER.exception("[fetch_biznesradar_sectors] Error fetching sector for %s: %s", clean_ticker, exc)
            result.errors += 1
            continue

    if sector_rows:
        df_sectors = pd.DataFrame(sector_rows)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        sectors_path = target_dir / f"sectors_{timestamp}.csv"
        df_sectors.to_csv(sectors_path, index=False)
        result.add_path(sectors_path)
        LOGGER.info("[fetch_biznesradar_sectors] Saved %d sector records to %s", len(sector_rows), sectors_path)
    else:
        LOGGER.info("[fetch_biznesradar_sectors] No sector data collected")

    return result


def _scrape_financial_table(url: str, concept_mapping: dict[str, str], ticker: str) -> list[dict]:
    """
    Pobiera tabelę finansową z BiznesRadar i konwertuje ją na listę rekordów.

    Args:
        url: URL do strony z tabelą finansową.
        concept_mapping: Słownik mapujący polskie nazwy na angielskie koncepty.
        ticker: Symbol tickera.

    Returns:
        Lista słowników z danymi finansowymi w formacie:
        [{"ticker": "PKO", "concept": "Revenues", "end": "2023-12-31", "value": 12345, "unit": "PLN"}, ...]
    """
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "html.parser")

    # Znajdź główną tabelę z danymi finansowymi
    # Tabela ma klasę "report-table"
    table = soup.find("table", class_="report-table")

    if not table:
        LOGGER.warning("[scrape_table] No financial table found at %s", url)
        return []

    records = []

    # Parsuj nagłówki (okresy - lata/kwartały)
    header_row = table.find("thead").find("tr") if table.find("thead") else table.find("tr")
    if not header_row:
        LOGGER.warning("[scrape_table] No header row found in table")
        return []

    # Dla danych kwartalnych używamy zarówno th jak i td (mogą być w pierwszym wierszu jako td)
    header_cells = header_row.find_all(["th", "td"])
    headers = [cell.get_text(strip=True) for cell in header_cells]

    # Pierwsze kolumny to zazwyczaj nazwa wskaźnika, kolejne to okresy
    # Format kwartalny: ["", "2019/Q3(wrz 19)", "2019/Q4(gru 19)", ...]
    # Format roczny: ["", "2020 (gru 20)", "2021 (gru 21)", ...]

    periods = []  # Lista tupli: (data, typ_raportu)
    for header in headers[1:]:  # Pomijamy pierwszą kolumnę (nazwa)
        period_info = _parse_period_header(header)
        if period_info:
            periods.append(period_info)

    if not periods:
        LOGGER.warning("[scrape_table] No valid periods found in headers")
        return []

    # Parsuj wiersze z danymi
    tbody = table.find("tbody") if table.find("tbody") else table
    rows = tbody.find_all("tr")

    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 2:
            continue

        # Pierwsza komórka to nazwa wskaźnika (concept)
        concept_pl = cells[0].get_text(strip=True)

        # Sprawdź czy ten wskaźnik jest w mapowaniu
        concept_en = concept_mapping.get(concept_pl)
        if not concept_en:
            # Pomiń niezmapowane wskaźniki
            continue

        # Reszta komórek to wartości dla poszczególnych okresów
        for i, cell in enumerate(cells[1:]):
            if i >= len(periods):
                break

            # Ekstraktuj wartość z zagnieżdżonej struktury span
            # Struktura: <td><span class="value"><span class="pv"><span>620 186</span></span></span></td>
            value_text = None

            # Próbuj znaleźć span.value > span.pv > span
            value_span = cell.find("span", class_="value")
            if value_span:
                pv_span = value_span.find("span", class_="pv")
                if pv_span:
                    inner_span = pv_span.find("span")
                    if inner_span:
                        value_text = inner_span.get_text(strip=True)

            # Jeśli nie znaleziono w strukturze span, użyj tekstu z całej komórki
            if not value_text:
                value_text = cell.get_text(strip=True)

            value = _parse_value(value_text)

            if value is None:
                continue  # Pomiń puste wartości

            # Rozpakuj informację o okresie (data, typ_raportu)
            period_date, report_type = periods[i]

            # Dodaj rekord
            records.append({
                "ticker": ticker,
                "concept": concept_en,
                "end": period_date,
                "value": value,
                "unit": "PLN",  # Wszystkie wartości w PLN
                "form": report_type,  # Q1, Q2, Q3, Q4 lub Annual
            })

    return records


def _parse_period_header(header: str) -> Optional[tuple[str, str]]:
    """
    Parsuje nagłówek okresu i zwraca datę końcową w formacie ISO oraz typ raportu.

    Przykłady:
    - "2023 (gru 23)" -> ("2023-12-31", "Annual")
    - "2019/Q3(wrz 19)" -> ("2019-09-30", "Q3")
    - "2020/Q1(mar 20)" -> ("2020-03-31", "Q1")
    - "2024 (K4 wrz 24)" -> ("2024-09-30", "Q4") (stary format)

    Args:
        header: Tekst nagłówka okresu.

    Returns:
        Tupla (data_końcowa w formacie ISO, typ_raportu) lub None.
    """
    header = header.strip()

    # Pattern dla nowego formatu kwartalnego: "2019/Q3(wrz 19)"
    new_quarterly_match = re.search(r"(\d{4})/Q(\d+)\((\w{3})\s*\d{2}\)", header)
    if new_quarterly_match:
        year = new_quarterly_match.group(1)
        quarter = new_quarterly_match.group(2)  # Numer kwartału (1, 2, 3, 4)
        month_pl = new_quarterly_match.group(3).lower()

        # Mapowanie polskich miesięcy
        month_map = {
            "sty": "01", "lut": "02", "mar": "03", "kwi": "04",
            "maj": "05", "cze": "06", "lip": "07", "sie": "08",
            "wrz": "09", "paź": "10", "lis": "11", "gru": "12",
        }

        month = month_map.get(month_pl, "03")

        # Ostatni dzień kwartału
        if month in ["03"]:
            return (f"{year}-03-31", f"Q{quarter}")
        elif month in ["06"]:
            return (f"{year}-06-30", f"Q{quarter}")
        elif month in ["09"]:
            return (f"{year}-09-30", f"Q{quarter}")
        elif month in ["12"]:
            return (f"{year}-12-31", f"Q{quarter}")

    # Pattern dla rocznych danych: "2023 (gru 23)"
    yearly_match = re.search(r"(\d{4})\s*\((\w{3})\s*\d{2}\)", header)
    if yearly_match:
        year = yearly_match.group(1)
        month_pl = yearly_match.group(2).lower()

        # Mapowanie polskich miesięcy
        month_map = {
            "sty": "01", "lut": "02", "mar": "03", "kwi": "04",
            "maj": "05", "cze": "06", "lip": "07", "sie": "08",
            "wrz": "09", "paź": "10", "lis": "11", "gru": "12",
        }

        month = month_map.get(month_pl, "12")

        # Ostatni dzień miesiąca (uproszczenie - używamy 31 dla końca roku)
        if month == "12":
            return (f"{year}-12-31", "Annual")
        else:
            # Dla innych miesięcy używamy 30 (uproszczenie dla kwartalnych)
            return (f"{year}-{month}-30", "Annual")

    # Pattern dla starego formatu kwartalnego: "2024 (K4 wrz 24)"
    old_quarterly_match = re.search(r"(\d{4})\s*\(K(\d)\s*(\w{3})\s*\d{2}\)", header)
    if old_quarterly_match:
        year = old_quarterly_match.group(1)
        quarter = old_quarterly_match.group(2)  # Numer kwartału (1, 2, 3, 4)
        month_pl = old_quarterly_match.group(3).lower()

        month_map = {
            "sty": "01", "lut": "02", "mar": "03", "kwi": "04",
            "maj": "05", "cze": "06", "lip": "07", "sie": "08",
            "wrz": "09", "paź": "10", "lis": "11", "gru": "12",
        }

        month = month_map.get(month_pl, "03")

        # Ostatni dzień kwartału
        if month in ["03"]:
            return (f"{year}-03-31", f"Q{quarter}")
        elif month in ["06"]:
            return (f"{year}-06-30", f"Q{quarter}")
        elif month in ["09"]:
            return (f"{year}-09-30", f"Q{quarter}")
        elif month in ["12"]:
            return (f"{year}-12-31", f"Q{quarter}")

    # Nie udało się sparsować
    return None


def _parse_value(value_text: str) -> Optional[float]:
    """
    Parsuje wartość finansową z tekstu.

    BiznesRadar używa formatu polskiego:
    - "123 456 789" (spacja jako separator tysięcy)
    - "12,34" (przecinek jako separator dziesiętny)
    - Może zawierać sufiks " tys. PLN" lub podobne

    Args:
        value_text: Tekst z wartością.

    Returns:
        Wartość jako float lub None jeśli nie udało się sparsować.
    """
    if not value_text or value_text == "-" or value_text == "":
        return None

    # Usuń sufiks jednostki i inne teksty
    value_text = re.sub(r"\s*(tys\.|mln|mld)\.?\s*(PLN|zł)?.*", "", value_text, flags=re.IGNORECASE)

    # Usuń spacje (separator tysięcy)
    value_text = value_text.replace(" ", "").replace("\xa0", "")

    # Zamień przecinek na kropkę
    value_text = value_text.replace(",", ".")

    try:
        # Konwersja na float
        value = float(value_text)

        # BiznesRadar podaje wartości w tysiącach PLN, więc mnożymy przez 1000
        # Aby uzyskać wartość w PLN (zgodnie z SEC API gdzie wartości są w USD)
        return value * 1000

    except (ValueError, AttributeError):
        return None


def _save_financials_json(ticker: str, records: list[dict], output_path: Path) -> None:
    """
    Zapisuje dane finansowe do pliku JSON w formacie kompatybilnym z SEC API.

    Args:
        ticker: Symbol tickera.
        records: Lista rekordów finansowych.
        output_path: Ścieżka do pliku wyjściowego.
    """
    data = {
        "ticker": ticker,
        "financials": records
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
