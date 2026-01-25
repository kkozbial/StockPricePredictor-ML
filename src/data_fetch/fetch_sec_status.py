"""
Moduł do pobierania metadanych SEC dla wykrywania delistingu i dat publikacji raportów.

Rozwiązuje:
1. Survivorship Bias - wykrywa status spółki (ACTIVE/DELISTED/DEREGISTERED/ZOMBIE)
2. Look-ahead Bias - mapuje daty publikacji raportów (filingDate) do dat sprawozdawczych (reportDate)
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOGGER = logging.getLogger("data_fetch.sec_status")


class SecMetadataFetcher:
    """
    Fetcher metadanych SEC dla wykrywania delistingu i dat publikacji raportów.

    Używa darmowego API SEC (data.sec.gov/submissions/) do:
    - Wykrywania statusu spółki (ACTIVE/DELISTED/DEREGISTERED/ZOMBIE)
    - Ekstrakcji dat publikacji raportów finansowych
    """

    # SEC wymaga User-Agent z kontaktem
    DEFAULT_USER_AGENT = "Stock Market Research Project contact@example.com"

    # SEC rate limit: 10 requests per second
    RATE_LIMIT_DELAY = 0.11  # 110ms między requestami = ~9 req/s (bezpieczny margines)

    # Formularze delistingu/deregistracji
    DELISTING_FORMS = {"25", "25-NSE"}
    DEREGISTRATION_FORMS = {"15-12G", "15-12B", "15-15D"}

    # Formularze raportów finansowych
    FINANCIAL_FORMS = {"10-K", "10-Q", "20-F", "40-F", "10-K/A", "10-Q/A"}

    # Próg dla statusu ZOMBIE (brak raportów finansowych przez X miesięcy)
    ZOMBIE_THRESHOLD_MONTHS = 18

    def __init__(
        self,
        user_agent: Optional[str] = None,
        rate_limit_delay: float = RATE_LIMIT_DELAY,
        cache_dir: Optional[Path] = None,
    ):
        """
        Inicjalizacja SecMetadataFetcher.

        Args:
            user_agent: User-Agent header (wymagany przez SEC, powinien zawierać email)
            rate_limit_delay: Opóźnienie między requestami w sekundach (default: 0.11s)
            cache_dir: Katalog do cache'owania odpowiedzi JSON (opcjonalnie)
        """
        self.user_agent = user_agent or self.DEFAULT_USER_AGENT
        self.rate_limit_delay = rate_limit_delay
        self.cache_dir = cache_dir
        self.last_request_time = 0.0

        # Konfiguracja session z retry logic
        self.session = self._create_session()

        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            LOGGER.info("Cache włączony w katalogu: %s", cache_dir)

    def _create_session(self) -> requests.Session:
        """Tworzy session z retry logic dla SEC API."""
        session = requests.Session()

        # Retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        # Headers wymagane przez SEC
        session.headers.update({
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov",
        })

        return session

    def _enforce_rate_limit(self) -> None:
        """Wymusza rate limit przez opóźnienie między requestami."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        if time_since_last_request < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last_request
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _format_cik(self, cik: str | int) -> str:
        """
        Formatuje CIK do 10-cyfrowego stringa z zerami wiodącymi.

        Args:
            cik: CIK jako string lub int

        Returns:
            CIK jako 10-cyfrowy string (np. "0001487197")
        """
        # Usuń wszystkie znaki niebędące cyframi
        cik_str = str(cik).strip()
        cik_digits = "".join(c for c in cik_str if c.isdigit())

        # Dopełnij zerami do 10 cyfr
        return cik_digits.zfill(10)

    def _get_cache_path(self, cik: str) -> Optional[Path]:
        """Zwraca ścieżkę do pliku cache dla danego CIK."""
        if not self.cache_dir:
            return None
        return self.cache_dir / f"sec_submissions_{cik}.json"

    def fetch_submissions(self, cik: str | int) -> dict:
        """
        Pobiera JSON submissions dla danego CIK z SEC API.

        Args:
            cik: CIK spółki (string lub int)

        Returns:
            Dict z danymi submissions z SEC API

        Raises:
            requests.HTTPError: Jeśli request się nie powiódł
        """
        cik_formatted = self._format_cik(cik)

        # Sprawdź cache
        cache_path = self._get_cache_path(cik_formatted)
        if cache_path and cache_path.exists():
            LOGGER.debug("Wczytywanie z cache: %s", cik_formatted)
            with open(cache_path, encoding="utf-8") as f:
                return json.load(f)

        # Pobierz z API
        url = f"https://data.sec.gov/submissions/CIK{cik_formatted}.json"

        self._enforce_rate_limit()

        LOGGER.debug("Pobieranie submissions dla CIK %s", cik_formatted)
        response = self.session.get(url, timeout=30)
        response.raise_for_status()

        data = response.json()

        # Zapisz do cache
        if cache_path:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        return data

    def _fetch_filing_file(self, cik: str | int, file_name: str) -> dict:
        """
        Pobiera dodatkowy plik zgłoszeń z SEC API (starsze dane).

        Args:
            cik: CIK spółki
            file_name: Nazwa pliku z submissions['filings']['files'] (np. "CIK0000789019-submissions-001.json")

        Returns:
            Dict z danymi zgłoszeń (ma te same klucze co 'recent': form, filingDate, reportDate, etc.)

        Raises:
            requests.HTTPError: Jeśli request się nie powiódł
        """
        cik_formatted = self._format_cik(cik)

        # Sprawdź cache
        if self.cache_dir:
            cache_path = self.cache_dir / f"sec_file_{file_name}"
            if cache_path.exists():
                LOGGER.debug("Wczytywanie z cache: %s", file_name)
                with open(cache_path, encoding="utf-8") as f:
                    return json.load(f)

        # Pobierz z API
        # URL format: https://data.sec.gov/submissions/CIK0000789019-submissions-001.json
        url = f"https://data.sec.gov/submissions/{file_name}"

        self._enforce_rate_limit()

        LOGGER.debug("Pobieranie pliku zgłoszeń %s dla CIK %s", file_name, cik_formatted)
        response = self.session.get(url, timeout=30)
        response.raise_for_status()

        data = response.json()

        # Zapisz do cache
        if self.cache_dir:
            cache_path = self.cache_dir / f"sec_file_{file_name}"
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        return data

    def _detect_status(
        self,
        forms: list[str],
        filing_dates: list[str],
    ) -> tuple[str, Optional[str], str, Optional[str]]:
        """
        Wykrywa status spółki na podstawie historii zgłoszeń.

        Args:
            forms: Lista typów formularzy z recent.form
            filing_dates: Lista dat zgłoszeń z recent.filingDate

        Returns:
            Tuple (status, delisting_date, reason, delisting_form)
        """
        if not forms or not filing_dates:
            return "UNKNOWN", None, "Brak danych o zgłoszeniach", None

        # Połącz formy z datami
        filings = list(zip(forms, filing_dates))

        # 1. Sprawdź formularze delistingu (25, 25-NSE)
        for form, date in filings:
            if form in self.DELISTING_FORMS:
                return "DELISTED", date, f"Found form {form}", form

        # 2. Sprawdź formularze deregistracji (15-12G, 15-12B, 15-15D)
        for form, date in filings:
            if form in self.DEREGISTRATION_FORMS:
                return "DEREGISTERED", date, f"Found form {form}", form

        # 3. Znajdź ostatni raport finansowy
        financial_reports = [
            (form, date) for form, date in filings
            if form in self.FINANCIAL_FORMS
        ]

        if not financial_reports:
            return "ZOMBIE", None, "Brak raportów finansowych w historii", None

        # Najnowszy raport finansowy
        latest_financial = max(financial_reports, key=lambda x: x[1])
        latest_date = datetime.strptime(latest_financial[1], "%Y-%m-%d")

        # 4. Sprawdź czy ostatni raport jest świeży
        threshold_date = datetime.now() - timedelta(days=30 * self.ZOMBIE_THRESHOLD_MONTHS)

        if latest_date < threshold_date:
            months_ago = (datetime.now() - latest_date).days // 30
            return "ZOMBIE", None, f"Last financial report {months_ago} months ago", None

        # 5. Spółka aktywna
        return "ACTIVE", None, "Recent financial reports found", None

    def generate_status_table(
        self,
        companies: list[dict[str, str]],
    ) -> pd.DataFrame:
        """
        Generuje tabelę statusów spółek (ACTIVE/DELISTED/DEREGISTERED/ZOMBIE).

        Args:
            companies: Lista słowników z kluczami 'cik' i 'ticker' (lub 'tickers')

        Returns:
            DataFrame z kolumnami:
            - ticker: Ticker spółki
            - cik: CIK (10-cyfrowy string)
            - status: ACTIVE/DELISTED/DEREGISTERED/ZOMBIE
            - delisting_date: Data delistingu (jeśli dotyczy)
            - delisting_form: Typ formularza delistingu (25, 25-NSE, 15-12G, etc.)
            - last_filing_date: Data ostatniego zgłoszenia
            - last_financial_report_date: Data ostatniego raportu finansowego
            - reason: Powód przypisania statusu
        """
        results = []

        for company in companies:
            cik = company.get("cik", "")

            # Obsługa różnych formatów tickerów
            ticker_field = company.get("ticker") or company.get("tickers")
            if isinstance(ticker_field, list):
                ticker = ticker_field[0] if ticker_field else "UNKNOWN"
            else:
                ticker = ticker_field or "UNKNOWN"

            try:
                data = self.fetch_submissions(cik)

                # Wyciągnij recent filings
                recent = data.get("filings", {}).get("recent", {})
                forms = recent.get("form", [])
                filing_dates = recent.get("filingDate", [])

                # Wykryj status
                status, delisting_date, reason, delisting_form = self._detect_status(forms, filing_dates)

                # Znajdź daty ostatnich zgłoszeń
                last_filing = filing_dates[0] if filing_dates else None

                # Znajdź ostatni raport finansowy
                financial_dates = [
                    date for form, date in zip(forms, filing_dates)
                    if form in self.FINANCIAL_FORMS
                ]
                last_financial = financial_dates[0] if financial_dates else None

                results.append({
                    "ticker": ticker,
                    "cik": self._format_cik(cik),
                    "status": status,
                    "delisting_date": delisting_date,
                    "delisting_form": delisting_form,
                    "last_filing_date": last_filing,
                    "last_financial_report_date": last_financial,
                    "reason": reason,
                })

                LOGGER.info("CIK %s (%s): %s - %s", cik, ticker, status, reason)

            except Exception as exc:
                LOGGER.error("Błąd przetwarzania CIK %s (%s): %s", cik, ticker, exc)
                results.append({
                    "ticker": ticker,
                    "cik": self._format_cik(cik),
                    "status": "ERROR",
                    "delisting_date": None,
                    "delisting_form": None,
                    "last_filing_date": None,
                    "last_financial_report_date": None,
                    "reason": str(exc),
                })

        return pd.DataFrame(results)

    def extract_filing_dates(
        self,
        companies: list[dict[str, str]],
    ) -> pd.DataFrame:
        """
        Ekstrahuje mapowanie dat raportów: reportDate (end) -> filingDate (publikacja).

        Rozwiązuje Look-ahead Bias przez dostarczenie daty faktycznej publikacji raportu.

        UWAGA: Pobiera zarówno z 'recent' (ostatnie ~1000 zgłoszeń) jak i 'files'
        (starsze zgłoszenia) aby uzyskać pełną historię.

        Args:
            companies: Lista słowników z kluczami 'cik' i 'ticker'

        Returns:
            DataFrame w formacie long z kolumnami:
            - ticker: Ticker spółki
            - cik: CIK (10-cyfrowy string)
            - form_type: Typ formularza (10-K, 10-Q, etc.)
            - end_date: Data końca okresu sprawozdawczego (reportDate) - KLUCZ DO JOINA
            - filing_date: Data publikacji raportu (filingDate) - BRAKUJĄCA DATA
            - accession_number: Numer akcesyjny SEC (dla weryfikacji)
        """
        results = []

        for company in companies:
            cik = company.get("cik", "")

            # Obsługa różnych formatów tickerów
            ticker_field = company.get("ticker") or company.get("tickers")
            if isinstance(ticker_field, list):
                ticker = ticker_field[0] if ticker_field else "UNKNOWN"
            else:
                ticker = ticker_field or "UNKNOWN"

            try:
                data = self.fetch_submissions(cik)

                # Wyciągnij recent filings
                recent = data.get("filings", {}).get("recent", {})
                forms = recent.get("form", [])
                filing_dates = recent.get("filingDate", [])
                report_dates = recent.get("reportDate", [])
                accession_numbers = recent.get("accessionNumber", [])

                # Zbierz wszystkie arrays (recent + files)
                all_forms = list(forms)
                all_filing_dates = list(filing_dates)
                all_report_dates = list(report_dates)
                all_accession_numbers = list(accession_numbers)

                # Pobierz starsze dane z 'files' endpoint (jeśli istnieją)
                files_list = data.get("filings", {}).get("files", [])
                for file_entry in files_list:
                    # Każdy file ma strukturę: {"name": "...", "filingCount": N, "filingFrom": "...", "filingTo": "..."}
                    file_name = file_entry.get("name", "")
                    if not file_name:
                        continue

                    # Pobierz dodatkowy JSON file z starszymi danymi
                    try:
                        file_data = self._fetch_filing_file(cik, file_name)
                        all_forms.extend(file_data.get("form", []))
                        all_filing_dates.extend(file_data.get("filingDate", []))
                        all_report_dates.extend(file_data.get("reportDate", []))
                        all_accession_numbers.extend(file_data.get("accessionNumber", []))

                    except Exception as file_exc:
                        LOGGER.warning("Błąd pobierania pliku %s dla CIK %s: %s", file_name, cik, file_exc)
                        continue

                # Filtruj tylko raporty finansowe
                for i, form in enumerate(all_forms):
                    if form not in self.FINANCIAL_FORMS:
                        continue

                    # Sprawdź czy mamy wszystkie potrzebne pola
                    filing_date = all_filing_dates[i] if i < len(all_filing_dates) else None
                    report_date = all_report_dates[i] if i < len(all_report_dates) else None
                    accession = all_accession_numbers[i] if i < len(all_accession_numbers) else None

                    # Pomiń rekordy bez report_date (np. formularze poprawkowe bez własnego okresu)
                    if not report_date or not filing_date:
                        continue

                    results.append({
                        "ticker": ticker,
                        "cik": self._format_cik(cik),
                        "form_type": form,
                        "end_date": report_date,  # KLUCZ - nazwa zgodna z tabelą financials
                        "filing_date": filing_date,  # BRAKUJĄCA DATA
                        "accession_number": accession,
                    })

                LOGGER.debug("CIK %s (%s): Znaleziono %d raportów finansowych",
                           cik, ticker, len([f for f in all_forms if f in self.FINANCIAL_FORMS]))

            except Exception as exc:
                LOGGER.error("Błąd ekstrakcji dat dla CIK %s (%s): %s", cik, ticker, exc)

        df = pd.DataFrame(results)

        # Konwersja dat
        if not df.empty:
            df["end_date"] = pd.to_datetime(df["end_date"])
            df["filing_date"] = pd.to_datetime(df["filing_date"])

        LOGGER.info("Wyekstrahowano %d rekordów mapowania dat raportów", len(df))

        return df


def main():
    """Przykład użycia."""
    import logging
    logging.basicConfig(level=logging.INFO)

    # Przykładowe dane wejściowe
    companies = [
        {"cik": "0001487197", "ticker": "BRFH"},
        {"cik": "0000320193", "ticker": "AAPL"},
        {"cik": "0001018724", "ticker": "AMZN"},
    ]

    # Inicjalizacja fetcher
    fetcher = SecMetadataFetcher(
        user_agent="Example Research contact@example.com",
        cache_dir=Path("data/cache/sec"),
    )

    # 1. Generuj tabelę statusów
    print("\n=== STATUS TABLE ===")
    status_df = fetcher.generate_status_table(companies)
    print(status_df.to_string(index=False))

    # 2. Ekstrahuj daty publikacji raportów
    print("\n=== FILING DATES MAPPING ===")
    filing_dates_df = fetcher.extract_filing_dates(companies)
    print(filing_dates_df.head(10).to_string(index=False))

    # Zapisz wyniki
    status_df.to_csv("data/sec_status.csv", index=False)
    filing_dates_df.to_csv("data/sec_filing_dates.csv", index=False)
    print("\n✅ Wyniki zapisane do CSV")


if __name__ == "__main__":
    main()
