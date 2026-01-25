"""
Testy jednostkowe dla modułu fetch_sec_status.py.

Testuje:
1. Formatowanie CIK
2. Wykrywanie statusu spółki (ACTIVE/DELISTED/DEREGISTERED/ZOMBIE)
3. Generowanie tabeli statusów
4. Ekstrakcję dat publikacji raportów
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data_fetch.fetch_sec_status import SecMetadataFetcher


class TestCIKFormatting:
    """Testy formatowania CIK."""

    def test_format_cik_string(self):
        """Test formatowania CIK jako string."""
        fetcher = SecMetadataFetcher()
        assert fetcher._format_cik("1487197") == "0001487197"
        assert fetcher._format_cik("0001487197") == "0001487197"
        assert fetcher._format_cik("320193") == "0000320193"

    def test_format_cik_integer(self):
        """Test formatowania CIK jako int."""
        fetcher = SecMetadataFetcher()
        assert fetcher._format_cik(1487197) == "0001487197"
        assert fetcher._format_cik(320193) == "0000320193"

    def test_format_cik_with_spaces(self):
        """Test formatowania CIK ze spacjami."""
        fetcher = SecMetadataFetcher()
        assert fetcher._format_cik("  1487197  ") == "0001487197"


class TestStatusDetection:
    """Testy wykrywania statusu spółki."""

    def test_detect_delisted_status(self):
        """Test wykrywania statusu DELISTED."""
        fetcher = SecMetadataFetcher()

        forms = ["10-K", "10-Q", "25-NSE", "8-K"]
        dates = ["2023-03-15", "2023-06-15", "2024-01-15", "2024-02-01"]

        status, delisting_date, reason = fetcher._detect_status(forms, dates)

        assert status == "DELISTED"
        assert delisting_date == "2024-01-15"
        assert "25-NSE" in reason

    def test_detect_deregistered_status(self):
        """Test wykrywania statusu DEREGISTERED."""
        fetcher = SecMetadataFetcher()

        forms = ["10-K", "10-Q", "15-12G"]
        dates = ["2023-03-15", "2023-06-15", "2024-01-15"]

        status, delisting_date, reason = fetcher._detect_status(forms, dates)

        assert status == "DEREGISTERED"
        assert delisting_date == "2024-01-15"
        assert "15-12G" in reason

    def test_detect_zombie_status(self):
        """Test wykrywania statusu ZOMBIE (brak raportów > 18 miesięcy)."""
        fetcher = SecMetadataFetcher()

        # Ostatni raport sprzed 24 miesięcy
        old_date = (datetime.now() - timedelta(days=24 * 30)).strftime("%Y-%m-%d")

        forms = ["10-K", "8-K"]
        dates = [old_date, old_date]

        status, delisting_date, reason = fetcher._detect_status(forms, dates)

        assert status == "ZOMBIE"
        assert delisting_date is None
        assert "months ago" in reason

    def test_detect_active_status(self):
        """Test wykrywania statusu ACTIVE."""
        fetcher = SecMetadataFetcher()

        # Ostatni raport z ostatniego miesiąca
        recent_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        forms = ["10-K", "10-Q", "8-K"]
        dates = ["2023-03-15", recent_date, recent_date]

        status, delisting_date, reason = fetcher._detect_status(forms, dates)

        assert status == "ACTIVE"
        assert delisting_date is None
        assert "Recent" in reason

    def test_detect_empty_filings(self):
        """Test wykrywania statusu przy braku zgłoszeń."""
        fetcher = SecMetadataFetcher()

        status, delisting_date, reason = fetcher._detect_status([], [])

        assert status == "UNKNOWN"
        assert delisting_date is None
        assert "Brak danych" in reason


class TestGenerateStatusTable:
    """Testy generowania tabeli statusów."""

    @patch.object(SecMetadataFetcher, "fetch_submissions")
    def test_generate_status_table_active(self, mock_fetch):
        """Test generowania tabeli statusów dla aktywnej spółki."""
        # Mock response z SEC API
        recent_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        mock_fetch.return_value = {
            "filings": {
                "recent": {
                    "form": ["10-K", "10-Q", "8-K"],
                    "filingDate": ["2023-03-15", recent_date, recent_date],
                }
            }
        }

        fetcher = SecMetadataFetcher()
        companies = [{"cik": "0000320193", "ticker": "AAPL"}]

        df = fetcher.generate_status_table(companies)

        assert len(df) == 1
        assert df.iloc[0]["ticker"] == "AAPL"
        assert df.iloc[0]["cik"] == "0000320193"
        assert df.iloc[0]["status"] == "ACTIVE"
        assert df.iloc[0]["last_filing_date"] == "2023-03-15"

    @patch.object(SecMetadataFetcher, "fetch_submissions")
    def test_generate_status_table_delisted(self, mock_fetch):
        """Test generowania tabeli statusów dla zdelistowanej spółki."""
        mock_fetch.return_value = {
            "filings": {
                "recent": {
                    "form": ["10-K", "25-NSE"],
                    "filingDate": ["2023-03-15", "2024-01-15"],
                }
            }
        }

        fetcher = SecMetadataFetcher()
        companies = [{"cik": "0001487197", "ticker": "BRFH"}]

        df = fetcher.generate_status_table(companies)

        assert len(df) == 1
        assert df.iloc[0]["status"] == "DELISTED"
        assert df.iloc[0]["delisting_date"] == "2024-01-15"

    @patch.object(SecMetadataFetcher, "fetch_submissions")
    def test_generate_status_table_error_handling(self, mock_fetch):
        """Test obsługi błędów przy generowaniu tabeli statusów."""
        mock_fetch.side_effect = Exception("API Error")

        fetcher = SecMetadataFetcher()
        companies = [{"cik": "0000000000", "ticker": "TEST"}]

        df = fetcher.generate_status_table(companies)

        assert len(df) == 1
        assert df.iloc[0]["status"] == "ERROR"
        assert "API Error" in df.iloc[0]["reason"]


class TestExtractFilingDates:
    """Testy ekstrakcji dat publikacji raportów."""

    @patch.object(SecMetadataFetcher, "fetch_submissions")
    def test_extract_filing_dates_basic(self, mock_fetch):
        """Test ekstrakcji dat publikacji raportów."""
        mock_fetch.return_value = {
            "filings": {
                "recent": {
                    "form": ["10-K", "10-Q", "8-K"],
                    "filingDate": ["2024-02-15", "2024-05-15", "2024-06-01"],
                    "reportDate": ["2023-12-31", "2024-03-31", ""],
                    "accessionNumber": ["0001234567-24-000001", "0001234567-24-000002", "0001234567-24-000003"],
                }
            }
        }

        fetcher = SecMetadataFetcher()
        companies = [{"cik": "0000320193", "ticker": "AAPL"}]

        df = fetcher.extract_filing_dates(companies)

        # Powinno być 2 rekordy (8-K nie ma reportDate)
        assert len(df) == 2
        assert df.iloc[0]["ticker"] == "AAPL"
        assert df.iloc[0]["form_type"] == "10-K"
        assert df.iloc[0]["end_date"] == pd.Timestamp("2023-12-31")
        assert df.iloc[0]["filing_date"] == pd.Timestamp("2024-02-15")

    @patch.object(SecMetadataFetcher, "fetch_submissions")
    def test_extract_filing_dates_filters_non_financial(self, mock_fetch):
        """Test filtrowania raportów niefinansowych."""
        mock_fetch.return_value = {
            "filings": {
                "recent": {
                    "form": ["10-K", "8-K", "S-1", "DEF 14A"],
                    "filingDate": ["2024-02-15", "2024-03-15", "2024-04-15", "2024-05-15"],
                    "reportDate": ["2023-12-31", "", "", ""],
                    "accessionNumber": ["0001234567-24-000001", "0001234567-24-000002", "0001234567-24-000003", "0001234567-24-000004"],
                }
            }
        }

        fetcher = SecMetadataFetcher()
        companies = [{"cik": "0000320193", "ticker": "AAPL"}]

        df = fetcher.extract_filing_dates(companies)

        # Tylko 10-K ma reportDate i jest raportem finansowym
        assert len(df) == 1
        assert df.iloc[0]["form_type"] == "10-K"

    @patch.object(SecMetadataFetcher, "fetch_submissions")
    def test_extract_filing_dates_handles_amendments(self, mock_fetch):
        """Test obsługi raportów poprawkowych (10-K/A, 10-Q/A)."""
        mock_fetch.return_value = {
            "filings": {
                "recent": {
                    "form": ["10-K", "10-K/A"],
                    "filingDate": ["2024-02-15", "2024-03-01"],
                    "reportDate": ["2023-12-31", "2023-12-31"],
                    "accessionNumber": ["0001234567-24-000001", "0001234567-24-000002"],
                }
            }
        }

        fetcher = SecMetadataFetcher()
        companies = [{"cik": "0000320193", "ticker": "AAPL"}]

        df = fetcher.extract_filing_dates(companies)

        # Powinno być 2 rekordy (oryginał + poprawka)
        assert len(df) == 2
        assert all(df["end_date"] == pd.Timestamp("2023-12-31"))


class TestRateLimiting:
    """Testy rate limiting."""

    def test_rate_limit_delay(self):
        """Test opóźnień między requestami."""
        import time

        fetcher = SecMetadataFetcher(rate_limit_delay=0.2)

        start_time = time.time()
        fetcher._enforce_rate_limit()
        first_call_time = time.time()
        fetcher._enforce_rate_limit()
        second_call_time = time.time()

        # Pierwsze wywołanie powinno być natychmiastowe
        assert first_call_time - start_time < 0.1

        # Drugie wywołanie powinno być opóźnione o ~0.2s
        assert second_call_time - first_call_time >= 0.2


class TestCaching:
    """Testy cache'owania odpowiedzi."""

    def test_cache_directory_creation(self, tmp_path):
        """Test tworzenia katalogu cache."""
        cache_dir = tmp_path / "cache"
        fetcher = SecMetadataFetcher(cache_dir=cache_dir)

        assert cache_dir.exists()

    @patch("requests.Session.get")
    def test_cache_hit(self, mock_get, tmp_path):
        """Test użycia cache przy powtórnym requestcie."""
        cache_dir = tmp_path / "cache"
        fetcher = SecMetadataFetcher(cache_dir=cache_dir)

        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"test": "data"}
        mock_get.return_value = mock_response

        # Pierwsze wywołanie - powinno trafić do API
        result1 = fetcher.fetch_submissions("0000320193")
        assert mock_get.call_count == 1

        # Drugie wywołanie - powinno użyć cache
        result2 = fetcher.fetch_submissions("0000320193")
        assert mock_get.call_count == 1  # Nie powinno być nowego requesta
        assert result1 == result2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
