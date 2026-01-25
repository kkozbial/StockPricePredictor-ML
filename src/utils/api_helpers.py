"""Pomocnicze funkcje do odpornej komunikacji z API."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import requests
from requests.adapters import HTTPAdapter, Retry


LOGGER = logging.getLogger("utils.api")


def build_session(retries: int = 3, backoff: float = 0.5, status_forcelist: Optional[list[int]] = None) -> requests.Session:
    """Zwraca obiekt Session z wlaczonym mechanizmem ponawiania."""
    status = status_forcelist or [429, 500, 502, 503, 504]
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=status,
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def safe_get(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 10.0,
) -> requests.Response:
    """Wykonuje zapytanie HTTP GET z obsługą ponawiania i opcjonalnych nagłówków."""
    session = build_session()
    LOGGER.debug("[safe_get] Requesting %s (params=%s)", url, params)
    response = session.get(url, params=params, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response

