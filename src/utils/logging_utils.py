"""Narzedzia do konfiguracji logowania w projekcie."""
from __future__ import annotations

import logging
import logging.config
from pathlib import Path

from .config_loader import DEFAULT_CONFIG_PATH, get_path_from_config, load_config
from .io_helpers import ensure_dir


def setup_logging(config_path: Path | None = None) -> None:
    """Konfiguruje logowanie na podstawie pliku logging.conf."""
    load_config()  # ensure configuration is readable before configuring logs
    log_cfg = Path(config_path) if config_path else DEFAULT_CONFIG_PATH.parent / "logging.conf"
    try:
        log_dir = get_path_from_config("paths", "logs")
        ensure_dir(log_dir)
    except Exception:
        # Nie przerywamy konfiguracji logowania, jesli katalogu nie da sie utworzyc.
        pass
    logging.config.fileConfig(log_cfg, disable_existing_loggers=False)
    logging.captureWarnings(True)
    logging.getLogger("root").info("Logging initialised using %s", log_cfg)


def get_data_path(category: str) -> Path:
    """Skraca dostep do sciezek danych z konfiguracji."""
    return get_path_from_config("paths", category)
