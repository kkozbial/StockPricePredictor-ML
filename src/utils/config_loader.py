"""Narzedzia do wczytywania konfiguracji projektu."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is optional at runtime
    yaml = None  # type: ignore


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "settings.yaml"


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Wczytuje plik YAML i zwraca go jako slownik."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if yaml is None:
        raise RuntimeError("PyYAML is required to load configuration.")
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def get_path_from_config(section: str, key: str, config: Optional[Dict[str, Any]] = None) -> Path:
    """Zwraca sciezke plikow zdefiniowana w konfiguracji."""
    cfg = config or load_config()
    try:
        location = cfg[section][key]
    except KeyError as exc:
        raise KeyError(f"Missing config path for {section}.{key}") from exc
    return (_PROJECT_ROOT / Path(location)).resolve()
