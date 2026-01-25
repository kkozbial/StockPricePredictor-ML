"""Pomoc przy odczycie i zapisie danych projektowych."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def ensure_dir(path: Path) -> Path:
    """Tworzy katalog jesli nie istnieje i zwraca go."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv(path: Path, parse_dates: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Wczytuje plik CSV do ramki danych."""
    return pd.read_csv(path, parse_dates=parse_dates)


def write_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Zapisuje ramke danych do CSV i tworzy brakujace katalogi."""
    ensure_dir(path.parent)
    df.to_csv(path, index=index)
