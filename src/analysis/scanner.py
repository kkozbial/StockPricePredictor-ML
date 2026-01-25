"""Skaner tickerów z konfigurowalnymi filtrami i kryteriami."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal

import pandas as pd

from .descriptive_stats import load_master_dataset

LOGGER = logging.getLogger("analysis.scanner")


class Operator(Enum):
    """Operatory porównania dla filtrów."""
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    EQ = "=="
    NE = "!="
    BETWEEN = "between"
    IN = "in"
    NOT_IN = "not_in"
    IS_NULL = "is_null"
    NOT_NULL = "not_null"


@dataclass
class FilterCondition:
    """Pojedynczy warunek filtrowania."""
    column: str
    operator: Operator
    value: Any = None
    value2: Any = None  # Dla BETWEEN

    def apply(self, df: pd.DataFrame) -> pd.Series:
        """Stosuje warunek do DataFrame i zwraca maskę boolean."""
        if self.column not in df.columns:
            raise ValueError(f"Column '{self.column}' not found in dataset")

        series = df[self.column]

        if self.operator == Operator.LT:
            return series < self.value
        elif self.operator == Operator.LE:
            return series <= self.value
        elif self.operator == Operator.GT:
            return series > self.value
        elif self.operator == Operator.GE:
            return series >= self.value
        elif self.operator == Operator.EQ:
            return series == self.value
        elif self.operator == Operator.NE:
            return series != self.value
        elif self.operator == Operator.BETWEEN:
            return (series >= self.value) & (series <= self.value2)
        elif self.operator == Operator.IN:
            return series.isin(self.value)
        elif self.operator == Operator.NOT_IN:
            return ~series.isin(self.value)
        elif self.operator == Operator.IS_NULL:
            return series.isnull()
        elif self.operator == Operator.NOT_NULL:
            return series.notnull()
        else:
            raise ValueError(f"Unknown operator: {self.operator}")


@dataclass
class ScannerConfig:
    """Konfiguracja skanera."""
    filters: list[FilterCondition] = field(default_factory=list)
    sort_by: str | None = None
    sort_ascending: bool = True
    limit: int | None = None
    columns_to_show: list[str] | None = None
    date_filter: str | None = None  # 'latest' lub 'YYYY-MM-DD'
    sector_filter: list[str] | None = None
    country_filter: str | None = None


# Predefiniowane filtry dla popularnych strategii
PRESET_FILTERS = {
    "value_stocks": [
        FilterCondition("pe_ratio", Operator.BETWEEN, 0, 15),
        FilterCondition("price_to_book", Operator.LT, 1.5),
        FilterCondition("dividend_yield", Operator.GT, 0.02),
    ],
    "growth_stocks": [
        FilterCondition("revenue_growth", Operator.GT, 0.15),
        FilterCondition("net_income_growth", Operator.GT, 0.10),
        FilterCondition("roe", Operator.GT, 0.15),
    ],
    "quality_stocks": [
        FilterCondition("roe", Operator.GT, 0.15),
        FilterCondition("debt_to_equity", Operator.LT, 1.0),
        FilterCondition("current_ratio", Operator.GT, 1.5),
        FilterCondition("operating_margin", Operator.GT, 0.10),
    ],
    "dividend_stocks": [
        FilterCondition("dividend_yield", Operator.GT, 0.03),
        FilterCondition("payout_ratio", Operator.BETWEEN, 0.2, 0.7),
        FilterCondition("debt_to_equity", Operator.LT, 1.5),
    ],
    "low_debt": [
        FilterCondition("debt_ratio", Operator.LT, 0.3),
        FilterCondition("debt_to_equity", Operator.LT, 0.5),
    ],
    "high_profitability": [
        FilterCondition("roe", Operator.GT, 0.20),
        FilterCondition("roa", Operator.GT, 0.10),
        FilterCondition("operating_margin", Operator.GT, 0.15),
    ],
    "undervalued": [
        FilterCondition("pe_ratio", Operator.LT, 12),
        FilterCondition("price_to_book", Operator.LT, 1.0),
        FilterCondition("price_to_sales", Operator.LT, 1.0),
    ],
    "small_cap_growth": [
        FilterCondition("close", Operator.LT, 20),
        FilterCondition("revenue_growth", Operator.GT, 0.20),
    ],
    "cash_rich": [
        FilterCondition("current_ratio", Operator.GT, 2.0),
        FilterCondition("quick_ratio", Operator.GT, 1.5),
        FilterCondition("ocf_ratio", Operator.GT, 0.10),
    ],
    "turnaround": [
        FilterCondition("net_income_growth", Operator.GT, 0.50),
        FilterCondition("revenue_growth", Operator.GT, 0.10),
        FilterCondition("pe_ratio", Operator.BETWEEN, 5, 20),
    ],
}

# Domyślne kolumny do wyświetlenia w wynikach
DEFAULT_DISPLAY_COLUMNS = [
    "ticker", "date", "sector", "close",
    "pe_ratio", "price_to_book", "roe", "roa",
    "debt_to_equity", "current_ratio", "dividend_yield",
    "revenue_growth", "net_income_growth",
]


def scan_tickers(
    df: pd.DataFrame | None = None,
    db_path: Path | None = None,
    config: ScannerConfig | None = None,
    filters: list[FilterCondition] | None = None,
    preset: str | None = None,
    sort_by: str | None = None,
    sort_ascending: bool = True,
    limit: int | None = 50,
    columns: list[str] | None = None,
    date_filter: str | None = "latest",
    sector: str | list[str] | None = None,
    country: str | None = None,
) -> pd.DataFrame:
    """
    Skanuje tickery według zadanych kryteriów.

    Args:
        df: DataFrame do analizy. Jeśli None, ładuje z bazy.
        db_path: Ścieżka do bazy (używana gdy df=None).
        config: Obiekt konfiguracji skanera (jeśli podany, nadpisuje inne parametry).
        filters: Lista warunków filtrowania.
        preset: Nazwa predefiniowanego zestawu filtrów.
        sort_by: Kolumna do sortowania.
        sort_ascending: Czy sortować rosnąco.
        limit: Maksymalna liczba wyników.
        columns: Kolumny do wyświetlenia w wynikach.
        date_filter: 'latest' dla najnowszych danych lub konkretna data 'YYYY-MM-DD'.
        sector: Filtr sektora (string lub lista).
        country: Filtr kraju ('US' lub 'PL').

    Returns:
        DataFrame z przefiltrowanymi tickerami.
    """
    if df is None:
        df = load_master_dataset(db_path)

    # Użyj konfiguracji lub parametrów
    if config:
        filters = config.filters
        sort_by = config.sort_by
        sort_ascending = config.sort_ascending
        limit = config.limit
        columns = config.columns_to_show
        date_filter = config.date_filter
        sector = config.sector_filter
        country = config.country_filter

    # Użyj presetu jeśli podano
    if preset:
        if preset not in PRESET_FILTERS:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(PRESET_FILTERS.keys())}")
        filters = PRESET_FILTERS[preset]

    # Filtr daty
    if date_filter == "latest":
        # Dla każdego tickera weź najnowszą datę
        latest_dates = df.groupby("ticker")["date"].max().reset_index()
        latest_dates.columns = ["ticker", "latest_date"]
        df = df.merge(latest_dates, on="ticker")
        df = df[df["date"] == df["latest_date"]].drop(columns=["latest_date"])
    elif date_filter:
        df = df[df["date"] == pd.to_datetime(date_filter)]

    # Filtr kraju
    if country and "country" in df.columns:
        df = df[df["country"] == country]

    # Filtr sektora
    if sector and "sector" in df.columns:
        if isinstance(sector, str):
            sector = [sector]
        df = df[df["sector"].isin(sector)]

    # Zastosuj filtry
    if filters:
        mask = pd.Series([True] * len(df), index=df.index)
        for condition in filters:
            try:
                mask &= condition.apply(df)
            except ValueError as e:
                LOGGER.warning("Skipping filter: %s", e)
        df = df[mask]

    # Sortowanie
    if sort_by and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=sort_ascending)

    # Limit
    if limit:
        df = df.head(limit)

    # Wybierz kolumny do wyświetlenia
    if columns is None:
        columns = [c for c in DEFAULT_DISPLAY_COLUMNS if c in df.columns]

    return df[columns].reset_index(drop=True)


def scan_by_indicator(
    df: pd.DataFrame | None = None,
    db_path: Path | None = None,
    indicator: str = "pe_ratio",
    min_value: float | None = None,
    max_value: float | None = None,
    percentile_range: tuple[float, float] | None = None,
    top_n: int | None = None,
    bottom_n: int | None = None,
    date_filter: str | None = "latest",
) -> pd.DataFrame:
    """
    Skanuje tickery według pojedynczego wskaźnika.

    Args:
        df: DataFrame do analizy. Jeśli None, ładuje z bazy.
        db_path: Ścieżka do bazy (używana gdy df=None).
        indicator: Nazwa wskaźnika do analizy.
        min_value: Minimalna wartość wskaźnika.
        max_value: Maksymalna wartość wskaźnika.
        percentile_range: Zakres percentyli (np. (10, 90)).
        top_n: Zwróć N spółek z najwyższymi wartościami.
        bottom_n: Zwróć N spółek z najniższymi wartościami.
        date_filter: Filtr daty.

    Returns:
        DataFrame z wynikami.
    """
    if df is None:
        df = load_master_dataset(db_path)

    if indicator not in df.columns:
        raise ValueError(f"Indicator '{indicator}' not found. Available: {df.select_dtypes(include='number').columns.tolist()}")

    filters = []

    # Filtr wartości
    if min_value is not None:
        filters.append(FilterCondition(indicator, Operator.GE, min_value))
    if max_value is not None:
        filters.append(FilterCondition(indicator, Operator.LE, max_value))

    # Filtr percentylowy
    if percentile_range:
        series = df[indicator].dropna()
        lower_bound = series.quantile(percentile_range[0] / 100)
        upper_bound = series.quantile(percentile_range[1] / 100)
        filters.append(FilterCondition(indicator, Operator.BETWEEN, lower_bound, upper_bound))

    # Określ sortowanie i limit
    if top_n:
        sort_by = indicator
        sort_ascending = False
        limit = top_n
    elif bottom_n:
        sort_by = indicator
        sort_ascending = True
        limit = bottom_n
    else:
        sort_by = indicator
        sort_ascending = True
        limit = 50

    return scan_tickers(
        df=df,
        filters=filters,
        sort_by=sort_by,
        sort_ascending=sort_ascending,
        limit=limit,
        date_filter=date_filter,
    )


def compare_tickers(
    df: pd.DataFrame | None = None,
    db_path: Path | None = None,
    tickers: list[str] | None = None,
    indicators: list[str] | None = None,
    date_filter: str | None = "latest",
) -> pd.DataFrame:
    """
    Porównuje wybrane tickery pod względem wskaźników.

    Args:
        df: DataFrame do analizy. Jeśli None, ładuje z bazy.
        db_path: Ścieżka do bazy (używana gdy df=None).
        tickers: Lista tickerów do porównania.
        indicators: Lista wskaźników do porównania.
        date_filter: Filtr daty.

    Returns:
        DataFrame z porównaniem.
    """
    if df is None:
        df = load_master_dataset(db_path)

    if tickers is None:
        raise ValueError("Please provide list of tickers to compare")

    if indicators is None:
        indicators = [
            "close", "pe_ratio", "price_to_book", "roe", "roa",
            "debt_to_equity", "current_ratio", "revenue_growth",
        ]

    # Filtr daty
    if date_filter == "latest":
        latest_dates = df.groupby("ticker")["date"].max().reset_index()
        latest_dates.columns = ["ticker", "latest_date"]
        df = df.merge(latest_dates, on="ticker")
        df = df[df["date"] == df["latest_date"]].drop(columns=["latest_date"])
    elif date_filter:
        df = df[df["date"] == pd.to_datetime(date_filter)]

    # Filtruj tickery
    df = df[df["ticker"].isin(tickers)]

    # Wybierz kolumny
    cols = ["ticker", "date"] + [c for c in indicators if c in df.columns]
    if "sector" in df.columns:
        cols.insert(2, "sector")

    result = df[cols].set_index("ticker")

    return result


def sector_ranking(
    df: pd.DataFrame | None = None,
    db_path: Path | None = None,
    indicator: str = "roe",
    aggregation: Literal["mean", "median", "max", "min"] = "median",
    date_filter: str | None = "latest",
    min_companies: int = 5,
) -> pd.DataFrame:
    """
    Tworzy ranking sektorów według wskaźnika.

    Args:
        df: DataFrame do analizy. Jeśli None, ładuje z bazy.
        db_path: Ścieżka do bazy (używana gdy df=None).
        indicator: Wskaźnik do rankingu.
        aggregation: Metoda agregacji ('mean', 'median', 'max', 'min').
        date_filter: Filtr daty.
        min_companies: Minimalna liczba spółek w sektorze.

    Returns:
        DataFrame z rankingiem sektorów.
    """
    if df is None:
        df = load_master_dataset(db_path)

    if indicator not in df.columns:
        raise ValueError(f"Indicator '{indicator}' not found")

    # Filtr daty
    if date_filter == "latest":
        latest_dates = df.groupby("ticker")["date"].max().reset_index()
        latest_dates.columns = ["ticker", "latest_date"]
        df = df.merge(latest_dates, on="ticker")
        df = df[df["date"] == df["latest_date"]].drop(columns=["latest_date"])
    elif date_filter:
        df = df[df["date"] == pd.to_datetime(date_filter)]

    # Grupuj po sektorze
    if "sector" not in df.columns:
        raise ValueError("Column 'sector' not found")

    agg_funcs = {
        "mean": "mean",
        "median": "median",
        "max": "max",
        "min": "min",
    }

    grouped = df.groupby("sector").agg({
        indicator: [agg_funcs[aggregation], "count", "std"],
        "ticker": "nunique",
    })

    grouped.columns = [f"{indicator}_{aggregation}", "observations", f"{indicator}_std", "n_companies"]

    # Filtruj sektory z minimalną liczbą spółek
    grouped = grouped[grouped["n_companies"] >= min_companies]

    # Sortuj
    grouped = grouped.sort_values(f"{indicator}_{aggregation}", ascending=False)

    return grouped.round(4)


def get_ticker_profile(
    df: pd.DataFrame | None = None,
    db_path: Path | None = None,
    ticker: str = "AAPL",
    periods: int = 12,
) -> dict:
    """
    Zwraca profil spółki z najważniejszymi wskaźnikami.

    Args:
        df: DataFrame do analizy. Jeśli None, ładuje z bazy.
        db_path: Ścieżka do bazy (używana gdy df=None).
        ticker: Symbol spółki.
        periods: Liczba ostatnich okresów do analizy trendu.

    Returns:
        Słownik z profilem spółki.
    """
    if df is None:
        df = load_master_dataset(db_path)

    ticker_df = df[df["ticker"] == ticker].sort_values("date")

    if len(ticker_df) == 0:
        raise ValueError(f"Ticker '{ticker}' not found in dataset")

    latest = ticker_df.iloc[-1]
    historical = ticker_df.tail(periods)

    profile = {
        "ticker": ticker,
        "sector": latest.get("sector"),
        "industry": latest.get("industry"),
        "latest_date": str(latest.get("date")),
        "current_values": {
            "close": round(latest.get("close", 0), 2) if pd.notna(latest.get("close")) else None,
            "pe_ratio": round(latest.get("pe_ratio", 0), 2) if pd.notna(latest.get("pe_ratio")) else None,
            "price_to_book": round(latest.get("price_to_book", 0), 2) if pd.notna(latest.get("price_to_book")) else None,
            "roe": round(latest.get("roe", 0), 4) if pd.notna(latest.get("roe")) else None,
            "roa": round(latest.get("roa", 0), 4) if pd.notna(latest.get("roa")) else None,
            "debt_to_equity": round(latest.get("debt_to_equity", 0), 2) if pd.notna(latest.get("debt_to_equity")) else None,
            "current_ratio": round(latest.get("current_ratio", 0), 2) if pd.notna(latest.get("current_ratio")) else None,
            "dividend_yield": round(latest.get("dividend_yield", 0), 4) if pd.notna(latest.get("dividend_yield")) else None,
        },
        "growth": {
            "revenue_growth": round(latest.get("revenue_growth", 0), 4) if pd.notna(latest.get("revenue_growth")) else None,
            "net_income_growth": round(latest.get("net_income_growth", 0), 4) if pd.notna(latest.get("net_income_growth")) else None,
        },
        "trends": {
            "price_trend": _calculate_trend(historical, "close"),
            "pe_trend": _calculate_trend(historical, "pe_ratio"),
            "roe_trend": _calculate_trend(historical, "roe"),
        },
        "data_points": len(ticker_df),
        "date_range": {
            "first": str(ticker_df["date"].min()),
            "last": str(ticker_df["date"].max()),
        },
    }

    return profile


def _calculate_trend(df: pd.DataFrame, column: str) -> str | None:
    """Oblicza trend dla kolumny (rosnący/malejący/stabilny)."""
    if column not in df.columns:
        return None

    series = df[column].dropna()
    if len(series) < 2:
        return None

    first_half = series.head(len(series) // 2).mean()
    second_half = series.tail(len(series) // 2).mean()

    if pd.isna(first_half) or pd.isna(second_half):
        return None

    pct_change = (second_half - first_half) / abs(first_half) if first_half != 0 else 0

    if pct_change > 0.05:
        return "rising"
    elif pct_change < -0.05:
        return "falling"
    else:
        return "stable"


def list_presets() -> dict[str, list[str]]:
    """Zwraca listę dostępnych presetów z opisem filtrów."""
    result = {}
    for name, filters in PRESET_FILTERS.items():
        result[name] = [
            f"{f.column} {f.operator.value} {f.value}" +
            (f" AND {f.value2}" if f.value2 else "")
            for f in filters
        ]
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    print("\n=== Available Presets ===")
    for name, filters in list_presets().items():
        print(f"\n{name}:")
        for f in filters:
            print(f"  - {f}")

    print("\n=== Value Stocks Scan ===")
    value_stocks = scan_tickers(preset="value_stocks", limit=10)
    print(value_stocks.to_string())

    print("\n=== Top 10 by ROE ===")
    top_roe = scan_by_indicator(indicator="roe", top_n=10)
    print(top_roe.to_string())

    print("\n=== Sector Ranking by ROE ===")
    sector_rank = sector_ranking(indicator="roe")
    print(sector_rank.to_string())
