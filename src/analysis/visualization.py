"""Wizualizacje rozkładów i wykresów analitycznych."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .descriptive_stats import load_master_dataset, DEFAULT_NUMERIC_COLS

LOGGER = logging.getLogger("analysis.visualization")

# Konfiguracja stylu
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def plot_distribution(
    df: pd.DataFrame | None = None,
    db_path: Path | None = None,
    column: str = "pe_ratio",
    output_path: Path | None = None,
    figsize: tuple[int, int] = (12, 5),
    bins: int | str = "auto",
    log_scale: bool = False,
    clip_percentile: tuple[float, float] | None = (1, 95),
) -> Path | None:
    """
    Tworzy wykres rozkładu zmiennej (histogram + KDE + boxplot).

    Args:
        df: DataFrame do analizy. Jeśli None, ładuje z bazy.
        db_path: Ścieżka do bazy (używana gdy df=None).
        column: Nazwa kolumny do wizualizacji.
        output_path: Ścieżka do zapisu. Jeśli None, wyświetla.
        figsize: Rozmiar figury.
        bins: Liczba przedziałów histogramu lub 'auto'.
        log_scale: Czy użyć skali logarytmicznej na osi X.
        clip_percentile: Percentyle do przycięcia danych (min, max) lub None.

    Returns:
        Ścieżka do zapisanego pliku lub None.
    """
    if df is None:
        df = load_master_dataset(db_path)

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataset")

    series = df[column].dropna()

    # Przycięcie ekstremów
    if clip_percentile:
        lower = series.quantile(clip_percentile[0] / 100)
        upper = series.quantile(clip_percentile[1] / 100)
        series = series[(series >= lower) & (series <= upper)]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Histogram + KDE
    ax1 = axes[0]
    sns.histplot(series, bins=bins, kde=True, ax=ax1, color="steelblue", alpha=0.7)
    ax1.axvline(series.mean(), color="red", linestyle="--", label=f"Mean: {series.mean():.2f}")
    ax1.axvline(series.median(), color="green", linestyle="-.", label=f"Median: {series.median():.2f}")
    ax1.set_xlabel(column)
    ax1.set_ylabel("Frequency")
    ax1.set_title(f"Rozkład: {column}")
    ax1.legend(fontsize=9)

    if log_scale and series.min() > 0:
        ax1.set_xscale("log")

    # Boxplot
    ax2 = axes[1]
    sns.boxplot(x=series, ax=ax2, color="steelblue")
    ax2.set_xlabel(column)
    ax2.set_title(f"Boxplot: {column}")

    # Statystyki w tekście
    stats_text = (
        f"n = {len(series):,}\n"
        f"Mean = {series.mean():.3f}\n"
        f"Std = {series.std():.3f}\n"
        f"Min = {series.min():.3f}\n"
        f"Max = {series.max():.3f}\n"
        f"Skew = {series.skew():.3f}"
    )
    ax2.text(
        0.98, 0.98, stats_text,
        transform=ax2.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        LOGGER.info("Saved distribution plot to %s", output_path)
        return output_path

    plt.show()
    return None


def plot_distributions_grid(
    df: pd.DataFrame | None = None,
    db_path: Path | None = None,
    columns: list[str] | None = None,
    output_path: Path | None = None,
    ncols: int = 3,
    figsize_per_plot: tuple[float, float] = (4, 3),
    clip_percentile: tuple[float, float] | None = (1, 95),
) -> Path | None:
    """
    Tworzy siatkę histogramów dla wielu zmiennych.

    Args:
        df: DataFrame do analizy. Jeśli None, ładuje z bazy.
        db_path: Ścieżka do bazy (używana gdy df=None).
        columns: Lista kolumn. Jeśli None, używa domyślnych.
        output_path: Ścieżka do zapisu.
        ncols: Liczba kolumn w siatce.
        figsize_per_plot: Rozmiar pojedynczego wykresu.
        clip_percentile: Percentyle do przycięcia danych.

    Returns:
        Ścieżka do zapisanego pliku lub None.
    """
    if df is None:
        df = load_master_dataset(db_path)

    if columns is None:
        columns = [c for c in DEFAULT_NUMERIC_COLS if c in df.columns][:12]

    n_plots = len(columns)
    nrows = (n_plots + ncols - 1) // ncols

    figsize = (figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if n_plots > 1 else [axes]

    for i, col in enumerate(columns):
        ax = axes[i]
        series = df[col].dropna()

        if clip_percentile and len(series) > 0:
            lower = series.quantile(clip_percentile[0] / 100)
            upper = series.quantile(clip_percentile[1] / 100)
            series = series[(series >= lower) & (series <= upper)]

        if len(series) > 0:
            sns.histplot(series, kde=True, ax=ax, color="steelblue", alpha=0.7)
            ax.axvline(series.median(), color="red", linestyle="--", linewidth=1)

        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
        ax.tick_params(axis="both", labelsize=8)

    # Ukryj puste osie
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Rozkłady zmiennych", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        LOGGER.info("Saved distributions grid to %s", output_path)
        return output_path

    plt.show()
    return None


def plot_distribution_by_sector(
    df: pd.DataFrame | None = None,
    db_path: Path | None = None,
    column: str = "pe_ratio",
    output_path: Path | None = None,
    figsize: tuple[int, int] = (14, 8),
    top_n_sectors: int = 10,
    clip_percentile: tuple[float, float] | None = (1, 95),
) -> Path | None:
    """
    Tworzy boxplot rozkładu zmiennej per sektor.

    Args:
        df: DataFrame do analizy. Jeśli None, ładuje z bazy.
        db_path: Ścieżka do bazy (używana gdy df=None).
        column: Kolumna do wizualizacji.
        output_path: Ścieżka do zapisu.
        figsize: Rozmiar figury.
        top_n_sectors: Liczba top sektorów do wyświetlenia.
        clip_percentile: Percentyle do przycięcia danych.

    Returns:
        Ścieżka do zapisanego pliku lub None.
    """
    if df is None:
        df = load_master_dataset(db_path)

    if column not in df.columns or "sector" not in df.columns:
        raise ValueError(f"Required columns not found")

    # Wybierz top N sektorów
    top_sectors = df["sector"].value_counts().head(top_n_sectors).index.tolist()
    plot_df = df[df["sector"].isin(top_sectors)][[column, "sector"]].dropna()

    # Przycięcie ekstremów
    if clip_percentile:
        lower = plot_df[column].quantile(clip_percentile[0] / 100)
        upper = plot_df[column].quantile(clip_percentile[1] / 100)
        plot_df = plot_df[(plot_df[column] >= lower) & (plot_df[column] <= upper)]

    # Sortuj sektory według mediany
    sector_medians = plot_df.groupby("sector")[column].median().sort_values(ascending=False)
    sector_order = sector_medians.index.tolist()

    fig, ax = plt.subplots(figsize=figsize)

    sns.boxplot(
        data=plot_df,
        x="sector",
        y=column,
        order=sector_order,
        ax=ax,
        palette="husl",
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Sektor", fontsize=11)
    ax.set_ylabel(column, fontsize=11)
    ax.set_title(f"Rozkład {column} wg sektora", fontsize=14, fontweight="bold")

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        LOGGER.info("Saved sector distribution to %s", output_path)
        return output_path

    plt.show()
    return None


def plot_violin_comparison(
    df: pd.DataFrame | None = None,
    db_path: Path | None = None,
    columns: list[str] | None = None,
    output_path: Path | None = None,
    figsize: tuple[int, int] = (14, 6),
    clip_percentile: tuple[float, float] | None = (1, 95),
) -> Path | None:
    """
    Tworzy wykres skrzypcowy dla porównania wielu zmiennych.

    Args:
        df: DataFrame do analizy. Jeśli None, ładuje z bazy.
        db_path: Ścieżka do bazy (używana gdy df=None).
        columns: Lista kolumn do porównania.
        output_path: Ścieżka do zapisu.
        figsize: Rozmiar figury.
        clip_percentile: Percentyle do przycięcia.

    Returns:
        Ścieżka do zapisanego pliku lub None.
    """
    if df is None:
        df = load_master_dataset(db_path)

    if columns is None:
        columns = ["roe", "roa", "ros", "gross_margin", "operating_margin"]
        columns = [c for c in columns if c in df.columns]

    # Przygotuj dane w formacie long
    plot_data = []
    for col in columns:
        series = df[col].dropna()
        if clip_percentile and len(series) > 0:
            lower = series.quantile(clip_percentile[0] / 100)
            upper = series.quantile(clip_percentile[1] / 100)
            series = series[(series >= lower) & (series <= upper)]
        for val in series:
            plot_data.append({"variable": col, "value": val})

    plot_df = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=figsize)

    sns.violinplot(
        data=plot_df,
        x="variable",
        y="value",
        ax=ax,
        palette="husl",
        inner="quartile",
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=10)
    ax.set_xlabel("Wskaźnik", fontsize=11)
    ax.set_ylabel("Wartość", fontsize=11)
    ax.set_title("Porównanie rozkładów wskaźników", fontsize=14, fontweight="bold")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        LOGGER.info("Saved violin plot to %s", output_path)
        return output_path

    plt.show()
    return None


def plot_time_series(
    df: pd.DataFrame | None = None,
    db_path: Path | None = None,
    ticker: str | None = None,
    columns: list[str] | None = None,
    output_path: Path | None = None,
    figsize: tuple[int, int] = (14, 6),
    normalize: bool = False,
) -> Path | None:
    """
    Tworzy wykres szeregu czasowego dla wybranego tickera.

    Args:
        df: DataFrame do analizy. Jeśli None, ładuje z bazy.
        db_path: Ścieżka do bazy (używana gdy df=None).
        ticker: Symbol spółki. Jeśli None, wybiera pierwszą dostępną.
        columns: Kolumny do wyświetlenia.
        output_path: Ścieżka do zapisu.
        figsize: Rozmiar figury.
        normalize: Czy znormalizować wartości do zakresu [0, 1].

    Returns:
        Ścieżka do zapisanego pliku lub None.
    """
    if df is None:
        df = load_master_dataset(db_path)

    if ticker is None:
        ticker = df["ticker"].iloc[0]

    if columns is None:
        columns = ["close", "pe_ratio", "roe"]
        columns = [c for c in columns if c in df.columns]

    ticker_df = df[df["ticker"] == ticker].sort_values("date")

    if len(ticker_df) == 0:
        raise ValueError(f"No data found for ticker: {ticker}")

    fig, ax = plt.subplots(figsize=figsize)

    for col in columns:
        series = ticker_df.set_index("date")[col]

        if normalize and series.notna().any():
            series = (series - series.min()) / (series.max() - series.min())

        ax.plot(series.index, series.values, label=col, linewidth=1.5)

    ax.set_xlabel("Data", fontsize=11)
    ax.set_ylabel("Wartość" + (" (znormalizowana)" if normalize else ""), fontsize=11)
    ax.set_title(f"Szereg czasowy: {ticker}", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        LOGGER.info("Saved time series plot to %s", output_path)
        return output_path

    plt.show()
    return None


def plot_price_history(
    df: pd.DataFrame,
    tickers: list[str],
    output_path: Path,
    figsize: tuple[int, int] = (12, 6),
) -> Path:
    """
    Rysuje historię cen zamknięcia dla wybranych spółek.

    Zachowana dla kompatybilności wstecznej.

    Args:
        df: DataFrame z danymi.
        tickers: Lista tickerów.
        output_path: Ścieżka do zapisu.
        figsize: Rozmiar figury.

    Returns:
        Ścieżka do zapisanego pliku.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for ticker in tickers:
        subset = df[df["ticker"] == ticker].sort_values("date")
        if len(subset) > 0:
            ax.plot(subset["date"], subset["close"], label=ticker, linewidth=1.5)

    ax.set_title("Historia cen", fontsize=14, fontweight="bold")
    ax.set_xlabel("Data", fontsize=11)
    ax.set_ylabel("Cena zamknięcia", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    LOGGER.info("Saved price history to %s", output_path)
    return output_path


def plot_sector_composition(
    df: pd.DataFrame | None = None,
    db_path: Path | None = None,
    output_path: Path | None = None,
    figsize: tuple[int, int] = (12, 6),
    top_n: int = 15,
) -> Path | None:
    """
    Tworzy wykres słupkowy składu sektorowego.

    Args:
        df: DataFrame do analizy. Jeśli None, ładuje z bazy.
        db_path: Ścieżka do bazy (używana gdy df=None).
        output_path: Ścieżka do zapisu.
        figsize: Rozmiar figury.
        top_n: Liczba top sektorów.

    Returns:
        Ścieżka do zapisanego pliku lub None.
    """
    if df is None:
        df = load_master_dataset(db_path)

    if "sector" not in df.columns:
        raise ValueError("Column 'sector' not found")

    # Policz unikalne tickery per sektor
    sector_counts = df.groupby("sector")["ticker"].nunique().sort_values(ascending=False)
    sector_counts = sector_counts.head(top_n)

    fig, ax = plt.subplots(figsize=figsize)

    colors = sns.color_palette("husl", len(sector_counts))
    bars = ax.barh(sector_counts.index[::-1], sector_counts.values[::-1], color=colors[::-1])

    # Dodaj wartości na słupkach
    for bar, val in zip(bars, sector_counts.values[::-1]):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                f"{val}", va="center", fontsize=9)

    ax.set_xlabel("Liczba spółek", fontsize=11)
    ax.set_ylabel("Sektor", fontsize=11)
    ax.set_title("Skład sektorowy datasetu", fontsize=14, fontweight="bold")

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        LOGGER.info("Saved sector composition to %s", output_path)
        return output_path

    plt.show()
    return None


def plot_missing_values(
    df: pd.DataFrame | None = None,
    db_path: Path | None = None,
    output_path: Path | None = None,
    figsize: tuple[int, int] = (14, 8),
    top_n: int = 30,
) -> Path | None:
    """
    Wizualizuje brakujące wartości w datasecie.

    Args:
        df: DataFrame do analizy. Jeśli None, ładuje z bazy.
        db_path: Ścieżka do bazy (używana gdy df=None).
        output_path: Ścieżka do zapisu.
        figsize: Rozmiar figury.
        top_n: Liczba kolumn do wyświetlenia.

    Returns:
        Ścieżka do zapisanego pliku lub None.
    """
    if df is None:
        df = load_master_dataset(db_path)

    # Oblicz % brakujących
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing_pct = missing_pct.head(top_n)

    fig, ax = plt.subplots(figsize=figsize)

    colors = ["red" if x > 50 else "orange" if x > 20 else "green" for x in missing_pct.values]
    bars = ax.barh(missing_pct.index[::-1], missing_pct.values[::-1], color=colors[::-1])

    # Dodaj wartości
    for bar, val in zip(bars, missing_pct.values[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=8)

    ax.set_xlabel("% brakujących wartości", fontsize=11)
    ax.set_ylabel("Kolumna", fontsize=11)
    ax.set_title("Brakujące wartości w datasecie", fontsize=14, fontweight="bold")
    ax.axvline(50, color="red", linestyle="--", alpha=0.5, label=">50%")
    ax.axvline(20, color="orange", linestyle="--", alpha=0.5, label=">20%")
    ax.legend(loc="lower right")

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        LOGGER.info("Saved missing values plot to %s", output_path)
        return output_path

    plt.show()
    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    # Przykładowe użycie
    print("Generating sample visualizations...")

    # Rozkład P/E
    plot_distribution(column="pe_ratio", output_path=Path("reports/dist_pe_ratio.png"))

    # Siatka rozkładów
    plot_distributions_grid(output_path=Path("reports/distributions_grid.png"))

    # Rozkład per sektor
    plot_distribution_by_sector(column="roe", output_path=Path("reports/roe_by_sector.png"))

    print("Done!")
