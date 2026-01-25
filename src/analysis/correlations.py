"""Analiza korelacji z wizualizacją heatmap."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .descriptive_stats import load_master_dataset, DEFAULT_NUMERIC_COLS

LOGGER = logging.getLogger("analysis.correlations")

# Grupy wskaźników do analizy korelacji
INDICATOR_GROUPS = {
    "valuation": ["pe_ratio", "price_to_book", "price_to_sales", "price_to_earnings"],
    "profitability": ["roe", "roa", "ros", "gross_margin", "operating_margin"],
    "leverage": ["debt_ratio", "debt_to_equity"],
    "liquidity": ["current_ratio", "quick_ratio"],
    "efficiency": ["asset_turnover", "inventory_turnover"],
    "cashflow": ["ocf_ratio", "capex_ratio", "free_cash_flow", "fcf_ttm"],
    "growth": ["revenue_growth", "net_income_growth"],
    "dividends": ["dividend_yield", "payout_ratio"],
    "macro": ["gdp_yoy", "cpi_yoy", "unemp_level", "rate_level"],
}


def correlation_matrix(
    df: pd.DataFrame | None = None,
    db_path: Path | None = None,
    columns: list[str] | None = None,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
) -> pd.DataFrame:
    """
    Oblicza macierz korelacji dla kolumn numerycznych.

    Args:
        df: DataFrame do analizy. Jeśli None, ładuje z bazy.
        db_path: Ścieżka do bazy (używana gdy df=None).
        columns: Lista kolumn do analizy. Jeśli None, używa domyślnych.
        method: Metoda korelacji ('pearson', 'spearman', 'kendall').

    Returns:
        Macierz korelacji jako DataFrame.
    """
    if df is None:
        df = load_master_dataset(db_path)

    if columns is None:
        columns = [c for c in DEFAULT_NUMERIC_COLS if c in df.columns]

    numeric_df = df[columns].select_dtypes(include="number")

    return numeric_df.corr(method=method)


def correlation_with_target(
    df: pd.DataFrame | None = None,
    db_path: Path | None = None,
    target_col: str = "revenue_growth",
    columns: list[str] | None = None,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
) -> pd.DataFrame:
    """
    Oblicza korelację wszystkich zmiennych z wybraną zmienną docelową.

    Args:
        df: DataFrame do analizy. Jeśli None, ładuje z bazy.
        db_path: Ścieżka do bazy (używana gdy df=None).
        target_col: Kolumna docelowa do korelacji.
        columns: Lista kolumn do analizy. Jeśli None, używa domyślnych.
        method: Metoda korelacji.

    Returns:
        DataFrame z korelacjami posortowany malejąco.
    """
    if df is None:
        df = load_master_dataset(db_path)

    if columns is None:
        columns = [c for c in DEFAULT_NUMERIC_COLS if c in df.columns]

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")

    # Dodaj target do kolumn jeśli go nie ma
    if target_col not in columns:
        columns = [target_col] + columns

    numeric_df = df[columns].select_dtypes(include="number")

    # Oblicz korelacje z targetem
    correlations = numeric_df.corr(method=method)[target_col].drop(target_col)

    result = pd.DataFrame({
        "variable": correlations.index,
        "correlation": correlations.values,
        "abs_correlation": abs(correlations.values),
    })

    return result.sort_values("abs_correlation", ascending=False).reset_index(drop=True)


def plot_correlation_heatmap(
    df: pd.DataFrame | None = None,
    db_path: Path | None = None,
    columns: list[str] | None = None,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
    output_path: Path | None = None,
    figsize: tuple[int, int] = (14, 12),
    cmap: str = "RdBu_r",
    annot: bool = True,
    title: str | None = None,
) -> Path | None:
    """
    Tworzy wizualizację heatmap macierzy korelacji.

    Args:
        df: DataFrame do analizy. Jeśli None, ładuje z bazy.
        db_path: Ścieżka do bazy (używana gdy df=None).
        columns: Lista kolumn. Jeśli None, używa domyślnych.
        method: Metoda korelacji.
        output_path: Ścieżka do zapisu wykresu. Jeśli None, wyświetla.
        figsize: Rozmiar figury.
        cmap: Paleta kolorów.
        annot: Czy wyświetlać wartości na heatmapie.
        title: Tytuł wykresu.

    Returns:
        Ścieżka do zapisanego pliku lub None.
    """
    corr_matrix = correlation_matrix(df, db_path, columns, method)

    fig, ax = plt.subplots(figsize=figsize)

    # Maska dla górnego trójkąta (opcjonalnie)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=annot,
        fmt=".2f",
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": f"Korelacja ({method})"},
        ax=ax,
        annot_kws={"size": 8},
    )

    if title is None:
        title = f"Macierz korelacji ({method.capitalize()})"
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        LOGGER.info("Saved correlation heatmap to %s", output_path)
        return output_path

    plt.show()
    return None


def plot_correlation_heatmap_grouped(
    df: pd.DataFrame | None = None,
    db_path: Path | None = None,
    group: str = "all",
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
    output_path: Path | None = None,
    figsize: tuple[int, int] | None = None,
) -> Path | None:
    """
    Tworzy heatmap dla wybranej grupy wskaźników.

    Args:
        df: DataFrame do analizy. Jeśli None, ładuje z bazy.
        db_path: Ścieżka do bazy (używana gdy df=None).
        group: Nazwa grupy z INDICATOR_GROUPS lub 'all'.
        method: Metoda korelacji.
        output_path: Ścieżka do zapisu.
        figsize: Rozmiar figury (automatyczny jeśli None).

    Returns:
        Ścieżka do zapisanego pliku lub None.
    """
    if df is None:
        df = load_master_dataset(db_path)

    if group == "all":
        # Pobierz wszystkie kolumny ze wszystkich grup
        columns = []
        for cols in INDICATOR_GROUPS.values():
            columns.extend([c for c in cols if c in df.columns])
        columns = list(dict.fromkeys(columns))  # Usuń duplikaty zachowując kolejność
    elif group in INDICATOR_GROUPS:
        columns = [c for c in INDICATOR_GROUPS[group] if c in df.columns]
    else:
        raise ValueError(f"Unknown group: {group}. Available: {list(INDICATOR_GROUPS.keys())}")

    if not columns:
        raise ValueError(f"No columns found for group: {group}")

    if figsize is None:
        n_cols = len(columns)
        figsize = (max(8, n_cols * 0.8), max(6, n_cols * 0.7))

    title = f"Korelacja - {group.replace('_', ' ').title()}" if group != "all" else "Macierz korelacji - Wszystkie wskaźniki"

    return plot_correlation_heatmap(
        df=df,
        columns=columns,
        method=method,
        output_path=output_path,
        figsize=figsize,
        title=title,
    )


def find_highly_correlated(
    df: pd.DataFrame | None = None,
    db_path: Path | None = None,
    columns: list[str] | None = None,
    threshold: float = 0.7,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
) -> pd.DataFrame:
    """
    Znajduje pary zmiennych o wysokiej korelacji.

    Args:
        df: DataFrame do analizy. Jeśli None, ładuje z bazy.
        db_path: Ścieżka do bazy (używana gdy df=None).
        columns: Lista kolumn. Jeśli None, używa domyślnych.
        threshold: Próg korelacji (wartość bezwzględna).
        method: Metoda korelacji.

    Returns:
        DataFrame z parami zmiennych o |korelacja| >= threshold.
    """
    corr_matrix = correlation_matrix(df, db_path, columns, method)

    # Znajdź pary powyżej progu
    pairs = []
    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            if i < j:  # Unikaj duplikatów i diagonali
                corr = corr_matrix.loc[col1, col2]
                if abs(corr) >= threshold:
                    pairs.append({
                        "variable_1": col1,
                        "variable_2": col2,
                        "correlation": round(corr, 4),
                        "abs_correlation": round(abs(corr), 4),
                    })

    result = pd.DataFrame(pairs)
    if len(result) > 0:
        result = result.sort_values("abs_correlation", ascending=False)

    return result.reset_index(drop=True)


def correlation_by_sector(
    df: pd.DataFrame | None = None,
    db_path: Path | None = None,
    var1: str = "pe_ratio",
    var2: str = "revenue_growth",
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
) -> pd.DataFrame:
    """
    Oblicza korelację między dwoma zmiennymi dla każdego sektora.

    Args:
        df: DataFrame do analizy. Jeśli None, ładuje z bazy.
        db_path: Ścieżka do bazy (używana gdy df=None).
        var1: Pierwsza zmienna.
        var2: Druga zmienna.
        method: Metoda korelacji.

    Returns:
        DataFrame z korelacjami per sektor.
    """
    if df is None:
        df = load_master_dataset(db_path)

    if "sector" not in df.columns:
        raise ValueError("Column 'sector' not found in dataset")

    results = []
    for sector in df["sector"].dropna().unique():
        sector_df = df[df["sector"] == sector][[var1, var2]].dropna()

        if len(sector_df) < 10:  # Minimum próbki
            continue

        corr = sector_df[var1].corr(sector_df[var2], method=method)

        results.append({
            "sector": sector,
            "n_samples": len(sector_df),
            "correlation": round(corr, 4),
            "abs_correlation": round(abs(corr), 4),
        })

    result = pd.DataFrame(results)
    if len(result) > 0:
        result = result.sort_values("abs_correlation", ascending=False)

    return result.reset_index(drop=True)


def plot_scatter_correlation(
    df: pd.DataFrame | None = None,
    db_path: Path | None = None,
    x_col: str = "pe_ratio",
    y_col: str = "revenue_growth",
    hue_col: str | None = "sector",
    output_path: Path | None = None,
    figsize: tuple[int, int] = (12, 8),
    alpha: float = 0.5,
    sample_size: int | None = 5000,
) -> Path | None:
    """
    Tworzy wykres punktowy z linią regresji.

    Args:
        df: DataFrame do analizy. Jeśli None, ładuje z bazy.
        db_path: Ścieżka do bazy (używana gdy df=None).
        x_col: Kolumna na osi X.
        y_col: Kolumna na osi Y.
        hue_col: Kolumna do kolorowania punktów (opcjonalna).
        output_path: Ścieżka do zapisu.
        figsize: Rozmiar figury.
        alpha: Przezroczystość punktów.
        sample_size: Liczba próbek do wyświetlenia (dla wydajności).

    Returns:
        Ścieżka do zapisanego pliku lub None.
    """
    if df is None:
        df = load_master_dataset(db_path)

    # Filtruj tylko kolumny potrzebne
    cols_needed = [x_col, y_col]
    if hue_col and hue_col in df.columns:
        cols_needed.append(hue_col)

    plot_df = df[cols_needed].dropna(subset=[x_col, y_col])

    # Próbkowanie dla wydajności
    if sample_size and len(plot_df) > sample_size:
        plot_df = plot_df.sample(n=sample_size, random_state=42)

    fig, ax = plt.subplots(figsize=figsize)

    if hue_col and hue_col in plot_df.columns:
        # Top 10 sektorów dla czytelności
        top_sectors = plot_df[hue_col].value_counts().head(10).index
        plot_df_filtered = plot_df[plot_df[hue_col].isin(top_sectors)]

        sns.scatterplot(
            data=plot_df_filtered,
            x=x_col,
            y=y_col,
            hue=hue_col,
            alpha=alpha,
            ax=ax,
        )
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    else:
        sns.scatterplot(data=plot_df, x=x_col, y=y_col, alpha=alpha, ax=ax)

    # Dodaj linię regresji
    sns.regplot(
        data=plot_df,
        x=x_col,
        y=y_col,
        scatter=False,
        color="red",
        line_kws={"linewidth": 2},
        ax=ax,
    )

    # Oblicz korelację
    corr = plot_df[x_col].corr(plot_df[y_col])
    ax.set_title(f"{x_col} vs {y_col}\nKorelacja Pearsona: {corr:.3f}", fontsize=12)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        LOGGER.info("Saved scatter plot to %s", output_path)
        return output_path

    plt.show()
    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    print("\n=== Highly Correlated Variables (|r| >= 0.7) ===")
    high_corr = find_highly_correlated(threshold=0.7)
    print(high_corr.to_string())

    print("\n=== Correlation with Revenue Growth ===")
    target_corr = correlation_with_target(target_col="revenue_growth")
    print(target_corr.head(15).to_string())

    print("\n=== Correlation by Sector (PE vs Revenue Growth) ===")
    sector_corr = correlation_by_sector(var1="pe_ratio", var2="revenue_growth")
    print(sector_corr.to_string())
