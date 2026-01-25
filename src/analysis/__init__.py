"""
Moduł analizy danych giełdowych.

Dostępne komponenty:
- descriptive_stats: Statystyki opisowe i jakość danych
- correlations: Macierze korelacji i heatmapy
- visualization: Wykresy rozkładów i wizualizacje
- scanner: Skanowanie tickerów według wskaźników

Przykładowe użycie:
    >>> from src.analysis import (
    ...     describe_dataset,
    ...     correlation_matrix,
    ...     plot_correlation_heatmap,
    ...     scan_tickers,
    ... )

    # Statystyki opisowe
    >>> stats = describe_dataset()

    # Macierz korelacji
    >>> corr = correlation_matrix()

    # Heatmapa korelacji
    >>> plot_correlation_heatmap(output_path="reports/correlation.png")

    # Skan value stocks
    >>> value_stocks = scan_tickers(preset="value_stocks")
"""
from __future__ import annotations

# Statystyki opisowe
from .descriptive_stats import (
    DEFAULT_NUMERIC_COLS,
    data_quality_summary,
    describe_by_group,
    describe_dataset,
    load_master_dataset,
    missing_values_report,
    outlier_report,
)

# Korelacje
from .correlations import (
    INDICATOR_GROUPS,
    correlation_by_sector,
    correlation_matrix,
    correlation_with_target,
    find_highly_correlated,
    plot_correlation_heatmap,
    plot_correlation_heatmap_grouped,
    plot_scatter_correlation,
)

# Wizualizacje
from .visualization import (
    plot_distribution,
    plot_distribution_by_sector,
    plot_distributions_grid,
    plot_missing_values,
    plot_price_history,
    plot_sector_composition,
    plot_time_series,
    plot_violin_comparison,
)

# Skaner
from .scanner import (
    PRESET_FILTERS,
    FilterCondition,
    Operator,
    ScannerConfig,
    compare_tickers,
    get_ticker_profile,
    list_presets,
    scan_by_indicator,
    scan_tickers,
    sector_ranking,
)

__all__ = [
    # descriptive_stats
    "load_master_dataset",
    "describe_dataset",
    "describe_by_group",
    "missing_values_report",
    "data_quality_summary",
    "outlier_report",
    "DEFAULT_NUMERIC_COLS",
    # correlations
    "correlation_matrix",
    "correlation_with_target",
    "plot_correlation_heatmap",
    "plot_correlation_heatmap_grouped",
    "find_highly_correlated",
    "correlation_by_sector",
    "plot_scatter_correlation",
    "INDICATOR_GROUPS",
    # visualization
    "plot_distribution",
    "plot_distributions_grid",
    "plot_distribution_by_sector",
    "plot_violin_comparison",
    "plot_time_series",
    "plot_price_history",
    "plot_sector_composition",
    "plot_missing_values",
    # scanner
    "scan_tickers",
    "scan_by_indicator",
    "compare_tickers",
    "sector_ranking",
    "get_ticker_profile",
    "list_presets",
    "FilterCondition",
    "Operator",
    "ScannerConfig",
    "PRESET_FILTERS",
]
