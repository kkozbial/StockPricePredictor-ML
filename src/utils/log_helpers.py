"""Pomocnicy do ustandaryzowanego logowania w krokach pipeline'u."""
from __future__ import annotations

import functools
import inspect
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional


@dataclass
class FetchResult:
    """Zagregowany wynik kroku pobierania danych."""

    updated: int = 0
    skipped: int = 0
    errors: int = 0
    saved_paths: Optional[list[str]] = None
    note: Optional[str] = None

    def add_path(self, path: Any) -> None:
        """Dodaje ścieżkę zapisanego pliku do wyniku."""
        if self.saved_paths is None:
            self.saved_paths = []
        self.saved_paths.append(str(path))


def format_counts(result: FetchResult, elapsed: float) -> str:
    """Zwraca zwięzły tekst podsumowujący liczniki do logowania."""
    return (
        f"updated={result.updated}, skipped={result.skipped}, "
        f"errors={result.errors}, elapsed={elapsed:.2f}s"
    )


def summarize_counts(result: FetchResult) -> str:
    """Zwraca liczniki bez czasu dla zbiorczych podsumowań."""
    return f"updated={result.updated}, skipped={result.skipped}, errors={result.errors}"


class ProgressTracker:
    """Klasa do śledzenia i logowania postępu w pętlach."""

    def __init__(self, total: int, module_name: str, checkpoint_percent: int = 10):
        """
        Inicjalizuje tracker postępu.

        Args:
            total: Całkowita liczba elementów do przetworzenia.
            module_name: Nazwa modułu dla logów (np. "fetch_dividends").
            checkpoint_percent: Co ile procent logować checkpoint (domyślnie 10%).
        """
        self.total = total
        self.module_name = module_name
        self.checkpoint_percent = checkpoint_percent
        self.logger = logging.getLogger(f"data_fetch.{module_name}")
        self.processed = 0
        self.last_checkpoint = 0
        self.start_time = time.perf_counter()

        # Statystyki
        self.stats = {"updated": 0, "skipped": 0, "errors": 0}

    def update(self, updated: bool = False, skipped: bool = False, error: bool = False) -> None:
        """
        Aktualizuje postęp o jeden element.

        Args:
            updated: Czy element został zaktualizowany.
            skipped: Czy element został pominięty.
            error: Czy wystąpił błąd.
        """
        self.processed += 1

        if updated:
            self.stats["updated"] += 1
        if skipped:
            self.stats["skipped"] += 1
        if error:
            self.stats["errors"] += 1

        # Sprawdź czy osiągnęliśmy checkpoint
        current_percent = (self.processed / self.total) * 100
        checkpoint_threshold = self.last_checkpoint + self.checkpoint_percent

        if current_percent >= checkpoint_threshold:
            elapsed = time.perf_counter() - self.start_time
            eta = (elapsed / self.processed) * (self.total - self.processed) if self.processed > 0 else 0

            self.logger.info(
                "[%s] Progress: %d%% (%d/%d) | Updated: %d, Skipped: %d, Errors: %d | "
                "Elapsed: %.1fs, ETA: %.1fs",
                self.module_name,
                int(current_percent),
                self.processed,
                self.total,
                self.stats["updated"],
                self.stats["skipped"],
                self.stats["errors"],
                elapsed,
                eta,
            )
            self.last_checkpoint = int(current_percent)

    def finish(self) -> None:
        """Loguje podsumowanie końcowe."""
        elapsed = time.perf_counter() - self.start_time
        self.logger.info(
            "[%s] COMPLETED: %d/%d processed in %.1fs | Updated: %d, Skipped: %d, Errors: %d",
            self.module_name,
            self.processed,
            self.total,
            elapsed,
            self.stats["updated"],
            self.stats["skipped"],
            self.stats["errors"],
        )


def _safe_len(obj: Any) -> Optional[int]:
    """Próbuje obliczyć len(obj) dla iterowalnych obiektów nie będących stringami."""
    if isinstance(obj, (str, bytes)):
        return None
    try:
        return len(obj)  # type: ignore[arg-type]
    except Exception:
        return None


def _infer_context(arguments: dict[str, Any]) -> Optional[str]:
    """Buduje krótki kontekst (tickers=, series=, itd.) dla logów startowych."""
    if "tickers" in arguments:
        length = _safe_len(arguments["tickers"])
        if length is not None:
            return f"tickers={length}"
    if "series_ids" in arguments:
        length = _safe_len(arguments["series_ids"])
        if length is not None:
            return f"series={length}"
    if "sources" in arguments:
        length = _safe_len(arguments["sources"])
        if length is not None:
            return f"sources={length}"
    return None


def log_step(name: str) -> Callable[[Callable[..., FetchResult | None]], Callable[..., FetchResult]]:
    """Dekorator standaryzujący logi START/DONE i mierzący czas wykonania."""

    def decorator(func: Callable[..., FetchResult | None]) -> Callable[..., FetchResult]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> FetchResult:
            pipeline_logger = logging.getLogger("pipeline")
            try:
                signature = inspect.signature(func)
                bound = signature.bind_partial(*args, **kwargs)
                context = _infer_context(bound.arguments)
            except Exception:
                context = None

            start_message = f"[{name}] Start"
            if context:
                start_message += f" ({context})"
            pipeline_logger.info(start_message)

            started = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            except Exception as exc:  # pragma: no cover - defensive guard
                elapsed = time.perf_counter() - started
                pipeline_logger.error(f"[{name}] FAILED: {exc} (elapsed={elapsed:.2f}s)")
                module_logger = logging.getLogger(func.__module__ or "pipeline")
                module_logger.exception("[%s] Wyjatek podczas wykonywania kroku.", name)
                return FetchResult(errors=1, note=str(exc))

            if not isinstance(result, FetchResult):
                result = FetchResult()

            elapsed = time.perf_counter() - started
            pipeline_logger.info(f"[{name}] Done ({format_counts(result, elapsed)})")
            return result

        return wrapper

    return decorator

