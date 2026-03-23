"""SQL Explorer — przeglądarka bazy DuckDB przez Streamlit.

Uruchamianie:
    streamlit run explorer.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Konfiguracja i połączenie
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent


def _get_db_path() -> Path:
    """Wczytuje ścieżkę do bazy z settings.yaml."""
    try:
        from src.utils.config_loader import load_config
        cfg = load_config()
        return (_PROJECT_ROOT / cfg["database"]["path"]).resolve()
    except Exception:
        return (_PROJECT_ROOT / "data" / "stock_market.db").resolve()


@st.cache_resource(show_spinner="Łączenie z bazą…")
def get_connection(db_path: str) -> duckdb.DuckDBPyConnection:
    """Otwiera połączenie read-only (cachowane przez Streamlit)."""
    return duckdb.connect(db_path, read_only=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_schemas_and_tables(conn: duckdb.DuckDBPyConnection) -> dict[str, list[tuple[str, int]]]:
    """Zwraca {schema: [(tabela, liczba_wierszy)]}."""
    rows = conn.execute(
        "SELECT table_schema, table_name FROM information_schema.tables "
        "WHERE table_schema NOT IN ('information_schema', 'pg_catalog') "
        "ORDER BY table_schema, table_name"
    ).fetchall()

    result: dict[str, list[tuple[str, int]]] = {}
    for schema, table in rows:
        count = 0
        try:
            count = conn.execute(f'SELECT COUNT(*) FROM "{schema}"."{table}"').fetchone()[0]
        except Exception:
            pass
        result.setdefault(schema, []).append((table, count))
    return result


def run_query(conn: duckdb.DuckDBPyConnection, sql: str) -> tuple[pd.DataFrame | None, str | None]:
    """Wykonuje SQL i zwraca (DataFrame, None) lub (None, komunikat_błędu)."""
    try:
        result = conn.execute(sql)
        if result.description:
            cols = [d[0] for d in result.description]
            data = result.fetchall()
            return pd.DataFrame(data, columns=cols), None
        return pd.DataFrame(), None
    except Exception as exc:
        return None, str(exc)


def _set_sql(sql: str) -> None:
    st.session_state["sql_input"] = sql


# ---------------------------------------------------------------------------
# Gotowe zapytania pomocnicze
# ---------------------------------------------------------------------------

QUICK_QUERIES: list[tuple[str, str]] = [
    (
        "Wszystkie tabele",
        "SELECT table_schema, table_name\n"
        "FROM information_schema.tables\n"
        "WHERE table_schema NOT IN ('information_schema', 'pg_catalog')\n"
        "ORDER BY table_schema, table_name;",
    ),
    (
        "Liczba wierszy (main)",
        "SELECT table_name,\n"
        "       estimated_size AS rows\n"
        "FROM duckdb_tables()\n"
        "WHERE schema_name = 'main'\n"
        "ORDER BY rows DESC;",
    ),
    (
        "Schematy",
        "SELECT DISTINCT table_schema\n"
        "FROM information_schema.tables\n"
        "WHERE table_schema NOT IN ('information_schema', 'pg_catalog')\n"
        "ORDER BY 1;",
    ),
    (
        "Rozmiar bazy",
        "SELECT database_size FROM pragma_database_size();",
    ),
]


# ---------------------------------------------------------------------------
# Aplikacja
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="SQL Explorer",
        page_icon="🔍",
        layout="wide",
    )

    db_path = _get_db_path()

    if not db_path.exists():
        st.error(f"Baza danych nie istnieje: `{db_path}`\nUruchom najpierw pipeline (BUILD/UPDATE).")
        st.stop()

    conn = get_connection(str(db_path))

    # Inicjalizuj session state
    if "sql_input" not in st.session_state:
        st.session_state["sql_input"] = "SELECT * FROM prices LIMIT 100;"
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None

    # -----------------------------------------------------------------------
    # Sidebar
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.title("SQL Explorer")
        st.caption(f"`{db_path.name}`")
        st.divider()

        # Schema browser
        st.subheader("Schema Browser")
        try:
            schemas = get_schemas_and_tables(conn)
            for schema, tables in schemas.items():
                with st.expander(f"**{schema}** ({len(tables)} tabel)", expanded=(schema == "main")):
                    for table, count in tables:
                        label = f"{table}  `{count:,}`"
                        if st.button(label, key=f"tbl_{schema}_{table}", use_container_width=True):
                            _set_sql(f'SELECT *\nFROM "{schema}"."{table}"\nLIMIT 100;')
                            st.rerun()
        except Exception as exc:
            st.warning(f"Nie można wczytać tabel: {exc}")

        st.divider()

        # Szybkie zapytania
        st.subheader("Szybkie zapytania")
        for label, sql in QUICK_QUERIES:
            if st.button(label, use_container_width=True):
                _set_sql(sql)
                st.rerun()

    # -----------------------------------------------------------------------
    # Główny obszar
    # -----------------------------------------------------------------------
    st.header("Edytor SQL")
    st.caption(f"Baza: `{db_path}`")

    sql = st.text_area(
        label="Zapytanie SQL",
        value=st.session_state["sql_input"],
        height=160,
        key="sql_input",
        placeholder="SELECT * FROM prices WHERE ticker = 'AAPL' LIMIT 50;",
        label_visibility="collapsed",
    )

    col_run, col_clear, _col_space = st.columns([1, 1, 8])
    run_clicked = col_run.button("▶  Uruchom", type="primary", use_container_width=True)
    if col_clear.button("✕  Wyczyść", use_container_width=True):
        _set_sql("")
        st.session_state["last_result"] = None
        st.rerun()

    # Uruchomienie zapytania
    if run_clicked and sql.strip():
        with st.spinner("Wykonywanie zapytania…"):
            df, err = run_query(conn, sql.strip())

        if err:
            st.error(f"**Błąd SQL:**\n```\n{err}\n```")
            st.session_state["last_result"] = None
        else:
            st.session_state["last_result"] = df
            # dodaj do historii (unikaj duplikatów z ostatniego wpisu)
            history: list[str] = st.session_state["history"]
            if not history or history[-1] != sql.strip():
                history.append(sql.strip())
                st.session_state["history"] = history[-10:]

    # Wyświetl wynik
    df_result: pd.DataFrame | None = st.session_state.get("last_result")
    if df_result is not None:
        if df_result.empty:
            st.info("Zapytanie wykonane — brak wierszy.")
        else:
            st.success(f"**{len(df_result):,}** wierszy · **{len(df_result.columns)}** kolumn")
            st.dataframe(df_result, use_container_width=True, height=420)
            st.download_button(
                label="⬇  Pobierz CSV",
                data=df_result.to_csv(index=False).encode("utf-8"),
                file_name="wynik.csv",
                mime="text/csv",
            )

    # Historia
    history: list[str] = st.session_state.get("history", [])
    if history:
        st.divider()
        st.subheader("Historia zapytań")
        for i, q in enumerate(reversed(history)):
            with st.expander(q[:80] + ("…" if len(q) > 80 else ""), expanded=False):
                st.code(q, language="sql")
                if st.button("Przywróć", key=f"hist_{i}"):
                    _set_sql(q)
                    st.rerun()


if __name__ == "__main__":
    main()
