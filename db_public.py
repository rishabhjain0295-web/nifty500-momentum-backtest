"""
Postgres access for the public app's "Saved Strategies" feature.

Shares the SAME Supabase project as the private
systematic-momentum-investing-portfolio portal (a deliberate choice --
simpler to manage one database than two). Because of that, every table
here is prefixed `public_` to avoid colliding with the private portal's
own tables (saved_strategies, deployments, trades, equity_snapshots,
broker_credentials) -- without the prefix, `CREATE TABLE IF NOT EXISTS
saved_strategies` would silently no-op against the private portal's
EXISTING, differently-shaped table instead of creating this one, and
every query below would fail on missing columns.

This is reachable by anyone who opens the public app, so it only ever
holds non-sensitive data: a strategy's name + parameters, and the
entry/exit history + current positions computed for it. No broker
credentials, no capital, no real trading state -- that's confined to
the private portal's own (differently-named) tables in this same
database.

Anyone can CREATE a strategy (no login on this app), so two lightweight
anti-abuse measures are built in here rather than at the UI layer:
  - owner_key: a random token generated at save time and shown to the
    creator ONCE -- required to delete that strategy later, so a random
    visitor can't delete someone else's. Not real auth, just enough
    friction to stop casual tampering.
  - MAX_STRATEGIES: a hard cap (see save_strategy) so the free-tier
    database can't be flooded, and so the weekly recompute job (which
    re-runs a full backtest for every saved strategy) stays bounded.

Entries/exits and current positions are fully REPLACED each week by
scripts/update_saved_strategies.py (delete-then-reinsert, not an
incremental diff) -- simpler and self-healing, since each week's run
recomputes everything fresh from that week's refreshed price data.
"""
import os
import secrets
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
import streamlit as st

MAX_STRATEGIES = 200


def _database_url() -> str | None:
    """Checks Streamlit secrets first (the interactive app), then the
    PUBLIC_STRATEGIES_DATABASE_URL environment variable (the weekly
    GitHub Actions job -- scripts/update_saved_strategies.py runs as a
    plain script with no .streamlit/secrets.toml file, so the workflow
    injects this as a repo secret instead, same pattern as GH_TOKEN in
    weekly-data-refresh.yml)."""
    try:
        if "public_strategies_database_url" in st.secrets:
            return st.secrets["public_strategies_database_url"]
    except Exception:
        pass  # no secrets.toml at all (e.g. running as a plain script) -- fall through
    return os.environ.get("PUBLIC_STRATEGIES_DATABASE_URL")


SCHEMA = """
CREATE TABLE IF NOT EXISTS public_saved_strategies (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    owner_key TEXT NOT NULL,
    parameters JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_computed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS public_strategy_events (
    id SERIAL PRIMARY KEY,
    strategy_id INTEGER NOT NULL REFERENCES public_saved_strategies(id) ON DELETE CASCADE,
    event_date DATE NOT NULL,
    symbol TEXT NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('entry', 'exit'))
);
CREATE INDEX IF NOT EXISTS idx_public_strategy_events_strategy_id ON public_strategy_events(strategy_id);

CREATE TABLE IF NOT EXISTS public_strategy_current_positions (
    id SERIAL PRIMARY KEY,
    strategy_id INTEGER NOT NULL REFERENCES public_saved_strategies(id) ON DELETE CASCADE,
    symbol TEXT NOT NULL,
    entry_date DATE,
    UNIQUE (strategy_id, symbol)
);
"""


class DatabaseNotConfigured(RuntimeError):
    """Raised by get_conn() when no database URL is available. Deliberately
    a plain exception, not a direct st.error()+st.stop() here -- this module
    is also imported by scripts/update_saved_strategies.py, a plain script
    with no Streamlit runtime, so it can't assume st.stop() is meaningful.
    Callers inside the app (app.py) catch this and decide how to degrade
    gracefully -- this feature being unconfigured shouldn't crash the rest
    of the page."""


@contextmanager
def get_conn():
    url = _database_url()
    if url is None:
        raise DatabaseNotConfigured(
            "No database URL found -- set public_strategies_database_url in "
            ".streamlit/secrets.toml, or PUBLIC_STRATEGIES_DATABASE_URL as an env var."
        )
    conn = psycopg2.connect(url)
    try:
        yield conn
    finally:
        conn.close()


def init_schema() -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA)
        conn.commit()


def count_strategies() -> int:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM public_saved_strategies")
            return cur.fetchone()[0]


def strategy_name_exists(name: str) -> bool:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM public_saved_strategies WHERE name = %s", (name,))
            return cur.fetchone() is not None


def save_strategy(name: str, parameters: dict) -> tuple[int, str]:
    """Raises ValueError if MAX_STRATEGIES has been reached. Returns
    (id, owner_key) -- the owner_key is generated here and only ever
    returned to the caller at creation time; it's not shown again."""
    import json

    if count_strategies() >= MAX_STRATEGIES:
        raise ValueError(
            f"This public demo caps saved strategies at {MAX_STRATEGIES} to keep the free-tier "
            "database and weekly recompute job bounded. Please try again later, or reuse an "
            "existing strategy close to what you want."
        )
    owner_key = secrets.token_hex(8)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO public_saved_strategies (name, owner_key, parameters) VALUES (%s, %s, %s) RETURNING id",
                (name, owner_key, json.dumps(parameters)),
            )
            strategy_id = cur.fetchone()[0]
        conn.commit()
        return strategy_id, owner_key


def list_strategies() -> list[dict]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, name, created_at, last_computed_at FROM public_saved_strategies ORDER BY name"
            )
            return [dict(row) for row in cur.fetchall()]


def get_strategy(strategy_id: int) -> dict | None:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM public_saved_strategies WHERE id = %s", (strategy_id,))
            row = cur.fetchone()
            return dict(row) if row else None


def delete_strategy(strategy_id: int, owner_key: str) -> bool:
    """Returns False (no-op) if owner_key doesn't match -- never raises
    for a wrong key, since that's an expected, not exceptional, case
    (someone guessing or mistyping)."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM public_saved_strategies WHERE id = %s AND owner_key = %s",
                (strategy_id, owner_key),
            )
            deleted = cur.rowcount > 0
        conn.commit()
        return deleted


def replace_strategy_events(strategy_id: int, events: list[dict]) -> None:
    """events: [{date, symbol, action}]. Deletes existing events for this
    strategy first -- the weekly job always recomputes the full trailing
    window fresh, never incrementally patches."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM public_strategy_events WHERE strategy_id = %s", (strategy_id,))
            if events:
                psycopg2.extras.execute_values(
                    cur,
                    "INSERT INTO public_strategy_events (strategy_id, event_date, symbol, action) VALUES %s",
                    [(strategy_id, e["date"], e["symbol"], e["action"]) for e in events],
                )
        conn.commit()


def replace_current_positions(strategy_id: int, positions: list[dict]) -> None:
    """positions: [{symbol, entry_date}]."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM public_strategy_current_positions WHERE strategy_id = %s", (strategy_id,))
            if positions:
                psycopg2.extras.execute_values(
                    cur,
                    "INSERT INTO public_strategy_current_positions (strategy_id, symbol, entry_date) VALUES %s",
                    [(strategy_id, p["symbol"], p["entry_date"]) for p in positions],
                )
        conn.commit()


def mark_computed(strategy_id: int) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE public_saved_strategies SET last_computed_at = now() WHERE id = %s", (strategy_id,))
        conn.commit()


def list_strategy_events(strategy_id: int, since_date) -> list[dict]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT event_date, symbol, action FROM public_strategy_events "
                "WHERE strategy_id = %s AND event_date >= %s ORDER BY event_date DESC, symbol",
                (strategy_id, since_date),
            )
            return [dict(row) for row in cur.fetchall()]


def list_current_positions(strategy_id: int) -> list[dict]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT symbol, entry_date FROM public_strategy_current_positions "
                "WHERE strategy_id = %s ORDER BY symbol",
                (strategy_id,),
            )
            return [dict(row) for row in cur.fetchall()]
