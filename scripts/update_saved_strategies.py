"""
Weekly recompute job for the public app's Saved Strategies feature -- run
as a step in .github/workflows/weekly-data-refresh.yml, after price data
has already been refreshed earlier in the same workflow run.

For every saved strategy: re-runs run_backtest with its saved parameters
against the freshly-refreshed price data, derives entry/exit events over
the trailing ~6 months from the resulting holdings_history, and the
current (latest) holdings -- replacing whatever was stored for that
strategy last week. Idempotent and self-healing: a strategy's computed
state is always fully rebuilt from scratch, never incrementally patched,
so a bug in one week's run can't compound into the next.

Needs PUBLIC_STRATEGIES_DATABASE_URL set as an env var (a GitHub Actions
repo secret in the deployed workflow) -- see db_public._database_url.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import db_public
from momentum_strategy import compute_holdings_history, current_positions, holdings_history_to_events

LOOKBACK_MONTHS_FOR_EVENTS = 6


def main():
    db_public.init_schema()
    strategies = db_public.list_strategies()
    print(f"Recomputing {len(strategies)} saved strategies...")

    failed = []
    for s in strategies:
        full = db_public.get_strategy(s["id"])
        name, params = full["name"], full["parameters"]
        try:
            holdings_history = compute_holdings_history(params)
            if not holdings_history:
                print(f"  {name}: no holdings computed (not enough data for these parameters) -- skipped")
                continue
            last_date = holdings_history[-1][0]
            since_date = last_date - pd.DateOffset(months=LOOKBACK_MONTHS_FOR_EVENTS)
            events = holdings_history_to_events(holdings_history, since_date)
            positions = current_positions(holdings_history)
            db_public.replace_strategy_events(s["id"], events)
            db_public.replace_current_positions(s["id"], positions)
            db_public.mark_computed(s["id"])
            print(f"  {name}: {len(events)} events (last {LOOKBACK_MONTHS_FOR_EVENTS}mo), {len(positions)} current positions")
        except Exception as e:
            print(f"  {name}: FAILED -- {e}", file=sys.stderr)
            failed.append(name)

    print(f"\nDone. {len(strategies) - len(failed)} succeeded, {len(failed)} failed.")
    if failed:
        print("Failed:", failed)


if __name__ == "__main__":
    main()
