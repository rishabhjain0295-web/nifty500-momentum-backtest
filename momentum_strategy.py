"""
Shared logic for turning a SAVED "Momentum Backtest" strategy's
parameters (see app.py's "Saved Strategies" sidebar section) into
holdings history and discrete entry/exit events -- used by both the
interactive app and the weekly recompute job
(scripts/update_saved_strategies.py). Kept Streamlit-free so the weekly
job (a plain script, no Streamlit runtime) can import it directly.

Only the CORE parameters that determine which stocks are held go here --
not display-only conveniences (log-scale toggle, mutual fund comparison
line) or overlays that don't cleanly map to discrete symbol-level
entry/exit events (stoploss, T+1 execution lag, MTF leverage, costs/
taxes). Same scoping principle the private portal's Saved Strategies
page already uses.
"""
import pandas as pd

from backtest_engine import (
    load_benchmark,
    load_gold_series,
    load_membership_matrix,
    load_prices,
    load_universe_symbols,
    run_backtest,
)

SAVEABLE_PARAM_KEYS = [
    "universe", "rebal_freq", "lookback_months", "skip_months", "hold_months",
    "n_stocks", "min_price", "price_col", "use_membership_filter", "weighting_mode",
    "use_exit_band", "exit_band_pct", "use_regime_filter", "gold_entry_lookback",
    "gold_exit_lookback", "max_volatility_pct", "use_risk_adjusted", "max_trailing_return_pct",
]


def compute_holdings_history(parameters: dict) -> list[tuple[pd.Timestamp, list[str]]]:
    """Runs run_backtest with the given saved parameters and returns just
    its holdings_history -- (rebalance_date, holdings) pairs. Mirrors
    app.py's own parameter-to-call mapping exactly, so a saved strategy's
    computed entries/exits never drift from what the interactive
    Backtest page would show for the same inputs."""
    is_weekly = parameters.get("rebal_freq") == "Weekly"
    price_freq = "W-FRI" if is_weekly else "ME"

    monthly_prices = load_prices(parameters["price_col"], price_freq)
    membership = (
        load_membership_matrix(monthly_prices.index, monthly_prices.columns)
        if parameters.get("use_membership_filter") else None
    )
    allowed_symbols = (
        None if parameters["universe"] == "Nifty 500"
        else load_universe_symbols(parameters["universe"])
    )
    bench_px = load_benchmark(parameters["price_col"], price_freq)
    gold_px = load_gold_series(parameters["price_col"]) if parameters.get("use_regime_filter") else None

    _, holdings_history = run_backtest(
        monthly_prices, membership,
        parameters["lookback_months"], parameters["skip_months"], parameters["hold_months"],
        parameters["n_stocks"], parameters["min_price"],
        parameters.get("use_exit_band", False), parameters.get("exit_band_pct", 0.0),
        parameters.get("use_regime_filter", False), bench_px, gold_px,
        parameters.get("gold_entry_lookback", 150), parameters.get("gold_exit_lookback", 55),
        parameters.get("weighting_mode", "equal_monthly"), allowed_symbols,
        max_volatility_pct=parameters.get("max_volatility_pct"),
        use_risk_adjusted=parameters.get("use_risk_adjusted", False),
        max_trailing_return_pct=parameters.get("max_trailing_return_pct"),
    )
    return holdings_history


def holdings_history_to_events(holdings_history, since_date: pd.Timestamp) -> list[dict]:
    """Diffs consecutive holdings snapshots into discrete (date, symbol,
    action) entry/exit events, restricted to transitions on/after
    since_date -- the inverse of what the private portal's execution
    engine does when it diffs a target portfolio against current
    holdings, applied here across a full historical holdings_history
    instead of just the latest step."""
    events = []
    prev: set[str] = set()
    for date, holdings in holdings_history:
        curr = set(holdings) if holdings != ["GOLD"] else set()
        if date >= since_date:
            for sym in curr - prev:
                events.append({"date": date.date(), "symbol": sym, "action": "entry"})
            for sym in prev - curr:
                events.append({"date": date.date(), "symbol": sym, "action": "exit"})
        prev = curr
    return events


def current_positions(holdings_history) -> list[dict]:
    """The latest holdings, each with the date it actually entered its
    CURRENT unbroken streak (walks backwards through holdings_history
    until a symbol is missing from an earlier snapshot) -- not just the
    most recent rebalance's date, which would understate how long a
    long-held survivor has actually been in the portfolio."""
    if not holdings_history:
        return []
    last_date, last_holdings = holdings_history[-1]
    if last_holdings == ["GOLD"]:
        return [{"symbol": "GOLD", "entry_date": last_date.date()}]

    entry_date = {sym: last_date for sym in last_holdings}
    active = set(last_holdings)
    for date, holdings in reversed(holdings_history[:-1]):
        if not active:
            break
        h = set(holdings) if holdings != ["GOLD"] else set()
        for sym in list(active):
            if sym in h:
                entry_date[sym] = date
            else:
                active.discard(sym)
    return [{"symbol": sym, "entry_date": d.date()} for sym, d in entry_date.items()]
