"""
Daily-resolution swing trading engine, layered on the SAME momentum
universe selection as backtest_engine.py (compute_momentum_ranking) but
with entirely different entry/exit mechanics: instead of holding the top
N stocks continuously and rebalancing monthly, this scans the top N
momentum stocks each month for breakout signals and trades them
individually with risk-based position sizing, a trailing/fixed stop, and
an R-multiple profit target.

Universe: at each monthly rebalance date, the top n_stocks by trailing
lookback_months return (skipping skip_months) become eligible for NEW
entries until the next rebalance -- reusing compute_momentum_ranking, so
this can never drift out of sync with the momentum backtest's own
ranking. A stock already in an open position when it drops out of next
month's universe is NOT force-closed -- the universe only gates new
entries; existing positions are managed to their own stop/target
regardless (a real trader's screener works the same way -- it finds
candidates, it doesn't manage open trades).

Entry strategies (entry_strategy):
  - "gap_up": buy at today's open when it gaps up gap_pct% or more above
    yesterday's close.
  - "donchian": buy when today's high breaks above the highest high of
    the trailing donchian_entry_lookback days (a stop-buy simulation --
    fill price is the worse of today's open and the breakout level).

Stop (exit_mode) -- the SAME mechanism produces every stop variant asked
for across this feature's spec:
  - "trailing_donchian" with exit_lookback_days=1: "previous day's low"
    (the gap-up strategy's stated stop).
  - "trailing_donchian" with exit_lookback_days=N: "N-day channel low"
    (the Donchian strategy's stated stop, recomputed daily so it's a
    genuine trailing stop, not fixed at entry).
  - "fixed_pct": stop = entry_price * (1 - stop_pct/100), fixed for the
    life of the position.
  Whichever mode, the exit fires when a day's low touches or breaches the
  stop; if the stock gaps below it, the fill is the worse of today's open
  and the stop level.

Position sizing -- risk-based, not equal-weight: qty is set so that if
the INITIAL stop (computed once at entry) is hit, the loss equals
risk_pct of capital_base. This alone is not safe to use unbounded: a
tight stop (small % away from entry) demands a huge position to reach
risk_pct of loss-if-stopped-out -- e.g. a 1% stop distance needs a
position worth 100% of capital just to risk 1%. Real risk-based systems
always pair this with a hard position-size ceiling (max_position_pct of
capital_base), which is what actually binds on tight-stop setups; qty is
also capped so a single trade never exceeds available cash (no leverage/
margin modeled). Either cap means realized risk on that trade is LESS
than risk_pct, never more.

Profit target: target_price = entry_price + risk_reward_ratio * (entry_price
- initial_stop_price), i.e. an R-multiple of the ORIGINAL risk (the target
doesn't move even if a trailing stop later tightens the effective risk).
Checked daily against that day's high; exits at the better of today's open
and the target if it gapped past it.

If a day's low would hit the stop AND its high would hit the target, the
stop is assumed to have triggered first (conservative -- daily OHLC bars
don't tell us the actual intraday sequence).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_engine import compute_momentum_ranking


def _build_universe_calendar(
    monthly_prices: pd.DataFrame,
    membership: pd.DataFrame | None,
    lookback_months: int,
    skip_months: int,
    n_stocks: int,
    min_price: float,
    allowed_symbols: set[str] | None = None,
    weakest: bool = False,
) -> list[tuple[pd.Timestamp, list[str]]]:
    """Top (or, if weakest=True, bottom) n_stocks by momentum rank at each
    monthly rebalance date -- the shared universe-selection logic behind
    every swing strategy in this module, so they can never rank differently
    from each other or from the Backtest page (all call
    compute_momentum_ranking directly).

    allowed_symbols, if given, restricts the ranking universe BEFORE taking
    the top/bottom n_stocks -- e.g. the Short Momentum (F&O) strategy only
    trades stocks that actually have tradeable stock futures, not the full
    momentum universe. weakest=True picks the n_stocks with the LOWEST
    trailing return instead of the highest (compute_momentum_ranking always
    returns descending-sorted, so this just reads from the tail)."""
    monthly_dates = monthly_prices.index
    min_history_months = lookback_months + skip_months + 1
    universe_by_period: list[tuple[pd.Timestamp, list[str]]] = []
    for i in range(min_history_months, len(monthly_dates)):
        today = monthly_dates[i]
        ranked = compute_momentum_ranking(monthly_prices, membership, today, lookback_months, skip_months, min_price)
        if ranked is None:
            continue
        if allowed_symbols is not None:
            ranked = ranked[ranked.index.isin(allowed_symbols)]
        if len(ranked) < n_stocks:
            continue
        picked = ranked.tail(n_stocks).index.tolist() if weakest else ranked.head(n_stocks).index.tolist()
        universe_by_period.append((today, picked))
    return universe_by_period


def run_swing_backtest(
    monthly_prices: pd.DataFrame,
    membership: pd.DataFrame | None,
    daily_open: pd.DataFrame,
    daily_high: pd.DataFrame,
    daily_low: pd.DataFrame,
    daily_close: pd.DataFrame,
    lookback_months: int,
    skip_months: int,
    n_stocks: int,
    min_price: float,
    entry_strategy: str = "gap_up",
    gap_pct: float = 1.0,
    donchian_entry_lookback: int = 20,
    exit_mode: str = "trailing_donchian",
    exit_lookback_days: int = 1,
    stop_pct: float = 8.0,
    risk_pct: float = 1.0,
    risk_reward_ratio: float = 2.0,
    max_position_pct: float = 20.0,
    capital_base: float = 1_000_000.0,
) -> dict:
    # 1. universe calendar: top n_stocks at each monthly rebalance
    universe_by_period = _build_universe_calendar(
        monthly_prices, membership, lookback_months, skip_months, n_stocks, min_price
    )

    empty = {
        "trades": pd.DataFrame(columns=["symbol", "entry_date", "entry_price", "stop_price", "target_price",
                                         "exit_date", "exit_price", "exit_reason", "qty", "risked_rs",
                                         "pnl_rs", "pnl_pct", "r_multiple", "hold_days", "status"]),
        "equity": pd.Series(dtype=float),
        "n_universe_periods": 0,
        "capital_base": capital_base,
    }
    if not universe_by_period:
        return empty

    rebalance_dates = pd.DatetimeIndex([u[0] for u in universe_by_period])
    start_date = rebalance_dates[0]
    last_daily_date = daily_close.index.max()
    if last_daily_date <= start_date:
        return empty
    trading_days = daily_close.index[(daily_close.index > start_date) & (daily_close.index <= last_daily_date)]
    if len(trading_days) == 0:
        return empty

    # 2. precompute vectorized rolling references ONCE (fast) instead of
    # re-slicing per stock per day (which would be very slow at this scale)
    prev_close = daily_close.shift(1)
    donchian_entry_upper = None
    if entry_strategy == "donchian":
        donchian_entry_upper = daily_high.rolling(donchian_entry_lookback).max().shift(1)
    trailing_exit_low = None
    if exit_mode == "trailing_donchian":
        trailing_exit_low = daily_low.rolling(exit_lookback_days).min().shift(1)

    cash = capital_base
    positions: dict[str, dict] = {}  # symbol -> entry_date, entry_price, qty, stop_price(initial), target_price
    trades: list[dict] = []
    equity_series = pd.Series(index=trading_days, dtype=float)

    def stop_level_for(sym: str, day: pd.Timestamp, entry_price: float) -> float:
        if exit_mode == "trailing_donchian":
            if sym in trailing_exit_low.columns and day in trailing_exit_low.index:
                return trailing_exit_low.at[day, sym]
            return np.nan
        return entry_price * (1 - stop_pct / 100.0)

    for day in trading_days:
        ridx = rebalance_dates.searchsorted(day, side="right") - 1
        universe = universe_by_period[ridx][1] if ridx >= 0 else []

        # --- exits: stop first (conservative), then target ---
        for sym in list(positions.keys()):
            if sym not in daily_low.columns or sym not in daily_open.columns or sym not in daily_high.columns:
                continue
            if day not in daily_low.index:
                continue
            today_low = daily_low.at[day, sym]
            today_high = daily_high.at[day, sym]
            today_open = daily_open.at[day, sym]
            if pd.isna(today_low) or pd.isna(today_high) or pd.isna(today_open):
                continue

            pos = positions[sym]
            current_stop = stop_level_for(sym, day, pos["entry_price"])
            exit_price = None
            exit_reason = None

            if pd.notna(current_stop) and today_low <= current_stop:
                exit_price = min(today_open, current_stop)
                exit_reason = exit_mode
            elif today_high >= pos["target_price"]:
                exit_price = max(today_open, pos["target_price"])
                exit_reason = "target"

            if exit_price is not None and exit_price > 0:
                qty = pos["qty"]
                cash += qty * exit_price
                pnl_rs = qty * (exit_price - pos["entry_price"])
                trades.append({
                    "symbol": sym, "entry_date": pos["entry_date"], "entry_price": pos["entry_price"],
                    "stop_price": pos["stop_price"], "target_price": pos["target_price"],
                    "exit_date": day, "exit_price": exit_price, "exit_reason": exit_reason,
                    "qty": qty, "risked_rs": pos["risked_rs"],
                    "pnl_rs": pnl_rs, "pnl_pct": exit_price / pos["entry_price"] - 1,
                    "r_multiple": pnl_rs / pos["risked_rs"] if pos["risked_rs"] > 0 else np.nan,
                    "hold_days": (day - pos["entry_date"]).days, "status": "closed",
                })
                del positions[sym]

        # --- entries ---
        for sym in universe:
            if sym in positions:
                continue
            if sym not in daily_open.columns or day not in daily_open.index:
                continue
            today_open = daily_open.at[day, sym]
            if pd.isna(today_open) or today_open <= 0:
                continue

            entry_price = None
            if entry_strategy == "gap_up":
                yc = prev_close.at[day, sym] if sym in prev_close.columns and day in prev_close.index else np.nan
                if pd.notna(yc) and yc > 0 and today_open >= yc * (1 + gap_pct / 100.0):
                    entry_price = today_open
            else:  # donchian
                today_high = daily_high.at[day, sym] if sym in daily_high.columns and day in daily_high.index else np.nan
                dh = donchian_entry_upper.at[day, sym] if sym in donchian_entry_upper.columns and day in donchian_entry_upper.index else np.nan
                if pd.notna(today_high) and pd.notna(dh) and today_high >= dh:
                    entry_price = max(today_open, dh)

            if entry_price is None or entry_price <= 0:
                continue

            initial_stop = stop_level_for(sym, day, entry_price)
            if pd.isna(initial_stop) or initial_stop >= entry_price:
                continue  # can't size a trade with zero/negative/undefined risk

            risk_per_share = entry_price - initial_stop
            risk_amount = capital_base * risk_pct / 100.0
            max_position_value = capital_base * max_position_pct / 100.0
            qty = risk_amount / risk_per_share
            qty = min(qty, max_position_value / entry_price, cash / entry_price)
            if qty <= 0:
                continue

            target_price = entry_price + risk_reward_ratio * risk_per_share
            cash -= qty * entry_price
            positions[sym] = {
                "entry_date": day, "entry_price": entry_price, "qty": qty,
                "stop_price": initial_stop, "target_price": target_price,
                "risked_rs": qty * risk_per_share,
            }

        # --- mark to market ---
        mtm = cash
        for sym, pos in positions.items():
            px = daily_close.at[day, sym] if sym in daily_close.columns and day in daily_close.index else np.nan
            mtm += pos["qty"] * (px if pd.notna(px) else pos["entry_price"])
        equity_series.at[day] = mtm

    # still-open positions at the end -> unrealized, included in the trade log
    for sym, pos in positions.items():
        px = daily_close.at[trading_days[-1], sym] if sym in daily_close.columns else np.nan
        if pd.isna(px):
            continue
        qty = pos["qty"]
        pnl_rs = qty * (px - pos["entry_price"])
        trades.append({
            "symbol": sym, "entry_date": pos["entry_date"], "entry_price": pos["entry_price"],
            "stop_price": pos["stop_price"], "target_price": pos["target_price"],
            "exit_date": pd.NaT, "exit_price": px, "exit_reason": "open",
            "qty": qty, "risked_rs": pos["risked_rs"],
            "pnl_rs": pnl_rs, "pnl_pct": px / pos["entry_price"] - 1,
            "r_multiple": pnl_rs / pos["risked_rs"] if pos["risked_rs"] > 0 else np.nan,
            "hold_days": (trading_days[-1] - pos["entry_date"]).days, "status": "open",
        })

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df = trades_df.sort_values("entry_date").reset_index(drop=True)

    return {
        "trades": trades_df,
        "equity": equity_series,
        "n_universe_periods": len(universe_by_period),
        "capital_base": capital_base,
    }


def run_ema_crossover_backtest(
    monthly_prices: pd.DataFrame,
    membership: pd.DataFrame | None,
    bar_open: pd.DataFrame,
    bar_close: pd.DataFrame,
    lookback_months: int,
    skip_months: int,
    n_stocks: int,
    min_price: float,
    ema_fast: int = 15,
    ema_slow: int = 50,
    risk_pct: float = 1.0,
    max_position_pct: float = 20.0,
    capital_base: float = 1_000_000.0,
) -> dict:
    """EMA(ema_fast)/EMA(ema_slow) crossover swing strategy. Timeframe-
    agnostic -- pass daily bars or hourly bars via bar_open/bar_close and
    it trades identically either way; the caller decides granularity (see
    pages/3_Swing_Trading.py, which offers both). Universe selection
    (_build_universe_calendar) is always monthly regardless of trading
    timeframe, matching every other strategy in this module.

    Entry: at the close of any bar where ema_fast > ema_slow, for a
    universe stock not currently held. Executed at the NEXT bar's open --
    the EMA needs that bar's own close to compute, so filling within the
    same bar the signal appeared on would be a lookahead violation (the
    other two strategies in this module don't need this delay because
    their signals -- a gap at the open, a high touching a pre-known
    breakout level -- are knowable without waiting for that bar to close).

    Exit: at the close of any bar where EITHER the bar's own close is
    below ema_slow (the spec's "initial" stop) OR ema_fast has closed
    below ema_slow (the spec's "trailing" stop) -- also executed at the
    next bar's open. Both conditions are checked every bar for the entire
    life of the position, not just "initially" vs "later"; this was the
    clearest coherent reading of a spec that named two stop rules without
    specifying when one hands off to the other. No fixed profit target --
    this is a pure trend-following exit, ride until one of the two stop
    conditions fires.

    Position sizing: risk-based, same convention as run_swing_backtest --
    qty sized so a stop-out loses risk_pct of capital_base, using the
    ema_slow reading from the SIGNAL bar (the most recently known value
    when the order is placed) as the initial stop distance. Capped by
    max_position_pct and available cash for the same reason documented in
    run_swing_backtest -- a tight stop distance can otherwise demand a
    position far larger than risk_pct alone would suggest.
    """
    universe_by_period = _build_universe_calendar(
        monthly_prices, membership, lookback_months, skip_months, n_stocks, min_price
    )

    empty = {
        "trades": pd.DataFrame(columns=["symbol", "entry_date", "entry_price", "stop_price",
                                         "exit_date", "exit_price", "exit_reason", "qty", "risked_rs",
                                         "pnl_rs", "pnl_pct", "r_multiple", "hold_days", "status"]),
        "equity": pd.Series(dtype=float),
        "n_universe_periods": 0,
        "capital_base": capital_base,
    }
    if not universe_by_period:
        return empty

    rebalance_dates = pd.DatetimeIndex([u[0] for u in universe_by_period])
    start_date = rebalance_dates[0]
    last_bar_date = bar_close.index.max()
    if last_bar_date <= start_date:
        return empty
    bars = bar_close.index[(bar_close.index > start_date) & (bar_close.index <= last_bar_date)]
    if len(bars) == 0:
        return empty

    ema_fast_series = bar_close.ewm(span=ema_fast, adjust=False).mean()
    ema_slow_series = bar_close.ewm(span=ema_slow, adjust=False).mean()

    cash = capital_base
    positions: dict[str, dict] = {}
    trades: list[dict] = []
    equity_series = pd.Series(index=bars, dtype=float)
    pending_entries: dict[str, float] = {}  # symbol -> stop level decided at the signal bar
    pending_exits: dict[str, str] = {}  # symbol -> exit_reason decided at the signal bar

    for bar in bars:
        ridx = rebalance_dates.searchsorted(bar, side="right") - 1
        universe = universe_by_period[ridx][1] if ridx >= 0 else []

        # --- execute signals queued from the PREVIOUS bar, at THIS bar's open ---
        for sym in list(pending_exits.keys()):
            reason = pending_exits.pop(sym)
            if sym not in positions or sym not in bar_open.columns or bar not in bar_open.index:
                continue
            open_px = bar_open.at[bar, sym]
            if pd.isna(open_px) or open_px <= 0:
                continue
            pos = positions.pop(sym)
            qty = pos["qty"]
            cash += qty * open_px
            pnl_rs = qty * (open_px - pos["entry_price"])
            trades.append({
                "symbol": sym, "entry_date": pos["entry_date"], "entry_price": pos["entry_price"],
                "stop_price": pos["stop_price"], "exit_date": bar, "exit_price": open_px,
                "exit_reason": reason, "qty": qty, "risked_rs": pos["risked_rs"],
                "pnl_rs": pnl_rs, "pnl_pct": open_px / pos["entry_price"] - 1,
                "r_multiple": pnl_rs / pos["risked_rs"] if pos["risked_rs"] > 0 else np.nan,
                "hold_days": (bar - pos["entry_date"]).days, "status": "closed",
            })

        for sym in list(pending_entries.keys()):
            initial_stop = pending_entries.pop(sym)
            if sym in positions or sym not in bar_open.columns or bar not in bar_open.index:
                continue
            open_px = bar_open.at[bar, sym]
            if pd.isna(open_px) or open_px <= 0 or initial_stop >= open_px:
                continue
            risk_per_share = open_px - initial_stop
            risk_amount = capital_base * risk_pct / 100.0
            max_position_value = capital_base * max_position_pct / 100.0
            qty = min(risk_amount / risk_per_share, max_position_value / open_px, cash / open_px)
            if qty <= 0:
                continue
            cash -= qty * open_px
            positions[sym] = {
                "entry_date": bar, "entry_price": open_px, "qty": qty,
                "stop_price": initial_stop, "risked_rs": qty * risk_per_share,
            }

        # --- evaluate THIS bar's close for new signals, queued for next bar ---
        for sym in list(positions.keys()):
            if sym in pending_exits or sym not in bar_close.columns or bar not in bar_close.index:
                continue
            c = bar_close.at[bar, sym]
            ef = ema_fast_series.at[bar, sym] if sym in ema_fast_series.columns else np.nan
            es = ema_slow_series.at[bar, sym] if sym in ema_slow_series.columns else np.nan
            if pd.isna(c) or pd.isna(ef) or pd.isna(es):
                continue
            if c < es:
                pending_exits[sym] = "close_below_ema_slow"
            elif ef < es:
                pending_exits[sym] = "ema_fast_below_ema_slow"

        for sym in universe:
            if sym in positions or sym in pending_entries:
                continue
            if sym not in bar_close.columns or bar not in bar_close.index:
                continue
            ef = ema_fast_series.at[bar, sym] if sym in ema_fast_series.columns else np.nan
            es = ema_slow_series.at[bar, sym] if sym in ema_slow_series.columns else np.nan
            if pd.isna(ef) or pd.isna(es):
                continue
            if ef > es:
                pending_entries[sym] = es

        # --- mark to market ---
        mtm = cash
        for sym, pos in positions.items():
            px = bar_close.at[bar, sym] if sym in bar_close.columns and bar in bar_close.index else np.nan
            mtm += pos["qty"] * (px if pd.notna(px) else pos["entry_price"])
        equity_series.at[bar] = mtm

    # still-open positions -> unrealized
    for sym, pos in positions.items():
        px = bar_close.at[bars[-1], sym] if sym in bar_close.columns else np.nan
        if pd.isna(px):
            continue
        qty = pos["qty"]
        pnl_rs = qty * (px - pos["entry_price"])
        trades.append({
            "symbol": sym, "entry_date": pos["entry_date"], "entry_price": pos["entry_price"],
            "stop_price": pos["stop_price"], "exit_date": pd.NaT, "exit_price": px,
            "exit_reason": "open", "qty": qty, "risked_rs": pos["risked_rs"],
            "pnl_rs": pnl_rs, "pnl_pct": px / pos["entry_price"] - 1,
            "r_multiple": pnl_rs / pos["risked_rs"] if pos["risked_rs"] > 0 else np.nan,
            "hold_days": (bars[-1] - pos["entry_date"]).days, "status": "open",
        })

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df = trades_df.sort_values("entry_date").reset_index(drop=True)

    return {
        "trades": trades_df,
        "equity": equity_series,
        "n_universe_periods": len(universe_by_period),
        "capital_base": capital_base,
    }


def run_short_ema_crossover_backtest(
    monthly_prices: pd.DataFrame,
    membership: pd.DataFrame | None,
    bar_open: pd.DataFrame,
    bar_close: pd.DataFrame,
    fno_symbols: set[str],
    lookback_months: int,
    skip_months: int,
    n_stocks: int,
    min_price: float,
    ema_fast: int = 15,
    ema_slow: int = 50,
    max_entries: int = 10,
    capital_base: float = 2_000_000.0,
) -> dict:
    """Short Momentum (F&O): the mirror image of run_ema_crossover_backtest,
    short-selling the WEAKEST momentum stocks within the F&O-eligible
    universe instead of going long the strongest. Timeframe-agnostic --
    pass hourly or 2h bars via bar_open/bar_close (see
    backtest_engine.load_2h_ohlc for the 2h resampling).

    Universe: at each monthly rebalance date, the n_stocks LOWEST trailing-
    return stocks among fno_symbols (real shorting of individual equities
    isn't allowed in the Indian cash market -- this is only viable via
    stock futures, hence restricting to F&O-eligible names; see
    get_fno_list.py). Reuses _build_universe_calendar(weakest=True,
    allowed_symbols=fno_symbols) so this can't drift out of sync with the
    long strategy's ranking formula.

    Entry: at the close of any bar where ema_fast < ema_slow, for a
    universe stock not currently shorted and while fewer than max_entries
    positions are open. Executed (short sale) at the NEXT bar's open, for
    the same lookahead reason documented in run_ema_crossover_backtest.

    Exit (cover): at the close of any bar where EITHER the bar's own close
    is above ema_slow (the spec's "initial" stop -- price recovering past
    the slow EMA invalidates the downtrend thesis) OR ema_fast has closed
    above ema_slow (the spec's "trailing" stop), both checked every bar for
    the life of the position. No fixed profit target. Executed at the next
    bar's open.

    Position sizing: EQUAL notional per slot (capital_base / max_entries),
    not risk-based -- the spec here gives a capital figure and a slot count
    rather than a risk %, so each of the up to max_entries concurrent
    shorts gets a fixed capital_base / max_entries of notional, sized off
    entry price. risked_rs (distance from entry to the initial ema_slow
    stop) is still recorded per trade for R-multiple reporting, it just
    isn't what determines qty here.

    Simplification: modeled as directly shorting the stock at its spot
    price (proceeds credited to cash at entry, debited at cover; mark-to-
    market values the open liability at the current close) -- economically
    close to a fully-margined stock future position, but ignores real
    futures mechanics (lot sizes, margin requirements, rollover, futures-
    spot basis). No slippage or costs modeled, consistent with the rest of
    this project's swing strategies.
    """
    universe_by_period = _build_universe_calendar(
        monthly_prices, membership, lookback_months, skip_months, n_stocks, min_price,
        allowed_symbols=fno_symbols, weakest=True,
    )

    empty = {
        "trades": pd.DataFrame(columns=["symbol", "entry_date", "entry_price", "stop_price",
                                         "exit_date", "exit_price", "exit_reason", "qty", "risked_rs",
                                         "pnl_rs", "pnl_pct", "r_multiple", "hold_days", "status"]),
        "equity": pd.Series(dtype=float),
        "n_universe_periods": 0,
        "capital_base": capital_base,
    }
    if not universe_by_period:
        return empty

    rebalance_dates = pd.DatetimeIndex([u[0] for u in universe_by_period])
    start_date = rebalance_dates[0]
    last_bar_date = bar_close.index.max()
    if last_bar_date <= start_date:
        return empty
    bars = bar_close.index[(bar_close.index > start_date) & (bar_close.index <= last_bar_date)]
    if len(bars) == 0:
        return empty

    ema_fast_series = bar_close.ewm(span=ema_fast, adjust=False).mean()
    ema_slow_series = bar_close.ewm(span=ema_slow, adjust=False).mean()
    notional_per_slot = capital_base / max_entries

    cash = capital_base
    positions: dict[str, dict] = {}
    trades: list[dict] = []
    equity_series = pd.Series(index=bars, dtype=float)
    pending_entries: dict[str, float] = {}  # symbol -> initial stop (ema_slow at signal bar)
    pending_exits: dict[str, str] = {}  # symbol -> exit_reason

    for bar in bars:
        ridx = rebalance_dates.searchsorted(bar, side="right") - 1
        universe = universe_by_period[ridx][1] if ridx >= 0 else []

        # --- execute signals queued from the PREVIOUS bar, at THIS bar's open ---
        for sym in list(pending_exits.keys()):
            reason = pending_exits.pop(sym)
            if sym not in positions or sym not in bar_open.columns or bar not in bar_open.index:
                continue
            open_px = bar_open.at[bar, sym]
            if pd.isna(open_px) or open_px <= 0:
                continue
            pos = positions.pop(sym)
            qty = pos["qty"]
            cash -= qty * open_px  # buy to cover
            pnl_rs = qty * (pos["entry_price"] - open_px)
            trades.append({
                "symbol": sym, "entry_date": pos["entry_date"], "entry_price": pos["entry_price"],
                "stop_price": pos["stop_price"], "exit_date": bar, "exit_price": open_px,
                "exit_reason": reason, "qty": qty, "risked_rs": pos["risked_rs"],
                "pnl_rs": pnl_rs, "pnl_pct": 1 - open_px / pos["entry_price"],
                "r_multiple": pnl_rs / pos["risked_rs"] if pos["risked_rs"] > 0 else np.nan,
                "hold_days": (bar - pos["entry_date"]).days, "status": "closed",
            })

        for sym in list(pending_entries.keys()):
            initial_stop = pending_entries.pop(sym)
            if sym in positions or len(positions) >= max_entries:
                continue
            if sym not in bar_open.columns or bar not in bar_open.index:
                continue
            open_px = bar_open.at[bar, sym]
            if pd.isna(open_px) or open_px <= 0 or initial_stop <= open_px:
                continue
            qty = notional_per_slot / open_px
            risk_per_share = initial_stop - open_px
            cash += qty * open_px  # short-sale proceeds
            positions[sym] = {
                "entry_date": bar, "entry_price": open_px, "qty": qty,
                "stop_price": initial_stop, "risked_rs": qty * risk_per_share,
            }

        # --- evaluate THIS bar's close for new signals, queued for next bar ---
        for sym in list(positions.keys()):
            if sym in pending_exits or sym not in bar_close.columns or bar not in bar_close.index:
                continue
            c = bar_close.at[bar, sym]
            ef = ema_fast_series.at[bar, sym] if sym in ema_fast_series.columns else np.nan
            es = ema_slow_series.at[bar, sym] if sym in ema_slow_series.columns else np.nan
            if pd.isna(c) or pd.isna(ef) or pd.isna(es):
                continue
            if c > es:
                pending_exits[sym] = "close_above_ema_slow"
            elif ef > es:
                pending_exits[sym] = "ema_fast_above_ema_slow"

        if len(positions) < max_entries:
            for sym in universe:
                if len(positions) + len(pending_entries) >= max_entries:
                    break
                if sym in positions or sym in pending_entries:
                    continue
                if sym not in bar_close.columns or bar not in bar_close.index:
                    continue
                ef = ema_fast_series.at[bar, sym] if sym in ema_fast_series.columns else np.nan
                es = ema_slow_series.at[bar, sym] if sym in ema_slow_series.columns else np.nan
                if pd.isna(ef) or pd.isna(es):
                    continue
                if ef < es:
                    pending_entries[sym] = es

        # --- mark to market: cash minus the current cost to cover every open short ---
        mtm = cash
        for sym, pos in positions.items():
            px = bar_close.at[bar, sym] if sym in bar_close.columns and bar in bar_close.index else np.nan
            mtm -= pos["qty"] * (px if pd.notna(px) else pos["entry_price"])
        equity_series.at[bar] = mtm

    # still-open positions -> unrealized
    for sym, pos in positions.items():
        px = bar_close.at[bars[-1], sym] if sym in bar_close.columns else np.nan
        if pd.isna(px):
            continue
        qty = pos["qty"]
        pnl_rs = qty * (pos["entry_price"] - px)
        trades.append({
            "symbol": sym, "entry_date": pos["entry_date"], "entry_price": pos["entry_price"],
            "stop_price": pos["stop_price"], "exit_date": pd.NaT, "exit_price": px,
            "exit_reason": "open", "qty": qty, "risked_rs": pos["risked_rs"],
            "pnl_rs": pnl_rs, "pnl_pct": 1 - px / pos["entry_price"],
            "r_multiple": pnl_rs / pos["risked_rs"] if pos["risked_rs"] > 0 else np.nan,
            "hold_days": (bars[-1] - pos["entry_date"]).days, "status": "open",
        })

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df = trades_df.sort_values("entry_date").reset_index(drop=True)

    return {
        "trades": trades_df,
        "equity": equity_series,
        "n_universe_periods": len(universe_by_period),
        "capital_base": capital_base,
    }
