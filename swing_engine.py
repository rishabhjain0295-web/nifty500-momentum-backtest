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

    # Bar -> period-index mapping, normalized to each bar's own month-start
    # before the searchsorted lookup -- see run_orb_backtest's comment on
    # this same line for why a raw timestamp comparison is wrong (it puts
    # the last trading day of each month into the NEXT month's universe).
    bar_month_starts = pd.DatetimeIndex([b.replace(day=1, hour=0, minute=0, second=0) for b in bars])
    ridx_of_bar = dict(zip(bars, rebalance_dates.searchsorted(bar_month_starts, side="right") - 1))

    cash = capital_base
    positions: dict[str, dict] = {}
    trades: list[dict] = []
    equity_series = pd.Series(index=bars, dtype=float)
    pending_entries: dict[str, float] = {}  # symbol -> stop level decided at the signal bar
    pending_exits: dict[str, str] = {}  # symbol -> exit_reason decided at the signal bar

    for bar in bars:
        ridx = ridx_of_bar[bar]
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
    use_target: bool = False,
    risk_reward_ratio: float = 2.0,
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
    Each trade records entry_rank -- the symbol's position (1 = weakest)
    in the WEAKEST-N universe as of the signal bar's most recent monthly
    rebalance. Two things can make a later-inspected trade look like it
    doesn't belong in "the bottom N you selected": (1) the universe only
    gates NEW entries -- a position stays open on its own stop/trailing-
    stop logic even after the stock's rank moves away from the bottom N
    at a later rebalance, so an old open trade can show a currently-
    unremarkable rank; (2) ranking is monthly, so a trade entered right
    at a rebalance boundary uses the OLD month's rank if the signal fired
    just before the boundary, not the new month's -- entry_rank always
    reflects the rank actually used at signal time, not "now".

    Exit (cover): at the close of any bar where EITHER the bar's own close
    is above ema_slow (the spec's "initial" stop -- price recovering past
    the slow EMA invalidates the downtrend thesis) OR ema_fast has closed
    above ema_slow (the spec's "trailing" stop), both checked every bar for
    the life of the position. Executed at the next bar's open.

    Optional profit target (use_target): if enabled, target_price = entry
    - risk_reward_ratio * (initial_stop - entry), i.e. an R-multiple of the
    ORIGINAL risk below entry, fixed for the life of the trade (doesn't
    move if the trailing stop later tightens the effective risk) -- same
    convention as run_swing_backtest's long-side target. Checked against
    each bar's CLOSE, same as the stop conditions -- there's no intrabar
    high/low for hourly/2h bars here (see load_hourly_ohlc), so, like the
    stop conditions, this can't detect a target touched and reversed
    within a single bar. Checked before the stop conditions each bar, but
    in practice they can't coincide: the target sits below entry, the stop
    conditions require price back above ema_slow (above entry), so at most
    one can be true on a given close. Off by default -- reproduces the
    original pure trend-following behavior (ride until a stop condition
    fires, no fixed exit).

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

    empty_cols = ["symbol", "entry_date", "entry_price", "entry_rank", "stop_price"]
    if use_target:
        empty_cols.append("target_price")
    empty_cols += ["exit_date", "exit_price", "exit_reason", "qty", "risked_rs",
                   "pnl_rs", "pnl_pct", "r_multiple", "hold_days", "status"]
    empty = {
        "trades": pd.DataFrame(columns=empty_cols),
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

    # Bar -> period-index mapping, normalized to each bar's own month-start
    # before the searchsorted lookup -- see run_orb_backtest's comment on
    # this same line for why a raw timestamp comparison is wrong (it puts
    # the last trading day of each month into the NEXT month's universe).
    bar_month_starts = pd.DatetimeIndex([b.replace(day=1, hour=0, minute=0, second=0) for b in bars])
    ridx_of_bar = dict(zip(bars, rebalance_dates.searchsorted(bar_month_starts, side="right") - 1))

    cash = capital_base
    positions: dict[str, dict] = {}
    trades: list[dict] = []
    equity_series = pd.Series(index=bars, dtype=float)
    pending_entries: dict[str, tuple[float, int]] = {}  # symbol -> (initial stop, rank at signal)
    pending_exits: dict[str, str] = {}  # symbol -> exit_reason

    for bar in bars:
        ridx = ridx_of_bar[bar]
        universe = universe_by_period[ridx][1] if ridx >= 0 else []
        # universe is ascending-momentum order (weakest last); rank 1 = weakest = the
        # strongest short candidate, mirroring how rank 1 means "strongest" for the
        # long strategies -- lets the trade log show exactly why a stock was picked.
        rank_of = {sym: len(universe) - i for i, sym in enumerate(universe)}

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
            trade_row = {
                "symbol": sym, "entry_date": pos["entry_date"], "entry_price": pos["entry_price"],
                "entry_rank": pos["entry_rank"], "stop_price": pos["stop_price"],
            }
            if use_target:
                trade_row["target_price"] = pos["target_price"]
            trade_row.update({
                "exit_date": bar, "exit_price": open_px,
                "exit_reason": reason, "qty": qty, "risked_rs": pos["risked_rs"],
                "pnl_rs": pnl_rs, "pnl_pct": 1 - open_px / pos["entry_price"],
                "r_multiple": pnl_rs / pos["risked_rs"] if pos["risked_rs"] > 0 else np.nan,
                "hold_days": (bar - pos["entry_date"]).days, "status": "closed",
            })
            trades.append(trade_row)

        for sym in list(pending_entries.keys()):
            initial_stop, entry_rank = pending_entries.pop(sym)
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
                "entry_rank": entry_rank,
            }
            if use_target:
                positions[sym]["target_price"] = open_px - risk_reward_ratio * risk_per_share

        # --- evaluate THIS bar's close for new signals, queued for next bar ---
        for sym in list(positions.keys()):
            if sym in pending_exits or sym not in bar_close.columns or bar not in bar_close.index:
                continue
            c = bar_close.at[bar, sym]
            ef = ema_fast_series.at[bar, sym] if sym in ema_fast_series.columns else np.nan
            es = ema_slow_series.at[bar, sym] if sym in ema_slow_series.columns else np.nan
            if pd.isna(c) or pd.isna(ef) or pd.isna(es):
                continue
            if use_target and c <= positions[sym]["target_price"]:
                pending_exits[sym] = "target_hit"
            elif c > es:
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
                    pending_entries[sym] = (es, rank_of[sym])

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
        trade_row = {
            "symbol": sym, "entry_date": pos["entry_date"], "entry_price": pos["entry_price"],
            "entry_rank": pos["entry_rank"], "stop_price": pos["stop_price"],
        }
        if use_target:
            trade_row["target_price"] = pos["target_price"]
        trade_row.update({
            "exit_date": pd.NaT, "exit_price": px,
            "exit_reason": "open", "qty": qty, "risked_rs": pos["risked_rs"],
            "pnl_rs": pnl_rs, "pnl_pct": 1 - px / pos["entry_price"],
            "r_multiple": pnl_rs / pos["risked_rs"] if pos["risked_rs"] > 0 else np.nan,
            "hold_days": (bars[-1] - pos["entry_date"]).days, "status": "open",
        })
        trades.append(trade_row)

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df = trades_df.sort_values("entry_date").reset_index(drop=True)

    return {
        "trades": trades_df,
        "equity": equity_series,
        "n_universe_periods": len(universe_by_period),
        "capital_base": capital_base,
    }


def _compute_orb_ranges(
    bar_high: pd.DataFrame,
    bar_low: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    universe_by_period: list[tuple[pd.Timestamp, list[str]]],
    period_bars: list[pd.DatetimeIndex],
    range_bars: int,
) -> list[dict[str, tuple[float, float]]]:
    """For each monthly period, {symbol: (range_high, range_low)} from the
    high/low of the first range_bars hourly bars of that period's FIRST
    TRADING DAY, for every symbol in that period's universe. Returns a list
    aligned by period index (parallel to universe_by_period/period_bars)."""
    ranges_by_period: list[dict[str, tuple[float, float]]] = []
    for pidx, (_, universe) in enumerate(universe_by_period):
        bars = period_bars[pidx]
        ranges: dict[str, tuple[float, float]] = {}
        if len(bars) > 0:
            first_day = bars[0].normalize()
            range_window = bars[bars.normalize() == first_day][:range_bars]
            for sym in universe:
                if sym not in bar_high.columns or sym not in bar_low.columns:
                    continue
                highs = bar_high.loc[bar_high.index.isin(range_window), sym].dropna()
                lows = bar_low.loc[bar_low.index.isin(range_window), sym].dropna()
                if highs.empty or lows.empty:
                    continue
                ranges[sym] = (highs.max(), lows.min())
        ranges_by_period.append(ranges)
    return ranges_by_period


def run_orb_backtest(
    monthly_prices: pd.DataFrame,
    membership: pd.DataFrame | None,
    bar_open: pd.DataFrame,
    bar_high: pd.DataFrame,
    bar_low: pd.DataFrame,
    bar_close: pd.DataFrame,
    fno_symbols: set[str] | None,
    lookback_months: int,
    skip_months: int,
    n_stocks: int,
    min_price: float,
    direction: str,
    range_minutes: int = 60,
    position_pct: float = 5.0,
    capital_base: float = 1_000_000.0,
) -> dict:
    """Opening Range Breakout (ORB), long or short (direction="long" or
    "short"), on hourly bars. Universe: for direction="long", the TOP
    n_stocks by momentum among all eligible Nifty 500 stocks (fno_symbols
    ignored); for direction="short", the BOTTOM n_stocks (weakest) among
    fno_symbols only (real shorting needs stock futures -- same reasoning
    as run_short_ema_crossover_backtest). Reuses _build_universe_calendar
    so this can't drift out of sync with the other strategies' ranking.

    Opening range: the high/low of the first range_minutes of the 1-hour
    chart on the FIRST TRADING DAY of each month, held fixed as that
    month's breakout/stop levels for every remaining bar of the month
    (recomputed fresh each month, not a rolling channel). range_minutes
    must be a multiple of 60 -- only hourly bars are available (see
    load_hourly_full_ohlc), so anything finer isn't computable here;
    range_bars = max(1, round(range_minutes / 60)) hourly bars are used.

    Entry (long): first bar CLOSE above the range high, for a universe
    stock not currently in a position. Entry (short): first bar CLOSE
    below the range low. Executed at the NEXT bar's open (same lookahead-
    avoidance lag as every other hourly strategy in this module -- the
    close that triggers the signal isn't known until the bar closes).

    Stop: long stops on a CLOSE below the range LOW (the opposite edge of
    the same range, not a computed distance); short stops on a CLOSE
    above the range HIGH. Also executed at the next bar's open. A stock
    stopped out mid-month is free to re-trigger the SAME entry condition
    again later in the SAME month, using the SAME range/stop levels --
    there's no limit on re-entries per month, only on available cash and
    the entry condition recurring.

    Exit (time-based): any position still open is force-closed on the
    LAST bar of the last trading day of the month, AT THAT BAR'S OWN
    CLOSE (not deferred to a next-bar open -- this is a scheduled square-
    off, not a reactive signal, and there may be no "next bar" left in
    the month to defer to). That bar is excluded from new-entry signal
    evaluation, since entering and immediately force-exiting on the same
    bar would be a pointless zero-duration trade. No position ever
    carries across a month boundary.

    Position sizing: a FIXED position_pct of capital_base per trade
    (e.g. 5%), not risk-based -- matches the spec (a flat % of capital,
    not a per-trade risk %). Capped by available cash. Every re-entry in
    a month sizes off the same capital_base, not fluctuating equity.

    Simplification (short direction only): modeled as directly shorting
    the stock at its spot price, same convention and same caveats as
    run_short_ema_crossover_backtest (economically close to a fully-
    margined stock future, ignoring real futures mechanics).
    """
    is_long = direction == "long"
    range_bars = max(1, round(range_minutes / 60))

    universe_by_period = _build_universe_calendar(
        monthly_prices, membership, lookback_months, skip_months, n_stocks, min_price,
        allowed_symbols=(None if is_long else fno_symbols), weakest=(not is_long),
    )

    empty = {
        "trades": pd.DataFrame(columns=["symbol", "entry_date", "entry_price", "entry_rank", "stop_price",
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

    # Bar -> period-index mapping, the SINGLE source of truth for every other
    # per-period lookup below (period_bars, ranges, universe/rank in the loop).
    # rebalance_dates are calendar month-end dates at MIDNIGHT, but bars carry
    # an intraday time -- comparing raw timestamps would put the last trading
    # day's own bars (e.g. Oct 31, 09:15-15:15) one period too late, since
    # "Oct 31 09:15" > "Oct 31 00:00". Normalizing each bar to the 1st of ITS
    # OWN calendar month before the searchsorted lookup fixes this: every bar
    # anywhere in October maps to the same period regardless of day or time.
    bar_month_starts = pd.DatetimeIndex([b.replace(day=1, hour=0, minute=0, second=0) for b in bars])
    bar_ridx = rebalance_dates.searchsorted(bar_month_starts, side="right") - 1
    ridx_of_bar = dict(zip(bars, bar_ridx))

    period_bars: list[pd.DatetimeIndex] = [bars[bar_ridx == pidx] for pidx in range(len(rebalance_dates))]
    last_bar_of_period = {pb[-1] for pb in period_bars if len(pb) > 0}

    ranges_by_period = _compute_orb_ranges(bar_high, bar_low, rebalance_dates, universe_by_period, period_bars, range_bars)

    notional_per_trade = capital_base * position_pct / 100.0

    cash = capital_base
    positions: dict[str, dict] = {}
    trades: list[dict] = []
    equity_series = pd.Series(index=bars, dtype=float)
    pending_entries: dict[str, tuple[float, int]] = {}  # symbol -> (stop_price, rank)
    pending_exits: dict[str, str] = {}  # symbol -> exit_reason

    for bar in bars:
        ridx = ridx_of_bar[bar]
        universe = universe_by_period[ridx][1] if ridx >= 0 else []
        ranges = ranges_by_period[ridx] if ridx >= 0 else {}
        is_month_end_bar = bar in last_bar_of_period
        rank_of = {sym: (i + 1 if is_long else len(universe) - i) for i, sym in enumerate(universe)}

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
            pnl_rs = qty * (open_px - pos["entry_price"]) if is_long else qty * (pos["entry_price"] - open_px)
            cash += qty * open_px if is_long else -qty * open_px
            trades.append({
                "symbol": sym, "entry_date": pos["entry_date"], "entry_price": pos["entry_price"],
                "entry_rank": pos["entry_rank"], "stop_price": pos["stop_price"],
                "exit_date": bar, "exit_price": open_px, "exit_reason": reason, "qty": qty,
                "risked_rs": pos["risked_rs"], "pnl_rs": pnl_rs,
                "pnl_pct": (open_px / pos["entry_price"] - 1) if is_long else (1 - open_px / pos["entry_price"]),
                "r_multiple": pnl_rs / pos["risked_rs"] if pos["risked_rs"] > 0 else np.nan,
                "hold_days": (bar - pos["entry_date"]).days, "status": "closed",
            })

        for sym in list(pending_entries.keys()):
            stop_price, entry_rank = pending_entries.pop(sym)
            if sym in positions:
                continue
            if sym not in bar_open.columns or bar not in bar_open.index:
                continue
            open_px = bar_open.at[bar, sym]
            if pd.isna(open_px) or open_px <= 0:
                continue
            if is_long and stop_price >= open_px:
                continue
            if not is_long and stop_price <= open_px:
                continue
            qty = min(notional_per_trade / open_px, cash / open_px)
            if qty <= 0:
                continue
            risk_per_share = abs(open_px - stop_price)
            cash -= qty * open_px if is_long else -qty * open_px
            positions[sym] = {
                "entry_date": bar, "entry_price": open_px, "qty": qty,
                "stop_price": stop_price, "risked_rs": qty * risk_per_share,
                "entry_rank": entry_rank,
            }

        # --- FORCE-CLOSE any open position at THIS bar's own close, if this is the
        # last bar of the month (scheduled square-off, not a next-bar-deferred signal) ---
        if is_month_end_bar:
            for sym in list(positions.keys()):
                if sym not in bar_close.columns or bar not in bar_close.index:
                    continue
                px = bar_close.at[bar, sym]
                if pd.isna(px) or px <= 0:
                    continue
                pos = positions.pop(sym)
                qty = pos["qty"]
                pnl_rs = qty * (px - pos["entry_price"]) if is_long else qty * (pos["entry_price"] - px)
                cash += qty * px if is_long else -qty * px
                trades.append({
                    "symbol": sym, "entry_date": pos["entry_date"], "entry_price": pos["entry_price"],
                    "entry_rank": pos["entry_rank"], "stop_price": pos["stop_price"],
                    "exit_date": bar, "exit_price": px, "exit_reason": "month_end", "qty": qty,
                    "risked_rs": pos["risked_rs"], "pnl_rs": pnl_rs,
                    "pnl_pct": (px / pos["entry_price"] - 1) if is_long else (1 - px / pos["entry_price"]),
                    "r_multiple": pnl_rs / pos["risked_rs"] if pos["risked_rs"] > 0 else np.nan,
                    "hold_days": (bar - pos["entry_date"]).days, "status": "closed",
                })
                pending_exits.pop(sym, None)
                pending_entries.pop(sym, None)
        else:
            # --- evaluate THIS bar's close for new signals, queued for next bar ---
            for sym in list(positions.keys()):
                if sym in pending_exits or sym not in bar_close.columns or bar not in bar_close.index:
                    continue
                if sym not in ranges:
                    continue
                c = bar_close.at[bar, sym]
                if pd.isna(c):
                    continue
                range_high, range_low = ranges[sym]
                if is_long and c < range_low:
                    pending_exits[sym] = "stoploss"
                elif not is_long and c > range_high:
                    pending_exits[sym] = "stoploss"

            for sym in universe:
                if sym in positions or sym in pending_entries:
                    continue
                if sym not in bar_close.columns or bar not in bar_close.index or sym not in ranges:
                    continue
                c = bar_close.at[bar, sym]
                if pd.isna(c):
                    continue
                range_high, range_low = ranges[sym]
                if is_long and c > range_high:
                    pending_entries[sym] = (range_low, rank_of[sym])
                elif not is_long and c < range_low:
                    pending_entries[sym] = (range_high, rank_of[sym])

        # --- mark to market ---
        mtm = cash
        for sym, pos in positions.items():
            px = bar_close.at[bar, sym] if sym in bar_close.columns and bar in bar_close.index else np.nan
            px = px if pd.notna(px) else pos["entry_price"]
            mtm += pos["qty"] * px if is_long else -pos["qty"] * px
        equity_series.at[bar] = mtm

    # still-open positions at the very end of available data -> unrealized
    for sym, pos in positions.items():
        px = bar_close.at[bars[-1], sym] if sym in bar_close.columns else np.nan
        if pd.isna(px):
            continue
        qty = pos["qty"]
        pnl_rs = qty * (px - pos["entry_price"]) if is_long else qty * (pos["entry_price"] - px)
        trades.append({
            "symbol": sym, "entry_date": pos["entry_date"], "entry_price": pos["entry_price"],
            "entry_rank": pos["entry_rank"], "stop_price": pos["stop_price"],
            "exit_date": pd.NaT, "exit_price": px, "exit_reason": "open", "qty": qty,
            "risked_rs": pos["risked_rs"], "pnl_rs": pnl_rs,
            "pnl_pct": (px / pos["entry_price"] - 1) if is_long else (1 - px / pos["entry_price"]),
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
