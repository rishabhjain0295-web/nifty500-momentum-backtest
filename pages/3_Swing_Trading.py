"""
Swing trading: scans a momentum-ranked universe (same ranking formula as
the Backtest page) each month for entries, trades them individually
instead of holding a rebalanced basket. Five strategy families:
  - Breakout (gap-up / Donchian channel), long, top-N momentum, with an
    R-multiple profit target.
  - EMA 15/50 crossover, long, top-N momentum, daily or hourly bars, no
    fixed target -- see run_ema_crossover_backtest's docstring for the
    hourly data history caveat. Hourly comes in two flavors: Yahoo
    (~2-3 years) or Upstox-derived (~4.5 years, resampled from the same
    30-min data as the RSI Reversal strategy below).
  - Short Momentum (F&O), short, BOTTOM-N (weakest) momentum among
    F&O-eligible stocks only, hourly or 2-hourly bars, no fixed target --
    see run_short_ema_crossover_backtest's docstring for the shorting
    simplification this implies.
  - ORB (Opening Range Breakout), long or short, hourly bars only (two
    flavors, same Yahoo/Upstox split as EMA crossover above). Long
    universe is top-N Nifty 500 momentum; short universe is bottom-N
    (weakest) F&O-eligible momentum, same reasoning as Short Momentum.
    See run_orb_backtest's docstring for the exact mechanics (opening
    range captured on each month's first trading day, held fixed for the
    month, re-entries allowed, forced exit at month end).
  - RSI Oversold Reversal, long, top-N momentum, 15-min/30-min/hourly
    bars, with an R-multiple target and a calendar-day time exit. See
    run_rsi_reversal_backtest's docstring. 15-min is capped at Yahoo
    Finance's ~60-day trailing window; 30-min instead comes from Upstox's
    Historical Candle Data API (~4.5 years, back to 2022-01-01, on par
    with hourly's ~2-3yr) -- see backtest_engine.load_30min_upstox_
    full_ohlc and scripts/download_upstox_30min_data.py.
See swing_engine.py for the full mechanics and the design assumptions
made where the source spec was ambiguous.

Position sizing (all five strategies) is either fixed capital throughout
(sizes every new position off the original starting capital, regardless
of how the account has actually grown or shrunk -- the long-standing
default) or compounding in steps (sizing capital ratchets UP by a
configurable % every time mark-to-market equity reaches another such
step, never back down on a drawdown) -- see the "Position sizing"
sidebar section and each run_*_backtest's capital_mode/compound_step_pct
parameters.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtest_engine import (
    NSE_UNIVERSES,
    ensure_15min_data,
    ensure_30min_data,
    ensure_hourly_data,
    ensure_stock_data,
)
from leverage_engine import apply_mtf_leverage
from streamlit_cache import (
    cached_load_2h_ohlc,
    cached_load_15min_full_ohlc,
    cached_load_30min_upstox_full_ohlc,
    cached_load_daily_ohlc,
    cached_load_fno_symbols,
    cached_load_hourly_full_ohlc,
    cached_load_hourly_ohlc,
    cached_load_hourly_upstox_full_ohlc,
    cached_load_hourly_upstox_ohlc,
    cached_load_membership,
    cached_load_prices,
    cached_load_universe_symbols,
)
from swing_engine import (
    run_ema_crossover_backtest,
    run_orb_backtest,
    run_rsi_reversal_backtest,
    run_short_ema_crossover_backtest,
    run_swing_backtest,
)

st.set_page_config(page_title="Nifty 500 Swing Trading", layout="wide")

with st.spinner("Fetching price data (first run only)..."):
    ensure_stock_data()

st.title("Swing Trading")
st.caption(
    "Trades individual breakouts within the top-N momentum universe, instead of holding the "
    "whole basket and rebalancing monthly. A stock's momentum rank only gates NEW entries -- "
    "once a position is open it's managed to its own stop/target regardless of whether the "
    "stock stays in next month's top N (like a screener finding candidates, not a portfolio "
    "rebalance)."
)

with st.sidebar:
    st.header("Strategy")
    strategy_label = st.radio(
        "Strategy type",
        ["Breakout (gap-up / Donchian)", "EMA 15/50 crossover", "Short Momentum (F&O)",
         "ORB (Opening Range Breakout)", "RSI Oversold Reversal"],
        index=0,
    )
    is_ema = strategy_label == "EMA 15/50 crossover"
    is_short = strategy_label == "Short Momentum (F&O)"
    is_orb = strategy_label == "ORB (Opening Range Breakout)"
    is_rsi = strategy_label == "RSI Oversold Reversal"

    orb_direction = "long"
    if is_orb:
        orb_direction_label = st.radio(
            "ORB direction", ["Long (Nifty 500 momentum)", "Short (F&O weakest momentum)"], index=0
        )
        orb_direction = "long" if orb_direction_label.startswith("Long") else "short"
    is_orb_short = is_orb and orb_direction == "short"

    st.header("Universe")
    universe_options = list(NSE_UNIVERSES.keys())
    universe_label = st.selectbox(
        "Universe", universe_options, index=universe_options.index("Nifty 500"),
        help="Restricts the momentum-ranking pool to one NSE index's CURRENT constituents (a "
             "snapshot, not point-in-time -- NSE doesn't publish historical inclusion/exclusion "
             "logs for these indices the way it does for the Nifty 500). For short-direction "
             "strategies this is intersected with the F&O list below, never replacing it."
    )
    universe_allowed_symbols = (
        cached_load_universe_symbols(universe_label) if universe_label != "Nifty 500" else None
    )
    if universe_allowed_symbols is not None:
        extra = (
            ", further intersected with the F&O list since this strategy shorts"
            if (is_short or is_orb_short) else ""
        )
        st.caption(f"Restricted to the current {len(universe_allowed_symbols)} constituents of {universe_label}{extra}.")

    st.header("Momentum universe")
    n_stocks = st.slider(
        "Number of stocks in universe", min_value=5, max_value=100, value=20, step=5,
        help="For Short Momentum / ORB-short this is the WEAKEST N stocks by trailing return, "
             "not the strongest."
    )
    lookback_months = st.slider("Ranking lookback (months)", min_value=1, max_value=24, value=10, step=1)
    skip_months = st.slider(
        "Skip period (months)", min_value=0, max_value=3, value=1, step=1,
        help="Excludes the most recent N months from the ranking lookback."
    )
    price_col = st.selectbox("Price field for ranking", ["Adj Close", "Close"], index=0)
    min_price = st.number_input("Minimum price filter (Rs)", min_value=0.0, value=10.0, step=5.0)
    use_membership_filter = st.checkbox("Enforce point-in-time index membership", value=True)
    if is_short or is_orb_short:
        st.caption(
            "Also restricted to currently F&O-eligible stocks (~208 names) -- shorting "
            "individual equities isn't viable in the Indian cash market, only via stock "
            "futures. This is a CURRENT F&O list applied across all history, not a "
            "point-in-time one (see get_fno_list.py)."
        )

    use_custom_start = st.checkbox(
        "Custom start date", value=False,
        help="Shows what this strategy would look like if you'd started trading it on a "
             "specific date, instead of from the earliest available history. The momentum "
             "ranking still uses real price history from before this date (so the first "
             "eligible trade isn't cold-started) -- only the point where capital actually "
             "starts opening positions moves to this date."
    )
    min_start_date = None
    if use_custom_start:
        min_start_date = pd.Timestamp(st.date_input(
            "Start date", value=pd.Timestamp.today() - pd.DateOffset(years=3),
        ))

    timeframe = "Daily"
    ema_fast = 15
    ema_slow = 50
    max_entries = 10
    entry_strategy = "gap_up"
    gap_pct = 1.0
    donchian_entry_lookback = 20
    exit_mode = "trailing_donchian"
    exit_lookback_days = 1
    stop_pct = 8.0
    risk_reward_ratio = 2.0
    range_minutes = 60
    max_reentries = 4
    orb_stop_mode = "range"
    position_pct = 5.0
    min_stop_pct = 1.0
    rsi_period = 14
    rsi_threshold = 31.0
    max_hold_days = 45
    max_stop_pct = 4.0

    use_target = False

    if is_short:
        st.header("Short Momentum (F&O)")
        timeframe = st.radio("Timeframe", ["Hourly (~2-3 years)", "2-Hourly (~2-3 years)"], index=0)
        st.caption(
            "Yahoo Finance only serves hourly data for roughly the trailing 2-3 years, unlike "
            "the daily data used elsewhere in this app which goes back to 2008. 2-Hourly bars "
            "are built by pairing consecutive hourly bars within each trading day (see "
            "backtest_engine.load_2h_ohlc), so they share the same ~2-3 year window."
        )
        ema_fast = st.number_input("Fast EMA span (bars)", min_value=2, max_value=100, value=15, step=1)
        ema_slow = st.number_input("Slow EMA span (bars)", min_value=5, max_value=300, value=50, step=1)
        st.caption(
            "Entry (short): fast EMA below slow EMA at a bar's close, filled at the next bar's "
            "open. Exit (cover): EITHER close above slow EMA OR fast EMA above slow EMA, also "
            "filled at the next bar's open -- checked every bar for the life of the trade."
        )
        min_stop_pct = st.slider(
            "Minimum stop distance for R-multiple reporting (%)", min_value=0.1, max_value=10.0,
            value=1.0, step=0.1,
            help="A fresh crossover often happens with price sitting right on ema_slow, which "
                 "otherwise makes risked_rs (and r_multiple) blow up from dividing by a near-zero "
                 "stop distance on an ordinary trade. This floors that distance for sizing/"
                 "reporting only -- the real cover trigger is always the CURRENT ema_slow, "
                 "unaffected by this setting."
        )
        use_target = st.checkbox("Enable profit target", value=False)
        if use_target:
            risk_reward_ratio = st.number_input(
                "Target: multiple of initial risk (target = entry - this x risk)",
                min_value=0.5, max_value=20.0, value=2.0, step=0.5,
                help="Risk = initial stop (ema_slow at signal) minus entry price. E.g. 2.0 = "
                     "target is twice that distance BELOW entry. Checked against each bar's "
                     "close (no intrabar high/low for hourly/2h bars), fixed for the life of "
                     "the trade even if the trailing stop later tightens."
            )
            st.caption("With the target on, cover fires on target OR either stop condition -- whichever comes first.")
        else:
            st.caption("No fixed profit target (default) -- pure trend-following exit via the stop conditions above.")
    elif is_orb:
        st.header("Opening Range Breakout")
        orb_timeframe = st.radio(
            "Timeframe", ["Hourly (Yahoo, ~2-3 years)", "Hourly (Upstox, ~4.5 years)"], index=0,
        )
        range_minutes = st.slider(
            "Opening range (minutes)", min_value=60, max_value=180, value=60, step=60,
            help="Must be a multiple of 60 -- only hourly bars are available, so anything finer "
                 "than one hourly bar isn't computable here. 60 = the first hourly bar of the "
                 "month's first trading day; 120 = the first two, etc."
        )
        max_reentries = st.slider(
            "Max re-entries per stock per month", min_value=0, max_value=4, value=4, step=1,
            help="0 = only the first entry each month, no re-entry after a stop-out. 4 = up to 4 "
                 "re-entries (5 entries total) at the same range levels if repeatedly stopped out. "
                 "Resets every month."
        )
        orb_stop_label = st.radio(
            "Stop mechanism", ["Opposite range edge (default)", "Fixed % stoploss"], index=0,
            help="Opposite edge: the stop is fixed off the RANGE, so every re-entry in a month "
                 "shares the same stop level. Fixed %: the stop is fixed off each entry's OWN "
                 "fill price instead, so re-entries at a different price get a different stop."
        )
        orb_stop_mode = "range" if orb_stop_label.startswith("Opposite") else "fixed_pct"
        if orb_stop_mode == "fixed_pct":
            stop_pct = st.slider(
                "Fixed stoploss (% from entry)", min_value=0.5, max_value=30.0, value=8.0, step=0.5,
                help="Long: stop = entry price x (1 - this%). Short: stop = entry price x (1 + this%)."
            )
        if orb_timeframe.startswith("Hourly (Yahoo"):
            st.caption(
                "Yahoo Finance only serves hourly data for roughly the trailing 2-3 years, unlike "
                "the daily data used elsewhere in this app which goes back to 2008."
            )
        else:
            st.caption(
                "Hourly bars built by pairing up the RSI Reversal strategy's 30-min Upstox bars "
                "(see backtest_engine.load_hourly_upstox_full_ohlc) -- real history back to "
                "2022-01-01, ~4.5 years, roughly double the Yahoo hourly option above. Same "
                "~329-symbol universe coverage caveat as the other Upstox-sourced options: a "
                "given month's top/bottom-N pick outside that set is silently skipped rather "
                "than erroring."
            )
        stop_desc = (
            f"a bar closes {'below' if orb_direction == 'long' else 'above'} the fixed stoploss "
            f"({stop_pct}% from entry)" if orb_stop_mode == "fixed_pct" else
            f"a bar closes {'below the range LOW' if orb_direction == 'long' else 'above the range HIGH'} "
            "(the opposite edge of the same range)"
        )
        if orb_direction == "long":
            st.caption(
                "Range = high/low of the opening window on the month's first trading day, held "
                f"fixed for the whole month. Entry: a bar closes above the range high, filled at "
                f"the next bar's open. Stop: {stop_desc}. Re-entry: if stopped out, allowed again "
                "later in the same month (up to the limit above) if price closes above the range "
                "high again. Exit: force-closed at the close of the last bar of the month if the "
                "stop hasn't hit."
            )
        else:
            st.caption(
                "Range = high/low of the opening window on the month's first trading day, held "
                f"fixed for the whole month. Entry (short): a bar closes below the range low, "
                f"filled at the next bar's open. Stop: {stop_desc}. Re-entry: if stopped out, "
                "allowed again later in the same month (up to the limit above) if price closes "
                "below the range low again. Exit: force-closed at the close of the last bar of "
                "the month if the stop hasn't hit."
            )
        st.caption(
            "Universe coverage: hourly data only exists for stocks that were ever in the top "
            "momentum universe or the F&O list when it was downloaded (~329 symbols) -- if a "
            "given month's top/bottom-N pick falls outside that set, it's silently skipped (no "
            "range, no trades) rather than erroring."
        )
    elif is_ema:
        st.header("EMA crossover")
        timeframe = st.radio(
            "Timeframe",
            ["Daily (full history)", "Hourly (Yahoo, ~2-3 years)", "Hourly (Upstox, ~4.5 years)"],
            index=0,
        )
        if timeframe.startswith("Hourly (Yahoo"):
            st.caption(
                "Yahoo Finance only serves hourly data for roughly the trailing 2-3 years "
                "(observed back to ~2023-09 for this project's download), unlike the daily "
                "data which goes back to 2008. Results below will only cover that shorter, "
                "more recent window, and only for the ~216 stocks that were actually in the "
                "top-30 momentum universe during it."
            )
        elif timeframe.startswith("Hourly (Upstox"):
            st.caption(
                "Hourly bars built by pairing up the RSI Reversal strategy's 30-min Upstox "
                "bars (see backtest_engine.load_hourly_upstox_ohlc) -- real history back to "
                "2022-01-01, ~4.5 years, roughly double the Yahoo hourly option above. Same "
                "~329-symbol universe coverage caveat as the other Upstox-sourced options: a "
                "given month's top-N pick outside that set is silently skipped rather than "
                "erroring."
            )
        ema_fast = st.number_input("Fast EMA span (bars)", min_value=2, max_value=100, value=15, step=1)
        ema_slow = st.number_input("Slow EMA span (bars)", min_value=5, max_value=300, value=50, step=1)
        st.caption(
            "Entry: fast EMA above slow EMA at a bar's close, filled at the next bar's open. "
            "Exit: EITHER close below slow EMA OR fast EMA below slow EMA, also filled at the "
            "next bar's open -- checked every bar for the life of the trade. No fixed profit "
            "target; this is a pure trend-following exit."
        )
        min_stop_pct = st.slider(
            "Minimum stop distance for sizing/R-multiple (%)", min_value=0.1, max_value=10.0,
            value=1.0, step=0.1,
            help="A fresh crossover often happens with price sitting right on ema_slow, which "
                 "otherwise both saturates position size at the max-position cap regardless of "
                 "how tight the real stop is, and makes risked_rs (and r_multiple) blow up from "
                 "dividing by a near-zero stop distance. This floors that distance for sizing/"
                 "reporting only -- the real exit trigger is always the CURRENT ema_slow, "
                 "unaffected by this setting."
        )
    elif is_rsi:
        st.header("RSI Oversold Reversal")
        timeframe = st.radio(
            "Timeframe", ["15 Min (~60 days only)", "30 Min (Upstox, ~4.5 years)", "1 Hour (~2-3 years)"],
            index=2,
        )
        if timeframe.startswith("15 Min"):
            st.caption(
                "Yahoo Finance caps intervals finer than 1 hour at a ~60-day trailing window -- "
                "much shorter than the ~2-3 years available at 1 hour, let alone the ~18 years "
                "of daily data used elsewhere in this app. Treat results here as a short recent "
                "sample, not a real multi-year backtest -- there's rarely room for more than a "
                "trade or two per stock to fully play out (especially with a 45-day time exit) "
                "in a 60-day window."
            )
        elif timeframe.startswith("30 Min"):
            st.caption(
                "Sourced from Upstox's Historical Candle Data API (not Yahoo Finance, which "
                "hard-caps sub-hourly intervals at ~60 days) -- real history back to 2022-01-01, "
                "~4.5 years, on par with the 1 Hour option below. Same ~329-symbol universe "
                "coverage caveat as the other intraday options: a given month's top-N pick "
                "outside that set is silently skipped rather than erroring."
            )
        else:
            st.caption(
                "Yahoo Finance only serves hourly data for roughly the trailing 2-3 years, "
                "unlike the daily data used elsewhere in this app which goes back to 2008."
            )
        rsi_period = st.number_input("RSI period (bars)", min_value=2, max_value=50, value=14, step=1)
        rsi_threshold = st.slider(
            "RSI oversold-recovery threshold", min_value=20.0, max_value=40.0, value=31.0, step=0.5,
            help="A bar is marked as an 'alert candle' the first time RSI closes at or above this "
                 "level, having closed below it the bar before -- a fresh recovery out of "
                 "oversold, not just 'RSI is currently above this'."
        )
        st.caption(
            "Entry: a bar closes above the alert candle's high, filled at the next bar's open. "
            "Only the MOST RECENT alert candle is active per stock -- a later RSI cross "
            "replaces an earlier untriggered one, and an alert is used up the moment a close "
            "breaks its high, whether or not the trade actually fills."
        )
        risk_reward_ratio = st.slider(
            "Target: multiple of initial risk (target = entry + this x risk)",
            min_value=2.0, max_value=50.0, value=5.0, step=0.5,
            help="Risk = entry price minus the alert candle's low. E.g. 5.0 = target is 5x that "
                 "distance above entry. The wide range (up to 50x) reflects how tight a stop can "
                 "be here -- capped by the max stop % below, but real candle ranges can still be "
                 "very small on a 15/30-min chart."
        )
        max_hold_days = st.number_input(
            "Time exit (calendar days)", min_value=1, max_value=365, value=45, step=1,
            help="If neither the stop nor the target has fired within this many calendar days "
                 "of entry, the position is closed at the next bar's open regardless."
        )
        max_stop_pct = st.slider(
            "Skip the trade if stoploss exceeds this % of entry", min_value=0.5, max_value=20.0,
            value=4.0, step=0.5,
            help="Checked at FILL time using the actual next-bar open, not the alert candle's "
                 "own close. If the alert candle's low implies a stop wider than this, the trade "
                 "is skipped entirely rather than taken with a larger-than-intended risk."
        )
        min_stop_pct = st.slider(
            "Minimum stop distance for sizing/R-multiple (%)", min_value=0.05, max_value=5.0,
            value=0.1, step=0.05,
            help="Floors the risk distance used for position sizing and r_multiple reporting "
                 "only, guarding the rare case of a near-zero-range alert candle. Rarely binds "
                 "here since the max-stop setting above already caps the top end."
        )
    else:
        st.header("Entry strategy")
        entry_label = st.radio("Entry signal", ["Gap-up breakout", "Donchian channel breakout"], index=0)
        entry_strategy = "gap_up" if entry_label == "Gap-up breakout" else "donchian"
        if entry_strategy == "gap_up":
            gap_pct = st.slider(
                "Gap-up threshold (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1,
                help="Buy at today's open when it gaps up this much or more above yesterday's close."
            )
        else:
            donchian_entry_lookback = st.slider(
                "Breakout lookback (days)", min_value=5, max_value=60, value=20, step=1,
                help="Buy when today's high breaks above the highest high of the trailing N days."
            )

        st.header("Exit / stop")
        exit_label = st.radio("Stop mechanism", ["Trailing low (Donchian-style)", "Fixed % stoploss"], index=0)
        exit_mode = "trailing_donchian" if exit_label == "Trailing low (Donchian-style)" else "fixed_pct"
        exit_lookback_days = 1 if entry_strategy == "gap_up" else donchian_entry_lookback
        if exit_mode == "trailing_donchian":
            exit_lookback_days = st.slider(
                "Trailing stop lookback (days)", min_value=1, max_value=60,
                value=(1 if entry_strategy == "gap_up" else donchian_entry_lookback), step=1,
                help="Stop = lowest low of the trailing N days, recomputed daily. N=1 is "
                     "'previous day's low'; N=20 (or your breakout lookback) is the 'channel low'."
            )
        else:
            stop_pct = st.slider("Fixed stoploss (% below entry)", min_value=0.5, max_value=30.0, value=8.0, step=0.5)

    st.header("Risk & money management")
    if is_short:
        capital_base = st.number_input(
            "Starting capital (Rs)", min_value=100_000.0, value=2_000_000.0, step=100_000.0
        )
        max_entries = st.slider(
            "Max concurrent short positions", min_value=1, max_value=30, value=10, step=1,
            help="Each open slot gets an equal share of starting capital as notional (capital / "
                 "this number), fixed regardless of stop distance -- not risk-based sizing like "
                 "the other two strategies, since the spec here is capital + a slot count."
        )
        risk_pct, max_position_pct = 1.0, 20.0  # unused by the short engine; kept defined for downstream code
    elif is_orb:
        capital_base = st.number_input("Starting capital (Rs)", min_value=100_000.0, value=1_000_000.0, step=100_000.0)
        position_pct = st.slider(
            "Position size (% of capital per stock)", min_value=1.0, max_value=100.0, value=5.0, step=0.5,
            help="A FIXED % of capital allocated to every entry (not risk-based). Every re-entry "
                 "within a month sizes off the same capital figure -- starting capital, or the "
                 "current compounded step if Position sizing below is set to compound -- and is "
                 "capped by available cash."
        )
        risk_pct, max_position_pct, risk_reward_ratio, max_entries = 1.0, 20.0, 2.0, 10  # unused, kept defined
    else:
        capital_base = st.number_input("Starting capital (Rs)", min_value=100_000.0, value=1_000_000.0, step=100_000.0)
        risk_pct = st.slider(
            "Risk per trade (% of capital)", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
            help="Position size is set so a stop-out loses exactly this % of starting capital -- "
                 "subject to the position-size cap below, which usually binds first on tight stops."
        )
        if not is_ema and not is_rsi:
            risk_reward_ratio = st.number_input(
                "Risk:reward ratio (target = entry + this x risk)", min_value=0.5, max_value=20.0, value=2.0, step=0.5,
                help="E.g. 2.0 = 1:2 -- target is twice the initial risk distance above entry. 10.0 = 1:10."
            )
        max_position_pct = st.slider(
            "Max position size (% of capital)", min_value=1.0, max_value=100.0, value=20.0, step=1.0,
            help="A hard ceiling on any single position's value. Necessary because pure risk-based "
                 "sizing can demand a huge position when the stop is tight -- e.g. a 1% stop distance "
                 "needs a position worth 100% of capital just to risk 1%. This cap (not risk_pct) is "
                 "usually what actually determines position size on tight-stop setups."
        )

    st.header("Position sizing")
    capital_mode_label = st.radio(
        "Capital used for sizing new positions", ["Fixed capital throughout (default)", "Compound in steps"],
        index=0,
        help="Fixed: every new position is always sized off the ORIGINAL starting capital, no "
             "matter how much the account has actually grown or shrunk -- what this page has "
             "always done. Compound in steps: sizing capital ratchets UP by a fixed % every time "
             "mark-to-market equity reaches another such step (e.g. 10L start, 50% steps -> once "
             "equity hits 15L, new positions size off 15L; once it hits 22.5L, off 22.5L; and so "
             "on). Only ratchets up -- a later drawdown below a step does NOT size back down."
    )
    capital_mode = "fixed" if capital_mode_label.startswith("Fixed") else "compounding_steps"
    compound_step_pct = 50.0
    if capital_mode == "compounding_steps":
        compound_step_pct = st.slider(
            "Compounding step (%)", min_value=10.0, max_value=200.0, value=50.0, step=5.0,
            help="How much mark-to-market equity must grow, from the last step reached, before "
                 "sizing capital steps up again."
        )

    st.header("MTF Leverage Overlay")
    use_leverage = st.checkbox(
        "Enable MTF leverage overlay", value=False,
        help="Optional, two-stage drawdown-triggered leverage (same engine as the Backtest page): "
             "adds extra MTF (Margin Trading Facility) exposure after a bad month, and more after "
             "a second bad month soon after. Its trigger/hold logic is inherently MONTHLY (a "
             "'month' with return <= -X%, hold for N months) -- since this page's own equity curve "
             "is bar-level (hourly/daily), the overlay runs on a MONTHLY-RESAMPLED copy of it "
             "instead, shown as a separate comparison below the main results, not applied to the "
             "granular trade log or equity chart above."
    )
    if use_leverage:
        lev_trigger1_pct = st.slider(
            "Leg 1 trigger: month return <= -X%", min_value=1.0, max_value=30.0, value=7.0, step=0.5,
            help="A resampled month with return at or below negative this % adds the first leg of leverage."
        )
        lev_leg1_pct = st.slider(
            "Leg 1 leverage added (%)", min_value=5.0, max_value=200.0, value=50.0, step=5.0,
            help="Extra exposure added as a fraction of current equity, e.g. 50% means 150% total exposure."
        )
        lev_trigger2_pct = st.slider(
            "Leg 2 trigger: another month return <= -X% within 2 months of leg 1", min_value=1.0,
            max_value=30.0, value=5.0, step=0.5,
            help="Within the 2 months AFTER leg 1's trigger month, a further month at or below "
                 "negative this % adds a second leg -- fires at most once per leg-1 cycle."
        )
        lev_leg2_pct = st.slider(
            "Leg 2 leverage added (%)", min_value=5.0, max_value=200.0, value=50.0, step=5.0,
            help="Additional exposure on top of leg 1, e.g. 50%+50% = 200% total exposure while both are active."
        )
        lev_hold_mode_label = st.radio(
            "Hold duration", ["Fixed number of months", "Until recovery above previous ATH"], index=0,
            help="Fixed months: each leg independently expires N months after its own trigger. "
                 "ATH recovery: a leg stays active until the strategy's OWN (unleveraged, "
                 "monthly-resampled) NAV recovers to X% above its all-time-high as of that leg's "
                 "trigger month."
        )
        lev_hold_mode = "fixed_months" if lev_hold_mode_label.startswith("Fixed") else "ath_recovery"
        lev_hold_months = 6
        lev_recovery_pct = 0.0
        if lev_hold_mode == "fixed_months":
            lev_hold_months = st.slider("Hold duration (months)", min_value=1, max_value=36, value=6, step=1)
        else:
            lev_recovery_pct = st.slider(
                "Recovery above previous ATH to revert (%)", min_value=0.0, max_value=50.0, value=0.0, step=1.0,
                help="0 = reverts as soon as the strategy's NAV merely reclaims its old high."
            )
        lev_annual_interest_pct = st.number_input(
            "MTF interest rate (% per year)", min_value=0.0, value=10.0, step=0.5,
            help="Charged monthly on the borrowed (leveraged) amount, deducted from equity "
                 "regardless of that month's P&L."
        )

monthly_prices = cached_load_prices(price_col)
membership = None
if use_membership_filter:
    membership = cached_load_membership(tuple(monthly_prices.index.values), tuple(monthly_prices.columns))

if is_orb:
    fno_symbols = cached_load_fno_symbols() if orb_direction == "short" else None
    if orb_timeframe.startswith("Hourly (Yahoo"):
        with st.spinner("Fetching hourly price data (first run only)..."):
            ensure_hourly_data()
        with st.spinner(f"Running ORB {orb_direction} backtest (hourly bars, ~2-3 year window)..."):
            bar_open, bar_high, bar_low, bar_close = cached_load_hourly_full_ohlc()
    else:
        with st.spinner("Fetching 30-minute price data (first run only)..."):
            ensure_30min_data()
        with st.spinner(f"Running ORB {orb_direction} backtest (hourly bars, ~4.5 year window)..."):
            bar_open, bar_high, bar_low, bar_close = cached_load_hourly_upstox_full_ohlc()
    result = run_orb_backtest(
        monthly_prices, membership, bar_open, bar_high, bar_low, bar_close, fno_symbols,
        lookback_months, skip_months, n_stocks, min_price,
        orb_direction, range_minutes, max_reentries, orb_stop_mode, stop_pct,
        position_pct, capital_base, allowed_symbols=universe_allowed_symbols, min_start_date=min_start_date,
        capital_mode=capital_mode, compound_step_pct=compound_step_pct,
    )
elif is_short:
    fno_symbols = cached_load_fno_symbols()
    with st.spinner("Fetching hourly price data (first run only)..."):
        ensure_hourly_data()
    if timeframe.startswith("2-Hourly"):
        with st.spinner("Running Short Momentum backtest (2-hourly bars, ~2-3 year window)..."):
            bar_open, bar_close = cached_load_2h_ohlc()
            result = run_short_ema_crossover_backtest(
                monthly_prices, membership, bar_open, bar_close, fno_symbols,
                lookback_months, skip_months, n_stocks, min_price,
                ema_fast, ema_slow, max_entries, min_stop_pct, capital_base,
                use_target, risk_reward_ratio, allowed_symbols=universe_allowed_symbols, min_start_date=min_start_date,
                capital_mode=capital_mode, compound_step_pct=compound_step_pct,
            )
    else:
        with st.spinner("Running Short Momentum backtest (hourly bars, ~2-3 year window)..."):
            bar_open, bar_close = cached_load_hourly_ohlc()
            result = run_short_ema_crossover_backtest(
                monthly_prices, membership, bar_open, bar_close, fno_symbols,
                lookback_months, skip_months, n_stocks, min_price,
                ema_fast, ema_slow, max_entries, min_stop_pct, capital_base,
                use_target, risk_reward_ratio, allowed_symbols=universe_allowed_symbols, min_start_date=min_start_date,
                capital_mode=capital_mode, compound_step_pct=compound_step_pct,
            )
elif is_ema:
    if timeframe.startswith("Hourly (Yahoo"):
        with st.spinner("Fetching hourly price data (first run only)..."):
            ensure_hourly_data()
        with st.spinner("Running EMA crossover backtest (hourly bars, ~2-3 year window)..."):
            bar_open, bar_close = cached_load_hourly_ohlc()
            result = run_ema_crossover_backtest(
                monthly_prices, membership, bar_open, bar_close,
                lookback_months, skip_months, n_stocks, min_price,
                ema_fast, ema_slow, risk_pct, max_position_pct, min_stop_pct, capital_base,
                allowed_symbols=universe_allowed_symbols, min_start_date=min_start_date,
                capital_mode=capital_mode, compound_step_pct=compound_step_pct,
            )
    elif timeframe.startswith("Hourly (Upstox"):
        with st.spinner("Fetching 30-minute price data (first run only)..."):
            ensure_30min_data()
        with st.spinner("Running EMA crossover backtest (hourly bars, ~4.5 year window)..."):
            bar_open, bar_close = cached_load_hourly_upstox_ohlc()
            result = run_ema_crossover_backtest(
                monthly_prices, membership, bar_open, bar_close,
                lookback_months, skip_months, n_stocks, min_price,
                ema_fast, ema_slow, risk_pct, max_position_pct, min_stop_pct, capital_base,
                allowed_symbols=universe_allowed_symbols, min_start_date=min_start_date,
                capital_mode=capital_mode, compound_step_pct=compound_step_pct,
            )
    else:
        with st.spinner("Running EMA crossover backtest (daily bars, full history)..."):
            daily_open, daily_high, daily_low, daily_close = cached_load_daily_ohlc()
            result = run_ema_crossover_backtest(
                monthly_prices, membership, daily_open, daily_close,
                lookback_months, skip_months, n_stocks, min_price,
                ema_fast, ema_slow, risk_pct, max_position_pct, min_stop_pct, capital_base,
                allowed_symbols=universe_allowed_symbols, min_start_date=min_start_date,
                capital_mode=capital_mode, compound_step_pct=compound_step_pct,
            )
elif is_rsi:
    if timeframe.startswith("15 Min"):
        with st.spinner("Fetching 15-minute price data (first run only)..."):
            ensure_15min_data()
        with st.spinner("Running RSI Reversal backtest (15-minute bars, ~60 day window)..."):
            bar_open, bar_high, bar_low, bar_close = cached_load_15min_full_ohlc()
    elif timeframe.startswith("30 Min"):
        with st.spinner("Fetching 30-minute price data (first run only)..."):
            ensure_30min_data()
        with st.spinner("Running RSI Reversal backtest (30-minute bars, ~4.5 year window)..."):
            bar_open, bar_high, bar_low, bar_close = cached_load_30min_upstox_full_ohlc()
    else:
        with st.spinner("Fetching hourly price data (first run only)..."):
            ensure_hourly_data()
        with st.spinner("Running RSI Reversal backtest (hourly bars, ~2-3 year window)..."):
            bar_open, bar_high, bar_low, bar_close = cached_load_hourly_full_ohlc()
    result = run_rsi_reversal_backtest(
        monthly_prices, membership, bar_open, bar_high, bar_low, bar_close,
        lookback_months, skip_months, n_stocks, min_price,
        rsi_period, rsi_threshold, risk_reward_ratio, max_hold_days, max_stop_pct, min_stop_pct,
        risk_pct, max_position_pct, capital_base, allowed_symbols=universe_allowed_symbols, min_start_date=min_start_date,
        capital_mode=capital_mode, compound_step_pct=compound_step_pct,
    )
else:
    daily_open, daily_high, daily_low, daily_close = cached_load_daily_ohlc()
    with st.spinner("Running swing backtest (daily scan across the full history)..."):
        result = run_swing_backtest(
            monthly_prices, membership, daily_open, daily_high, daily_low, daily_close,
            lookback_months, skip_months, n_stocks, min_price,
            entry_strategy, gap_pct, donchian_entry_lookback,
            exit_mode, exit_lookback_days, stop_pct,
            risk_pct, risk_reward_ratio, max_position_pct, capital_base,
            allowed_symbols=universe_allowed_symbols, min_start_date=min_start_date,
            capital_mode=capital_mode, compound_step_pct=compound_step_pct,
        )

trades = result["trades"]
equity = result["equity"]

if min_start_date is not None and not trades.empty:
    st.info(
        f"Showing results as if this strategy started trading on **{min_start_date.date()}** "
        f"({len(trades)} trades since). Momentum ranking still uses real price history from "
        "before this date."
    )

if capital_mode == "compounding_steps" and not equity.empty:
    n_steps = 0
    sc, mult = capital_base, 1 + compound_step_pct / 100.0
    for mtm in equity.values:
        while mtm >= sc * mult:
            sc *= mult
            n_steps += 1
    st.info(
        f"Position sizing compounded in **{compound_step_pct:.0f}% steps** -- capital used for "
        f"new positions stepped up {n_steps} time(s) over the backtest, from "
        f"Rs {capital_base:,.0f} to Rs {sc:,.0f}. Only ratchets up, never back down on a drawdown."
    )

if trades.empty or equity.empty:
    st.warning(
        "No trades were generated with these parameters -- try a lower gap threshold, a "
        "shorter Donchian lookback, a larger universe, (for the EMA/Short Momentum strategies) "
        "a shorter fast/slow EMA span, (for ORB) a shorter opening range, (for RSI Reversal) "
        "a higher RSI threshold, an earlier custom start date, or (for 15-min specifically) a "
        "longer timeframe -- it's the only one still capped at a ~60 day window."
    )
    st.stop()

has_target = "target_price" in trades.columns

leverage_result = None
monthly_equity_unlevered = None
if use_leverage:
    monthly_equity_unlevered = equity.resample("ME").last().dropna()
    monthly_rets = monthly_equity_unlevered.pct_change().dropna()
    if len(monthly_rets) > 0:
        leverage_result = apply_mtf_leverage(
            monthly_rets, lev_trigger1_pct, lev_leg1_pct, lev_trigger2_pct, lev_leg2_pct,
            lev_hold_mode, lev_hold_months, lev_recovery_pct, lev_annual_interest_pct,
        )


def fmt_rs(x: float) -> str:
    return f"Rs {x:,.0f}"


def fmt_pct(x: float) -> str:
    return "-" if pd.isna(x) else f"{x:.2%}"


final_equity = equity.iloc[-1]
total_return = final_equity / capital_base - 1
n_years = (equity.index[-1] - equity.index[0]).days / 365.25
cagr = (final_equity / capital_base) ** (1 / n_years) - 1 if n_years > 0 and final_equity > 0 else float("nan")
closed = trades[trades["status"] == "closed"]
win_rate = (closed["pnl_rs"] > 0).mean() if len(closed) else float("nan")
avg_r = closed["r_multiple"].mean() if len(closed) else float("nan")
running_max = equity.cummax()
dd = equity / running_max - 1
max_dd = dd.min()

st.subheader("Results")
cols = st.columns(5)
cols[0].metric("Starting capital", fmt_rs(capital_base))
cols[1].metric("Ending equity", fmt_rs(final_equity), delta=fmt_pct(total_return))
cols[2].metric("CAGR", fmt_pct(cagr))
cols[3].metric("Max drawdown", fmt_pct(max_dd))
cols[4].metric("Trades", str(len(trades)))

cols2 = st.columns(4)
cols2[0].metric("Win rate", fmt_pct(win_rate))
cols2[1].metric("Avg R-multiple (expectancy)", f"{avg_r:.2f}" if pd.notna(avg_r) else "-")
if has_target:
    breakeven_wr = 1 / (1 + risk_reward_ratio)
    cols2[2].metric("Breakeven win rate needed", fmt_pct(breakeven_wr))
else:
    breakeven_wr = float("nan")
    cols2[2].metric("Breakeven win rate needed", "n/a (no fixed target)")
cols2[3].metric("Open positions now", str((trades["status"] == "open").sum()))

if pd.notna(avg_r) and avg_r < 0:
    if has_target and is_short:
        st.warning(
            f"Negative expectancy ({avg_r:.2f}R per trade on average): this parameter combination "
            f"loses money on average per trade, even before compounding. The target gets hit less "
            f"often than the {breakeven_wr:.0%} win rate this risk:reward ratio needs to break "
            f"even -- try a smaller target multiple, a wider stop (larger EMA spans), or disable "
            f"the target and let the trailing-stop exit run."
        )
    elif has_target:
        st.warning(
            f"Negative expectancy ({avg_r:.2f}R per trade on average): this parameter combination "
            f"loses money on average per trade, even before compounding. A tight stop relative to "
            f"the entry signal (e.g. previous-day-low right after a gap) gets hit by ordinary "
            f"volatility more often than the {breakeven_wr:.0%} win rate this risk:reward ratio "
            f"needs to break even -- try a wider stop, a different risk:reward, or the other entry "
            f"strategy."
        )
    else:
        st.warning(
            f"Negative expectancy ({avg_r:.2f}R per trade on average): this parameter combination "
            f"loses money on average per trade, even before compounding. Trend-following systems "
            f"like this one often have a low win rate but rely on a few large winners to make up "
            f"for many small stop-outs -- check the R-multiple distribution below before concluding "
            f"the parameters are bad."
        )

st.subheader("Equity curve")
fig = go.Figure()
fig.add_trace(go.Scatter(x=equity.index, y=equity.values, name="Equity"))
fig.add_hline(y=capital_base, line_dash="dot", line_color="gray", annotation_text="starting capital")
fig.update_layout(
    yaxis_title="Portfolio value (Rs)",
    margin=dict(t=30, l=10, r=10, b=10), height=400,
)
st.plotly_chart(fig, use_container_width=True)

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Drawdown")
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=dd.index, y=dd.values, fill="tozeroy", name="Drawdown"))
    fig_dd.update_layout(yaxis_tickformat=".0%", margin=dict(t=30, l=10, r=10, b=10), height=300)
    st.plotly_chart(fig_dd, use_container_width=True)

with col_b:
    st.subheader("R-multiple distribution")
    fig_r = go.Figure()
    fig_r.add_trace(go.Histogram(x=closed["r_multiple"].clip(-3, 10), nbinsx=40))
    fig_r.add_vline(x=0, line_color="gray")
    fig_r.update_layout(
        xaxis_title="R-multiple (P&L / amount risked)", yaxis_title="Trade count",
        margin=dict(t=30, l=10, r=10, b=10), height=300,
    )
    st.plotly_chart(fig_r, use_container_width=True)

st.subheader("Exit reasons")
st.dataframe(
    trades["exit_reason"].value_counts().rename_axis("Exit reason").reset_index(name="Count"),
    hide_index=True,
)

if leverage_result is not None:
    with st.expander("MTF leverage overlay (monthly-resampled)"):
        st.caption(
            "This strategy's equity curve is bar-level (hourly/daily), but the leverage engine's "
            "trigger/hold logic is inherently monthly -- so this overlay runs on a MONTHLY "
            "resample of the equity curve above, shown here as a separate comparison. It does "
            "NOT change the KPIs, equity chart, or trade log above."
        )
        leveraged_rets = leverage_result["leveraged_rets"]
        leveraged_equity = capital_base * (1 + leveraged_rets).cumprod()
        total_interest_rs = leverage_result["interest_series"].sum() * capital_base

        lev_final = leveraged_equity.iloc[-1] if len(leveraged_equity) else float("nan")
        lev_n_years = (leveraged_equity.index[-1] - monthly_equity_unlevered.index[0]).days / 365.25
        lev_cagr = (lev_final / capital_base) ** (1 / lev_n_years) - 1 if lev_n_years > 0 and lev_final > 0 else float("nan")
        unlev_final = monthly_equity_unlevered.iloc[-1]
        unlev_cagr = (unlev_final / capital_base) ** (1 / lev_n_years) - 1 if lev_n_years > 0 and unlev_final > 0 else float("nan")
        lev_dd = (leveraged_equity / leveraged_equity.cummax() - 1).min() if len(leveraged_equity) else float("nan")
        unlev_dd = (monthly_equity_unlevered / monthly_equity_unlevered.cummax() - 1).min()

        st.caption(
            f"{leverage_result['n_tranches']} tranche(s) triggered. "
            f"Total interest paid: Rs {total_interest_rs:,.0f} (on a Rs {capital_base:,.0f} capital base)."
        )
        lev_cols = st.columns(4)
        lev_cols[0].metric("CAGR (monthly, unleveraged)", fmt_pct(unlev_cagr))
        lev_cols[1].metric("CAGR (monthly, leveraged)", fmt_pct(lev_cagr))
        lev_cols[2].metric("Max drawdown (unleveraged)", fmt_pct(unlev_dd))
        lev_cols[3].metric("Max drawdown (leveraged)", fmt_pct(lev_dd))

        fig_lev = go.Figure()
        fig_lev.add_trace(go.Scatter(
            x=monthly_equity_unlevered.index, y=monthly_equity_unlevered.values, name="Unleveraged (monthly)"
        ))
        fig_lev.add_trace(go.Scatter(x=leveraged_equity.index, y=leveraged_equity.values, name="Leveraged (monthly)"))
        fig_lev.update_layout(
            yaxis_title="Portfolio value (Rs)", margin=dict(t=30, l=10, r=10, b=10), height=350,
        )
        st.plotly_chart(fig_lev, use_container_width=True)

        if leverage_result["events"]:
            ev_df = pd.DataFrame(leverage_result["events"])
            ev_df["date"] = ev_df["date"].dt.date
            ev_df["start"] = ev_df["start"].dt.date
            ev_df["end"] = ev_df["end"].dt.date
            st.dataframe(
                ev_df.rename(columns={
                    "date": "Trigger month", "leg": "Leg", "month_return": "Month return",
                    "start": "Active from", "end": "Active until",
                }).style.format({"Month return": "{:.2%}"}),
                hide_index=True, height=250,
            )
        else:
            st.caption("No leverage triggers with these parameters.")

has_entry_rank = "entry_rank" in trades.columns

with st.expander("Trade log"):
    if has_entry_rank and is_orb:
        rank1_meaning = "weakest" if is_orb_short else "strongest"
        st.caption(
            f"entry_rank = the stock's position (1 = {rank1_meaning}) in that month's universe "
            "as of the monthly rebalance in effect when the entry fired -- ranking only updates "
            "monthly, so a re-entry later in the same month still uses the rank from the month's "
            "start. ORB positions never carry across a month boundary, so this always reflects "
            "the rank that was actually active for that specific trade."
        )
    elif has_entry_rank:
        st.caption(
            "entry_rank = the stock's position (1 = weakest) in the weakest-N F&O universe as "
            "of the monthly rebalance in effect when the entry SIGNAL fired -- not its rank "
            "today. A trade can look out of place if you check it later: the universe only "
            "gates new entries (an open short isn't force-closed when the stock's rank later "
            "improves), and ranking only updates monthly."
        )
    display_cols = ["symbol", "entry_date", "entry_price"]
    if has_entry_rank:
        display_cols.append("entry_rank")
    display_cols.append("stop_price")
    if has_target:
        display_cols.append("target_price")
    display_cols += ["exit_date", "exit_price", "exit_reason", "qty", "risked_rs",
                      "pnl_rs", "pnl_pct", "r_multiple", "hold_days", "status"]
    fmt_map = {
        "entry_price": "{:.2f}", "stop_price": "{:.2f}",
        "exit_price": "{:.2f}", "qty": "{:.1f}", "risked_rs": "Rs {:,.0f}",
        "pnl_rs": "Rs {:,.0f}", "pnl_pct": "{:.2%}", "r_multiple": "{:.2f}",
    }
    if has_target:
        fmt_map["target_price"] = "{:.2f}"
    st.dataframe(
        trades[display_cols].sort_values("entry_date", ascending=False).style.format(fmt_map),
        hide_index=True, height=400,
    )
    st.download_button(
        "Download trade log as CSV",
        trades.to_csv(index=False).encode("utf-8"),
        file_name="swing_trades.csv",
        mime="text/csv",
    )

st.divider()
if is_orb:
    st.caption(
        f"Universe recomputed at {result['n_universe_periods']} monthly rebalance points using the "
        "same ranking formula as the Backtest page (compute_momentum_ranking)"
        + (", restricted to F&O-eligible stocks and taking the WEAKEST names instead of the "
           "strongest" if is_orb_short else "")
        + ". Position sizing is a FIXED % of starting capital per trade (see sidebar), not "
          "risk-based, capped by available cash. No position ever carries across a month "
          "boundary -- every trade opens and closes within the same calendar month."
        + (" Modeled as directly shorting the stock at its spot price -- economically close to "
           "a fully-margined stock future, but ignoring real futures mechanics (lot sizes, "
           "margin, rollover, futures-spot basis)." if is_orb_short else "")
        + " Costs, taxes, and slippage are not modeled here yet."
    )
elif is_short:
    st.caption(
        f"Universe recomputed at {result['n_universe_periods']} monthly rebalance points using the "
        "same ranking formula as the Backtest page (compute_momentum_ranking), restricted to "
        "F&O-eligible stocks and taking the WEAKEST names instead of the strongest. Position "
        "sizing is equal notional per slot (starting capital / max concurrent positions), not "
        "risk-based. Modeled as directly shorting the stock at its spot price -- economically "
        "close to a fully-margined stock future, but ignoring real futures mechanics (lot sizes, "
        "margin, rollover, futures-spot basis). Costs, taxes, and slippage are not modeled here yet."
    )
elif is_rsi:
    st.caption(
        f"Universe recomputed at {result['n_universe_periods']} monthly rebalance points using the "
        "same ranking formula as the Backtest page (compute_momentum_ranking). Position sizing is "
        "risk-based (see sidebar), capped by the max-position-size setting and by available cash "
        "-- a trade whose implied stop exceeds the max-stop-% setting is skipped entirely, never "
        "taken with a larger-than-intended risk. Target and stop are both FIXED at entry (the "
        "target doesn't trail); the time exit closes anything neither has caught within the "
        "configured number of calendar days. Idle cash between trades earns 0% (a "
        "simplification). Costs, taxes, and slippage are not modeled here yet."
    )
else:
    st.caption(
        f"Universe recomputed at {result['n_universe_periods']} monthly rebalance points using the "
        "same ranking formula as the Backtest page (compute_momentum_ranking), so this can't drift "
        "out of sync with it. Position sizing is risk-based (see sidebar), not equal-weight -- "
        "actual position values vary trade to trade depending on stop distance, capped by the "
        "max-position-size setting. Idle cash between trades earns 0% (a simplification). Costs, "
        "taxes, and slippage are not modeled here yet."
    )
