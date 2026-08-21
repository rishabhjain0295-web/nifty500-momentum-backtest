"""
EMA + relative-strength "2-of-3 vote" rotation strategy.

Three criteria, generalized to a selectable "equity leg" rotated against
GOLDBEES (see EQUITY_LEGS below):
  1. The leg's spot index close > 250-day EMA (of that spot index)
  2. The leg's spot index close > 50-day EMA (of that spot index)
  3. Relative strength: the leg's ETF 88-trading-day return > GOLDBEES's
     88-trading-day return

The vote is checked once a month, on the last trading day of the month.
If at least `min_votes` of the 3 are true at that check, the target is
the equity leg's ETF; otherwise GOLDBEES (100% in one or the other, no
cash state). The actual rebalance only happens on the first trading day
of the *following* month -- and even then only if the target differs
from the current holding. That execution day's own return is still
earned by whatever was held coming into it (the trade is placed at that
day's close); the new holding only starts earning returns from the next
trading day onward -- no intraday lookahead, matching the convention used
by the other engines in this project.

Equity legs and their real data coverage (Yahoo Finance, as of this
project's data pull):
  - NIFTY50:     NIFTY50 spot back to 2008-01-01; NIFTYBEES ETF from
                 2009-01-02. Longest, most reliable leg.
  - MIDCAP150:   Nifty Midcap 150 spot AND MIDCAPBEES (Nippon MID150BEES)
                 both only exist from ~2019 -- this is a real methodology
                 launch date, not a data gap (Nippon renamed the fund when
                 the Midcap 150 index launched).
  - SMALLCAP250: Nifty Smallcap 250 spot has long history back to 2008,
                 but there is no Nippon "BEES" smallcap ETF -- the ETF leg
                 uses HDFCSML250 (HDFC Nifty Smallcap 250 ETF), which only
                 started trading 2023-02-21. Backtest for this leg is
                 necessarily short (~3.5 years), not 2008-present.
  - NIFTY500:    NIFTY500 spot back to 2008-01-01, but the only tradable
                 broad Nifty 500 ETF found is MONIFTY500 (Motilal Oswal
                 Nifty500 ETF), which only started trading 2023-09-29.
                 Backtest for this leg is necessarily short (~3 years).

For every leg, the EMA criteria use the *spot index's own* (often longer)
calendar for warmup, then align onto the ETF's trading calendar -- so the
EMAs are as pre-warmed as the data allows, but the live decision can never
start before the ETF itself has both RS-lookback and trading history.

No transaction costs or slippage modeled. Reuses clean_bad_ticks() to
repair known Yahoo Finance vendor glitches in the ETF/index price series.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from etf_rotation_engine import (  # noqa: F401
    clean_bad_ticks,
    perf_stats,
    annual_returns,
    annual_max_drawdown,
    yearly_table,
)

ROOT = Path(__file__).resolve().parent
ETF_DIR = ROOT / "data" / "etfs"
INDEX_DIR = ROOT / "data" / "index"

EMA_LONG = 250
EMA_SHORT = 50
RS_LOOKBACK = 88
MIN_VOTES = 2

EQUITY_LEGS = {
    "NIFTY50": {
        "label": "Nifty 50 (NIFTYBEES)",
        "spot_file": "NIFTY50.csv",
        "etf_symbol": "NIFTYBEES",
        "spot_start": "2008-01-01",
        "etf_start": "2009-01-02",
        "note": "Longest history of all four legs.",
    },
    "MIDCAP150": {
        "label": "Nifty Midcap 150 (MIDCAPBEES)",
        "spot_file": "NIFTYMIDCAP150.csv",
        "etf_symbol": "MIDCAPBEES",
        "spot_start": "2019-01-14",
        "etf_start": "2019-06-20",
        "note": "Both the index and MIDCAPBEES (Nippon MID150BEES) launched in 2019 -- a real "
                "methodology start, not a data gap.",
    },
    "SMALLCAP250": {
        "label": "Nifty Smallcap 250 (HDFCSML250)",
        "spot_file": "NIFTYSMLCAP250.csv",
        "etf_symbol": "HDFCSML250",
        "spot_start": "2008-01-01",
        "etf_start": "2023-02-21",
        "note": "No Nippon 'BEES' smallcap fund exists -- uses HDFC's Nifty Smallcap 250 ETF, "
                "which only started trading 2023-02-21. Backtest is necessarily short (~3.5 years).",
    },
    "NIFTY500": {
        "label": "Nifty 500 (MONIFTY500)",
        "spot_file": "NIFTY500.csv",
        "etf_symbol": "MONIFTY500",
        "spot_start": "2008-01-01",
        "etf_start": "2023-09-29",
        "note": "Only tradable broad Nifty 500 ETF found is Motilal Oswal's MONIFTY500, which "
                "only started trading 2023-09-29. Backtest is necessarily short (~3 years).",
    },
}
DEFAULT_EQUITY_LEG = "NIFTY50"


def load_data(equity_leg: str = DEFAULT_EQUITY_LEG, price_col: str = "Close"):
    """Returns (spot, prices, equity_symbol) for the chosen leg. `prices` has
    keys [equity_symbol, "GOLDBEES"]."""
    leg = EQUITY_LEGS[equity_leg]
    equity_symbol = leg["etf_symbol"]

    spot_df = pd.read_csv(INDEX_DIR / leg["spot_file"], index_col=0, parse_dates=True)
    spot = clean_bad_ticks(spot_df[price_col].dropna().sort_index())

    prices = {}
    for sym in [equity_symbol, "GOLDBEES"]:
        df = pd.read_csv(ETF_DIR / f"{sym}.csv", index_col=0, parse_dates=True)
        s = df[price_col].dropna().sort_index()
        if sym == "MIDCAPBEES":
            s = s[s.index >= "2019-06-20"]  # drop stray isolated first row, see etf_rotation_engine.py
        prices[sym] = clean_bad_ticks(s)

    return spot, prices, equity_symbol


def run_backtest(spot: pd.Series, prices: dict[str, pd.Series], equity_symbol: str = "NIFTYBEES",
                  ema_long: int = EMA_LONG, ema_short: int = EMA_SHORT, rs_lookback: int = RS_LOOKBACK,
                  min_votes: int = MIN_VOTES):
    calendar = prices[equity_symbol].index

    # EMAs computed on the spot index's own (often longer) trading calendar,
    # so they're as warmed-up as possible -- then aligned onto the ETF
    # calendar for the vote.
    ema_long_full = spot.ewm(span=ema_long, adjust=False).mean()
    ema_short_full = spot.ewm(span=ema_short, adjust=False).mean()
    spot_on_cal = spot.reindex(calendar).ffill()
    ema250 = ema_long_full.reindex(calendar).ffill()
    ema50 = ema_short_full.reindex(calendar).ffill()

    equity_px = prices[equity_symbol].reindex(calendar).ffill()
    gold_px = prices["GOLDBEES"].reindex(calendar).ffill()
    equity_rs_ret = equity_px.pct_change(rs_lookback)
    gold_rs_ret = gold_px.pct_change(rs_lookback)

    crit1 = spot_on_cal > ema250
    crit2 = spot_on_cal > ema50
    crit3 = equity_rs_ret > gold_rs_ret
    rs_valid = equity_rs_ret.notna() & gold_rs_ret.notna()

    vote_count = crit1.astype(int) + crit2.astype(int) + crit3.fillna(False).astype(int)

    # Decision dates: last trading day of each month. Execution dates: first
    # trading day of the *next* month. A decision made at month M's close
    # only gets acted on at month M+1's first trading day.
    month_periods = calendar.to_period("M")
    cal_series = calendar.to_series()
    month_end_dates = cal_series.groupby(month_periods).last()
    month_start_dates = cal_series.groupby(month_periods).first()
    ordered_periods = sorted(month_end_dates.index)

    exec_decisions = {}  # execution date -> (target, decision date)
    for prev_period, this_period in zip(ordered_periods, ordered_periods[1:]):
        decision_date = month_end_dates[prev_period]
        if not rs_valid.loc[decision_date]:
            continue
        target = equity_symbol if vote_count.loc[decision_date] >= min_votes else "GOLDBEES"
        exec_decisions[month_start_dates[this_period]] = (target, decision_date)

    holding = pd.Series(index=calendar, dtype=object)
    signal_log = []
    current = equity_symbol  # default before RS criterion is computable
    for t in calendar:
        if t in exec_decisions:
            target, decision_date = exec_decisions[t]
            if target != current:
                current = target
                signal_log.append({
                    "exec_date": t, "decision_date": decision_date, "position": current,
                    "votes": int(vote_count.loc[decision_date]),
                    "crit_ema250": bool(crit1.loc[decision_date]),
                    "crit_ema50": bool(crit2.loc[decision_date]),
                    "crit_rs": bool(crit3.loc[decision_date]),
                })
        holding.loc[t] = current

    equity_ret = equity_px.pct_change()
    gold_ret = gold_px.pct_change()
    prev_holding = holding.shift(1)
    prev_holding.iloc[0] = equity_symbol

    rets = pd.Series(index=calendar, dtype=float)
    is_equity = prev_holding == equity_symbol
    is_gold = prev_holding == "GOLDBEES"
    rets[is_equity] = equity_ret[is_equity]
    rets[is_gold] = gold_ret[is_gold]

    diagnostics = pd.DataFrame({
        "spot": spot_on_cal, "ema250": ema250, "ema50": ema50,
        "crit_ema250": crit1, "crit_ema50": crit2, "crit_rs": crit3,
        "vote_count": vote_count, "holding": holding,
    })

    return rets.dropna(), pd.DataFrame(signal_log), diagnostics


def run_full_backtest(equity_leg: str = DEFAULT_EQUITY_LEG, price_col: str = "Close",
                       ema_long: int = EMA_LONG, ema_short: int = EMA_SHORT,
                       rs_lookback: int = RS_LOOKBACK, min_votes: int = MIN_VOTES):
    spot, prices, equity_symbol = load_data(equity_leg, price_col)
    strat_rets, signal_log, diagnostics = run_backtest(
        spot, prices, equity_symbol, ema_long, ema_short, rs_lookback, min_votes
    )

    bench_px = prices[equity_symbol]
    bench_rets = bench_px.pct_change().reindex(strat_rets.index).dropna()
    strat_rets = strat_rets.reindex(bench_rets.index)

    return {
        "strategy_returns": strat_rets,
        "benchmark_returns": bench_rets,
        "strategy_stats": perf_stats(strat_rets),
        "benchmark_stats": perf_stats(bench_rets),
        "signal_log": signal_log,
        "diagnostics": diagnostics,
        "equity_symbol": equity_symbol,
    }
