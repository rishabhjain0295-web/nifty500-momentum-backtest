"""
ETF relative-strength (RS) rotation strategy.

Rule:
  - Candidates: JUNIORBEES, GOLDBEES, MIDCAPBEES. Reference: NIFTYBEES.
  - At each quarterly rebalance date, score each candidate by its trailing
    LOOKBACK_DAYS (55 trading days) return minus NIFTYBEES's trailing
    return over the same window ("relative strength" = excess return vs
    Nifty). Hold whichever candidate has the highest RS score.
  - If no candidate beats NIFTYBEES (max RS <= 0), hold NIFTYBEES itself
    instead (absolute-momentum fallback).
  - MIDCAPBEES (Nippon renamed the underlying fund MID150BEES) only has
    clean daily data from 2019-06-20 onward -- it is excluded from the
    candidate set at any rebalance date where it doesn't yet have
    LOOKBACK_DAYS of its own trading history. Before that the candidate
    set is JUNIORBEES + GOLDBEES only.

Plain pandas implementation, decision and execution both use the
rebalance date's own closing price (same simplification the repo's
stock-momentum engine uses) -- no intraday lookahead-safe execution,
no slippage, no transaction costs.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ETF_DIR = ROOT / "data" / "etfs"

CANDIDATES = ["JUNIORBEES", "MIDCAPBEES", "GOLDBEES"]
REFERENCE = "NIFTYBEES"
LOOKBACK_DAYS = 55


def clean_bad_ticks(price: pd.Series, drop_threshold: float = 0.85, recover_band: float = 0.03) -> pd.Series:
    """Repair isolated data-vendor glitches: a run of one or more days
    where the price craters (ratio to the last known-good print falls
    below `drop_threshold`) and then fully reverts to within
    `recover_band` of the pre-crash level on the next print. That
    round-trip pattern is a vendor tick error (e.g. a stray /100 scaling),
    not a real market move -- a genuine crash doesn't fully recover on
    the very next print. Repaired via linear interpolation."""
    s = price.copy()
    good_val = s.iloc[0]
    bad_idx: list[int] = []
    for i in range(1, len(s)):
        ratio = s.iloc[i] / good_val
        if ratio < drop_threshold:
            bad_idx.append(i)
            continue
        if bad_idx and abs(ratio - 1) < recover_band:
            for j in bad_idx:
                s.iloc[j] = np.nan
        bad_idx = []
        good_val = s.iloc[i]
    return s.interpolate(method="linear")


def load_etf_prices(price_col: str = "Close") -> dict[str, pd.Series]:
    """Load each ETF's price series, keyed by symbol. MIDCAPBEES's stray
    first row (2019-02-04, isolated 136 days before continuous data starts
    2019-06-20) is dropped. Known Yahoo Finance vendor glitches (e.g. the
    2019-12-19/20 tick errors on GOLDBEES/NIFTYBEES/BANKBEES) are repaired
    via clean_bad_ticks."""
    out = {}
    for sym in CANDIDATES + [REFERENCE]:
        f = ETF_DIR / f"{sym}.csv"
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        s = df[price_col].dropna().sort_index()
        if sym == "MIDCAPBEES":
            s = s[s.index >= "2019-06-20"]
        s = clean_bad_ticks(s)
        out[sym] = s
    return out


def quarterly_rebalance_dates(calendar: pd.DatetimeIndex, min_start: pd.Timestamp) -> pd.DatetimeIndex:
    """First trading day of each calendar quarter present in `calendar`,
    from the first quarter fully at/after `min_start`."""
    cal = calendar[calendar >= min_start]
    quarters = cal.to_series().groupby(cal.to_period("Q")).first()
    return pd.DatetimeIndex(quarters.values)


def trailing_return(price: pd.Series, date: pd.Timestamp, lookback_days: int) -> float | None:
    """Return over the trailing `lookback_days` trading days of `price`'s
    own history, ending at `date`. None if not enough history yet."""
    if date not in price.index:
        return None
    loc = price.index.get_loc(date)
    if loc < lookback_days:
        return None
    return price.iloc[loc] / price.iloc[loc - lookback_days] - 1.0


def run_backtest(prices: dict[str, pd.Series], lookback_days: int = LOOKBACK_DAYS):
    """Returns (daily portfolio returns, holdings_history list of dicts).

    On a rebalance date, the day's own return is still earned by whatever
    was held coming into that date (decided at the *previous* rebalance);
    the newly-chosen asset starts earning returns from the next trading
    day onward. This mirrors the no-lookahead convention used by the
    repo's stock-momentum engine (backtest_engine.py)."""
    nifty = prices[REFERENCE]
    calendar = nifty.index
    min_start = calendar[lookback_days]  # need lookback history for Junior+Gold+Nifty
    rebal_dates = set(quarterly_rebalance_dates(calendar, min_start))

    all_syms = CANDIDATES + [REFERENCE]
    price_df = pd.DataFrame({s: prices[s].reindex(calendar).ffill() for s in all_syms})
    rets_df = price_df.pct_change()

    holdings_history = []
    daily_rets = pd.Series(index=calendar, dtype=float)
    current_asset = None

    for date in calendar:
        if current_asset is not None:
            daily_rets.loc[date] = rets_df.loc[date, current_asset]

        if date not in rebal_dates:
            continue

        nifty_ret = trailing_return(nifty, date, lookback_days)
        scores = {}
        for cand in CANDIDATES:
            cand_ret = trailing_return(prices[cand], date, lookback_days)
            if cand_ret is None or nifty_ret is None:
                continue
            scores[cand] = cand_ret - nifty_ret

        if scores:
            best_cand = max(scores, key=scores.get)
            best_score = scores[best_cand]
        else:
            best_cand, best_score = None, None

        chosen = best_cand if (best_score is not None and best_score > 0) else REFERENCE

        holdings_history.append({
            "rebalance_date": date,
            "chosen": chosen,
            **{f"rs_{c}": scores.get(c) for c in CANDIDATES},
        })
        current_asset = chosen

    return daily_rets.dropna(), pd.DataFrame(holdings_history)


def perf_stats(rets: pd.Series, freq: int = 252) -> dict:
    cum = (1 + rets).cumprod()
    n_years = len(rets) / freq
    cagr = cum.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 else np.nan
    vol = rets.std() * np.sqrt(freq)
    sharpe = (rets.mean() * freq) / vol if vol > 0 else np.nan
    running_max = cum.cummax()
    drawdown = cum / running_max - 1
    max_dd = drawdown.min()
    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "MaxDrawdown": max_dd,
        "TotalReturn": cum.iloc[-1] - 1,
    }


def annual_returns(rets: pd.Series) -> pd.Series:
    return (1 + rets).groupby(rets.index.year).apply(lambda x: x.prod() - 1)


def annual_max_drawdown(rets: pd.Series) -> pd.Series:
    cum = (1 + rets).cumprod()
    dd = cum / cum.cummax() - 1
    return dd.groupby(dd.index.year).min()


def yearly_table(strat_rets: pd.Series, bench_rets: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "strategy_return": annual_returns(strat_rets),
        "benchmark_return": annual_returns(bench_rets),
        "strategy_max_drawdown": annual_max_drawdown(strat_rets),
        "benchmark_max_drawdown": annual_max_drawdown(bench_rets),
    })


def run_full_backtest(price_col: str = "Close", lookback_days: int = LOOKBACK_DAYS):
    prices = load_etf_prices(price_col)
    strat_rets, holdings_history = run_backtest(prices, lookback_days)

    bench_px = prices[REFERENCE]
    bench_rets = bench_px.pct_change().reindex(strat_rets.index).dropna()
    strat_rets = strat_rets.reindex(bench_rets.index)

    return {
        "strategy_returns": strat_rets,
        "benchmark_returns": bench_rets,
        "strategy_stats": perf_stats(strat_rets),
        "benchmark_stats": perf_stats(bench_rets),
        "holdings_history": holdings_history,
        "prices": prices,
    }
