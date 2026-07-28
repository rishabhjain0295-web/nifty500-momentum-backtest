"""
Shared cross-sectional momentum backtest engine, used by both
scripts/momentum_backtest.py (CLI) and app.py (Streamlit portal).

Strategy (classic Jegadeesh-Titman style N-M momentum):
  - At each month end, rank stocks by their return over the trailing
    lookback_months, skipping the most recent skip_months (to avoid
    short-term reversal).
  - Go long the top n_stocks by that ranking, equal-weighted.
  - Hold for hold_months, then rebalance.

Point-in-time universe: when membership is passed to run_backtest, a stock
is only eligible to be ranked/held at a given rebalance date if the
membership calendar (data/nifty500_membership_calendar.csv, built by
scripts/build_membership_calendar.py) says it was actually a Nifty 500
constituent on that date. See that script's docstring for how the calendar
was built and its known limitations (best-effort post-2020, ~25% of
historical names unresolved to a live ticker).

Plain pandas implementation -- no lookahead-safe execution-cost modeling,
no slippage. Ignores dividends unless price_col is 'Adj Close'.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
STOCKS_DIR = ROOT / "data" / "stocks"
INDEX_DIR = ROOT / "data" / "index"
ETF_DIR = ROOT / "data" / "etfs"
MEMBERSHIP_CSV = ROOT / "data" / "nifty500_membership_calendar.csv"

# data/stocks/ is too large (300MB+) to commit to git -- it's fetched from a
# GitHub Release asset on first run instead. Set via the DATA_ARCHIVE_URL
# environment variable (or Streamlit secrets) in deployment; falls back to
# this constant, which must be updated after the release is created.
DATA_ARCHIVE_URL = "https://github.com/rishabhjain0295-web/nifty500-momentum-backtest/releases/download/data-v1/stocks.zip"


def ensure_stock_data(archive_url: str | None = None) -> None:
    """Download and extract data/stocks/ from a GitHub Release asset if it's
    not already present -- needed on a fresh cloud container where data/
    isn't in git. No-op if data/stocks/ already has files (e.g. local dev)."""
    import os
    import zipfile
    from io import BytesIO

    import requests

    if STOCKS_DIR.exists() and any(STOCKS_DIR.glob("*.csv")):
        return

    url = archive_url or os.environ.get("DATA_ARCHIVE_URL") or DATA_ARCHIVE_URL
    STOCKS_DIR.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(BytesIO(resp.content)) as zf:
        zf.extractall(STOCKS_DIR)


def load_prices(price_col: str = "Adj Close") -> pd.DataFrame:
    """Load all stock CSVs into one wide DataFrame of monthly prices, symbol columns."""
    frames = {}
    for f in STOCKS_DIR.glob("*.csv"):
        sym = f.stem
        try:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
        except Exception:
            continue
        if price_col not in df.columns or df.empty:
            continue
        s = df[price_col].dropna()
        if s.empty:
            continue
        frames[sym] = s
    if not frames:
        raise RuntimeError(f"No usable price data found in {STOCKS_DIR}")
    wide = pd.DataFrame(frames).sort_index()
    return wide.resample("ME").last()


def load_benchmark(price_col: str = "Adj Close") -> pd.Series:
    f = INDEX_DIR / "NIFTY500.csv"
    df = pd.read_csv(f, index_col=0, parse_dates=True)
    col = price_col if price_col in df.columns else "Close"
    s = df[col].dropna().sort_index()
    return s.resample("ME").last()


def load_gold_series(price_col: str = "Adj Close") -> pd.Series:
    """GOLDBEES (Nippon India ETF Gold BeES) as a proxy for domestic INR gold
    prices -- an actual investable instrument, unlike a USD gold index."""
    f = ETF_DIR / "GOLDBEES.csv"
    df = pd.read_csv(f, index_col=0, parse_dates=True)
    col = price_col if price_col in df.columns else "Close"
    s = df[col].dropna().sort_index()
    return s.resample("ME").last()


def load_membership_matrix(dates: pd.DatetimeIndex, symbols: pd.Index) -> pd.DataFrame:
    """Boolean (date x symbol) matrix: was `symbol` a Nifty 500 constituent as of `date`."""
    cal = pd.read_csv(MEMBERSHIP_CSV, parse_dates=["start", "end"])
    mat = pd.DataFrame(False, index=dates, columns=symbols)
    for sym, grp in cal.groupby("symbol"):
        if sym not in mat.columns:
            continue
        member = pd.Series(False, index=dates)
        for _, row in grp.iterrows():
            if pd.isna(row["end"]):
                member |= dates >= row["start"]
            else:
                member |= (dates >= row["start"]) & (dates <= row["end"])
        mat[sym] = member
    return mat


def run_backtest(
    monthly_prices: pd.DataFrame,
    membership: pd.DataFrame | None,
    lookback_months: int = 12,
    skip_months: int = 1,
    hold_months: int = 1,
    n_stocks: int = 30,
    min_price: float = 10.0,
    use_exit_band: bool = False,
    exit_band_pct: float = 0.0,
    use_regime_filter: bool = False,
    nifty500_index: pd.Series | None = None,
    gold_series: pd.Series | None = None,
    gold_entry_lookback: int = 150,
    gold_exit_lookback: int = 55,
    weighting_mode: str = "equal_monthly",
) -> tuple[pd.Series, list[tuple[pd.Timestamp, list[str]]]]:
    """Returns (monthly portfolio returns, [(rebalance_date, holdings), ...]).

    Entries are always drawn from the top n_stocks by momentum rank. Exits
    normally happen as soon as a held stock's rank drops out of the top
    n_stocks too -- but with use_exit_band=True, a held stock is only exited
    once its rank falls below n_stocks * (1 + exit_band_pct/100). This is a
    standard turnover-reduction technique: widening the exit threshold lets
    a stock hovering near the cutoff stay put instead of round-tripping in
    and out on every rebalance. exit_band_pct=0 (or use_exit_band=False)
    reproduces the plain top-N-in/top-N-out behavior exactly.

    weighting_mode controls how the portfolio is weighted between rebalances:
      - "equal_monthly" (default): weights reset to equal every month, even
        between rebalances. A stock's return that month never lets it grow
        (or shrink) as a share of the portfolio for next month -- every
        month is a fresh equal-weighted average. This is a common academic-
        backtest simplification (implicit frictionless monthly rebalancing)
        but isn't literally "buy and hold" between rebalances.
      - "drift": equal-weighted only at the moment of each rebalance (shares
        are set so each holding starts at 1/n_stocks of portfolio value);
        between rebalances, weights drift naturally with each stock's own
        performance -- winners become a larger share of the portfolio,
        laggards a smaller one, until the next rebalance resets everyone
        (including survivors) back to equal weight. This is how real
        equal-weight index funds/ETFs actually rebalance.
    Note: tax_cost_engine.py's per-trade gain sizing already assumes each
    position is 1/n_stocks of NAV at entry and drifts with that stock's own
    return until exit -- i.e. it implicitly matches "drift" regardless of
    which weighting_mode produced the return series. This is an existing
    approximation, not new to this parameter.

    Regime filter (use_regime_filter=True, requires nifty500_index and
    gold_series): at each rebalance, while in the "momentum" regime, compare
    the Nifty 500 index's trailing gold_entry_lookback-month return to
    gold's. If gold's is higher, switch to a "gold" regime -- the entire
    portfolio becomes gold (a synthetic "GOLD" position; no individual
    stocks held) until, at a later rebalance, the Nifty 500's trailing
    gold_exit_lookback-month return exceeds gold's, switching back to
    momentum. A synthetic "GOLD" entry in holdings_history isn't a real
    monthly_prices symbol, so cost/tax simulation (tax_cost_engine.py)
    silently skips it -- only the equity leg's costs/taxes are modeled.
    """
    min_history_months = lookback_months + skip_months + 1
    monthly_rets = monthly_prices.pct_change()
    dates = monthly_prices.index

    gold_rets = nifty_trail_entry = gold_trail_entry = nifty_trail_exit = gold_trail_exit = None
    if use_regime_filter and nifty500_index is not None and gold_series is not None:
        nifty_aligned = nifty500_index.reindex(dates)
        gold_aligned = gold_series.reindex(dates)
        gold_rets = gold_aligned.pct_change()
        nifty_trail_entry = nifty_aligned / nifty_aligned.shift(gold_entry_lookback) - 1
        gold_trail_entry = gold_aligned / gold_aligned.shift(gold_entry_lookback) - 1
        nifty_trail_exit = nifty_aligned / nifty_aligned.shift(gold_exit_lookback) - 1
        gold_trail_exit = gold_aligned / gold_aligned.shift(gold_exit_lookback) - 1

    portfolio_rets = pd.Series(index=dates, dtype=float)
    holdings_history: list[tuple[pd.Timestamp, list[str]]] = []
    current_holdings: list[str] = []
    regime = "momentum"
    months_held = 0
    shares: dict[str, float] = {}   # only used in "drift" mode
    nav = 1.0                       # running portfolio value, drives drift-mode reallocation sizing

    for i in range(min_history_months, len(dates)):
        today = dates[i]
        holding_something = bool(current_holdings) or regime == "gold"

        if regime == "gold":
            r = gold_rets.loc[today] if gold_rets is not None else np.nan
            r = r if pd.notna(r) else 0.0
        elif current_holdings:
            if weighting_mode == "drift" and shares:
                px_today = monthly_prices.loc[today, current_holdings]
                value_today = sum(
                    shares[s] * px_today[s] for s in current_holdings if pd.notna(px_today.get(s))
                )
                r = value_today / nav - 1 if nav > 0 else 0.0
            else:
                r = monthly_rets.loc[today, current_holdings].mean()
                if pd.isna(r):
                    r = 0.0
        else:
            r = 0.0

        portfolio_rets.loc[today] = r
        nav *= (1 + r)

        months_held += 1
        if months_held < hold_months and holding_something:
            continue

        if nifty_trail_entry is not None:
            if regime == "momentum":
                n150, g150 = nifty_trail_entry.loc[today], gold_trail_entry.loc[today]
                if pd.notna(n150) and pd.notna(g150) and g150 > n150:
                    regime = "gold"
            elif regime == "gold":
                n55, g55 = nifty_trail_exit.loc[today], gold_trail_exit.loc[today]
                if pd.notna(n55) and pd.notna(g55) and n55 > g55:
                    regime = "momentum"

        if regime == "gold":
            current_holdings = []
            shares = {}
            holdings_history.append((today, ["GOLD"]))
            months_held = 0
            continue

        end_idx = i - skip_months
        start_idx = end_idx - lookback_months
        if start_idx < 0:
            continue

        px_start = monthly_prices.iloc[start_idx]
        px_end = monthly_prices.iloc[end_idx]
        last_price = monthly_prices.iloc[i]

        eligible = (px_start > 0) & (px_end > 0) & (last_price >= min_price)
        if membership is not None:
            eligible &= membership.loc[today]
        mom = (px_end / px_start - 1.0)[eligible].dropna()

        if len(mom) < n_stocks:
            continue

        ranked = mom.sort_values(ascending=False)
        if use_exit_band and exit_band_pct > 0:
            exit_threshold_rank = n_stocks * (1 + exit_band_pct / 100.0)
            rank_of = {sym: pos + 1 for pos, sym in enumerate(ranked.index)}
            survivors = [s for s in current_holdings if rank_of.get(s, float("inf")) <= exit_threshold_rank]
            needed = n_stocks - len(survivors)
            if needed > 0:
                survivors_set = set(survivors)
                new_entrants = [s for s in ranked.index if s not in survivors_set][:needed]
            else:
                new_entrants = []
            current_holdings = survivors + new_entrants
        else:
            current_holdings = ranked.head(n_stocks).index.tolist()

        holdings_history.append((today, current_holdings))
        months_held = 0

        if weighting_mode == "drift" and current_holdings:
            px_now = monthly_prices.loc[today, current_holdings]
            per_stock_value = nav / len(current_holdings)
            shares = {
                s: (per_stock_value / px_now[s]) if pd.notna(px_now.get(s)) and px_now[s] > 0 else 0.0
                for s in current_holdings
            }

    return portfolio_rets.dropna(), holdings_history


def perf_stats(rets: pd.Series, freq: int = 12) -> dict:
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
    """Compounded return for each calendar year present in `rets`."""
    return (1 + rets).groupby(rets.index.year).apply(lambda x: x.prod() - 1)


def annual_max_drawdown(rets: pd.Series) -> pd.Series:
    """Worst peak-to-trough drawdown observed *during* each calendar year,
    where the peak may have been set in an earlier year (running cumulative
    max from inception) -- this is the drawdown an investor actually lived
    through in that year, not a within-year-only reset."""
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


def run_full_backtest(
    price_col: str = "Adj Close",
    lookback_months: int = 12,
    skip_months: int = 1,
    hold_months: int = 1,
    n_stocks: int = 30,
    min_price: float = 10.0,
    use_membership_filter: bool = True,
    use_exit_band: bool = False,
    exit_band_pct: float = 0.0,
    use_regime_filter: bool = False,
    gold_entry_lookback: int = 150,
    gold_exit_lookback: int = 55,
    weighting_mode: str = "equal_monthly",
):
    """End-to-end: load data, run strategy, align to benchmark. Returns a dict."""
    monthly_prices = load_prices(price_col)
    membership = None
    if use_membership_filter:
        membership = load_membership_matrix(monthly_prices.index, monthly_prices.columns)

    bench_px = load_benchmark(price_col)
    gold_px = load_gold_series(price_col) if use_regime_filter else None

    strat_rets, holdings_history = run_backtest(
        monthly_prices, membership, lookback_months, skip_months, hold_months, n_stocks, min_price,
        use_exit_band, exit_band_pct,
        use_regime_filter, bench_px, gold_px, gold_entry_lookback, gold_exit_lookback,
        weighting_mode,
    )

    bench_rets = bench_px.pct_change().reindex(strat_rets.index).dropna()
    strat_rets = strat_rets.reindex(bench_rets.index)

    return {
        "strategy_returns": strat_rets,
        "benchmark_returns": bench_rets,
        "strategy_stats": perf_stats(strat_rets),
        "benchmark_stats": perf_stats(bench_rets),
        "holdings_history": holdings_history,
        "monthly_prices": monthly_prices,
        "membership": membership,
    }
