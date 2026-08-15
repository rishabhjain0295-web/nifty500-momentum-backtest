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


def load_daily_prices(price_col: str = "Adj Close") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads daily close (price_col) and daily open prices, WITHOUT resampling
    to monthly. Only used by the stoploss/re-entry overlay (apply_stoploss),
    which needs day-by-day granularity that the rest of the engine discards
    by working in monthly_prices. Note Open is raw (not split/dividend
    adjusted the way Adj Close is) -- a minor inconsistency around corporate
    actions when price_col='Adj Close', not corrected for here."""
    close_frames = {}
    open_frames = {}
    for f in STOCKS_DIR.glob("*.csv"):
        sym = f.stem
        try:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
        except Exception:
            continue
        if price_col not in df.columns or "Open" not in df.columns or df.empty:
            continue
        c = df[price_col].dropna()
        o = df["Open"].dropna()
        if c.empty or o.empty:
            continue
        close_frames[sym] = c
        open_frames[sym] = o
    if not close_frames:
        raise RuntimeError(f"No usable daily price data found in {STOCKS_DIR}")
    daily_close = pd.DataFrame(close_frames).sort_index()
    daily_open = pd.DataFrame(open_frames).sort_index()
    return daily_close, daily_open


def load_daily_ohlc() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads raw (unadjusted) daily Open/High/Low/Close for every stock --
    used by swing_engine.py, which trades on actual price action (gaps,
    N-day highs/lows) rather than total-return series. Deliberately uses
    'Close', not 'Adj Close': gaps and Donchian channels are about what a
    trader actually saw on the chart, and mixing an adjusted close with raw
    High/Low would misalign around dividend/split dates."""
    open_frames, high_frames, low_frames, close_frames = {}, {}, {}, {}
    for f in STOCKS_DIR.glob("*.csv"):
        sym = f.stem
        try:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
        except Exception:
            continue
        if not {"Open", "High", "Low", "Close"}.issubset(df.columns) or df.empty:
            continue
        o, h, l, c = df["Open"].dropna(), df["High"].dropna(), df["Low"].dropna(), df["Close"].dropna()
        if o.empty or h.empty or l.empty or c.empty:
            continue
        open_frames[sym] = o
        high_frames[sym] = h
        low_frames[sym] = l
        close_frames[sym] = c
    if not close_frames:
        raise RuntimeError(f"No usable daily OHLC data found in {STOCKS_DIR}")
    daily_open = pd.DataFrame(open_frames).sort_index()
    daily_high = pd.DataFrame(high_frames).sort_index()
    daily_low = pd.DataFrame(low_frames).sort_index()
    daily_close = pd.DataFrame(close_frames).sort_index()
    return daily_open, daily_high, daily_low, daily_close


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


def load_liquid_series(price_col: str = "Adj Close") -> pd.Series:
    """LIQUIDBEES (Nippon India ETF Liquid BeES) as a proxy for a liquid/
    money-market fund -- used by the SIP simulator (sip_engine.py) as the
    defensive sleeve for the dynamic allocation mode."""
    f = ETF_DIR / "LIQUIDBEES.csv"
    df = pd.read_csv(f, index_col=0, parse_dates=True)
    col = price_col if price_col in df.columns else "Close"
    s = df[col].dropna().sort_index()
    return s.resample("ME").last()


def load_current_universe() -> pd.DataFrame:
    """The CURRENT Nifty 500 constituent list (Company Name, Symbol, ...),
    from data/nifty500_list.csv (see scripts/get_nifty500_list.py). Used by
    the live stock ranker (pages/1_Stock_Ranker.py) to restrict rankings to
    today's actual investable universe -- data/stocks/ has ~970 symbols
    (970 = 500 current + historical/delisted names kept for backtesting;
    see build_membership_calendar.py), most of which aren't current
    constituents and shouldn't show up in a live ranking."""
    f = ROOT / "data" / "nifty500_list.csv"
    return pd.read_csv(f)


def compute_momentum_ranking(
    monthly_prices: pd.DataFrame,
    membership: pd.DataFrame | None,
    as_of_date: pd.Timestamp,
    lookback_months: int,
    skip_months: int,
    min_price: float,
) -> pd.Series | None:
    """Trailing lookback_months return (skipping the most recent skip_months)
    for every stock eligible as of as_of_date, sorted descending (first
    entry = rank 1 = highest momentum). Returns None if as_of_date isn't in
    monthly_prices or doesn't have enough trailing history.

    Eligibility: valid (>0) price at both the lookback start and end, last
    price >= min_price, and -- if membership is given -- an actual Nifty
    500 constituent as of as_of_date per the point-in-time calendar. This
    is the single source of truth for the ranking formula, shared by
    run_backtest (historical simulation) and the live stock ranker
    (pages/1_Stock_Ranker.py, current snapshot).
    """
    dates = monthly_prices.index
    if as_of_date not in dates:
        return None
    i = dates.get_loc(as_of_date)
    end_idx = i - skip_months
    start_idx = end_idx - lookback_months
    if start_idx < 0:
        return None

    px_start = monthly_prices.iloc[start_idx]
    px_end = monthly_prices.iloc[end_idx]
    last_price = monthly_prices.iloc[i]

    eligible = (px_start > 0) & (px_end > 0) & (last_price >= min_price)
    if membership is not None:
        eligible &= membership.loc[as_of_date]
    mom = (px_end / px_start - 1.0)[eligible].dropna()
    return mom.sort_values(ascending=False)


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

        ranked = compute_momentum_ranking(monthly_prices, membership, today, lookback_months, skip_months, min_price)
        if ranked is None or len(ranked) < n_stocks:
            continue

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


def apply_stoploss(
    monthly_prices: pd.DataFrame,
    daily_close: pd.DataFrame,
    daily_open: pd.DataFrame,
    holdings_history: list[tuple[pd.Timestamp, list[str]]],
    stoploss_pct: float,
    max_reentries: int,
) -> dict[pd.Timestamp, float]:
    """Re-simulates portfolio returns at daily granularity for every
    momentum-regime holding period in holdings_history (GOLD-regime periods
    are left untouched by the caller -- this only returns values for months
    it actually recomputed), applying a per-stock stoploss with optional
    re-entry:

      - Each stock slot is bought at the rebalance date's close (taken from
        monthly_prices, the same value the momentum ranking already used --
        not re-derived from daily_close, which can have a different last
        trading day than the "ME" period-end label).
      - Each subsequent trading day, if held and that day's close is <=
        entry_price * (1 - stoploss_pct/100), the position is stopped out
        at that close; the slot then holds cash (0 return) for following
        days.
      - While in cash, if re-entries remain (reentry_count < max_reentries)
        and a day's close rises back above the ORIGINAL entry price for
        this holding period (a fixed reference -- not the post-stop price,
        and not reset by earlier re-entries), the slot re-enters at the
        *next* trading day's open. A fresh stoploss is set from this new
        entry price (each re-entry gets its own stop, anchored to what was
        actually paid).
      - Once re-entries are exhausted, the slot stays in cash for the rest
        of the holding period.

    Every held stock is weighted 1/n_stocks (whatever "n_stocks" the period
    actually held, matching exit-band survivor counts if applicable) --
    this always uses per-stock discrete entry/exit tracking, i.e. "drift"-
    style position sizing, regardless of the separately-selectable
    weighting_mode: a stoploss fundamentally requires knowing what price a
    stock was actually bought at, which "equal_monthly" (reset to equal
    weight every month) doesn't preserve.

    Returns {month_end_date: return} for the recomputed months only. The
    caller merges this into the baseline portfolio_rets from run_backtest,
    leaving GOLD-regime and pre-first-rebalance months unchanged.
    """
    stop_frac = stoploss_pct / 100.0
    monthly_result: dict[pd.Timestamp, float] = {}
    last_daily_date = daily_close.index.max()

    for idx in range(len(holdings_history)):
        start_date, holdings = holdings_history[idx]
        if holdings == ["GOLD"]:
            continue
        end_date = holdings_history[idx + 1][0] if idx + 1 < len(holdings_history) else last_daily_date

        symbols = [s for s in holdings if s in monthly_prices.columns and s in daily_close.columns]
        n = len(symbols)
        if n == 0:
            continue

        period_dates = daily_close.index[(daily_close.index > start_date) & (daily_close.index <= end_date)]
        if len(period_dates) == 0:
            continue
        daily_port_ret = pd.Series(0.0, index=period_dates)

        for sym in symbols:
            entry_price = monthly_prices.loc[start_date, sym]
            if pd.isna(entry_price) or entry_price <= 0:
                continue
            initial_entry_price = entry_price

            close = daily_close[sym].reindex(period_dates)
            open_ = daily_open[sym].reindex(period_dates)

            in_stock = True
            reentry_count = 0
            pending_reentry = False
            prev_price = entry_price

            for d_i, d in enumerate(period_dates):
                px_close = close.iloc[d_i]
                if pd.isna(px_close):
                    continue

                if pending_reentry:
                    px_open = open_.iloc[d_i]
                    pending_reentry = False
                    if pd.notna(px_open) and px_open > 0:
                        entry_price = px_open
                        in_stock = True
                        daily_port_ret.loc[d] += (px_close / px_open - 1) / n
                        prev_price = px_close
                    continue

                if in_stock:
                    if pd.notna(prev_price) and prev_price > 0:
                        daily_port_ret.loc[d] += (px_close / prev_price - 1) / n
                    if px_close <= entry_price * (1 - stop_frac):
                        in_stock = False
                    prev_price = px_close
                else:
                    if reentry_count < max_reentries and px_close > initial_entry_price:
                        pending_reentry = True
                        reentry_count += 1
                    prev_price = px_close

        monthly_from_daily = (1 + daily_port_ret).resample("ME").prod() - 1
        for m_date, r in monthly_from_daily.items():
            monthly_result[m_date] = r

    return monthly_result


def apply_execution_lag(
    monthly_prices: pd.DataFrame,
    daily_close: pd.DataFrame,
    daily_open: pd.DataFrame,
    holdings_history: list[tuple[pd.Timestamp, list[str]]],
) -> dict[pd.Timestamp, float]:
    """Re-simulates portfolio returns reflecting T+1-open execution: the
    rebalance SIGNAL is still generated from the month-end close (unchanged
    ranking/selection in run_backtest -- this does not affect which stocks
    get picked), but ENTRIES and EXITS are executed on the next trading day,
    not at the month-end close itself.

      - Stocks continuing to be held across a rebalance (survivors) are
        unaffected -- no execution needed, they just keep compounding
        exactly as the baseline calculation already has them.
      - Stocks being DROPPED continue to be held (and accrue return) through
        the next trading day's OPEN, at which point they're sold. This adds
        a small return sliver (month-end close -> next-day open) that the
        baseline calculation misses entirely for these stocks.
      - Stocks being ADDED are bought at the next trading day's OPEN, so
        they only start accruing return from that point -- the baseline
        calculation overstates them by including the month-end-close-to-
        next-day-open gap they didn't actually experience.

    Only the period FROM each rebalance date TO the next is touched (the
    prior period, which ends at that rebalance date, is unaffected -- the
    old holdings genuinely were held through that close in both models).

    Returns {month_end_date: return} for every month following a rebalance
    where the holdings list actually changed. Months with no change, GOLD-
    regime months, and the very first holding period (nothing to compare
    against, since there's no "previous" holdings list yet) are left
    untouched by the caller.
    """
    monthly_result: dict[pd.Timestamp, float] = {}
    last_daily_date = daily_close.index.max()

    prev_holdings: set[str] = set()
    for idx in range(len(holdings_history)):
        start_date, holdings = holdings_history[idx]
        if holdings == ["GOLD"]:
            prev_holdings = set()
            continue
        curr_holdings = set(holdings)
        end_date = holdings_history[idx + 1][0] if idx + 1 < len(holdings_history) else last_daily_date

        added = curr_holdings - prev_holdings
        dropped = prev_holdings - curr_holdings
        survivors = curr_holdings & prev_holdings
        n = len(curr_holdings)

        if n == 0 or (not added and not dropped):
            prev_holdings = curr_holdings
            continue

        period_dates = daily_close.index[(daily_close.index > start_date) & (daily_close.index <= end_date)]
        if len(period_dates) == 0:
            prev_holdings = curr_holdings
            continue

        daily_port_ret = pd.Series(0.0, index=period_dates)
        exec_date = period_dates[0]

        for sym in dropped:
            if sym not in monthly_prices.columns or sym not in daily_open.columns:
                continue
            last_price = monthly_prices.loc[start_date, sym]
            exec_open = daily_open.loc[exec_date, sym]
            if pd.isna(last_price) or last_price <= 0 or pd.isna(exec_open) or exec_open <= 0:
                continue
            daily_port_ret.loc[exec_date] += (exec_open / last_price - 1) / n

        for sym in added:
            if sym not in daily_open.columns or sym not in daily_close.columns:
                continue
            exec_open = daily_open.loc[exec_date, sym]
            exec_close = daily_close.loc[exec_date, sym]
            if pd.isna(exec_open) or exec_open <= 0 or pd.isna(exec_close):
                continue
            daily_port_ret.loc[exec_date] += (exec_close / exec_open - 1) / n
            prev_price = exec_close
            for d in period_dates[1:]:
                px = daily_close.loc[d, sym]
                if pd.isna(px):
                    continue
                if pd.notna(prev_price) and prev_price > 0:
                    daily_port_ret.loc[d] += (px / prev_price - 1) / n
                prev_price = px

        for sym in survivors:
            if sym not in monthly_prices.columns or sym not in daily_close.columns:
                continue
            prev_price = monthly_prices.loc[start_date, sym]
            if pd.isna(prev_price) or prev_price <= 0:
                continue
            for d in period_dates:
                px = daily_close.loc[d, sym]
                if pd.isna(px):
                    continue
                if pd.notna(prev_price) and prev_price > 0:
                    daily_port_ret.loc[d] += (px / prev_price - 1) / n
                prev_price = px

        monthly_from_daily = (1 + daily_port_ret).resample("ME").prod() - 1
        for m_date, r in monthly_from_daily.items():
            monthly_result[m_date] = r

        prev_holdings = curr_holdings

    return monthly_result


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


def build_trade_log(
    monthly_prices: pd.DataFrame,
    strat_rets: pd.Series,
    holdings_history: list[tuple[pd.Timestamp, list[str]]],
    n_stocks: int,
    capital_base_rs: float = 1_000_000.0,
) -> pd.DataFrame:
    """Reconstructs discrete buy/sell trades (entry/exit date, price, qty,
    P&L) from the rebalance holdings history.

    Every position is sized as if bought outright at 1/n_stocks of portfolio
    NAV on the date it enters the basket, held at that fixed share count
    until it exits, then sold -- this is the natural "quantity" concept for
    a trade blotter, so it's used here regardless of which weighting_mode
    (see run_backtest) generated the return curve: even under "equal_monthly"
    weighting, a real trader can't literally re-buy/re-sell fractional
    amounts every month for free, so a trade log has to describe discrete
    orders. Position sizing uses the GROSS (pre-cost, pre-tax) equity curve,
    independent of tax_cost_engine.py -- this is a plain summary of what the
    strategy did, not a costed simulation.

    Includes still-open positions at the end of the backtest as unrealized
    (status="open"), marked to the last available price.

    The "GOLD" synthetic regime marker (see run_backtest) has no entry/exit
    price in monthly_prices and is excluded from this log.
    """
    nav = (1 + strat_rets).cumprod()
    open_positions: dict[str, tuple] = {}  # symbol -> (entry_date, entry_price, entry_nav, entry_n_active)
    trades = []
    prev_holdings: set[str] = set()

    for date, holdings in holdings_history:
        new_holdings = set(holdings) - {"GOLD"}
        equity_prev = prev_holdings - {"GOLD"}
        added = new_holdings - equity_prev
        dropped = equity_prev - new_holdings
        n_active = len(new_holdings) if new_holdings else n_stocks
        nav_at_date = nav.loc[date] if date in nav.index else None

        for sym in dropped:
            entry_date, entry_price, entry_nav, entry_n_active = open_positions.pop(sym)
            exit_price = monthly_prices.loc[date, sym] if sym in monthly_prices.columns else None
            if exit_price is None or pd.isna(exit_price):
                continue
            position_value = (entry_nav / entry_n_active) * capital_base_rs
            qty = position_value / entry_price
            trades.append({
                "symbol": sym, "entry_date": entry_date, "entry_price": entry_price,
                "exit_date": date, "exit_price": exit_price, "qty": qty,
                "pnl_rs": qty * (exit_price - entry_price), "pnl_pct": exit_price / entry_price - 1,
                "hold_days": (date - entry_date).days, "status": "closed",
            })

        if nav_at_date is not None:
            for sym in added:
                entry_price = monthly_prices.loc[date, sym] if sym in monthly_prices.columns else None
                if entry_price is None or pd.isna(entry_price):
                    continue
                open_positions[sym] = (date, entry_price, nav_at_date, n_active)

        prev_holdings = new_holdings

    last_date = monthly_prices.index[-1]
    for sym, (entry_date, entry_price, entry_nav, entry_n_active) in open_positions.items():
        px_series = monthly_prices[sym].dropna() if sym in monthly_prices.columns else pd.Series(dtype=float)
        if px_series.empty:
            continue
        last_price = px_series.iloc[-1]
        position_value = (entry_nav / entry_n_active) * capital_base_rs
        qty = position_value / entry_price
        trades.append({
            "symbol": sym, "entry_date": entry_date, "entry_price": entry_price,
            "exit_date": pd.NaT, "exit_price": last_price, "qty": qty,
            "pnl_rs": qty * (last_price - entry_price), "pnl_pct": last_price / entry_price - 1,
            "hold_days": (last_date - entry_date).days, "status": "open",
        })

    if not trades:
        return pd.DataFrame(columns=["symbol", "entry_date", "entry_price", "exit_date",
                                      "exit_price", "qty", "pnl_rs", "pnl_pct", "hold_days", "status"])
    return pd.DataFrame(trades).sort_values("entry_date").reset_index(drop=True)


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
    use_stoploss: bool = False,
    stoploss_pct: float = 10.0,
    max_reentries: int = 0,
    use_execution_lag: bool = False,
):
    """End-to-end: load data, run strategy, align to benchmark. Returns a dict.

    Note: use_stoploss and use_execution_lag don't currently compose -- each
    independently recomputes a changed month's return from scratch rather
    than layering on top of the other. If both are enabled, execution_lag
    is applied second and wins for any month both would have touched.
    """
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

    if use_stoploss or use_execution_lag:
        daily_close, daily_open = load_daily_prices(price_col)
        if use_stoploss:
            overlay = apply_stoploss(
                monthly_prices, daily_close, daily_open, holdings_history, stoploss_pct, max_reentries
            )
            for m_date, r in overlay.items():
                if m_date in strat_rets.index:
                    strat_rets.loc[m_date] = r
        if use_execution_lag:
            overlay = apply_execution_lag(monthly_prices, daily_close, daily_open, holdings_history)
            for m_date, r in overlay.items():
                if m_date in strat_rets.index:
                    strat_rets.loc[m_date] = r

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
