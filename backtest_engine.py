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
HOURLY_DIR = ROOT / "data" / "hourly"
FIFTEEN_MIN_DIR = ROOT / "data" / "15min"
THIRTY_MIN_DIR = ROOT / "data" / "30min"
MUTUAL_FUND_DIR = ROOT / "data" / "mutual_funds"
MEMBERSHIP_CSV = ROOT / "data" / "nifty500_membership_calendar.csv"

# data/stocks/ and data/hourly/ are too large to commit to git -- both are
# fetched from GitHub Release assets on first run instead. Set via the
# DATA_ARCHIVE_URL / HOURLY_ARCHIVE_URL environment variables (or Streamlit
# secrets) in deployment; falls back to these constants, which must be
# updated after the release is created.
DATA_ARCHIVE_URL = "https://github.com/rishabhjain0295-web/nifty500-momentum-backtest/releases/download/data-v1/stocks.zip"
HOURLY_ARCHIVE_URL = "https://github.com/rishabhjain0295-web/nifty500-momentum-backtest/releases/download/data-v1/hourly.zip"
FIFTEEN_MIN_ARCHIVE_URL = "https://github.com/rishabhjain0295-web/nifty500-momentum-backtest/releases/download/data-v1/15min.zip"
THIRTY_MIN_ARCHIVE_URL = "https://github.com/rishabhjain0295-web/nifty500-momentum-backtest/releases/download/data-v1/30min.zip"
MUTUAL_FUND_ARCHIVE_URL = "https://github.com/rishabhjain0295-web/nifty500-momentum-backtest/releases/download/data-v1/mutual_funds.zip"


def _ensure_data_from_archive(target_dir: Path, archive_url: str | None, env_var: str, fallback_url: str) -> None:
    """Shared download/extract logic behind ensure_stock_data and
    ensure_hourly_data -- fetches a GitHub Release zip asset into target_dir
    if it isn't already populated. No-op if target_dir already has files
    (e.g. local dev, where the data was downloaded directly)."""
    import os
    import zipfile
    from io import BytesIO

    import requests

    if target_dir.exists() and any(target_dir.glob("*.csv")):
        return

    url = archive_url or os.environ.get(env_var) or fallback_url
    target_dir.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(BytesIO(resp.content)) as zf:
        zf.extractall(target_dir)


def ensure_stock_data(archive_url: str | None = None) -> None:
    """Download and extract data/stocks/ from a GitHub Release asset if it's
    not already present -- needed on a fresh cloud container where data/
    isn't in git. No-op if data/stocks/ already has files (e.g. local dev)."""
    _ensure_data_from_archive(STOCKS_DIR, archive_url, "DATA_ARCHIVE_URL", DATA_ARCHIVE_URL)


def ensure_hourly_data(archive_url: str | None = None) -> None:
    """Download and extract data/hourly/ (used by the EMA-crossover swing
    strategy's hourly variant) from a GitHub Release asset if it's not
    already present. No-op if data/hourly/ already has files."""
    _ensure_data_from_archive(HOURLY_DIR, archive_url, "HOURLY_ARCHIVE_URL", HOURLY_ARCHIVE_URL)


def ensure_15min_data(archive_url: str | None = None) -> None:
    """Download and extract data/15min/ (used by the RSI Oversold Reversal
    swing strategy's 15-min variant) from a GitHub Release asset if it's
    not already present. No-op if data/15min/ already has files."""
    _ensure_data_from_archive(FIFTEEN_MIN_DIR, archive_url, "FIFTEEN_MIN_ARCHIVE_URL", FIFTEEN_MIN_ARCHIVE_URL)


def ensure_30min_data(archive_url: str | None = None) -> None:
    """Download and extract data/30min/ (used by the RSI Oversold Reversal
    swing strategy's 30-min variant) from a GitHub Release asset if it's
    not already present. Unlike every other ensure_*_data function, the
    SOURCE data (scripts/download_upstox_30min_data.py) needs a private
    Upstox API token that never leaves the machine that ran it -- this
    only ever fetches the resulting PRICE DATA the deployed app needs,
    same bootstrap-once model as ensure_hourly_data/ensure_15min_data."""
    _ensure_data_from_archive(THIRTY_MIN_DIR, archive_url, "THIRTY_MIN_ARCHIVE_URL", THIRTY_MIN_ARCHIVE_URL)


def ensure_mutual_fund_data(archive_url: str | None = None) -> None:
    """Download and extract data/mutual_funds/ (used by the Mutual Fund
    Comparison page) from a GitHub Release asset if it's not already
    present -- see scripts/download_mutual_funds.py for the source (a
    free, no-auth AMFI-backed API, unlike the Upstox-sourced intraday
    data), same bootstrap-once model as every other ensure_*_data."""
    _ensure_data_from_archive(MUTUAL_FUND_DIR, archive_url, "MUTUAL_FUND_ARCHIVE_URL", MUTUAL_FUND_ARCHIVE_URL)


def _consolidated_parquet_path(field: str) -> Path:
    slug = field.lower().replace(" ", "")
    return STOCKS_DIR / f"_consolidated_{slug}.parquet"


def _build_wide_daily_field(field: str) -> pd.DataFrame:
    """The expensive path: opens and parses every symbol's CSV
    individually to pull out one field ('Adj Close', 'Close', or
    'Open') and concats them into one wide (date x symbol) DataFrame.
    Only runs when no pre-built consolidated Parquet cache exists yet --
    see load_wide_daily_field's docstring for why that's the uncommon
    case."""
    frames = {}
    for f in STOCKS_DIR.glob("*.csv"):
        sym = f.stem
        try:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
        except Exception:
            continue
        if field not in df.columns or df.empty:
            continue
        s = df[field].dropna()
        if s.empty:
            continue
        frames[sym] = s
    if not frames:
        raise RuntimeError(f"No usable '{field}' price data found in {STOCKS_DIR}")
    return pd.DataFrame(frames).sort_index()


def _clean_bad_ticks(wide: pd.DataFrame) -> pd.DataFrame:
    """Detects and corrects "spike-and-revert" bad ticks: a single day
    where a symbol's price jumps by an extreme ratio then reverts almost
    exactly the next trading day. Found via real evidence, not
    speculation: 209 such incidents across 34 distinct dates (1998-2020)
    in this project's own data, up to 97 DIFFERENT symbols affected on a
    single day (2005-07-28), each by a different, unrelated ratio (2x,
    5x, 11x, ...) -- a vendor data-quality glitch for that date, not a
    real synchronized market move (a real event wouldn't hit unrelated
    stocks by wildly different, individually-reverting amounts). Only
    replaces the specific flagged (date, symbol) cells -- with the
    average of the immediately adjacent (trusted) days -- never touches
    genuine multi-day gaps (a stock not yet listed, delisted, etc.),
    since the detection requires both neighboring days to already be
    valid numbers."""
    ratio = wide / wide.shift(1)
    reverts = ratio * ratio.shift(-1)
    spike = ((ratio > 1.5) | (ratio < 2 / 3)) & (reverts > 0.9) & (reverts < 1.1)
    if not spike.to_numpy().any():
        return wide
    replacement = (wide.shift(1) + wide.shift(-1)) / 2.0
    return wide.where(~spike, replacement)


def load_wide_daily_field(field: str) -> pd.DataFrame:
    """Full daily (unresampled) wide price history for one OHLC-ish
    field ('Adj Close', 'Close', or 'Open'), symbol columns -- the
    shared, expensive-to-build input behind load_prices and
    load_daily_prices.

    Reads a pre-built consolidated Parquet cache
    (STOCKS_DIR/_consolidated_<field>.parquet) if one exists: one fast
    binary read instead of opening and parsing ~1000 individual CSVs on
    every fresh session -- that CSV-by-CSV parse used to be the
    dominant cost of a cold app start, and on the Backtest page it was
    paid out TWICE (once each for load_prices and load_daily_prices,
    which used to loop independently). scripts/refresh_weekly_data.py
    rebuilds this cache every week and ships it inside stocks.zip, so
    the deployed app doesn't have to build it itself.

    Falls back to parsing the CSVs directly -- and opportunistically
    writes the Parquet cache for next time -- if no cache exists yet,
    e.g. a fresh local clone before the weekly refresh script has ever
    run against it. Either way, the result has already been through
    _clean_bad_ticks (see its docstring) before being cached, so every
    caller benefits without needing to know about it."""
    parquet_path = _consolidated_parquet_path(field)
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except Exception:
            pass  # fall through and rebuild if the cached file is somehow unreadable
    wide = _clean_bad_ticks(_build_wide_daily_field(field))
    try:
        wide.to_parquet(parquet_path, compression="zstd")
    except Exception:
        pass  # best-effort cache -- a read-only filesystem shouldn't break loading
    return wide


def load_prices(price_col: str = "Adj Close", freq: str = "ME") -> pd.DataFrame:
    """Wide DataFrame of period-end prices, symbol columns -- see
    load_wide_daily_field for where the underlying daily data comes
    from. freq is any pandas resample rule -- "ME" (the default,
    calendar month-end) for the monthly rebalancing engine everywhere else
    in this app, or "W-FRI" (calendar week ending Friday) for the Backtest
    page's weekly rebalancing option. run_backtest and
    compute_momentum_ranking don't care which -- they treat the DataFrame's
    rows as generic rebalance-eligible periods, using integer row offsets
    for lookback/skip, not calendar-month arithmetic -- so lookback/skip/
    hold counts passed alongside a "W-FRI" frame are interpreted in WEEKS,
    not months."""
    return load_wide_daily_field(price_col).resample(freq).last()


def load_daily_prices(price_col: str = "Adj Close") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads daily close (price_col) and daily open prices, WITHOUT resampling
    to monthly. Only used by the stoploss/re-entry overlay (apply_stoploss)
    and T+1 execution lag (apply_execution_lag), which need day-by-day
    granularity that the rest of the engine discards by working in
    monthly_prices.

    Open is scaled onto the SAME adjustment basis as price_col (NOT the
    raw print) when price_col='Adj Close' -- Yahoo only ever publishes a
    split/dividend-adjusted CLOSE, never an adjusted Open, so Open is
    scaled by that day's own (price_col / raw Close) ratio. Mixing a raw
    Open directly against an Adj Close-based price (monthly_prices or
    daily_close) is a real bug, not a "minor inconsistency": Adj Close is
    backward-adjusted for EVERY split/bonus up to TODAY, so for an old
    date, a raw Open can be many times larger than that same day's Adj
    Close purely from splits that hadn't happened yet -- found via
    apply_execution_lag's T+1 calculation, which mixed exactly these two
    bases and showed VEDL "returning" +258% on a single execution day in
    Nov 2005 (Adj Close 14.16 vs raw Open 50.75, same stock, one day
    apart) purely from this mismatch, not a real price move. If price_col
    is already 'Close' (raw), no scaling is needed -- Open is returned as
    printed.

    Restricts both frames to symbols present in all the fields needed
    (a symbol missing any of them is dropped), matching the old
    single-pass-per-file behavior even though each field is now loaded
    (and cached) independently via load_wide_daily_field."""
    daily_close = load_wide_daily_field(price_col)
    daily_open_raw = load_wide_daily_field("Open")
    common_symbols = daily_close.columns.intersection(daily_open_raw.columns)
    daily_close = daily_close[common_symbols]
    daily_open_raw = daily_open_raw[common_symbols]

    if price_col == "Close":
        return daily_close, daily_open_raw

    daily_close_raw = load_wide_daily_field("Close")
    common_symbols = common_symbols.intersection(daily_close_raw.columns)
    daily_close = daily_close[common_symbols]
    daily_open_raw = daily_open_raw[common_symbols]
    daily_close_raw = daily_close_raw[common_symbols]

    adj_factor = daily_close / daily_close_raw.reindex(daily_close.index)
    adj_factor = adj_factor.replace([np.inf, -np.inf], np.nan)
    daily_open = daily_open_raw * adj_factor.reindex(daily_open_raw.index)
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


def load_hourly_ohlc() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads hourly Open/Close from data/hourly/ (see
    scripts/download_hourly_data.py) for the hourly EMA-crossover swing
    strategy. Only Open/Close are needed (the EMA strategy trades on
    closes and fills at opens, no high/low channel logic), and only ~216
    symbols exist here -- whichever stocks appeared in the top-30 momentum
    universe within Yahoo Finance's ~2-3 year hourly data window, not the
    full ~970-symbol daily universe. Timestamps are converted from IST
    (+05:30, as downloaded) to naive local wall-clock time, matching the
    naive DatetimeIndex convention used by the daily loaders -- comparing
    a tz-aware and a tz-naive index would raise, and the monthly rebalance
    calendar this gets aligned against is itself naive.
    """
    open_frames, close_frames = {}, {}
    for f in HOURLY_DIR.glob("*.csv"):
        sym = f.stem
        try:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
        except Exception:
            continue
        if not {"Open", "Close"}.issubset(df.columns) or df.empty:
            continue
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        o, c = df["Open"].dropna(), df["Close"].dropna()
        if o.empty or c.empty:
            continue
        open_frames[sym] = o
        close_frames[sym] = c
    if not close_frames:
        raise RuntimeError(f"No usable hourly data found in {HOURLY_DIR}")
    hourly_open = pd.DataFrame(open_frames).sort_index()
    hourly_close = pd.DataFrame(close_frames).sort_index()
    return hourly_open, hourly_close


def load_hourly_full_ohlc() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Like load_hourly_ohlc, but also loads High/Low -- needed by the ORB
    (Opening Range Breakout) swing strategy to capture the actual high/low
    of the opening range bar(s), not just their close. Same data/hourly/
    source, same ~2-3 year window, same tz-naive conversion."""
    open_frames, high_frames, low_frames, close_frames = {}, {}, {}, {}
    for f in HOURLY_DIR.glob("*.csv"):
        sym = f.stem
        try:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
        except Exception:
            continue
        if not {"Open", "High", "Low", "Close"}.issubset(df.columns) or df.empty:
            continue
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        o, h, l, c = df["Open"].dropna(), df["High"].dropna(), df["Low"].dropna(), df["Close"].dropna()
        if o.empty or h.empty or l.empty or c.empty:
            continue
        open_frames[sym] = o
        high_frames[sym] = h
        low_frames[sym] = l
        close_frames[sym] = c
    if not close_frames:
        raise RuntimeError(f"No usable hourly data found in {HOURLY_DIR}")
    return (
        pd.DataFrame(open_frames).sort_index(),
        pd.DataFrame(high_frames).sort_index(),
        pd.DataFrame(low_frames).sort_index(),
        pd.DataFrame(close_frames).sort_index(),
    )


def load_2h_ohlc() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resamples load_hourly_ohlc()'s bars into 2-hour bars, for the Short
    Momentum (F&O) swing strategy's "2 hourly" timeframe option. NSE's
    session doesn't start on an even clock hour (9:15, not 9:00), so a
    plain pandas .resample('2h') would misalign against the actual hourly
    bars (typically ~9:15/10:15/11:15/12:15/13:15/14:15/15:15). Instead we
    pair up each day's hourly bars in trading order starting from that
    day's first bar -- (9:15,10:15), (11:15,12:15), (13:15,14:15), (15:15)
    -- open = the pair's first bar's open, close = the pair's last bar's
    close, labeled with the last bar's timestamp.
    """
    hourly_open, hourly_close = load_hourly_ohlc()
    day = pd.Series(hourly_close.index.date, index=hourly_close.index)
    pair_seq = hourly_close.groupby(day).cumcount() // 2
    pair_key = day.astype(str) + "_" + pair_seq.astype(str)
    bar_ts = pd.Series(hourly_close.index, index=hourly_close.index)

    label = bar_ts.groupby(pair_key).last()
    open_2h = hourly_open.groupby(pair_key).first()
    close_2h = hourly_close.groupby(pair_key).last()
    open_2h.index = label.reindex(open_2h.index).values
    close_2h.index = label.reindex(close_2h.index).values
    return open_2h.sort_index(), close_2h.sort_index()


def load_15min_full_ohlc() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Open/High/Low/Close from data/15min/ (see scripts/download_15min_
    data.py), for the RSI Oversold Reversal swing strategy's finer
    timeframe options -- the only strategy in this app needing sub-hourly
    bars. Yahoo Finance caps intervals finer than 1h at a 60-DAY trailing
    window (not ~2-3yr like load_hourly_full_ohlc, let alone ~18yr of
    daily data) -- a hard limit, not a download choice, so results from
    this are a short recent sample, not a real multi-year backtest."""
    open_frames, high_frames, low_frames, close_frames = {}, {}, {}, {}
    for f in FIFTEEN_MIN_DIR.glob("*.csv"):
        sym = f.stem
        try:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
        except Exception:
            continue
        if not {"Open", "High", "Low", "Close"}.issubset(df.columns) or df.empty:
            continue
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        o, h, l, c = df["Open"].dropna(), df["High"].dropna(), df["Low"].dropna(), df["Close"].dropna()
        if o.empty or h.empty or l.empty or c.empty:
            continue
        open_frames[sym] = o
        high_frames[sym] = h
        low_frames[sym] = l
        close_frames[sym] = c
    if not close_frames:
        raise RuntimeError(f"No usable 15-minute data found in {FIFTEEN_MIN_DIR}")
    return (
        pd.DataFrame(open_frames).sort_index(),
        pd.DataFrame(high_frames).sort_index(),
        pd.DataFrame(low_frames).sort_index(),
        pd.DataFrame(close_frames).sort_index(),
    )


def load_30min_upstox_full_ohlc() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Open/High/Low/Close from data/30min/ (see scripts/download_upstox_
    30min_data.py), for the RSI Oversold Reversal swing strategy's 30-min
    timeframe. Unlike every other intraday loader in this project, this is
    NOT Yahoo Finance -- Yahoo hard-caps anything finer than 1h at a ~60-day
    trailing window, so genuine multi-year 30-min history requires a
    different source. This one comes from Upstox's Historical Candle Data
    V3 API (a free, no-cost, read-only Analytics Token -- see the download
    script's docstring), which serves 30-min bars back to 2022-01-01 --
    confirmed empirically, not just from Upstox's own docs -- giving ~4.5
    years of real history instead of ~60 days."""
    open_frames, high_frames, low_frames, close_frames = {}, {}, {}, {}
    for f in THIRTY_MIN_DIR.glob("*.csv"):
        sym = f.stem
        try:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
        except Exception:
            continue
        if not {"Open", "High", "Low", "Close"}.issubset(df.columns) or df.empty:
            continue
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        o, h, l, c = df["Open"].dropna(), df["High"].dropna(), df["Low"].dropna(), df["Close"].dropna()
        if o.empty or h.empty or l.empty or c.empty:
            continue
        open_frames[sym] = o
        high_frames[sym] = h
        low_frames[sym] = l
        close_frames[sym] = c
    if not close_frames:
        raise RuntimeError(f"No usable 30-minute data found in {THIRTY_MIN_DIR}")
    return (
        pd.DataFrame(open_frames).sort_index(),
        pd.DataFrame(high_frames).sort_index(),
        pd.DataFrame(low_frames).sort_index(),
        pd.DataFrame(close_frames).sort_index(),
    )


def load_hourly_upstox_ohlc() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resamples load_30min_upstox_full_ohlc()'s Open/Close into hourly
    bars, for the EMA crossover swing strategy's hourly timeframe --
    same ~4.5 year history (back to 2022-01-01) as the 30-min data itself,
    instead of load_hourly_ohlc's Yahoo-sourced ~2-3 year window. Same
    day-pair grouping idea as load_2h_ohlc, but pairing 30-min bars
    instead of hourly ones: each day's 30-min bars are paired in trading
    order starting from that day's first bar -- (9:15,9:45), (10:15,
    10:45), ..., (15:15 alone, no partner since the session ends at
    15:30) -- open = the pair's first bar's open, close = the pair's
    last bar's close, labeled with the pair's FIRST bar's timestamp so
    the result lines up exactly with load_hourly_ohlc's own labeling
    convention (9:15/10:15/11:15/.../15:15)."""
    open_30, _, _, close_30 = load_30min_upstox_full_ohlc()
    day = pd.Series(close_30.index.date, index=close_30.index)
    pair_seq = close_30.groupby(day).cumcount() // 2
    pair_key = day.astype(str) + "_" + pair_seq.astype(str)
    bar_ts = pd.Series(close_30.index, index=close_30.index)

    label = bar_ts.groupby(pair_key).first()
    open_1h = open_30.groupby(pair_key).first()
    close_1h = close_30.groupby(pair_key).last()
    open_1h.index = label.reindex(open_1h.index).values
    close_1h.index = label.reindex(close_1h.index).values
    return open_1h.sort_index(), close_1h.sort_index()


def load_hourly_upstox_full_ohlc() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Like load_hourly_upstox_ohlc, but also resamples High/Low -- needed
    by the ORB (Opening Range Breakout) swing strategy to capture the
    actual high/low of the opening range bar(s), not just their close.
    Same day-pair grouping from 30-min Upstox bars, same ~4.5 year
    window (back to 2022-01-01) instead of load_hourly_full_ohlc's
    Yahoo-sourced ~2-3 years: open = pair's first bar's open, high = max
    of the pair, low = min of the pair, close = pair's last bar's close,
    labeled with the pair's first bar's timestamp."""
    open_30, high_30, low_30, close_30 = load_30min_upstox_full_ohlc()
    day = pd.Series(close_30.index.date, index=close_30.index)
    pair_seq = close_30.groupby(day).cumcount() // 2
    pair_key = day.astype(str) + "_" + pair_seq.astype(str)
    bar_ts = pd.Series(close_30.index, index=close_30.index)

    label = bar_ts.groupby(pair_key).first()
    open_1h = open_30.groupby(pair_key).first()
    high_1h = high_30.groupby(pair_key).max()
    low_1h = low_30.groupby(pair_key).min()
    close_1h = close_30.groupby(pair_key).last()
    for df in (open_1h, high_1h, low_1h, close_1h):
        df.index = label.reindex(df.index).values
    return open_1h.sort_index(), high_1h.sort_index(), low_1h.sort_index(), close_1h.sort_index()


def load_benchmark(price_col: str = "Adj Close", freq: str = "ME") -> pd.Series:
    f = INDEX_DIR / "NIFTY500.csv"
    df = pd.read_csv(f, index_col=0, parse_dates=True)
    col = price_col if price_col in df.columns else "Close"
    s = df[col].dropna().sort_index()
    return s.resample(freq).last()


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


# Instruments available to the Correction Lumpsum page (pages/5_Correction_Lumpsum.py).
# Indices (not directly investable) are used for Nifty/midcap/smallcap since they have
# the longest, cleanest history -- a "correction" is naturally a statement about the
# index. Gold uses GOLDBEES (the ETF, not a USD gold index) for the same reason it's
# used everywhere else in this project: it's the actual investable instrument.
CORRECTION_INSTRUMENTS: dict[str, tuple[Path, str]] = {
    "Nifty 50": (INDEX_DIR, "NIFTY50.csv"),
    "Nifty 500": (INDEX_DIR, "NIFTY500.csv"),
    "Nifty Midcap 150": (INDEX_DIR, "NIFTYMIDCAP150.csv"),
    "Nifty Smallcap 250": (INDEX_DIR, "NIFTYSMLCAP250.csv"),
    "Gold (GOLDBEES)": (ETF_DIR, "GOLDBEES.csv"),
}


def load_correction_instrument_daily(name: str, price_col: str = "Close") -> pd.Series:
    """DAILY (not monthly) price series for one of CORRECTION_INSTRUMENTS,
    for the Correction Lumpsum page -- detecting a % correction from a
    rolling peak needs day-level resolution, unlike the monthly series
    used by the momentum backtest and SIP pages."""
    base_dir, fname = CORRECTION_INSTRUMENTS[name]
    df = pd.read_csv(base_dir / fname, index_col=0, parse_dates=True)
    col = price_col if price_col in df.columns else "Close"
    return df[col].dropna().sort_index()


# Curated list of well-known Direct Growth mutual fund schemes for the
# Mutual Fund Comparison page (see scripts/download_mutual_funds.py) --
# NOT an exhaustive or AUM-ranked list (no free API gives real-time AUM
# rankings), just hand-picked well-known funds across major equity
# categories, spanning several AMCs. Direct Growth specifically, not
# Regular -- Regular plans carry distributor commission drag baked into
# a lower NAV, an easier bar for the strategy to beat, not a fair
# comparison. Several of these are funds whose old marketing names
# (e.g. "HDFC Top 100", "ICICI Prudential Bluechip", "SBI Bluechip",
# "Kotak Emerging Equity", "Quant Active Fund") no longer match their
# CURRENT AMFI scheme name after a rename -- every scheme_code below was
# verified against the live api.mfapi.in scheme list, not guessed from
# the old marketing name. scheme_code is the mfapi.in/AMFI code.
MUTUAL_FUNDS: dict[str, dict] = {
    # Large Cap
    "HDFC Large Cap Fund": {"scheme_code": 119018, "category": "Large Cap"},
    "ICICI Prudential Large Cap Fund": {"scheme_code": 120586, "category": "Large Cap"},
    "SBI Large Cap Fund": {"scheme_code": 119598, "category": "Large Cap"},
    "Nippon India Large Cap Fund": {"scheme_code": 118632, "category": "Large Cap"},
    "Mirae Asset Large Cap Fund": {"scheme_code": 118825, "category": "Large Cap"},
    # Flexi Cap
    "Parag Parikh Flexi Cap Fund": {"scheme_code": 122639, "category": "Flexi Cap"},
    "HDFC Flexi Cap Fund": {"scheme_code": 118955, "category": "Flexi Cap"},
    "Kotak Flexi Cap Fund": {"scheme_code": 120166, "category": "Flexi Cap"},
    "Franklin India Flexi Cap Fund": {"scheme_code": 118535, "category": "Flexi Cap"},
    # Multi Cap
    "Nippon India Multi Cap Fund": {"scheme_code": 118650, "category": "Multi Cap"},
    "Quant Multi Cap Fund": {"scheme_code": 120823, "category": "Multi Cap"},
    "ICICI Prudential Multi Cap Fund": {"scheme_code": 120599, "category": "Multi Cap"},
    # Mid Cap
    "HDFC Mid Cap Fund": {"scheme_code": 118989, "category": "Mid Cap"},
    "Kotak Mid Cap Fund": {"scheme_code": 119775, "category": "Mid Cap"},
    "Axis Midcap Fund": {"scheme_code": 120505, "category": "Mid Cap"},
    "DSP Midcap Fund": {"scheme_code": 119071, "category": "Mid Cap"},
    # Small Cap
    "SBI Small Cap Fund": {"scheme_code": 125497, "category": "Small Cap"},
    "Nippon India Small Cap Fund": {"scheme_code": 118778, "category": "Small Cap"},
    "Axis Small Cap Fund": {"scheme_code": 125354, "category": "Small Cap"},
    "Quant Small Cap Fund": {"scheme_code": 120828, "category": "Small Cap"},
    # ELSS
    "Axis ELSS Tax Saver Fund": {"scheme_code": 120503, "category": "ELSS"},
    "Mirae Asset ELSS Tax Saver Fund": {"scheme_code": 135781, "category": "ELSS"},
    "SBI ELSS Tax Saver Fund": {"scheme_code": 119723, "category": "ELSS"},
    # Index / Passive
    "UTI Nifty 50 Index Fund": {"scheme_code": 120716, "category": "Index"},
    "HDFC Nifty 50 Index Fund": {"scheme_code": 119063, "category": "Index"},
    # Momentum (thematically the closest comparison to this project's own strategy)
    "UTI Nifty 200 Momentum 30 Index Fund": {"scheme_code": 148703, "category": "Momentum"},
    "quant Momentum Fund": {"scheme_code": 152189, "category": "Momentum"},
    "Axis Nifty500 Momentum 50 Index Fund": {"scheme_code": 153244, "category": "Momentum"},
}


def load_mutual_fund_nav(name: str) -> pd.Series:
    """DAILY NAV series for one of MUTUAL_FUNDS, from data/mutual_funds/
    <scheme_code>.csv (see scripts/download_mutual_funds.py). Like
    load_correction_instrument_daily, this is daily (not monthly) --
    the Mutual Fund Comparison page resamples to month-end itself,
    matching load_benchmark's convention, so both sides of the
    comparison share the exact same rebalance-date grid."""
    scheme_code = MUTUAL_FUNDS[name]["scheme_code"]
    f = MUTUAL_FUND_DIR / f"{scheme_code}.csv"
    df = pd.read_csv(f, index_col=0, parse_dates=True)
    return df["NAV"].dropna().sort_index()


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


def load_fno_symbols() -> set[str]:
    """Current NSE F&O (futures & options) eligible stock symbols, from
    data/fno_stocks.csv (see scripts/get_fno_list.py). Used by the Short
    Momentum (F&O) swing strategy to restrict its universe to stocks that
    actually have tradeable stock futures -- shorting isn't otherwise
    viable for individual equities in the Indian cash market. Like
    load_current_universe, this is a CURRENT snapshot applied across all
    history, not a point-in-time calendar (a stock's F&O eligibility does
    change over time, but NSE doesn't publish a historical version of this
    list the way it does index membership)."""
    f = ROOT / "data" / "fno_stocks.csv"
    return set(pd.read_csv(f)["Symbol"])


# The "Universe" selector offered above stock selection on every page (see
# scripts/get_nse_universes.py). "Nifty 500 (point-in-time)" isn't in this
# dict -- it's the app's existing default (no allowed_symbols restriction at
# all, just the point-in-time membership calendar already in use), kept that
# way specifically so picking it is a no-op rather than swapping the
# survivorship-bias-aware historical universe for a current-only snapshot.
# Every OTHER entry here IS a current snapshot (NSE doesn't publish
# historical inclusion/exclusion logs for these the way it does for the
# (Nifty) 500), so selecting one applies that current list across all
# history -- same simplification as load_fno_symbols above.
NSE_UNIVERSES: dict[str, str] = {
    "Nifty 50": "nifty50",
    "Nifty Next 50": "niftynext50",
    "Nifty Midcap Select": "niftymidcapselect",
    "Nifty 100": "nifty100",
    "Nifty 200": "nifty200",
    "Nifty 500": "nifty500",
    "Nifty Total Market": "niftytotalmarket",
    "Nifty Microcap 250": "niftymicrocap250",
    "Nifty Smallcap 250": "niftysmallcap250",
    "Nifty Alpha 50": "niftyalpha50",
}


def load_universe_symbols(name: str) -> set[str]:
    """Current constituent symbols for one of NSE_UNIVERSES, from
    data/universes/<key>.csv (see scripts/get_nse_universes.py)."""
    key = NSE_UNIVERSES[name]
    f = ROOT / "data" / "universes" / f"{key}.csv"
    return set(pd.read_csv(f)["Symbol"])


def _momentum_window(monthly_prices: pd.DataFrame, as_of_date: pd.Timestamp, lookback_months: int, skip_months: int):
    """Shared window resolution behind compute_momentum_ranking and
    compute_trailing_volatility -- both need the EXACT same
    [start_idx, end_idx] slice (the lookback window ending skip_months
    before as_of_date), so this is the one place that logic lives.
    Returns None if as_of_date isn't in monthly_prices or there isn't
    enough trailing history, else (start_idx, end_idx)."""
    dates = monthly_prices.index
    if as_of_date not in dates:
        return None
    i = dates.get_loc(as_of_date)
    end_idx = i - skip_months
    start_idx = end_idx - lookback_months
    if start_idx < 0:
        return None
    return start_idx, end_idx


def compute_trailing_volatility(
    monthly_prices: pd.DataFrame,
    as_of_date: pd.Timestamp,
    lookback_months: int,
    skip_months: int,
) -> pd.Series | None:
    """Annualized volatility (std dev of monthly returns x sqrt(12)) over
    the SAME lookback window compute_momentum_ranking measures its return
    over -- e.g. lookback_months=10, skip_months=1 means "volatility of
    the 10 months of returns ending 1 month before as_of_date", pairing
    exactly with that window's own trailing return. No separate lookback
    parameter of its own, deliberately -- a risk-adjusted return is return
    over risk measured over the SAME period, not two independently-tuned
    windows. Returns None under the same conditions compute_momentum_
    ranking does (not enough history)."""
    window = _momentum_window(monthly_prices, as_of_date, lookback_months, skip_months)
    if window is None:
        return None
    start_idx, end_idx = window
    window_prices = monthly_prices.iloc[start_idx:end_idx + 1]
    return window_prices.pct_change().std() * np.sqrt(12)


def compute_momentum_ranking(
    monthly_prices: pd.DataFrame,
    membership: pd.DataFrame | None,
    as_of_date: pd.Timestamp,
    lookback_months: int,
    skip_months: int,
    min_price: float,
    allowed_symbols: set[str] | None = None,
    max_volatility_pct: float | None = None,
    use_risk_adjusted: bool = False,
    max_trailing_return_pct: float | None = None,
) -> pd.Series | None:
    """Trailing lookback_months return (skipping the most recent skip_months)
    for every stock eligible as of as_of_date, sorted descending (first
    entry = rank 1 = highest momentum). Returns None if as_of_date isn't in
    monthly_prices or doesn't have enough trailing history.

    Eligibility: valid (>0) price at both the lookback start and end, last
    price >= min_price, -- if membership is given -- an actual Nifty 500
    constituent as of as_of_date per the point-in-time calendar, and -- if
    allowed_symbols is given -- a member of that set. allowed_symbols is
    the "Universe" selector (see NSE_UNIVERSES / load_universe_symbols): a
    CURRENT snapshot of some other NSE index (Nifty 50, Smallcap 250,
    etc.), not a point-in-time history like membership -- NSE doesn't
    publish historical inclusion/exclusion logs for these the way it does
    for the (Nifty) 500. This is the single source of truth for the
    ranking formula, shared by run_backtest (historical simulation), the
    live stock ranker (pages/1_Stock_Ranker.py, current snapshot), and
    swing_engine.py's strategies.

    max_volatility_pct, if given, additionally requires a stock's trailing
    ANNUALIZED volatility (see compute_trailing_volatility -- same window
    as the return itself) to be at or below this %, e.g. 60.0 excludes
    anything more volatile than 60%/yr. use_risk_adjusted, if True, scores
    and sorts by (trailing return / that same volatility) instead of the
    raw return -- a Sharpe-like ratio, not a percentage; stocks with
    zero/NaN volatility are excluded first to avoid an undefined ratio.
    Both are opt-in and back-compatible -- omitting them reproduces
    exactly today's absolute-return-only behavior, which is what every
    caller other than app.py and the Stock Ranker still does.

    max_trailing_return_pct, if given, excludes a stock whose RAW trailing
    return (before any risk-adjustment) exceeds this % -- a data-sanity
    guard, not a strategy choice. Found via real evidence: 110 of ~1,089
    symbols have an implausible (>3x single-day) price jump somewhere in
    their pre-2008 history (almost certainly unadjusted splits/mergers/
    symbol reuse in that older Yahoo Finance data, not real moves -- see
    e.g. KANSAINER, ~40x between May 2004 and June 2005). A stock like
    that dominates the top-N ranking for the ~lookback_months its window
    straddles the bad jump, which is what made 2003/2005's backtest years
    look "exceptional" before this existed. Checked against the RAW
    return regardless of use_risk_adjusted, since the artifact is in the
    return itself, not in how it's scored.
    """
    window = _momentum_window(monthly_prices, as_of_date, lookback_months, skip_months)
    if window is None:
        return None
    start_idx, end_idx = window

    px_start = monthly_prices.iloc[start_idx]
    px_end = monthly_prices.iloc[end_idx]
    last_price = monthly_prices.iloc[monthly_prices.index.get_loc(as_of_date)]

    eligible = (px_start > 0) & (px_end > 0) & (last_price >= min_price)
    if membership is not None:
        eligible &= membership.loc[as_of_date]
    if allowed_symbols is not None:
        eligible &= eligible.index.isin(allowed_symbols)

    mom = px_end / px_start - 1.0
    score = mom

    if max_trailing_return_pct is not None:
        eligible &= mom <= max_trailing_return_pct / 100.0

    if max_volatility_pct is not None or use_risk_adjusted:
        volatility = compute_trailing_volatility(monthly_prices, as_of_date, lookback_months, skip_months)
        if max_volatility_pct is not None:
            eligible &= volatility.notna() & (volatility <= max_volatility_pct / 100.0)
        if use_risk_adjusted:
            eligible &= volatility.notna() & (volatility > 1e-9)
            score = mom / volatility

    score = score[eligible].dropna()
    return score.sort_values(ascending=False)


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
    allowed_symbols: set[str] | None = None,
    max_volatility_pct: float | None = None,
    use_risk_adjusted: bool = False,
    max_trailing_return_pct: float | None = None,
) -> tuple[pd.Series, list[tuple[pd.Timestamp, list[str]]]]:
    """Returns (monthly portfolio returns, [(rebalance_date, holdings), ...]).

    allowed_symbols, if given, is the "Universe" selector -- see
    compute_momentum_ranking's docstring (a CURRENT snapshot of some other
    NSE index, applied on top of whatever membership already restricts).

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

        ranked = compute_momentum_ranking(
            monthly_prices, membership, today, lookback_months, skip_months, min_price, allowed_symbols,
            max_volatility_pct=max_volatility_pct, use_risk_adjusted=use_risk_adjusted,
            max_trailing_return_pct=max_trailing_return_pct,
        )
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
    resample_freq: str = "ME",
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

    Returns {period_end_date: return} for the recomputed periods only,
    resampled with resample_freq (default "ME" -- must match whatever
    period grid monthly_prices/holdings_history actually use, e.g. "W-FRI"
    for weekly rebalancing, or the merge back into portfolio_rets won't
    line up). The caller merges this into the baseline portfolio_rets from
    run_backtest, leaving GOLD-regime and pre-first-rebalance periods
    unchanged.
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

        period_from_daily = (1 + daily_port_ret).resample(resample_freq).prod() - 1
        for p_date, r in period_from_daily.items():
            monthly_result[p_date] = r

    return monthly_result


def apply_execution_lag(
    monthly_prices: pd.DataFrame,
    daily_close: pd.DataFrame,
    daily_open: pd.DataFrame,
    holdings_history: list[tuple[pd.Timestamp, list[str]]],
    resample_freq: str = "ME",
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

    Returns {period_end_date: return} for every period following a
    rebalance where the holdings list actually changed, resampled with
    resample_freq (default "ME" -- must match monthly_prices/
    holdings_history's own period grid, e.g. "W-FRI" for weekly
    rebalancing). Periods with no change, GOLD-regime periods, and the
    very first holding period (nothing to compare against, since there's
    no "previous" holdings list yet) are left untouched by the caller.
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

        period_from_daily = (1 + daily_port_ret).resample(resample_freq).prod() - 1
        for p_date, r in period_from_daily.items():
            monthly_result[p_date] = r

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


def _round_half_up(x: float) -> int:
    """Round-half-up (0.5 rounds to 1, not Python's round-half-to-even) --
    used for share quantities, since a real trader can't buy a fractional
    share and "round to even" would be a strange, unexplainable rule for
    a quantity a user is looking at directly in a trade log."""
    import math
    return math.floor(x + 0.5)


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
    orders -- qty is additionally rounded to the nearest whole share
    (round-half-up, so 100.5 becomes 101, see _round_half_up) for the
    same reason: a trader can't buy 0.5 of a share either. pnl_rs/
    pnl_pct are computed from this rounded qty, so they reflect what a
    real position would have actually earned, not the unrounded
    theoretical sizing. Position sizing uses the GROSS (pre-cost, pre-tax) equity curve,
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
            qty = _round_half_up(position_value / entry_price)
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
        qty = _round_half_up(position_value / entry_price)
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
