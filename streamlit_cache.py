"""
Shared @st.cache_data-wrapped loaders, used by both app.py and pages/*.py so
navigating between pages doesn't reload the same price data twice. Kept
separate from backtest_engine.py so that module stays Streamlit-agnostic
(usable from the plain CLI script without pulling in a Streamlit runtime).
"""
import pandas as pd
import streamlit as st

from backtest_engine import (
    load_2h_ohlc,
    load_15min_full_ohlc,
    load_30min_upstox_full_ohlc,
    load_benchmark,
    load_correction_instrument_daily,
    load_current_universe,
    load_daily_ohlc,
    load_daily_prices,
    load_fno_symbols,
    load_gold_series,
    load_hourly_full_ohlc,
    load_hourly_ohlc,
    load_hourly_upstox_full_ohlc,
    load_hourly_upstox_ohlc,
    load_liquid_series,
    load_membership_matrix,
    load_mutual_fund_nav,
    load_prices,
    load_universe_symbols,
)
from ema_rs_rotation_engine import load_data as load_ema_rs_data


@st.cache_data(show_spinner="Loading stock price history...")
def cached_load_prices(price_col: str, freq: str = "ME") -> pd.DataFrame:
    return load_prices(price_col, freq)


@st.cache_data(show_spinner="Loading benchmark...")
def cached_load_benchmark(price_col: str, freq: str = "ME") -> pd.Series:
    return load_benchmark(price_col, freq)


@st.cache_data(show_spinner="Loading gold price history...")
def cached_load_gold(price_col: str) -> pd.Series:
    return load_gold_series(price_col)


@st.cache_data(show_spinner="Loading daily price history...")
def cached_load_daily_prices(price_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_daily_prices(price_col)


@st.cache_data(show_spinner="Loading point-in-time membership calendar...")
def cached_load_membership(dates: tuple, symbols: tuple) -> pd.DataFrame:
    return load_membership_matrix(pd.DatetimeIndex(dates), pd.Index(symbols))


@st.cache_data(show_spinner="Loading current Nifty 500 constituent list...")
def cached_load_current_universe() -> pd.DataFrame:
    return load_current_universe()


@st.cache_data(show_spinner="Loading liquid fund (LIQUIDBEES) price history...")
def cached_load_liquid(price_col: str) -> pd.Series:
    return load_liquid_series(price_col)


@st.cache_data(show_spinner="Loading daily OHLC history for swing trading (this can take ~20s)...")
def cached_load_daily_ohlc() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return load_daily_ohlc()


@st.cache_data(show_spinner="Loading hourly price history for the EMA swing strategy...")
def cached_load_hourly_ohlc() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_hourly_ohlc()


@st.cache_data(show_spinner="Loading hourly OHLC history for the ORB strategy...")
def cached_load_hourly_full_ohlc() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return load_hourly_full_ohlc()


@st.cache_data(show_spinner="Building 2-hour bars from hourly price history...")
def cached_load_2h_ohlc() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_2h_ohlc()


@st.cache_data(show_spinner="Loading 15-minute OHLC history for the RSI Reversal strategy (~60 day window)...")
def cached_load_15min_full_ohlc() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return load_15min_full_ohlc()


@st.cache_data(show_spinner="Loading 30-minute price history (Upstox, ~4.5 years)...")
def cached_load_30min_upstox_full_ohlc() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return load_30min_upstox_full_ohlc()


@st.cache_data(show_spinner="Building hourly bars from 30-minute Upstox price history (~4.5 years)...")
def cached_load_hourly_upstox_ohlc() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_hourly_upstox_ohlc()


@st.cache_data(show_spinner="Building hourly OHLC bars from 30-minute Upstox price history for ORB (~4.5 years)...")
def cached_load_hourly_upstox_full_ohlc() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return load_hourly_upstox_full_ohlc()


@st.cache_data(show_spinner="Loading F&O eligible stock list...")
def cached_load_fno_symbols() -> set[str]:
    return load_fno_symbols()


@st.cache_data(show_spinner="Loading universe constituent list...")
def cached_load_universe_symbols(name: str) -> set[str]:
    return load_universe_symbols(name)


@st.cache_data(show_spinner="Loading price history for correction-triggered lumpsum...")
def cached_load_correction_instrument(name: str, price_col: str) -> pd.Series:
    return load_correction_instrument_daily(name, price_col)


@st.cache_data(show_spinner="Loading mutual fund NAV history...")
def cached_load_mutual_fund_nav(name: str) -> pd.Series:
    return load_mutual_fund_nav(name)


@st.cache_data(show_spinner="Loading spot index / ETF price history...")
def cached_load_ema_rs_data(equity_leg: str, price_col: str) -> tuple[pd.Series, dict, str]:
    return load_ema_rs_data(equity_leg, price_col)
