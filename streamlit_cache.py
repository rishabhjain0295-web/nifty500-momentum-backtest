"""
Shared @st.cache_data-wrapped loaders, used by both app.py and pages/*.py so
navigating between pages doesn't reload the same price data twice. Kept
separate from backtest_engine.py so that module stays Streamlit-agnostic
(usable from the plain CLI script without pulling in a Streamlit runtime).
"""
import pandas as pd
import streamlit as st

from backtest_engine import (
    load_benchmark,
    load_current_universe,
    load_daily_ohlc,
    load_daily_prices,
    load_gold_series,
    load_hourly_ohlc,
    load_liquid_series,
    load_membership_matrix,
    load_prices,
)


@st.cache_data(show_spinner="Loading stock price history...")
def cached_load_prices(price_col: str) -> pd.DataFrame:
    return load_prices(price_col)


@st.cache_data(show_spinner="Loading benchmark...")
def cached_load_benchmark(price_col: str) -> pd.Series:
    return load_benchmark(price_col)


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
