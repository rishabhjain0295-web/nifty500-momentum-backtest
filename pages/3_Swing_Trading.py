"""
Swing trading: scans the top-N momentum universe (same ranking as the
Backtest page) each month for entries, trades them individually with
risk-based position sizing. Two strategy families are offered:
breakout (gap-up / Donchian channel, with an R-multiple profit target)
and EMA 15/50 crossover (trend-following, daily or hourly bars, no fixed
target -- see run_ema_crossover_backtest's docstring for the hourly data
history caveat). See swing_engine.py for the full mechanics and the
design assumptions made where the source spec was ambiguous.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtest_engine import ensure_hourly_data, ensure_stock_data
from streamlit_cache import (
    cached_load_daily_ohlc,
    cached_load_hourly_ohlc,
    cached_load_membership,
    cached_load_prices,
)
from swing_engine import run_ema_crossover_backtest, run_swing_backtest

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
    st.header("Momentum universe")
    n_stocks = st.slider("Number of stocks in universe", min_value=5, max_value=100, value=20, step=5)
    lookback_months = st.slider("Ranking lookback (months)", min_value=1, max_value=24, value=10, step=1)
    skip_months = st.slider(
        "Skip period (months)", min_value=0, max_value=3, value=1, step=1,
        help="Excludes the most recent N months from the ranking lookback."
    )
    price_col = st.selectbox("Price field for ranking", ["Adj Close", "Close"], index=0)
    min_price = st.number_input("Minimum price filter (Rs)", min_value=0.0, value=10.0, step=5.0)
    use_membership_filter = st.checkbox("Enforce point-in-time index membership", value=True)

    st.header("Strategy")
    strategy_label = st.radio(
        "Strategy type", ["Breakout (gap-up / Donchian)", "EMA 15/50 crossover"], index=0
    )
    is_ema = strategy_label == "EMA 15/50 crossover"

    timeframe = "Daily"
    ema_fast = 15
    ema_slow = 50
    entry_strategy = "gap_up"
    gap_pct = 1.0
    donchian_entry_lookback = 20
    exit_mode = "trailing_donchian"
    exit_lookback_days = 1
    stop_pct = 8.0
    risk_reward_ratio = 2.0

    if is_ema:
        st.header("EMA crossover")
        timeframe = st.radio(
            "Timeframe", ["Daily (full history)", "Hourly (~2-3 years only)"], index=0
        )
        if timeframe.startswith("Hourly"):
            st.caption(
                "Yahoo Finance only serves hourly data for roughly the trailing 2-3 years "
                "(observed back to ~2023-09 for this project's download), unlike the daily "
                "data which goes back to 2008. Results below will only cover that shorter, "
                "more recent window, and only for the ~216 stocks that were actually in the "
                "top-30 momentum universe during it."
            )
        ema_fast = st.number_input("Fast EMA span (bars)", min_value=2, max_value=100, value=15, step=1)
        ema_slow = st.number_input("Slow EMA span (bars)", min_value=5, max_value=300, value=50, step=1)
        st.caption(
            "Entry: fast EMA above slow EMA at a bar's close, filled at the next bar's open. "
            "Exit: EITHER close below slow EMA OR fast EMA below slow EMA, also filled at the "
            "next bar's open -- checked every bar for the life of the trade. No fixed profit "
            "target; this is a pure trend-following exit."
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
    capital_base = st.number_input("Starting capital (Rs)", min_value=100_000.0, value=1_000_000.0, step=100_000.0)
    risk_pct = st.slider(
        "Risk per trade (% of capital)", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
        help="Position size is set so a stop-out loses exactly this % of starting capital -- "
             "subject to the position-size cap below, which usually binds first on tight stops."
    )
    if not is_ema:
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

monthly_prices = cached_load_prices(price_col)
membership = None
if use_membership_filter:
    membership = cached_load_membership(tuple(monthly_prices.index.values), tuple(monthly_prices.columns))

if is_ema:
    if timeframe.startswith("Hourly"):
        with st.spinner("Fetching hourly price data (first run only)..."):
            ensure_hourly_data()
        with st.spinner("Running EMA crossover backtest (hourly bars, ~2-3 year window)..."):
            bar_open, bar_close = cached_load_hourly_ohlc()
            result = run_ema_crossover_backtest(
                monthly_prices, membership, bar_open, bar_close,
                lookback_months, skip_months, n_stocks, min_price,
                ema_fast, ema_slow, risk_pct, max_position_pct, capital_base,
            )
    else:
        with st.spinner("Running EMA crossover backtest (daily bars, full history)..."):
            daily_open, daily_high, daily_low, daily_close = cached_load_daily_ohlc()
            result = run_ema_crossover_backtest(
                monthly_prices, membership, daily_open, daily_close,
                lookback_months, skip_months, n_stocks, min_price,
                ema_fast, ema_slow, risk_pct, max_position_pct, capital_base,
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
        )

trades = result["trades"]
equity = result["equity"]

if trades.empty or equity.empty:
    st.warning(
        "No trades were generated with these parameters -- try a lower gap threshold, a "
        "shorter Donchian lookback, a larger universe, or (for the EMA strategy) a shorter "
        "fast/slow span."
    )
    st.stop()

has_target = "target_price" in trades.columns


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
    if has_target:
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

with st.expander("Trade log"):
    display_cols = ["symbol", "entry_date", "entry_price", "stop_price"]
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
st.caption(
    f"Universe recomputed at {result['n_universe_periods']} monthly rebalance points using the "
    "same ranking formula as the Backtest page (compute_momentum_ranking), so this can't drift "
    "out of sync with it. Position sizing is risk-based (see sidebar), not equal-weight -- "
    "actual position values vary trade to trade depending on stop distance, capped by the "
    "max-position-size setting. Idle cash between trades earns 0% (a simplification). Costs, "
    "taxes, and slippage are not modeled here yet."
)
