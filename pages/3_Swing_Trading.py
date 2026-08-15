"""
Swing trading: scans the top-N momentum universe (same ranking as the
Backtest page) each month for breakout entries, trades them individually
with risk-based position sizing, a stop, and an R-multiple profit target.
See swing_engine.py for the full mechanics and the design assumptions
made where the source spec was ambiguous.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtest_engine import ensure_stock_data
from streamlit_cache import cached_load_daily_ohlc, cached_load_membership, cached_load_prices
from swing_engine import run_swing_backtest

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

    st.header("Entry strategy")
    entry_label = st.radio("Entry signal", ["Gap-up breakout", "Donchian channel breakout"], index=0)
    entry_strategy = "gap_up" if entry_label == "Gap-up breakout" else "donchian"
    gap_pct = 1.0
    donchian_entry_lookback = 20
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
    stop_pct = 8.0
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
        "shorter Donchian lookback, or a larger universe."
    )
    st.stop()


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
breakeven_wr = 1 / (1 + risk_reward_ratio)
cols2[2].metric("Breakeven win rate needed", fmt_pct(breakeven_wr))
cols2[3].metric("Open positions now", str((trades["status"] == "open").sum()))

if pd.notna(avg_r) and avg_r < 0:
    st.warning(
        f"Negative expectancy ({avg_r:.2f}R per trade on average): this parameter combination "
        f"loses money on average per trade, even before compounding. A tight stop relative to "
        f"the entry signal (e.g. previous-day-low right after a gap) gets hit by ordinary "
        f"volatility more often than the {breakeven_wr:.0%} win rate this risk:reward ratio "
        f"needs to break even -- try a wider stop, a different risk:reward, or the other entry "
        f"strategy."
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
    display_cols = ["symbol", "entry_date", "entry_price", "stop_price", "target_price",
                     "exit_date", "exit_price", "exit_reason", "qty", "risked_rs",
                     "pnl_rs", "pnl_pct", "r_multiple", "hold_days", "status"]
    st.dataframe(
        trades[display_cols].sort_values("entry_date", ascending=False).style.format({
            "entry_price": "{:.2f}", "stop_price": "{:.2f}", "target_price": "{:.2f}",
            "exit_price": "{:.2f}", "qty": "{:.1f}", "risked_rs": "Rs {:,.0f}",
            "pnl_rs": "Rs {:,.0f}", "pnl_pct": "{:.2%}", "r_multiple": "{:.2f}",
        }),
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
