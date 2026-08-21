"""
Standalone portal for the EMA + relative-strength "N-of-3 vote" ETF
rotation strategy. Single-purpose app, separate from the main
nifty500-momentum multi-page portal -- run this file directly and only
this strategy's page loads.

Run with:
    streamlit run ema_rs_portal/app.py

See ../ema_rs_rotation_engine.py for the full strategy mechanics: three
criteria (equity leg's spot index vs a long EMA, vs a short EMA, and the
leg's ETF vs GOLDBEES relative strength) checked on the last trading day
of each month, with the rebalance executed on the following month's first
trading day, only if the target changed. The equity leg (Nifty 50,
Midcap 150, Smallcap 250, or Nifty 500) is selectable below.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ema_rs_rotation_engine import (
    DEFAULT_EQUITY_LEG,
    EMA_LONG,
    EMA_SHORT,
    EQUITY_LEGS,
    MIN_VOTES,
    RS_LOOKBACK,
    load_data,
    perf_stats,
    run_backtest,
    yearly_table,
)

st.set_page_config(page_title="EMA+RS ETF Rotation Backtest", layout="wide")


@st.cache_data(show_spinner="Loading spot index / ETF price history...")
def cached_load_data(equity_leg: str, price_col: str):
    return load_data(equity_leg, price_col)


def fmt_pct(x: float) -> str:
    return "-" if pd.isna(x) else f"{x:.2%}"


def fmt_ratio(x: float) -> str:
    return "-" if pd.isna(x) else f"{x:.2f}"


st.title("EMA + Relative-Strength ETF Rotation Backtest")
st.caption(
    "Rotates 100% between a chosen equity ETF and GOLDBEES. Three criteria are checked on the "
    "last trading day of each month: (1) the equity leg's spot index above its long EMA, "
    "(2) above its short EMA, (3) the equity ETF's trailing return beats GOLDBEES's over the RS "
    "lookback. If at least the threshold number agree, the target is the equity ETF; otherwise "
    "GOLDBEES. The rebalance only executes on the first trading day of the following month, and "
    "only if the target differs from the current holding."
)

with st.sidebar:
    st.header("Equity leg")
    leg_key = st.selectbox(
        "Rotate against GOLDBEES using",
        list(EQUITY_LEGS.keys()),
        format_func=lambda k: EQUITY_LEGS[k]["label"],
        index=list(EQUITY_LEGS.keys()).index(DEFAULT_EQUITY_LEG),
    )
    leg = EQUITY_LEGS[leg_key]
    st.caption(leg["note"])

    st.header("Strategy parameters")
    ema_long = st.slider("Long EMA period (days)", min_value=50, max_value=400, value=EMA_LONG, step=10)
    ema_short = st.slider("Short EMA period (days)", min_value=5, max_value=200, value=EMA_SHORT, step=5)
    if ema_short >= ema_long:
        st.warning("Short EMA period is >= long EMA period -- the two EMA criteria will move almost in lockstep.")
    rs_lookback = st.slider(
        "Relative-strength lookback (trading days)", min_value=10, max_value=300, value=RS_LOOKBACK, step=1,
        help=f"{leg['etf_symbol']}'s trailing return over this many trading days is compared against "
             "GOLDBEES's trailing return over the same window."
    )
    min_votes = st.radio(
        f"Votes needed to hold {leg['etf_symbol']}", [1, 2, 3], index=MIN_VOTES - 1, horizontal=True,
        help=f"Out of the 3 criteria above. 1 = hold {leg['etf_symbol']} unless all 3 favour GOLDBEES "
             f"(aggressive). 3 = only hold {leg['etf_symbol']} when every criterion agrees (defensive, "
             "more time in GOLDBEES)."
    )

    st.header("Data & display")
    price_col = st.selectbox("Price field", ["Close", "Adj Close"], index=0)
    log_scale = st.checkbox("Log scale equity curve", value=True)

spot, prices, equity_symbol = cached_load_data(leg_key, price_col)

with st.spinner("Running backtest..."):
    strat_rets, signal_log, diagnostics = run_backtest(
        spot, prices, equity_symbol, ema_long, ema_short, rs_lookback, min_votes
    )

bench_px = prices[equity_symbol]
bench_rets = bench_px.pct_change().reindex(strat_rets.index).dropna()
strat_rets = strat_rets.reindex(bench_rets.index)

if len(strat_rets) == 0:
    st.warning(
        "No valid decisions were produced with these parameters -- the RS lookback is likely "
        "longer than the available price history for this leg. Try a shorter lookback."
    )
    st.stop()

n_years = (bench_rets.index[-1] - bench_rets.index[0]).days / 365.25
if n_years < 5:
    st.info(
        f"**{leg['label']}** only has ~{n_years:.1f} years of ETF history ({bench_px.index.min().date()} "
        f"to {bench_px.index.max().date()}) -- {leg['note']} Treat these results as indicative, not a "
        "long-run track record."
    )

strat_stats = perf_stats(strat_rets)
bench_stats = perf_stats(bench_rets)

st.subheader("Performance")
cols = st.columns(5)
labels = ["CAGR", "Volatility", "Sharpe", "MaxDrawdown", "TotalReturn"]
for col, label in zip(cols, labels):
    fmt = fmt_ratio if label == "Sharpe" else fmt_pct
    strat_v = strat_stats[label]
    bench_v = bench_stats[label]
    delta = None
    if label != "MaxDrawdown" and not pd.isna(strat_v) and not pd.isna(bench_v):
        d = strat_v - bench_v
        delta = f"{fmt(d)} vs bench"
    col.metric(label, fmt(strat_v), delta=delta)
    col.caption(f"{equity_symbol}: {fmt(bench_v)}")

st.subheader("Equity curve")
strat_cum = (1 + strat_rets).cumprod()
bench_cum = (1 + bench_rets).cumprod()
fig = go.Figure()
fig.add_trace(go.Scatter(x=strat_cum.index, y=strat_cum.values, name="EMA+RS strategy", line=dict(width=2)))
fig.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum.values, name=f"{equity_symbol} buy & hold", line=dict(width=2)))
fig.update_layout(
    yaxis_type="log" if log_scale else "linear",
    yaxis_title="Growth of Rs 1",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(t=30, l=10, r=10, b=10),
    height=450,
)
st.plotly_chart(fig, use_container_width=True)

col_dd, col_spot = st.columns(2)
with col_dd:
    st.subheader("Drawdown")
    dd = strat_cum / strat_cum.cummax() - 1
    bench_dd = bench_cum / bench_cum.cummax() - 1
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=dd.index, y=dd.values, name="Strategy", fill="tozeroy"))
    fig_dd.add_trace(go.Scatter(x=bench_dd.index, y=bench_dd.values, name=equity_symbol, line=dict(dash="dot")))
    fig_dd.update_layout(
        yaxis_tickformat=".0%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=30, l=10, r=10, b=10),
        height=350,
    )
    st.plotly_chart(fig_dd, use_container_width=True)

with col_spot:
    st.subheader(f"{leg['label']} spot vs {ema_long}/{ema_short}-day EMA")
    fig_spot = go.Figure()
    fig_spot.add_trace(go.Scatter(x=diagnostics.index, y=diagnostics["spot"], name="Spot index", line=dict(width=1.5)))
    fig_spot.add_trace(go.Scatter(x=diagnostics.index, y=diagnostics["ema250"], name=f"{ema_long}-day EMA", line=dict(width=1.5)))
    fig_spot.add_trace(go.Scatter(x=diagnostics.index, y=diagnostics["ema50"], name=f"{ema_short}-day EMA", line=dict(width=1.5)))
    fig_spot.update_layout(
        yaxis_type="log" if log_scale else "linear",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=30, l=10, r=10, b=10),
        height=350,
    )
    st.plotly_chart(fig_spot, use_container_width=True)

st.subheader("Calendar-year returns")
yt = yearly_table(strat_rets, bench_rets)
fig_yr = go.Figure()
fig_yr.add_trace(go.Bar(x=yt.index, y=yt["strategy_return"], name="Strategy"))
fig_yr.add_trace(go.Bar(x=yt.index, y=yt["benchmark_return"], name=equity_symbol))
fig_yr.update_layout(
    barmode="group",
    yaxis_title="Calendar-year return",
    yaxis_tickformat=".0%",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(t=30, l=10, r=10, b=10),
    height=350,
    xaxis=dict(type="category"),
)
st.plotly_chart(fig_yr, use_container_width=True)

with st.expander("Year-wise table"):
    st.dataframe(yt.style.format("{:.2%}"), height=300)
    st.download_button(
        "Download year-wise table as CSV",
        yt.to_csv().encode("utf-8"),
        file_name=f"ema_rs_rotation_{leg_key.lower()}_yearly.csv",
        mime="text/csv",
    )

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Current holding")
    last_holding = diagnostics["holding"].iloc[-1]
    last_date = diagnostics.index[-1]
    last_votes = int(diagnostics["vote_count"].iloc[-1])
    st.info(f"As of {last_date.date()}: **{last_holding}** ({last_votes}/3 criteria currently favour {equity_symbol})")
    st.write(f"**Long EMA criterion:** {'above' if diagnostics['crit_ema250'].iloc[-1] else 'below'} the {ema_long}-day EMA")
    st.write(f"**Short EMA criterion:** {'above' if diagnostics['crit_ema50'].iloc[-1] else 'below'} the {ema_short}-day EMA")
    rs_state = diagnostics["crit_rs"].iloc[-1]
    st.write(f"**Relative strength:** {equity_symbol} {'beating' if rs_state else 'lagging'} GOLDBEES over {rs_lookback} days")

with col_b:
    st.subheader("Data coverage")
    st.write(f"**{leg['label']} spot history:** {spot.index.min().date()} to {spot.index.max().date()}")
    st.write(f"**{equity_symbol} ETF history:** {bench_px.index.min().date()} to {bench_px.index.max().date()}")
    st.write(f"**Position flips:** {len(signal_log)}")
    st.write(f"**Backtest days:** {len(strat_rets)}")

st.subheader("Signal log")
st.caption("Every executed position change: the month-end date the criteria were checked, and the next month's first trading day it was acted on.")
if signal_log.empty:
    st.caption("No position flips with these parameters.")
else:
    display_log = signal_log.copy()
    display_log["decision_date"] = display_log["decision_date"].dt.date
    display_log["exec_date"] = display_log["exec_date"].dt.date
    st.dataframe(
        display_log[["decision_date", "exec_date", "position", "votes", "crit_ema250", "crit_ema50", "crit_rs"]]
        .sort_values("exec_date", ascending=False),
        hide_index=True, height=400,
    )
    st.download_button(
        "Download signal log as CSV",
        signal_log.to_csv(index=False).encode("utf-8"),
        file_name=f"ema_rs_rotation_{leg_key.lower()}_signal_log.csv",
        mime="text/csv",
    )

with st.expander("Daily returns (strategy vs benchmark)"):
    out = pd.DataFrame({"strategy_return": strat_rets, "benchmark_return": bench_rets})
    st.dataframe(out.style.format("{:.2%}"), height=300)
    st.download_button(
        "Download daily returns as CSV",
        out.to_csv().encode("utf-8"),
        file_name=f"ema_rs_rotation_{leg_key.lower()}_results.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "No slippage, brokerage, or transaction costs are deducted. Spot indices are used for the two EMA "
    "criteria on their own (often longer) trading calendar, so EMAs are as warmed-up as the data allows "
    "before the ETF itself has price history -- the RS lookback and the ETF's own inception are what "
    "actually delay the first live decision. Nifty 50 and Midcap 150 have Nippon 'BEES' ETFs; Smallcap "
    "250 and Nifty 500 use the longest-history tradable ETF found for that index (HDFC and Motilal Oswal "
    "respectively), both of which only started trading in 2023. Known Yahoo Finance vendor data glitches "
    "in the underlying series (e.g. GOLDBEES/NIFTYBEES/BANKBEES on 2019-12-19/20, NIFTYBEES on "
    "2010-10-06) are repaired via interpolation before the backtest runs."
)
