"""
Interactive portal for the Nifty 500 momentum backtest.

Run with:
    streamlit run app.py

Lets you customise the number of stocks held, the rebalancing period, and
the trailing-return lookback period (plus skip period, min price filter,
and whether to enforce point-in-time index membership), and see the
resulting strategy performance vs. the Nifty 500 benchmark update live.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtest_engine import (
    ensure_stock_data,
    load_benchmark,
    load_membership_matrix,
    load_prices,
    perf_stats,
    run_backtest,
    yearly_table,
)
from tax_cost_engine import CostParams, TaxParams, simulate_costs_and_taxes

st.set_page_config(page_title="Nifty 500 Momentum Backtest", layout="wide")

with st.spinner("Fetching price data (first run only)..."):
    ensure_stock_data()


@st.cache_data(show_spinner="Loading stock price history...")
def cached_load_prices(price_col: str) -> pd.DataFrame:
    return load_prices(price_col)


@st.cache_data(show_spinner="Loading benchmark...")
def cached_load_benchmark(price_col: str) -> pd.Series:
    return load_benchmark(price_col)


@st.cache_data(show_spinner="Loading point-in-time membership calendar...")
def cached_load_membership(dates: tuple, symbols: tuple) -> pd.DataFrame:
    return load_membership_matrix(pd.DatetimeIndex(dates), pd.Index(symbols))


def fmt_pct(x: float) -> str:
    return "-" if pd.isna(x) else f"{x:.2%}"


def fmt_ratio(x: float) -> str:
    return "-" if pd.isna(x) else f"{x:.2f}"


st.title("Nifty 500 Momentum Backtest")
st.caption(
    "Cross-sectional momentum: rank stocks by trailing return, go long the top N "
    "equal-weighted, rebalance periodically."
)

with st.sidebar:
    st.header("Strategy parameters")
    n_stocks = st.slider("Number of stocks held", min_value=5, max_value=100, value=30, step=5)
    hold_months = st.slider("Rebalancing period (months)", min_value=1, max_value=12, value=1, step=1)
    lookback_months = st.slider("Trailing return lookback (months)", min_value=1, max_value=24, value=12, step=1)
    skip_months = st.slider(
        "Skip period (months)", min_value=0, max_value=3, value=1, step=1,
        help="Excludes the most recent N months from the lookback, to avoid short-term reversal effects."
    )

    st.header("Universe & data")
    price_col = st.selectbox("Price field", ["Adj Close", "Close"], index=0)
    min_price = st.number_input("Minimum price filter (Rs)", min_value=0.0, value=10.0, step=5.0)
    use_membership_filter = st.checkbox(
        "Enforce point-in-time index membership", value=True,
        help="If off, every stock ever downloaded is treated as always investable "
             "(survivorship-biased). If on, only stocks that were actually Nifty 500 "
             "constituents as of each rebalance date are eligible."
    )
    log_scale = st.checkbox("Log scale equity curve", value=True)

    st.header("Costs & taxes (India)")
    apply_costs_taxes = st.checkbox(
        "Apply transaction costs & capital gains tax", value=False,
        help="Simulates STT, stamp duty, exchange charges, slippage, and STCG/LTCG tax "
             "(with loss carryforward) on every trade implied by the rebalances above. "
             "Approximate -- not tax advice."
    )
    with st.expander("Cost & tax assumptions", expanded=False):
        capital_base = st.number_input(
            "Capital base (Rs)", min_value=100_000.0, value=1_000_000.0, step=100_000.0,
            help="Needed to convert % gains into absolute rupees for the LTCG exemption threshold."
        )
        slippage_pct = st.number_input("Slippage / market impact, one-way (%)", min_value=0.0, value=0.15, step=0.05)
        stt_pct = st.number_input("STT, each side (%)", min_value=0.0, value=0.10, step=0.01)
        stamp_duty_pct = st.number_input("Stamp duty, buy side (%)", min_value=0.0, value=0.015, step=0.005, format="%.3f")
        exchange_pct = st.number_input("Exchange + SEBI charges, each side (%)", min_value=0.0, value=0.0035, step=0.001, format="%.4f")
        brokerage_pct = st.number_input("Brokerage, each side (%)", min_value=0.0, value=0.0, step=0.01)
        gst_pct = st.number_input("GST on brokerage+exchange charges (%)", min_value=0.0, value=18.0, step=1.0)
        stcg_rate_pct = st.number_input("STCG rate, <12mo holding (%)", min_value=0.0, value=20.0, step=1.0)
        ltcg_rate_pct = st.number_input("LTCG rate, >=12mo holding (%)", min_value=0.0, value=12.5, step=0.5)
        ltcg_exemption = st.number_input("LTCG exemption per year (Rs)", min_value=0.0, value=125_000.0, step=25_000.0)
        cess_pct = st.number_input("Health & education cess on tax (%)", min_value=0.0, value=4.0, step=1.0)

monthly_prices = cached_load_prices(price_col)

membership = None
if use_membership_filter:
    membership = cached_load_membership(
        tuple(monthly_prices.index.values), tuple(monthly_prices.columns)
    )

strat_rets, holdings_history = run_backtest(
    monthly_prices, membership, lookback_months, skip_months, hold_months, n_stocks, min_price
)

bench_px = cached_load_benchmark(price_col)
bench_rets = bench_px.pct_change().reindex(strat_rets.index).dropna()
strat_rets = strat_rets.reindex(bench_rets.index)

strat_stats = perf_stats(strat_rets)
bench_stats = perf_stats(bench_rets)

cost_tax_result = None
post_cost_stats = None
post_tax_stats = None
if apply_costs_taxes and len(strat_rets) > 0:
    cost_tax_result = simulate_costs_and_taxes(
        monthly_prices, strat_rets, holdings_history, n_stocks,
        cost=CostParams(
            slippage_pct=slippage_pct, stt_buy_pct=stt_pct, stt_sell_pct=stt_pct,
            stamp_duty_pct=stamp_duty_pct, exchange_charges_pct=exchange_pct,
            brokerage_pct=brokerage_pct, gst_pct=gst_pct,
        ),
        tax=TaxParams(
            capital_base_rs=capital_base, stcg_rate_pct=stcg_rate_pct, ltcg_rate_pct=ltcg_rate_pct,
            ltcg_exemption_rs=ltcg_exemption, cess_pct=cess_pct,
        ),
    )
    post_cost_stats = perf_stats(cost_tax_result["nav_post_cost"].pct_change().dropna())
    post_tax_stats = perf_stats(cost_tax_result["nav_net"].pct_change().dropna())

st.subheader("Performance")
cols = st.columns(5)
labels = ["CAGR", "Volatility", "Sharpe", "MaxDrawdown", "TotalReturn"]
for col, label in zip(cols, labels):
    fmt = fmt_ratio if label == "Sharpe" else fmt_pct
    strat_v = strat_stats[label]
    bench_v = bench_stats[label]
    delta = None
    if label not in ("MaxDrawdown",) and not pd.isna(strat_v) and not pd.isna(bench_v):
        delta = f"{fmt(strat_v - bench_v) if label != 'Sharpe' else fmt_ratio(strat_v - bench_v)} vs bench"
    col.metric(label, fmt(strat_v), delta=delta)
    col.caption(f"Benchmark: {fmt(bench_v)}")

if post_tax_stats is not None:
    st.caption("Gross (pre-cost, pre-tax) shown above. Post-cost and post-cost-and-tax below.")
    cols_pc = st.columns(5)
    for col, label in zip(cols_pc, labels):
        fmt = fmt_ratio if label == "Sharpe" else fmt_pct
        gross_v = strat_stats[label]
        pc_v = post_cost_stats[label]
        delta = None
        if not pd.isna(gross_v) and not pd.isna(pc_v):
            d = pc_v - gross_v
            delta = f"{fmt(d) if label != 'Sharpe' else fmt_ratio(d)} vs gross"
        col.metric(f"{label} (post-cost)", fmt(pc_v), delta=delta)
    cols_pt = st.columns(5)
    for col, label in zip(cols_pt, labels):
        fmt = fmt_ratio if label == "Sharpe" else fmt_pct
        gross_v = strat_stats[label]
        pt_v = post_tax_stats[label]
        delta = None
        if not pd.isna(gross_v) and not pd.isna(pt_v):
            d = pt_v - gross_v
            delta = f"{fmt(d) if label != 'Sharpe' else fmt_ratio(d)} vs gross"
        col.metric(f"{label} (post-tax)", fmt(pt_v), delta=delta)

    st.markdown(
        f"**Total transaction cost paid:** Rs {cost_tax_result['total_cost_rs']:,.0f}  |  "
        f"**Total tax paid:** Rs {cost_tax_result['total_tax_rs']:,.0f}  "
        f"(on a Rs {capital_base:,.0f} capital base)  |  "
        f"**Closed trades:** {len(cost_tax_result['closed_trades'])} "
        f"({cost_tax_result['n_stcg_trades']} short-term, {cost_tax_result['n_ltcg_trades']} long-term)"
    )

if len(strat_rets) == 0:
    st.warning(
        "No months produced a valid portfolio -- likely `Number of stocks held` is larger "
        "than the number of eligible stocks available early in the sample. Try lowering it "
        "or shortening the lookback."
    )
else:
    st.subheader("Equity curve")
    strat_cum = (1 + strat_rets).cumprod()
    bench_cum = (1 + bench_rets).cumprod()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strat_cum.index, y=strat_cum.values, name="Momentum strategy (gross)", line=dict(width=2)))
    if cost_tax_result is not None:
        fig.add_trace(go.Scatter(
            x=cost_tax_result["nav_post_cost"].index, y=cost_tax_result["nav_post_cost"].values,
            name="Post-cost", line=dict(width=2, dash="dash"),
        ))
        fig.add_trace(go.Scatter(
            x=cost_tax_result["nav_net"].index, y=cost_tax_result["nav_net"].values,
            name="Post-cost & tax", line=dict(width=2, dash="dot"),
        ))
    fig.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum.values, name="Nifty 500", line=dict(width=2)))
    fig.update_layout(
        yaxis_type="log" if log_scale else "linear",
        yaxis_title="Growth of Rs 1",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=30, l=10, r=10, b=10),
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Drawdown")
    dd = strat_cum / strat_cum.cummax() - 1
    bench_dd = bench_cum / bench_cum.cummax() - 1
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=dd.index, y=dd.values, name="Momentum strategy", fill="tozeroy"))
    fig_dd.add_trace(go.Scatter(x=bench_dd.index, y=bench_dd.values, name="Nifty 500", line=dict(dash="dot")))
    fig_dd.update_layout(
        yaxis_tickformat=".0%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=30, l=10, r=10, b=10),
        height=300,
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    st.subheader("Year-wise returns and drawdown")
    st.caption(
        "Max drawdown per year is the worst peak-to-trough decline *experienced during* that "
        "year, measured from the running cumulative peak since inception (the peak may have "
        "been set in an earlier year). First/last year may be partial periods."
    )
    yt = yearly_table(strat_rets, bench_rets)

    fig_yr = go.Figure()
    fig_yr.add_trace(go.Bar(x=yt.index, y=yt["strategy_return"], name="Momentum strategy"))
    fig_yr.add_trace(go.Bar(x=yt.index, y=yt["benchmark_return"], name="Nifty 500"))
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

    fig_yr_dd = go.Figure()
    fig_yr_dd.add_trace(go.Bar(x=yt.index, y=yt["strategy_max_drawdown"], name="Momentum strategy"))
    fig_yr_dd.add_trace(go.Bar(x=yt.index, y=yt["benchmark_max_drawdown"], name="Nifty 500"))
    fig_yr_dd.update_layout(
        barmode="group",
        yaxis_title="Max drawdown during year",
        yaxis_tickformat=".0%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=30, l=10, r=10, b=10),
        height=350,
        xaxis=dict(type="category"),
    )
    st.plotly_chart(fig_yr_dd, use_container_width=True)

    with st.expander("Year-wise table"):
        st.dataframe(yt.style.format("{:.2%}"), height=300)
        st.download_button(
            "Download year-wise table as CSV",
            yt.to_csv().encode("utf-8"),
            file_name="momentum_backtest_yearly.csv",
            mime="text/csv",
        )

    if cost_tax_result is not None:
        with st.expander("Trade log (costs & taxes)"):
            trades = cost_tax_result["closed_trades"].copy()
            if not trades.empty:
                trades["hold_months"] = (trades["hold_days"] / 30.44).round(1)
                trades["gain_pct_of_capital"] = trades["rupee_gain"] / capital_base
                st.dataframe(
                    trades[["symbol", "entry_date", "exit_date", "hold_months", "is_ltcg",
                            "rupee_gain", "gain_pct_of_capital"]]
                    .sort_values("exit_date")
                    .style.format({
                        "rupee_gain": "Rs {:,.0f}",
                        "gain_pct_of_capital": "{:.2%}",
                        "hold_months": "{:.1f}",
                    }),
                    height=300,
                )
                st.download_button(
                    "Download trade log as CSV",
                    trades.to_csv(index=False).encode("utf-8"),
                    file_name="momentum_backtest_trades.csv",
                    mime="text/csv",
                )
            st.caption("Tax paid by financial year-end (Rs):")
            tax_by_year = cost_tax_result["tax_paid"][cost_tax_result["tax_paid"] > 0]
            if not tax_by_year.empty:
                st.dataframe(tax_by_year.to_frame("tax_paid_rs").style.format("Rs {:,.0f}"), height=200)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.subheader("Current holdings")
        if holdings_history:
            last_date, last_holdings = holdings_history[-1]
            st.caption(f"As of {last_date.date()} ({len(last_holdings)} stocks, equal-weighted)")
            st.dataframe(pd.DataFrame({"Symbol": last_holdings}), hide_index=True, height=300)
        else:
            st.caption("No rebalance has occurred yet with these parameters.")

    with col_b:
        st.subheader("Data coverage")
        st.write(f"**Stocks loaded:** {monthly_prices.shape[1]}")
        st.write(f"**Date range:** {monthly_prices.index.min().date()} to {monthly_prices.index.max().date()}")
        if membership is not None:
            avg_eligible = membership.sum(axis=1).replace(0, np.nan).mean()
            st.write(f"**Avg. eligible stocks/month (point-in-time):** {avg_eligible:.0f}")
        st.write(f"**Rebalances:** {len(holdings_history)}")

    with st.expander("Monthly returns (strategy vs benchmark)"):
        out = pd.DataFrame({"strategy_return": strat_rets, "benchmark_return": bench_rets})
        st.dataframe(out.style.format("{:.2%}"), height=300)
        st.download_button(
            "Download monthly returns as CSV",
            out.to_csv().encode("utf-8"),
            file_name="momentum_backtest_results.csv",
            mime="text/csv",
        )

st.divider()
st.caption(
    "Point-in-time membership calendar is best-effort: authoritative (NSE-published) for "
    "1998-2020, approximated from periodic snapshot diffs for 2020-2026. ~25% of historical "
    "constituent names could not be resolved to a live ticker -- those are almost entirely "
    "companies that were delisted, merged, or liquidated, with no tradable price series "
    "available anywhere. See build_membership_calendar.py for details."
)
