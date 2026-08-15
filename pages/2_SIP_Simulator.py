"""
SIP simulator: compares a plain fixed-amount monthly SIP into the momentum
strategy against a dynamic version that splits contributions between the
strategy and a liquid fund (LIQUIDBEES), tactically going all-in on the
strategy during drawdowns. See sip_engine.py for the exact mechanics and a
documented ambiguity in the source spec (whether exiting "aggressive" mode
force-rebalances existing holdings back to the target split -- currently:
no, only future contributions change).
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtest_engine import apply_execution_lag, ensure_stock_data, run_backtest
from sip_engine import simulate_dynamic_sip, simulate_plain_sip
from streamlit_cache import (
    cached_load_daily_prices,
    cached_load_liquid,
    cached_load_membership,
    cached_load_prices,
)

st.set_page_config(page_title="Nifty 500 SIP Simulator", layout="wide")

with st.spinner("Fetching price data (first run only)..."):
    ensure_stock_data()

st.title("SIP Simulator")
st.caption(
    "Compares a plain monthly SIP into the momentum strategy against a dynamic version "
    "that splits between the strategy and a liquid fund, going all-in on the strategy "
    "during drawdowns. Runs on the core momentum strategy (rank/select/exit-band/T+1 "
    "execution) -- stoploss and the gold regime filter from the Backtest page aren't "
    "included here yet."
)

with st.sidebar:
    st.header("Strategy parameters")
    n_stocks = st.slider("Number of stocks held", min_value=5, max_value=100, value=10, step=5)
    hold_months = st.slider("Rebalancing period (months)", min_value=1, max_value=12, value=1, step=1)
    lookback_months = st.slider("Trailing return lookback (months)", min_value=1, max_value=24, value=10, step=1)
    skip_months = st.slider(
        "Skip period (months)", min_value=0, max_value=3, value=1, step=1,
        help="Excludes the most recent N months from the lookback, to avoid short-term reversal effects."
    )
    use_exit_band = st.checkbox(
        "Custom exit criteria", value=True,
        help="Off: a held stock exits as soon as its rank falls out of the top N. On: a held "
             "stock is only exited once its rank falls below N x (1 + exit%)."
    )
    exit_band_pct = 0.0
    if use_exit_band:
        exit_band_pct = st.slider("Exit threshold (% beyond N)", min_value=0, max_value=300, value=80, step=10)
        exit_rank_preview = int(round(n_stocks * (1 + exit_band_pct / 100)))
        st.caption(f"With {n_stocks} stocks and {exit_band_pct}%, exit rank is {exit_rank_preview}.")

    use_execution_lag = st.checkbox(
        "T+1 execution (signal at month-end close, trade at next open)", value=False,
        help="Realistic execution timing: the rebalance ranking is still computed from the "
             "month-end close (unchanged), but entries and exits are executed on the NEXT "
             "trading day. A stock being dropped is held (and keeps accruing return) through "
             "that next day's open, where it's sold; a stock being added is bought at that "
             "open, so it only starts accruing return from there. Requires daily price data "
             "(a bit slower to compute)."
    )

    st.header("Universe & data")
    price_col = st.selectbox("Price field", ["Adj Close", "Close"], index=0)
    min_price = st.number_input("Minimum price filter (Rs)", min_value=0.0, value=10.0, step=5.0)
    use_membership_filter = st.checkbox("Enforce point-in-time index membership", value=True)

    st.header("SIP parameters")
    sip_amount = st.number_input("Monthly SIP amount (Rs)", min_value=500.0, value=10000.0, step=500.0)
    strategy_alloc_pct = st.slider(
        "Normal-mode allocation to strategy (%)", min_value=10, max_value=100, value=70, step=5,
        help="The rest goes to the liquid fund (LIQUIDBEES). E.g. 70 means Rs 7,000 of a "
             "Rs 10,000 SIP goes to the strategy, Rs 3,000 to the liquid fund."
    )
    drawdown_trigger_pct = st.slider(
        "Drawdown that triggers full allocation (%)", min_value=1, max_value=50, value=15, step=1,
        help="When the strategy's own NAV falls this far below its running all-time high, the "
             "entire liquid fund balance converts into the strategy and 100% of new SIP goes "
             "to the strategy too."
    )
    recovery_pct = st.slider(
        "Recovery above previous ATH to revert (%)", min_value=1, max_value=100, value=20, step=1,
        help="Stays in full-allocation mode until the strategy's NAV climbs this far above the "
             "PREVIOUS all-time high (the peak that was being drawn down from) -- not any new "
             "high made during the recovery itself."
    )

monthly_prices = cached_load_prices(price_col)
membership = None
if use_membership_filter:
    membership = cached_load_membership(tuple(monthly_prices.index.values), tuple(monthly_prices.columns))
liquid_px = cached_load_liquid(price_col)

strat_rets, holdings_history = run_backtest(
    monthly_prices, membership, lookback_months, skip_months, hold_months, n_stocks, min_price,
    use_exit_band, exit_band_pct,
)

if use_execution_lag and len(strat_rets) > 0:
    daily_close, daily_open = cached_load_daily_prices(price_col)
    exec_lag_overlay = apply_execution_lag(monthly_prices, daily_close, daily_open, holdings_history)
    for m_date, r in exec_lag_overlay.items():
        if m_date in strat_rets.index:
            strat_rets.loc[m_date] = r

if len(strat_rets) == 0:
    st.warning(
        "No months produced a valid portfolio with these parameters -- try lowering the "
        "number of stocks held or shortening the lookback."
    )
    st.stop()

liquid_rets = liquid_px.pct_change().reindex(strat_rets.index)

plain = simulate_plain_sip(strat_rets, sip_amount)
dynamic = simulate_dynamic_sip(
    strat_rets, liquid_rets, sip_amount, strategy_alloc_pct, drawdown_trigger_pct, recovery_pct
)


def fmt_rs(x: float) -> str:
    return f"Rs {x:,.0f}"


def fmt_pct(x: float) -> str:
    return "-" if pd.isna(x) else f"{x:.2%}"


st.subheader("Results")
cols = st.columns(4)
cols[0].metric("Total invested", fmt_rs(plain["total_invested"]))
cols[1].metric("Plain SIP final value", fmt_rs(plain["final_value"]), delta=fmt_pct(plain["xirr"]) + " XIRR")
cols[2].metric("Dynamic SIP final value", fmt_rs(dynamic["final_value"]), delta=fmt_pct(dynamic["xirr"]) + " XIRR")
diff = dynamic["final_value"] - plain["final_value"]
cols[3].metric("Dynamic vs Plain", fmt_rs(diff), delta=f"{diff / plain['final_value']:.1%}" if plain["final_value"] else None)

st.caption(
    f"Dynamic mode spent {(dynamic['state'] == 'aggressive').sum()} of {len(dynamic['state'])} months "
    f"in full-allocation mode, across {len(dynamic['transitions']) // 2} trigger/recovery cycle(s) "
    f"(a cycle may be incomplete if still in aggressive mode at the end of the backtest)."
)

st.subheader("Portfolio value over time")
fig = go.Figure()
fig.add_trace(go.Scatter(x=plain["invested"].index, y=plain["invested"].values,
                          name="Total invested", line=dict(dash="dot", color="gray")))
fig.add_trace(go.Scatter(x=plain["value"].index, y=plain["value"].values, name="Plain SIP"))
fig.add_trace(go.Scatter(x=dynamic["value"].index, y=dynamic["value"].values, name="Dynamic SIP"))
fig.update_layout(
    yaxis_type="log", yaxis_title="Portfolio value (Rs, log scale)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(t=30, l=10, r=10, b=10), height=450,
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Drawdown")
strat_cum = (1 + strat_rets).cumprod()
strat_dd = strat_cum / strat_cum.cummax() - 1
plain_value_dd = plain["value"] / plain["value"].cummax() - 1
dynamic_value_dd = dynamic["value"] / dynamic["value"].cummax() - 1

dd_cols = st.columns(3)
dd_cols[0].metric("Strategy NAV max drawdown", fmt_pct(strat_dd.min()))
dd_cols[1].metric("Plain SIP portfolio max drawdown", fmt_pct(plain_value_dd.min()))
dd_cols[2].metric("Dynamic SIP portfolio max drawdown", fmt_pct(dynamic_value_dd.min()))
st.caption(
    "Strategy NAV drawdown is the underlying strategy's own price-based drawdown (what "
    "drives the dynamic mode's trigger, dashed line below). Portfolio drawdown is each SIP's "
    "actual invested value relative to ITS OWN running peak -- the more relevant number for "
    "'how underwater was my actual money.' Early in a SIP, ongoing contributions can make this "
    "noticeably shallower than the pure NAV drawdown; late in a long-running SIP, once "
    "compounded value dwarfs the monthly contribution, the two converge. Note the dynamic "
    "mode's goal isn't necessarily a shallower portfolio drawdown than plain SIP -- going "
    "further into the strategy during a fall doesn't reduce how far it falls, it's a bet on "
    "the recovery. Compare the two lines below rather than assuming dynamic is 'safer.'"
)

fig_dd1 = go.Figure()
fig_dd1.add_trace(go.Scatter(x=strat_dd.index, y=strat_dd.values, name="Strategy NAV drawdown", fill="tozeroy"))
fig_dd1.add_hline(
    y=-drawdown_trigger_pct / 100, line_dash="dash", line_color="red",
    annotation_text=f"trigger ({-drawdown_trigger_pct}%)", annotation_position="bottom right",
)
fig_dd1.update_layout(
    yaxis_tickformat=".0%", yaxis_title="Drawdown",
    margin=dict(t=30, l=10, r=10, b=10), height=300,
)
st.plotly_chart(fig_dd1, use_container_width=True)

fig_dd2 = go.Figure()
fig_dd2.add_trace(go.Scatter(x=plain_value_dd.index, y=plain_value_dd.values, name="Plain SIP"))
fig_dd2.add_trace(go.Scatter(x=dynamic_value_dd.index, y=dynamic_value_dd.values, name="Dynamic SIP"))
fig_dd2.update_layout(
    yaxis_tickformat=".0%", yaxis_title="Portfolio value drawdown",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(t=30, l=10, r=10, b=10), height=300,
)
st.plotly_chart(fig_dd2, use_container_width=True)

col_a, col_b = st.columns([1, 1])
with col_a:
    st.subheader("Dynamic SIP: strategy vs liquid split")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=dynamic["strategy_value"].index, y=dynamic["strategy_value"].values,
                               name="In strategy", stackgroup="one"))
    fig2.add_trace(go.Scatter(x=dynamic["liquid_value"].index, y=dynamic["liquid_value"].values,
                               name="In liquid fund", stackgroup="one"))
    fig2.update_layout(
        yaxis_title="Rs", legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=30, l=10, r=10, b=10), height=350,
    )
    st.plotly_chart(fig2, use_container_width=True)

with col_b:
    st.subheader("Trigger/recovery timeline")
    if dynamic["transitions"]:
        tdf = pd.DataFrame(dynamic["transitions"], columns=["Date", "Event"])
        tdf["Date"] = tdf["Date"].dt.date
        st.dataframe(tdf, hide_index=True, height=350)
    else:
        st.caption("No drawdown large enough to trigger full allocation with these parameters.")

st.divider()
st.caption(
    "Both simulations use a units/NAV model like a real mutual fund SIP: each contribution "
    "buys units at that month's NAV, so total value = units held x NAV. The drawdown/all-time-"
    "high tracking for the trigger uses the strategy's own price path only, independent of "
    "contribution timing. XIRR is computed via Newton-Raphson on the full contribution + "
    "final-value cashflow series (Act/365 day count)."
)
