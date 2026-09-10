"""
SIP simulator: a plain fixed-amount monthly SIP into the momentum
strategy. Optionally (off by default) adds a Dynamic SIP comparison that
splits contributions between the strategy and a liquid fund (LIQUIDBEES),
tactically going all-in on the strategy during drawdowns. See sip_engine.py
for the exact mechanics and a documented ambiguity in the source spec
(whether exiting "aggressive" mode force-rebalances existing holdings back
to the target split -- currently: no, only future contributions change).

Optionally overlays two-stage MTF leverage (see leverage_engine.py) on the
underlying strategy's own returns, applied BEFORE all of the above -- so
Plain SIP, Dynamic SIP, and the Lumpsum sleeve all reflect it uniformly
when enabled.

Optionally also runs the SAME plain SIP mechanics into one curated Direct
Growth mutual fund scheme (see backtest_engine.MUTUAL_FUNDS), shown
alongside Plain/Dynamic SIP for reference -- a hand-picked list, not
exhaustive or AUM-ranked.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtest_engine import (
    MUTUAL_FUNDS,
    NSE_UNIVERSES,
    apply_execution_lag,
    ensure_mutual_fund_data,
    ensure_stock_data,
    run_backtest,
)
from leverage_engine import apply_mtf_leverage
from sip_engine import simulate_dynamic_sip, simulate_lumpsum_on_drawdown, simulate_plain_sip
from streamlit_cache import (
    cached_load_daily_prices,
    cached_load_liquid,
    cached_load_membership,
    cached_load_mutual_fund_nav,
    cached_load_prices,
    cached_load_universe_symbols,
)

st.set_page_config(page_title="Nifty 500 SIP Simulator", layout="wide")

with st.spinner("Fetching price data (first run only)..."):
    ensure_stock_data()
    ensure_mutual_fund_data()

st.title("SIP Simulator")
st.caption(
    "A plain monthly SIP into the momentum strategy. Optionally (enable in the sidebar) "
    "compare against a Dynamic SIP that splits between the strategy and a liquid fund, going "
    "all-in on the strategy during drawdowns. Runs on the core momentum strategy "
    "(rank/select/exit-band/T+1 execution) -- stoploss and the gold regime filter from the "
    "Backtest page aren't included here yet."
)

with st.sidebar:
    st.header("Universe")
    universe_options = list(NSE_UNIVERSES.keys())
    universe_label = st.selectbox(
        "Universe", universe_options, index=universe_options.index("Nifty 500"),
        help="Restricts stock selection to one NSE index's CURRENT constituents (a snapshot, "
             "not point-in-time -- NSE doesn't publish historical inclusion/exclusion logs for "
             "these indices the way it does for the Nifty 500). Nifty 500 uses the point-in-time "
             "membership calendar below instead when that toggle is on, so picking it here "
             "doesn't restrict anything further."
    )
    allowed_symbols = cached_load_universe_symbols(universe_label) if universe_label != "Nifty 500" else None
    if allowed_symbols is not None:
        st.caption(f"Restricted to the current {len(allowed_symbols)} constituents of {universe_label}.")

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

    use_custom_start = st.checkbox(
        "Custom start date", value=False,
        help="Shows what the SIP would look like if you'd started contributing on a specific "
             "date, instead of from the earliest available history. The ranking lookback still "
             "uses real price history from before this date. Note: the drawdown/all-time-high "
             "reference used by the dynamic SIP and lumpsum triggers below also resets fresh at "
             "this date (it doesn't know about any peak that existed before you started), "
             "consistent with everything else on this page."
    )
    custom_start_date = None
    if use_custom_start:
        custom_start_date = st.date_input(
            "Start date", value=pd.Timestamp.today() - pd.DateOffset(years=5),
        )

    st.header("MTF Leverage Overlay")
    use_leverage = st.checkbox(
        "Enable MTF leverage overlay", value=False,
        help="Optional, two-stage drawdown-triggered leverage on the underlying momentum "
             "strategy's own returns -- applied BEFORE the SIP/Dynamic SIP/Lumpsum simulations "
             "below, so all of them reflect the leveraged returns uniformly when this is on "
             "(Dynamic SIP's own drawdown trigger will then react to the leveraged, larger "
             "swings too)."
    )
    if use_leverage:
        lev_trigger1_pct = st.slider(
            "Leg 1 trigger: month return <= -X%", min_value=1.0, max_value=30.0, value=7.0, step=0.5,
        )
        lev_leg1_pct = st.slider(
            "Leg 1 leverage added (%)", min_value=5.0, max_value=200.0, value=50.0, step=5.0,
        )
        lev_trigger2_pct = st.slider(
            "Leg 2 trigger: another month return <= -X% within 2 months of leg 1", min_value=1.0,
            max_value=30.0, value=5.0, step=0.5,
        )
        lev_leg2_pct = st.slider(
            "Leg 2 leverage added (%)", min_value=5.0, max_value=200.0, value=50.0, step=5.0,
        )
        lev_hold_mode_label = st.radio(
            "Hold duration", ["Fixed number of months", "Until recovery above previous ATH"], index=0,
        )
        lev_hold_mode = "fixed_months" if lev_hold_mode_label.startswith("Fixed") else "ath_recovery"
        lev_hold_months = 6
        lev_recovery_pct = 0.0
        if lev_hold_mode == "fixed_months":
            lev_hold_months = st.slider("Hold duration (months)", min_value=1, max_value=36, value=6, step=1)
        else:
            lev_recovery_pct = st.slider(
                "Recovery above previous ATH to revert (%)", min_value=0.0, max_value=50.0, value=0.0, step=1.0,
            )
        lev_annual_interest_pct = st.number_input(
            "MTF interest rate (% per year)", min_value=0.0, value=10.0, step=0.5,
        )

    st.header("SIP parameters")
    sip_amount = st.number_input("Monthly SIP amount (Rs)", min_value=500.0, value=10000.0, step=500.0)

    st.header("Dynamic SIP")
    use_dynamic = st.checkbox(
        "Enable Dynamic SIP", value=False,
        help="Optional: a version that splits contributions between the strategy and a liquid "
             "fund (LIQUIDBEES) normally, tactically going all-in on the strategy during "
             "drawdowns. Off by default -- Plain SIP alone is shown unless you turn this on."
    )
    strategy_alloc_pct = 70
    drawdown_trigger_pct = 15
    recovery_pct = 20
    if use_dynamic:
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

    st.header("Lumpsum on drawdown")
    use_lumpsum = st.checkbox(
        "Add opportunistic lumpsum", value=False,
        help="An independent sleeve of money, separate from the SIP above: invests a fixed "
             "lumpsum into the strategy each time it dips below its all-time high by a "
             "threshold, then waits for a confirmed recovery before it's willing to fire again. "
             "Tracked on its own, and also shown combined with each SIP mode for reference."
    )
    lumpsum_amount = 100000.0
    lumpsum_drawdown_pct = 15.0
    lumpsum_reset_pct = 0.0
    if use_lumpsum:
        lumpsum_amount = st.number_input("Lumpsum amount per trigger (Rs)", min_value=1000.0, value=100000.0, step=5000.0)
        lumpsum_drawdown_pct = st.slider(
            "Drawdown that triggers a lumpsum (%)", min_value=1, max_value=50, value=15, step=1,
            help="Invests the lumpsum the first time the strategy's NAV falls this far below "
                 "its running all-time high."
        )
        lumpsum_reset_pct = st.slider(
            "Recovery above pre-drawdown peak to re-arm (%)", min_value=0, max_value=100, value=0, step=1,
            help="0 = ready to trigger again as soon as the strategy merely reclaims its old "
                 "high. Higher = requires a confirmed rally past the old peak by this much "
                 "before it's willing to fire on the next dip."
        )

    st.header("Compare against a mutual fund")
    fund_choice = st.selectbox(
        "Mutual fund (optional)", ["None"] + list(MUTUAL_FUNDS.keys()),
        format_func=lambda n: n if n == "None" else f"[{MUTUAL_FUNDS[n]['category']}] {n}",
        help="Runs the SAME plain monthly SIP (same amount, same cadence) into one Direct "
             "Growth mutual fund scheme instead of the strategy, shown alongside Plain/Dynamic "
             "SIP for reference. A hand-picked list of well-known funds (not exhaustive or "
             "AUM-ranked), Direct Growth only (not Regular, which carries distributor "
             "commission drag). Source: api.mfapi.in. If the fund's history is shorter than the "
             "comparison window, its SIP simply starts later."
    )
    compare_fund = fund_choice if fund_choice != "None" else None

monthly_prices = cached_load_prices(price_col)
membership = None
if use_membership_filter:
    membership = cached_load_membership(tuple(monthly_prices.index.values), tuple(monthly_prices.columns))
liquid_px = cached_load_liquid(price_col)

strat_rets, holdings_history = run_backtest(
    monthly_prices, membership, lookback_months, skip_months, hold_months, n_stocks, min_price,
    use_exit_band, exit_band_pct,
    allowed_symbols=allowed_symbols,
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

leverage_result = None
if use_leverage:
    leverage_result = apply_mtf_leverage(
        strat_rets, lev_trigger1_pct, lev_leg1_pct, lev_trigger2_pct, lev_leg2_pct,
        lev_hold_mode, lev_hold_months, lev_recovery_pct, lev_annual_interest_pct,
    )
    strat_rets = leverage_result["leveraged_rets"]

if custom_start_date is not None:
    start_ts = pd.Timestamp(custom_start_date)
    strat_rets = strat_rets[strat_rets.index >= start_ts]
    holdings_history = [(d, h) for d, h in holdings_history if d >= start_ts]
    if len(strat_rets) == 0:
        st.warning(
            f"No months fall on or after {start_ts.date()} -- pick an earlier start date, "
            "or the strategy's data doesn't extend that recently."
        )
        st.stop()
    st.info(f"Showing results as if the SIP started on **{start_ts.date()}** ({len(strat_rets)} months since).")

liquid_rets = liquid_px.pct_change().reindex(strat_rets.index)

plain = simulate_plain_sip(strat_rets, sip_amount)
dynamic = None
if use_dynamic:
    dynamic = simulate_dynamic_sip(
        strat_rets, liquid_rets, sip_amount, strategy_alloc_pct, drawdown_trigger_pct, recovery_pct
    )
lumpsum = None
if use_lumpsum:
    lumpsum = simulate_lumpsum_on_drawdown(strat_rets, lumpsum_amount, lumpsum_drawdown_pct, lumpsum_reset_pct)

fund_sip = None
if compare_fund is not None:
    fund_nav = cached_load_mutual_fund_nav(compare_fund).resample("ME").last()
    fund_rets = fund_nav.pct_change().dropna()
    if custom_start_date is not None:
        fund_rets = fund_rets[fund_rets.index >= pd.Timestamp(custom_start_date)]
    if len(fund_rets) > 1:
        fund_sip = simulate_plain_sip(fund_rets, sip_amount)
    else:
        st.sidebar.warning(f"{compare_fund}: no data in the selected window.")


def fmt_rs(x: float) -> str:
    return f"Rs {x:,.0f}"


def fmt_pct(x: float) -> str:
    return "-" if pd.isna(x) else f"{x:.2%}"


st.subheader("Results")
cols = st.columns(3 if use_dynamic else 2)
cols[0].metric("Total invested", fmt_rs(plain["total_invested"]))
cols[1].metric("Plain SIP final value", fmt_rs(plain["final_value"]), delta=fmt_pct(plain["xirr"]) + " XIRR")
if use_dynamic:
    cols[2].metric("Dynamic SIP final value", fmt_rs(dynamic["final_value"]), delta=fmt_pct(dynamic["xirr"]) + " XIRR")
    diff = dynamic["final_value"] - plain["final_value"]
    st.caption(
        f"Dynamic vs Plain: {fmt_rs(diff)} ({diff / plain['final_value']:.1%})" if plain["final_value"] else ""
    )

if fund_sip is not None:
    fcols = st.columns(3 if use_dynamic else 2)
    fcols[0].metric(
        f"{compare_fund}: SIP final value", fmt_rs(fund_sip["final_value"]),
        delta=fmt_pct(fund_sip["xirr"]) + " XIRR",
    )
    diff_plain_fund = plain["final_value"] - fund_sip["final_value"]
    fcols[1].metric(
        "Plain SIP vs Fund", fmt_rs(diff_plain_fund),
        delta=f"{diff_plain_fund / fund_sip['final_value']:.1%}" if fund_sip["final_value"] else None,
    )
    if use_dynamic:
        diff_dynamic_fund = dynamic["final_value"] - fund_sip["final_value"]
        fcols[2].metric(
            "Dynamic SIP vs Fund", fmt_rs(diff_dynamic_fund),
            delta=f"{diff_dynamic_fund / fund_sip['final_value']:.1%}" if fund_sip["final_value"] else None,
        )
    st.caption(
        f"{compare_fund} SIP invested Rs {fund_sip['total_invested']:,.0f} over "
        f"{len(fund_sip['value'])} months (its own window -- starts later than the strategy's "
        f"if the fund's history is shorter)."
    )

if use_dynamic:
    st.caption(
        f"Dynamic mode spent {(dynamic['state'] == 'aggressive').sum()} of {len(dynamic['state'])} months "
        f"in full-allocation mode, across {len(dynamic['transitions']) // 2} trigger/recovery cycle(s) "
        f"(a cycle may be incomplete if still in aggressive mode at the end of the backtest)."
    )

if leverage_result is not None:
    with st.expander(f"MTF leverage overlay: {leverage_result['n_tranches']} tranche(s) triggered"):
        st.caption(
            "Interest is already deducted from the strategy returns used everywhere below (Plain "
            "SIP, Dynamic SIP, Lumpsum) -- there's no single rupee capital base to report a total "
            "interest figure against on this page, since a SIP invests incrementally over time "
            "rather than as one lump sum."
        )
        if leverage_result["events"]:
            ev_df = pd.DataFrame(leverage_result["events"])
            ev_df["date"] = ev_df["date"].dt.date
            ev_df["start"] = ev_df["start"].dt.date
            ev_df["end"] = ev_df["end"].dt.date
            st.dataframe(
                ev_df.rename(columns={
                    "date": "Trigger month", "leg": "Leg", "month_return": "Month return",
                    "start": "Active from", "end": "Active until",
                }).style.format({"Month return": "{:.2%}"}),
                hide_index=True, height=200,
            )
        else:
            st.caption("No leverage triggers with these parameters.")

st.subheader("Portfolio value over time")
fig = go.Figure()
fig.add_trace(go.Scatter(x=plain["invested"].index, y=plain["invested"].values,
                          name="Total invested", line=dict(dash="dot", color="gray")))
fig.add_trace(go.Scatter(x=plain["value"].index, y=plain["value"].values, name="Plain SIP"))
if use_dynamic:
    fig.add_trace(go.Scatter(x=dynamic["value"].index, y=dynamic["value"].values, name="Dynamic SIP"))
if fund_sip is not None:
    fig.add_trace(go.Scatter(
        x=fund_sip["value"].index, y=fund_sip["value"].values,
        name=f"{compare_fund} SIP", line=dict(dash="dashdot"),
    ))
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

dynamic_value_dd = None
if use_dynamic:
    dynamic_value_dd = dynamic["value"] / dynamic["value"].cummax() - 1

fund_value_dd = None
if fund_sip is not None:
    fund_value_dd = fund_sip["value"] / fund_sip["value"].cummax() - 1

dd_items = [("Strategy NAV max drawdown", strat_dd.min()), ("Plain SIP portfolio max drawdown", plain_value_dd.min())]
if use_dynamic:
    dd_items.append(("Dynamic SIP portfolio max drawdown", dynamic_value_dd.min()))
if fund_sip is not None:
    dd_items.append((f"{compare_fund} SIP max drawdown", fund_value_dd.min()))
dd_cols = st.columns(len(dd_items))
for col, (label, value) in zip(dd_cols, dd_items):
    col.metric(label, fmt_pct(value))
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
if use_dynamic:
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
if use_dynamic:
    fig_dd2.add_trace(go.Scatter(x=dynamic_value_dd.index, y=dynamic_value_dd.values, name="Dynamic SIP"))
if fund_value_dd is not None:
    fig_dd2.add_trace(go.Scatter(x=fund_value_dd.index, y=fund_value_dd.values, name=f"{compare_fund} SIP"))
fig_dd2.update_layout(
    yaxis_tickformat=".0%", yaxis_title="Portfolio value drawdown",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(t=30, l=10, r=10, b=10), height=300,
)
st.plotly_chart(fig_dd2, use_container_width=True)

if use_dynamic:
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

if lumpsum is not None:
    st.divider()
    st.subheader("Lumpsum on drawdown")
    st.caption(
        "An independent sleeve, tracked entirely separately from the SIP above -- not blended "
        "into either mode's totals. Shown on its own, plus combined with each SIP mode below "
        "for reference (simple addition of the two value series)."
    )
    lcols = st.columns(4)
    lcols[0].metric("Lumpsum invested", fmt_rs(lumpsum["total_invested"]))
    lcols[1].metric("Lumpsum current value", fmt_rs(lumpsum["final_value"]), delta=fmt_pct(lumpsum["xirr"]) + " XIRR")
    lcols[2].metric("Triggers fired", str(lumpsum["n_triggers"]))
    lump_dd = lumpsum["value"] / lumpsum["value"].cummax() - 1 if lumpsum["total_invested"] > 0 else lumpsum["value"] * 0
    lcols[3].metric("Lumpsum sleeve max drawdown", fmt_pct(lump_dd.min()) if lumpsum["total_invested"] > 0 else "-")

    if lumpsum["n_triggers"] == 0:
        st.info("No drawdown reached the trigger threshold with these parameters -- the lumpsum was never deployed.")
    else:
        fig_lump = go.Figure()
        fig_lump.add_trace(go.Scatter(x=lumpsum["invested"].index, y=lumpsum["invested"].values,
                                       name="Lumpsum invested", line=dict(dash="dot", color="gray")))
        fig_lump.add_trace(go.Scatter(x=lumpsum["value"].index, y=lumpsum["value"].values, name="Lumpsum sleeve value"))
        fig_lump.update_layout(
            yaxis_title="Rs", legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=30, l=10, r=10, b=10), height=350,
        )
        st.plotly_chart(fig_lump, use_container_width=True)

        st.markdown("**Combined with each SIP mode**")
        combined_plain = plain["value"] + lumpsum["value"]
        fig_combined = go.Figure()
        fig_combined.add_trace(go.Scatter(x=plain["value"].index, y=plain["value"].values, name="Plain SIP alone"))
        fig_combined.add_trace(go.Scatter(x=combined_plain.index, y=combined_plain.values, name="Plain SIP + Lumpsum"))
        if use_dynamic:
            combined_dynamic = dynamic["value"] + lumpsum["value"]
            fig_combined.add_trace(go.Scatter(x=dynamic["value"].index, y=dynamic["value"].values, name="Dynamic SIP alone"))
            fig_combined.add_trace(go.Scatter(x=combined_dynamic.index, y=combined_dynamic.values, name="Dynamic SIP + Lumpsum"))
        fig_combined.update_layout(
            yaxis_type="log", yaxis_title="Portfolio value (Rs, log scale)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=30, l=10, r=10, b=10), height=400,
        )
        st.plotly_chart(fig_combined, use_container_width=True)

        combined_cols = st.columns(2 if use_dynamic else 1)
        combined_cols[0].metric("Plain SIP + Lumpsum final value", fmt_rs(combined_plain.iloc[-1]))
        if use_dynamic:
            combined_cols[1].metric("Dynamic SIP + Lumpsum final value", fmt_rs(combined_dynamic.iloc[-1]))

        with st.expander("Lumpsum trigger dates"):
            trig_df = pd.DataFrame(lumpsum["triggers"], columns=["Date", "Detail"])
            trig_df["Date"] = trig_df["Date"].dt.date
            st.dataframe(trig_df, hide_index=True, height=250)

st.divider()
st.caption(
    "Both simulations use a units/NAV model like a real mutual fund SIP: each contribution "
    "buys units at that month's NAV, so total value = units held x NAV. The drawdown/all-time-"
    "high tracking for the trigger uses the strategy's own price path only, independent of "
    "contribution timing. XIRR is computed via Newton-Raphson on the full contribution + "
    "final-value cashflow series (Act/365 day count)."
)
