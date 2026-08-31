"""
Correction Lumpsum: backtests investing a fixed lumpsum whenever an index
or GOLDBEES corrects a customizable % from its running all-time high, using
the exact same opportunistic-lumpsum mechanics as the SIP Simulator page's
"Lumpsum on drawdown" sleeve (sip_engine.simulate_lumpsum_on_drawdown) --
one lumpsum per correction episode, re-armed only after a confirmed
recovery above the pre-drawdown peak, not one lumpsum per day spent
underwater. Runs on DAILY prices (unlike the monthly series used
elsewhere in this app), since detecting a correction needs day-level
resolution.

Two modes:
  - Single instrument: pick ONE index/ETF, invest whenever IT corrects.
  - Multiple instruments (combined): pick several; EACH one independently
    triggers its own lumpsum off its OWN price history (same threshold %
    and lumpsum amount applied to each), combined into one portfolio and
    one money-weighted XIRR (sip_engine.run_multi_correction_lumpsum).

Instruments: Nifty 50 / Nifty 500 / Nifty Midcap 150 / Nifty Smallcap 250
use their NSE index price series (not directly investable, but the
longest, cleanest history -- a "correction" is naturally a statement
about the index); Gold uses GOLDBEES, the actual investable ETF, same as
everywhere else in this project. See backtest_engine.CORRECTION_INSTRUMENTS.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtest_engine import CORRECTION_INSTRUMENTS, ensure_stock_data
from sip_engine import run_multi_correction_lumpsum, simulate_lumpsum_on_drawdown
from streamlit_cache import cached_load_correction_instrument

st.set_page_config(page_title="Correction Lumpsum", layout="wide")

with st.spinner("Fetching price data (first run only)..."):
    ensure_stock_data()

st.title("Correction Lumpsum")
st.caption(
    "Invests a fixed lumpsum whenever a chosen index or GOLDBEES corrects a threshold % from "
    "its running all-time high -- one lumpsum per correction episode (armed again only after a "
    "confirmed recovery above the pre-drawdown peak), not one lumpsum per day spent underwater."
)

INSTRUMENT_NAMES = list(CORRECTION_INSTRUMENTS.keys())

with st.sidebar:
    st.header("Mode")
    mode_label = st.radio("Mode", ["Single instrument", "Multiple instruments (combined)"], index=0)
    is_multi = mode_label == "Multiple instruments (combined)"

    st.header("Instrument(s)")
    if is_multi:
        selected = st.multiselect(
            "Instruments -- each corrects and triggers independently off its own price history",
            INSTRUMENT_NAMES, default=["Nifty 50", "Gold (GOLDBEES)"],
        )
    else:
        selected_one = st.selectbox("Instrument", INSTRUMENT_NAMES, index=0)
        selected = [selected_one]
    price_col = st.selectbox("Price field", ["Close", "Adj Close"], index=0)
    st.caption(
        "Nifty 50 / 500 / Midcap 150 / Smallcap 250 use NSE's index price series (not directly "
        "investable, used for its longer/cleaner history). Nifty Midcap 150 only has data from "
        "2019; the others go back to 2008-2009."
    )

    st.header("Correction trigger")
    lumpsum_amount = st.number_input("Lumpsum amount per trigger (Rs)", min_value=1000.0, value=100000.0, step=5000.0)
    drawdown_trigger_pct = st.slider(
        "Correction that triggers a lumpsum (%)", min_value=1, max_value=50, value=15, step=1,
        help="Invests the lumpsum the first time price falls this far below its running all-time "
             "high (per instrument, if multiple are selected)."
    )
    reset_recovery_pct = st.slider(
        "Recovery above pre-correction peak to re-arm (%)", min_value=0, max_value=100, value=0, step=1,
        help="0 = ready to trigger again as soon as price merely reclaims its old high. Higher = "
             "requires a confirmed rally past the old peak by this much before it's willing to "
             "fire on the next correction. Each correction episode only ever fires ONE lumpsum, "
             "however deep it gets -- this isn't a buy-every-day-it's-down strategy."
    )

    st.header("Data & display")
    use_custom_start = st.checkbox(
        "Custom start date", value=False,
        help="Shows what this would look like if you'd started watching for corrections from a "
             "specific date, instead of from the earliest available history. The all-time-high "
             "reference resets fresh at this date too (it's YOUR all-time-high-since-you-started, "
             "not the instrument's true all-time high from before then) -- same convention as the "
             "SIP Simulator page's lumpsum sleeve, which uses this identical mechanism."
    )
    custom_start_date = None
    if use_custom_start:
        custom_start_date = st.date_input(
            "Start date", value=pd.Timestamp.today() - pd.DateOffset(years=5),
        )

if not selected:
    st.warning("Pick at least one instrument in the sidebar.")
    st.stop()

returns_by_name: dict[str, pd.Series] = {}
prices_by_name: dict[str, pd.Series] = {}
for name in selected:
    px = cached_load_correction_instrument(name, price_col)
    prices_by_name[name] = px
    returns_by_name[name] = px.pct_change().dropna()

if custom_start_date is not None:
    start_ts = pd.Timestamp(custom_start_date)
    returns_by_name = {n: r[r.index >= start_ts] for n, r in returns_by_name.items()}
    prices_by_name = {n: p[p.index >= start_ts] for n, p in prices_by_name.items()}
    if all(len(r) == 0 for r in returns_by_name.values()):
        st.warning(f"No data falls on or after {start_ts.date()} -- pick an earlier start date.")
        st.stop()
    st.info(f"Showing results as if watching for corrections started on **{start_ts.date()}**.")

if is_multi:
    with st.spinner("Running combined correction-lumpsum backtest..."):
        result = run_multi_correction_lumpsum(returns_by_name, lumpsum_amount, drawdown_trigger_pct, reset_recovery_pct)
else:
    with st.spinner("Running correction-lumpsum backtest..."):
        result = simulate_lumpsum_on_drawdown(
            returns_by_name[selected[0]], lumpsum_amount, drawdown_trigger_pct, reset_recovery_pct
        )


def fmt_rs(x: float) -> str:
    return f"Rs {x:,.0f}"


def fmt_pct(x: float) -> str:
    return "-" if pd.isna(x) else f"{x:.2%}"


st.subheader("Results")
cols = st.columns(4)
cols[0].metric("Total invested", fmt_rs(result["total_invested"]))
cols[1].metric("Current value", fmt_rs(result["final_value"]), delta=fmt_pct(result["xirr"]) + " XIRR")
cols[2].metric("Triggers fired", str(result["n_triggers"]))
gain = result["final_value"] - result["total_invested"]
cols[3].metric(
    "Unrealized gain/loss", fmt_rs(gain),
    delta=f"{gain / result['total_invested']:.1%}" if result["total_invested"] else None,
)

if result["n_triggers"] == 0:
    st.info(
        "No instrument corrected by the trigger threshold with these parameters -- the lumpsum "
        "was never deployed. Try a smaller correction %, a different instrument, or a longer "
        "history (e.g. Nifty 50 instead of Nifty Midcap 150, which only starts in 2019)."
    )
    st.stop()

st.subheader("Portfolio value over time")
fig = go.Figure()
fig.add_trace(go.Scatter(x=result["invested"].index, y=result["invested"].values,
                          name="Total invested", line=dict(dash="dot", color="gray")))
fig.add_trace(go.Scatter(x=result["value"].index, y=result["value"].values, name="Portfolio value"))
fig.update_layout(
    yaxis_title="Rs", legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(t=30, l=10, r=10, b=10), height=400,
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Instrument correction(s) vs trigger threshold")
st.caption(
    "Each instrument's own price drawdown from its running all-time high. A lumpsum fires the "
    "first time a line crosses below the dashed threshold, then that line is free to run deeper "
    "without firing again until it recovers and corrects a second time."
)
fig_dd = go.Figure()
for name, px in prices_by_name.items():
    dd = px / px.cummax() - 1
    fig_dd.add_trace(go.Scatter(x=dd.index, y=dd.values, name=name))
fig_dd.add_hline(
    y=-drawdown_trigger_pct / 100, line_dash="dash", line_color="red",
    annotation_text=f"trigger (-{drawdown_trigger_pct}%)", annotation_position="bottom right",
)
fig_dd.update_layout(
    yaxis_tickformat=".0%", yaxis_title="Drawdown from all-time high",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(t=30, l=10, r=10, b=10), height=350,
)
st.plotly_chart(fig_dd, use_container_width=True)

if is_multi and len(result["per_instrument"]) > 1:
    st.subheader("Per-instrument breakdown")
    rows = []
    for name, res in result["per_instrument"].items():
        rows.append({
            "Instrument": name, "Invested": res["total_invested"], "Current value": res["final_value"],
            "XIRR": res["xirr"], "Triggers": res["n_triggers"],
        })
    bdf = pd.DataFrame(rows)
    st.dataframe(
        bdf.style.format({"Invested": "Rs {:,.0f}", "Current value": "Rs {:,.0f}", "XIRR": "{:.2%}"}),
        hide_index=True,
    )

with st.expander("Trigger dates"):
    if is_multi:
        trig_rows = []
        for name, res in result["per_instrument"].items():
            for d, detail in res["triggers"]:
                trig_rows.append({"Date": d.date(), "Instrument": name, "Detail": detail})
        trig_df = pd.DataFrame(trig_rows).sort_values("Date")
        st.dataframe(trig_df, hide_index=True, height=300)
    else:
        trig_df = pd.DataFrame(result["triggers"], columns=["Date", "Detail"])
        trig_df["Date"] = trig_df["Date"].dt.date
        st.dataframe(trig_df, hide_index=True, height=300)

st.divider()
st.caption(
    "Uses a units/NAV model like a real fund purchase: each trigger buys units at that day's "
    "price, so value = units held x current price. XIRR is computed via Newton-Raphson on the "
    "full trigger + final-value cashflow series (Act/365 day count) -- for the combined mode "
    "this is ONE money-weighted return across all instruments' merged cashflows, not an average "
    "of their individual XIRRs. Idle cash waiting for the next correction earns 0% (a "
    "simplification, same as elsewhere in this app). No costs, taxes, or slippage modeled."
)
