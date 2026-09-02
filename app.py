"""
Interactive portal for the Nifty 500 momentum backtest.

Run with:
    streamlit run app.py

Lets you customise the number of stocks held, the rebalancing period, and
the trailing-return lookback period (plus skip period, min price filter,
and whether to enforce point-in-time index membership), and see the
resulting strategy performance vs. the Nifty 500 benchmark update live.

Rebalancing frequency is itself customizable -- Monthly (the original
mode, hold 1-12 months, lookback 1-24 months) or Weekly (hold 1-6 weeks,
lookback 4-104 weeks). Weekly mode re-derives the whole strategy on a
weekly price grid (see backtest_engine.load_prices' freq param) rather
than just rebalancing a monthly ranking more often. The GOLDBEES regime
filter is monthly-only for now -- its 150/55 lookback windows are
calibrated in months and aren't auto-converted to weeks.

Optionally overlays one Direct Growth mutual fund scheme (see
backtest_engine.MUTUAL_FUNDS) alongside the Nifty 500 benchmark, in the
KPIs, equity curve, and drawdown chart -- a hand-picked list of well-known
funds, not exhaustive or AUM-ranked.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtest_engine import (
    MUTUAL_FUNDS,
    NSE_UNIVERSES,
    apply_execution_lag,
    apply_stoploss,
    build_trade_log,
    ensure_mutual_fund_data,
    ensure_stock_data,
    perf_stats,
    run_backtest,
    yearly_table,
)
from leverage_engine import apply_mtf_leverage
from streamlit_cache import (
    cached_load_benchmark,
    cached_load_daily_prices,
    cached_load_gold,
    cached_load_membership,
    cached_load_mutual_fund_nav,
    cached_load_prices,
    cached_load_universe_symbols,
)
from tax_cost_engine import CostParams, TaxParams, simulate_costs_and_taxes

st.set_page_config(page_title="Nifty 500 Momentum Backtest", layout="wide")

with st.spinner("Fetching price data (first run only)..."):
    ensure_stock_data()
    ensure_mutual_fund_data()


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
        st.caption(
            f"Restricted to the current {len(allowed_symbols)} constituents of {universe_label}. "
            "If this universe includes stocks outside the Nifty 500 (e.g. Microcap 250), turn "
            "off 'Enforce point-in-time index membership' below, or they'll also need to have "
            "been a Nifty 500 constituent at each rebalance date to be picked."
        )

    st.header("Strategy parameters")
    n_stocks = st.slider("Number of stocks held", min_value=5, max_value=100, value=30, step=5)

    rebal_freq_label = st.radio(
        "Rebalancing frequency", ["Monthly", "Weekly"], index=0,
        help="Weekly re-derives the entire strategy (price series, ranking lookback/skip, "
             "rebalancing) on a weekly grid instead of monthly -- not just a faster rebalance "
             "of the same monthly ranking. The GOLDBEES regime filter isn't available in weekly "
             "mode yet (its lookback windows are calibrated in months)."
    )
    is_weekly = rebal_freq_label == "Weekly"
    period_word = "week" if is_weekly else "month"
    price_freq = "W-FRI" if is_weekly else "ME"

    if is_weekly:
        hold_months = st.slider("Rebalancing period (weeks)", min_value=1, max_value=6, value=1, step=1)
        lookback_months = st.slider(
            "Trailing return lookback (weeks)", min_value=4, max_value=104, value=52, step=1,
            help="52 weeks is the weekly equivalent of the monthly mode's 12-month default."
        )
        skip_months = st.slider(
            "Skip period (weeks)", min_value=0, max_value=12, value=4, step=1,
            help="Excludes the most recent N weeks from the lookback, to avoid short-term reversal effects."
        )
    else:
        hold_months = st.slider("Rebalancing period (months)", min_value=1, max_value=12, value=1, step=1)
        lookback_months = st.slider("Trailing return lookback (months)", min_value=1, max_value=24, value=12, step=1)
        skip_months = st.slider(
            "Skip period (months)", min_value=0, max_value=3, value=1, step=1,
            help="Excludes the most recent N months from the lookback, to avoid short-term reversal effects."
        )

    weighting_label = st.radio(
        "Rebalancing weight method",
        [f"Equal weight every {period_word}", "Equal weight at rebalance, drift between"],
        index=0,
        help=f"Equal weight every {period_word}: weights reset to exactly 1/N every {period_word}, "
             "even between rebalances -- a winner's gain never lets it grow as a share of the "
             f"portfolio for next {period_word}. Equal weight at rebalance, drift between: weights "
             "are set to 1/N only at each rebalance; between rebalances a stock's own performance "
             "lets its weight drift up or down (winners compound, laggards shrink) until the next "
             "rebalance resets everyone -- this is how real equal-weight index funds/ETFs actually work."
    )
    weighting_mode = "equal_monthly" if weighting_label.startswith("Equal weight every") else "drift"

    use_stoploss = st.checkbox(
        "Per-stock stoploss", value=False,
        help="Optional daily-level risk control, independent of the rebalancing period. Each "
             "stock is bought at its rebalance-day close; on any later trading day its close "
             "falls to stoploss% below where it was bought, it's sold that same close and the "
             "slot holds cash. New stocks are only ever added at the next scheduled rebalance -- "
             "a stopped-out slot doesn't get replaced with a different stock mid-period. "
             "Requires daily price data (a bit slower to compute) and always uses discrete "
             "per-stock entry/exit tracking, overriding the weighting method above for pricing "
             "purposes while it's on."
    )
    stoploss_pct = 10.0
    max_reentries = 0
    if use_stoploss:
        stoploss_pct = st.slider(
            "Stoploss (% below buy price)", min_value=1.0, max_value=50.0, value=10.0, step=1.0,
        )
        max_reentries = st.slider(
            "Number of re-entries allowed", min_value=0, max_value=5, value=0, step=1,
            help="0 = once stopped out, the slot stays in cash for the rest of the holding "
                 "period. N > 0: after a stop, if the SAME stock's close later rises back above "
                 "its original buy price for this holding period, it's bought again at the "
                 "next day's open (a fresh stoploss is set from that new price) -- up to N "
                 "times per holding period."
        )

    use_execution_lag = st.checkbox(
        f"T+1 execution (signal at {period_word}-end close, trade at next open)", value=False,
        help=f"Realistic execution timing: the rebalance ranking is still computed from the "
             f"{period_word}-end close (unchanged), but entries and exits are executed on the "
             "NEXT trading day. A stock being dropped is held (and keeps accruing return) through "
             "that next day's open, where it's sold; a stock being added is bought at that "
             "open, so it only starts accruing return from there. Stocks that stay held across "
             "a rebalance are unaffected. Requires daily price data (a bit slower to compute)."
    )
    if use_stoploss and use_execution_lag:
        st.caption(
            f"Note: stoploss and T+1 execution don't currently compose -- for any {period_word} "
            "where both would apply, T+1 execution's calculation wins (it doesn't account for "
            "stops/re-entries happening intra-period)."
        )

    use_exit_band = st.checkbox(
        "Custom exit criteria", value=False,
        help="Optional turnover-reduction rule. Off: a held stock exits as soon as its "
             "momentum rank falls out of the top N (same as entries). On: a held stock "
             "is only exited once its rank falls below N x (1 + exit%) -- new stocks are "
             "still only ever added from the top N."
    )
    exit_band_pct = 0.0
    if use_exit_band:
        exit_band_pct = st.slider(
            "Exit threshold (% beyond N)", min_value=0, max_value=300, value=100, step=10,
        )
        exit_rank_preview = int(round(n_stocks * (1 + exit_band_pct / 100)))
        st.caption(
            f"With {n_stocks} stocks and {exit_band_pct}% exit criteria, a held stock "
            f"exits once its rank falls below {exit_rank_preview}."
        )

    use_regime_filter = False
    gold_entry_lookback = 150
    gold_exit_lookback = 55
    if is_weekly:
        st.caption(
            "GOLDBEES defensive rotation isn't available in weekly mode yet -- its 150/55 "
            "lookback windows are calibrated in months, and reinterpreting them as weeks would "
            "silently change the regime timing rather than just rebalancing faster. Switch back "
            "to Monthly to use it."
        )
    else:
        use_regime_filter = st.checkbox(
            "GOLDBEES defensive rotation", value=False,
            help="Optional regime filter. At each rebalance: while running the momentum "
                 "portfolio, if GOLDBEES's trailing return beats the Nifty 500's over the "
                 "entry lookback, switch the ENTIRE portfolio to GOLDBEES. While in GOLDBEES, "
                 "switch back to the momentum portfolio once the Nifty 500's trailing return "
                 "beats GOLDBEES's over the (shorter) exit lookback. Uses GOLDBEES (Nippon "
                 "India ETF Gold BeES) as the actual investable gold instrument, not a spot "
                 "gold index."
        )
    if use_regime_filter:
        gold_entry_lookback = st.number_input(
            "GOLDBEES entry lookback (months)", min_value=1, max_value=200, value=150, step=1,
            help="Trailing-return period compared between Nifty 500 and GOLDBEES to decide "
                 "whether to switch INTO GOLDBEES."
        )
        gold_exit_lookback = st.number_input(
            "GOLDBEES exit lookback (months)", min_value=1, max_value=200, value=55, step=1,
            help="Trailing-return period compared between Nifty 500 and GOLDBEES to decide "
                 "whether to switch BACK to the momentum portfolio."
        )
        st.caption(
            "Note: GOLDBEES data starts 2009-01-02, so the entry check (needing "
            f"{gold_entry_lookback} months of gold history) can't produce a signal until "
            "roughly that many months after gold's inception -- expect this filter to be "
            "inactive for the earlier years of the backtest. Also, the cost/tax simulation "
            "below charges the equity-side cost of liquidating/rebuilding the stock "
            "portfolio around a gold rotation, but not the gold ETF leg's own cost or tax."
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

    use_custom_start = st.checkbox(
        "Custom start date", value=False,
        help="Shows what the backtest would look like if you'd started investing on a "
             "specific date, instead of from the earliest available history. The ranking "
             f"lookback still uses real price history from BEFORE this date (so the first "
             f"{period_word} after it is ranked correctly, not cold-started) -- only the "
             "displayed performance, trade log, and rebalance count are re-based to begin "
             "from this date, as if capital were first deployed then."
    )
    custom_start_date = None
    if use_custom_start:
        custom_start_date = st.date_input(
            "Start date", value=pd.Timestamp.today() - pd.DateOffset(years=5),
        )

    st.header("Compare against a mutual fund")
    fund_choice = st.selectbox(
        "Mutual fund (optional)", ["None"] + list(MUTUAL_FUNDS.keys()),
        format_func=lambda n: n if n == "None" else f"[{MUTUAL_FUNDS[n]['category']}] {n}",
        help="Adds one Direct Growth mutual fund scheme alongside the Nifty 500 benchmark, "
             "everywhere the benchmark is shown -- KPIs, equity curve, drawdown. A hand-picked "
             "list of well-known funds (not exhaustive or AUM-ranked), Direct Growth only (not "
             "Regular, which carries distributor commission drag). Source: api.mfapi.in. If the "
             "fund's history starts later than the strategy's, its line/stats simply begin later "
             "-- combine with Custom start date above for an apples-to-apples window."
    )
    compare_fund = fund_choice if fund_choice != "None" else None

    st.header("MTF Leverage Overlay")
    use_leverage = st.checkbox(
        "Enable MTF leverage overlay", value=False,
        help="Optional, two-stage drawdown-triggered leverage: adds extra MTF (Margin Trading "
             "Facility) exposure after a bad month, and more after a second bad month soon "
             "after, on top of the strategy's own returns. Applied IN PLACE to the strategy's "
             "returns -- like the stoploss/T+1 execution overlays above, this changes the same "
             "'Momentum strategy' line everywhere (KPIs, equity curve, trade log), rather than "
             "adding a separate comparison line. Toggle it on/off to compare with vs. without."
    )
    if use_leverage:
        lev_trigger1_pct = st.slider(
            "Leg 1 trigger: month return <= -X%", min_value=1.0, max_value=30.0, value=7.0, step=0.5,
            help="A month with return at or below negative this % adds the first leg of leverage."
        )
        lev_leg1_pct = st.slider(
            "Leg 1 leverage added (%)", min_value=5.0, max_value=200.0, value=50.0, step=5.0,
            help="Extra exposure added as a fraction of current equity, e.g. 50% means 150% total exposure."
        )
        lev_trigger2_pct = st.slider(
            "Leg 2 trigger: another month return <= -X% within 2 months of leg 1", min_value=1.0,
            max_value=30.0, value=5.0, step=0.5,
            help="Within the 2 months AFTER leg 1's trigger month, a further month at or below "
                 "negative this % adds a second leg -- fires at most once per leg-1 cycle."
        )
        lev_leg2_pct = st.slider(
            "Leg 2 leverage added (%)", min_value=5.0, max_value=200.0, value=50.0, step=5.0,
            help="Additional exposure on top of leg 1, e.g. 50%+50% = 200% total exposure while both are active."
        )
        lev_hold_mode_label = st.radio(
            "Hold duration", ["Fixed number of months", "Until recovery above previous ATH"], index=0,
            help="Fixed months: each leg independently expires N months after its own trigger. "
                 "ATH recovery: a leg stays active until the strategy's OWN (unleveraged) NAV "
                 "recovers to X% above its all-time-high as of that leg's trigger month -- can "
                 "run much longer than a fixed window during a slow recovery."
        )
        lev_hold_mode = "fixed_months" if lev_hold_mode_label.startswith("Fixed") else "ath_recovery"
        lev_hold_months = 6
        lev_recovery_pct = 0.0
        if lev_hold_mode == "fixed_months":
            lev_hold_months = st.slider("Hold duration (months)", min_value=1, max_value=36, value=6, step=1)
        else:
            lev_recovery_pct = st.slider(
                "Recovery above previous ATH to revert (%)", min_value=0.0, max_value=50.0, value=0.0, step=1.0,
                help="0 = reverts as soon as the strategy's NAV merely reclaims its old high."
            )
        lev_annual_interest_pct = st.number_input(
            "MTF interest rate (% per year)", min_value=0.0, value=10.0, step=0.5,
            help="Charged monthly on the borrowed (leveraged) amount, deducted from equity "
                 "regardless of that month's P&L."
        )

    st.header("Costs & taxes (India)")
    apply_costs_taxes = st.checkbox(
        "Apply transaction costs & capital gains tax", value=False,
        help="Simulates STT, stamp duty, exchange charges, slippage, and STCG/LTCG tax "
             "(with loss carryforward) on every trade implied by the rebalances above. "
             "Approximate -- not tax advice."
    )
    if apply_costs_taxes and use_stoploss:
        st.caption(
            "Note: this cost/tax simulation only sees rebalance-level entries/exits, not the "
            "intra-period stops and re-entries the stoploss overlay adds to the return curve -- "
            "so it under-counts real trading costs/tax events while stoploss is on."
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

monthly_prices = cached_load_prices(price_col, price_freq)

membership = None
if use_membership_filter:
    membership = cached_load_membership(
        tuple(monthly_prices.index.values), tuple(monthly_prices.columns)
    )

bench_px = cached_load_benchmark(price_col, price_freq)
gold_px = cached_load_gold(price_col) if use_regime_filter else None

strat_rets, holdings_history = run_backtest(
    monthly_prices, membership, lookback_months, skip_months, hold_months, n_stocks, min_price,
    use_exit_band, exit_band_pct,
    use_regime_filter, bench_px, gold_px, gold_entry_lookback, gold_exit_lookback,
    weighting_mode, allowed_symbols,
)

if use_stoploss or use_execution_lag:
    daily_close, daily_open = cached_load_daily_prices(price_col)
    if use_stoploss:
        stoploss_overlay = apply_stoploss(
            monthly_prices, daily_close, daily_open, holdings_history, stoploss_pct, max_reentries,
            resample_freq=price_freq,
        )
        for m_date, r in stoploss_overlay.items():
            if m_date in strat_rets.index:
                strat_rets.loc[m_date] = r
    if use_execution_lag:
        exec_lag_overlay = apply_execution_lag(
            monthly_prices, daily_close, daily_open, holdings_history, resample_freq=price_freq
        )
        for m_date, r in exec_lag_overlay.items():
            if m_date in strat_rets.index:
                strat_rets.loc[m_date] = r

leverage_result = None
if use_leverage and len(strat_rets) > 0:
    leverage_result = apply_mtf_leverage(
        strat_rets, lev_trigger1_pct, lev_leg1_pct, lev_trigger2_pct, lev_leg2_pct,
        lev_hold_mode, lev_hold_months, lev_recovery_pct, lev_annual_interest_pct,
    )
    strat_rets = leverage_result["leveraged_rets"]

bench_rets = bench_px.pct_change().reindex(strat_rets.index).dropna()
strat_rets = strat_rets.reindex(bench_rets.index)

if custom_start_date is not None:
    start_ts = pd.Timestamp(custom_start_date)
    strat_rets = strat_rets[strat_rets.index >= start_ts]
    bench_rets = bench_rets[bench_rets.index >= start_ts]
    holdings_history = [(d, h) for d, h in holdings_history if d >= start_ts]
    if len(strat_rets) == 0:
        st.warning(
            f"No rebalances fall on or after {start_ts.date()} -- pick an earlier start date, "
            "or the strategy's data doesn't extend that recently."
        )
        st.stop()

perf_freq = 52 if is_weekly else 12
strat_stats = perf_stats(strat_rets, freq=perf_freq)
bench_stats = perf_stats(bench_rets, freq=perf_freq)

fund_rets = None
fund_stats = None
if compare_fund is not None:
    fund_nav = cached_load_mutual_fund_nav(compare_fund).resample(price_freq).last()
    fund_rets = fund_nav.pct_change().dropna()
    if custom_start_date is not None:
        fund_rets = fund_rets[fund_rets.index >= pd.Timestamp(custom_start_date)]
    if len(fund_rets) > 1:
        fund_stats = perf_stats(fund_rets, freq=perf_freq)
    else:
        st.sidebar.warning(f"{compare_fund}: no data in the selected window.")
        fund_rets = None

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
    post_cost_stats = perf_stats(cost_tax_result["nav_post_cost"].pct_change().dropna(), freq=perf_freq)
    post_tax_stats = perf_stats(cost_tax_result["nav_net"].pct_change().dropna(), freq=perf_freq)

if custom_start_date is not None:
    st.info(
        f"Showing results as if capital were first deployed on **{pd.Timestamp(custom_start_date).date()}** "
        f"({len(holdings_history)} rebalances since). Ranking still uses real price history from "
        "before this date, so the first rebalance shown isn't cold-started."
    )

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
    if fund_stats is not None:
        col.caption(f"{compare_fund}: {fmt(fund_stats[label])}")

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
        f"No {period_word}s produced a valid portfolio -- likely `Number of stocks held` is "
        "larger than the number of eligible stocks available early in the sample (a smaller "
        "Universe like Nifty 50 or Alpha 50 makes this more likely). Try lowering it, shortening "
        "the lookback, or picking a larger Universe."
    )
else:
    strat_cum = (1 + strat_rets).cumprod()

    st.subheader("Investment value")
    start_capital = st.number_input(
        "Starting capital (Rs)", min_value=1_000.0, value=100_000.0, step=10_000.0,
        help="What this becomes today, compounding at the strategy's own gross returns shown below."
    )
    final_value = start_capital * strat_cum.iloc[-1]
    st.metric(
        f"Value today (invested {strat_cum.index[0].date()})", f"Rs {final_value:,.0f}",
        delta=fmt_pct(strat_cum.iloc[-1] - 1) + " total return",
    )

    st.subheader("Equity curve")
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
    if fund_rets is not None:
        fund_cum = (1 + fund_rets).cumprod()
        fig.add_trace(go.Scatter(x=fund_cum.index, y=fund_cum.values, name=compare_fund, line=dict(width=2, dash="dashdot")))
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
    if fund_rets is not None:
        fund_dd = fund_cum / fund_cum.cummax() - 1
        fig_dd.add_trace(go.Scatter(x=fund_dd.index, y=fund_dd.values, name=compare_fund, line=dict(dash="dashdot")))
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

    if use_regime_filter and holdings_history:
        with st.expander("GOLDBEES regime timeline"):
            segments = []
            seg_start, seg_regime = holdings_history[0][0], (holdings_history[0][1] == ["GOLD"])
            for date, holdings in holdings_history[1:]:
                is_gold = holdings == ["GOLD"]
                if is_gold != seg_regime:
                    segments.append((seg_start, date, "GOLDBEES" if seg_regime else "Momentum"))
                    seg_start, seg_regime = date, is_gold
            segments.append((seg_start, holdings_history[-1][0], "GOLDBEES" if seg_regime else "Momentum"))
            seg_df = pd.DataFrame(segments, columns=["start", "last_rebalance", "regime"])
            n_switches = sum(1 for s in segments if s[2] == "GOLDBEES")
            st.caption(f"{n_switches} switch(es) into GOLDBEES over the backtest period.")
            st.dataframe(seg_df, hide_index=True, height=250)

    if leverage_result is not None:
        with st.expander("MTF leverage overlay: tranche log & interest"):
            total_interest_rs = leverage_result["interest_series"].sum() * capital_base
            st.caption(
                f"{leverage_result['n_tranches']} tranche(s) triggered over the backtest period. "
                f"Total interest paid: Rs {total_interest_rs:,.0f} (on a Rs {capital_base:,.0f} capital base)."
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
                    hide_index=True, height=250,
                )
            else:
                st.caption("No leverage triggers with these parameters.")

    st.subheader("All trades")
    st.caption(
        "Every entry/exit the strategy took, reconstructed as discrete buy/sell orders "
        "(qty sized at 1/N of portfolio value on entry, held until exit). Independent of "
        "the cost/tax settings above -- this is a plain summary of strategy activity. "
        "Still-open positions at the end of the backtest are marked unrealized."
    )
    if use_stoploss:
        st.warning(
            "Stoploss is on, but this table still only reflects REBALANCE-level entries/exits "
            "(one row per stock per holding period) -- it does not yet break out the intra-period "
            "stop-outs and re-entries the stoploss overlay simulates for the return curve above. "
            "The equity curve and performance stats above do already include that activity; this "
            "trade log's P&L for a period spanning a stop-out just won't match it."
        )
    trade_log_capital = st.number_input(
        "Portfolio size for quantity sizing (Rs)", min_value=100_000.0, value=1_000_000.0,
        step=100_000.0, key="trade_log_capital",
    )
    trade_log = build_trade_log(monthly_prices, strat_rets, holdings_history, n_stocks, trade_log_capital)
    if trade_log.empty:
        st.caption("No trades yet with these parameters.")
    else:
        n_closed = (trade_log["status"] == "closed").sum()
        n_open = (trade_log["status"] == "open").sum()
        realized_pnl = trade_log.loc[trade_log["status"] == "closed", "pnl_rs"].sum()
        unrealized_pnl = trade_log.loc[trade_log["status"] == "open", "pnl_rs"].sum()
        st.markdown(
            f"**{n_closed} closed** trades (realized P&L: Rs {realized_pnl:,.0f})  |  "
            f"**{n_open} open** positions (unrealized P&L: Rs {unrealized_pnl:,.0f})"
        )
        st.dataframe(
            trade_log.sort_values("entry_date", ascending=False).style.format({
                "entry_price": "{:.2f}", "exit_price": "{:.2f}", "qty": "{:.1f}",
                "pnl_rs": "Rs {:,.0f}", "pnl_pct": "{:.2%}",
            }),
            hide_index=True, height=400,
        )
        st.download_button(
            "Download all trades as CSV",
            trade_log.to_csv(index=False).encode("utf-8"),
            file_name="all_trades.csv",
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
            if last_holdings == ["GOLD"]:
                st.caption(f"As of {last_date.date()}")
                st.info("Currently in the **gold defensive regime** -- 100% GOLDBEES, no individual stocks held.")
            else:
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
            st.write(f"**Avg. eligible stocks/{period_word} (point-in-time):** {avg_eligible:.0f}")
        st.write(f"**Rebalances:** {len(holdings_history)}")

    with st.expander(f"{period_word.capitalize()}ly returns (strategy vs benchmark)"):
        out = pd.DataFrame({"strategy_return": strat_rets, "benchmark_return": bench_rets})
        st.dataframe(out.style.format("{:.2%}"), height=300)
        st.download_button(
            f"Download {period_word}ly returns as CSV",
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
