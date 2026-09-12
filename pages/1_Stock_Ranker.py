"""
Live Nifty 500 momentum ranker: ranks all CURRENT Nifty 500 constituents by
trailing return, using the exact same formula as the backtest engine
(compute_momentum_ranking in backtest_engine.py) -- shows what the momentum
backtest's next scheduled rebalance would pick, as of the latest complete
period in the downloaded data, and lets you check an existing holding's
current rank against a custom exit threshold.

Rebalancing frequency (Monthly/Weekly) mirrors the Backtest page's option
of the same name -- Weekly re-derives the ranking on a weekly price grid
(lookback/skip in weeks, not months), it isn't just a relabeled monthly
view. See app.py's module docstring for why this needs a different price
series (backtest_engine.load_prices' freq param), not just a different
"as of" date.

Optionally ranks by RISK-ADJUSTED return (trailing return / trailing
annualized volatility, see backtest_engine.compute_trailing_volatility)
instead of absolute return, and/or filters out anything above a chosen
volatility threshold -- same two options as the Backtest page, mirrored
here so a live ranking and a historical backtest can use the identical
selection rule.
"""
import pandas as pd
import streamlit as st

from backtest_engine import NSE_UNIVERSES, compute_momentum_ranking, compute_trailing_volatility, ensure_stock_data
from streamlit_cache import cached_load_current_universe, cached_load_prices, cached_load_universe_symbols

st.set_page_config(page_title="Nifty 500 Stock Ranker", layout="wide")

with st.spinner("Fetching price data (first run only)..."):
    ensure_stock_data()

st.title("Nifty 500 Stock Ranker")
st.caption(
    "Ranks the current Nifty 500 constituents by trailing return, using the same formula "
    "as the momentum backtest (see the Backtest page) -- useful for checking what your next "
    "scheduled rebalance would pick, or monitoring an existing portfolio's ranks against "
    "your exit rule."
)

with st.sidebar:
    st.header("Universe")
    universe_options = list(NSE_UNIVERSES.keys()) + ["All ~970 symbols (no restriction)"]
    universe_label = st.selectbox(
        "Universe", universe_options, index=universe_options.index("Nifty 500"),
        help="Restricts ranking to one NSE index's CURRENT constituents. 'All ~970 symbols' "
             "ranks every symbol in the dataset, including historical/delisted names kept for "
             "backtesting -- not investable, shown for reference only."
    )
    allowed_symbols = None if universe_label.startswith("All") else cached_load_universe_symbols(universe_label)
    if allowed_symbols is not None:
        st.caption(
            f"Restricted to the current {len(allowed_symbols)} constituents of {universe_label}. "
            "Company name/industry lookup below is still keyed off the Nifty 500 list, so a "
            "stock outside it (e.g. from Microcap 250) shows '-' for those two columns."
        )

    st.header("Ranking parameters")
    rebal_freq_label = st.radio(
        "Rebalancing frequency", ["Monthly", "Weekly"], index=0,
        help="Weekly re-derives the ranking on a weekly price grid (lookback/skip in weeks, "
             "not months) and shows the ranking as of the latest complete WEEK instead of month."
    )
    is_weekly = rebal_freq_label == "Weekly"
    period_word = "week" if is_weekly else "month"
    price_freq = "W-FRI" if is_weekly else "ME"

    if is_weekly:
        lookback_months = st.slider(
            "Trailing return lookback (weeks)", min_value=4, max_value=104, value=52, step=1,
            help="52 weeks is the weekly equivalent of the monthly mode's 12-month default."
        )
        skip_months = st.slider(
            "Skip period (weeks)", min_value=0, max_value=12, value=4, step=1,
            help="Excludes the most recent N weeks from the lookback, to avoid short-term reversal effects."
        )
    else:
        lookback_months = st.slider("Trailing return lookback (months)", min_value=1, max_value=24, value=10, step=1)
        skip_months = st.slider(
            "Skip period (months)", min_value=0, max_value=3, value=1, step=1,
            help="Excludes the most recent N months from the lookback, to avoid short-term reversal effects."
        )
    min_price = st.number_input("Minimum price filter (Rs)", min_value=0.0, value=10.0, step=5.0)

    st.header("Volatility & risk-adjusted ranking")
    ranking_method_label = st.radio(
        "Ranking method", ["Absolute trailing return (default)", "Risk-adjusted (return / volatility)"],
        index=0,
        help="Risk-adjusted divides each stock's trailing return by its own trailing ANNUALIZED "
             "volatility (std dev of the same lookback window's returns x sqrt(12)) -- a "
             "Sharpe-like score, favoring smoother trends over merely bigger ones. Stocks with "
             "zero/undefined volatility are excluded, since the ratio wouldn't be meaningful."
    )
    use_risk_adjusted = ranking_method_label.startswith("Risk-adjusted")
    use_volatility_filter = st.checkbox(
        "Volatility filter", value=False,
        help="Excludes any stock whose trailing annualized volatility (same definition as above) "
             "exceeds the threshold below, before ranking -- independent of the ranking method "
             "chosen above."
    )
    max_volatility_pct = None
    if use_volatility_filter:
        max_volatility_pct = st.slider(
            "Max annualized volatility (%)", min_value=10.0, max_value=200.0, value=60.0, step=5.0,
        )

    price_col = st.selectbox("Price field", ["Adj Close", "Close"], index=0)
    top_n = st.number_input("Highlight top N (buy zone)", min_value=1, max_value=100, value=10, step=1)
    exit_rank = st.number_input(
        "Exit rank threshold", min_value=int(top_n), max_value=500, value=18, step=1,
        help="Ranks from N+1 up to this are a 'hold if already owned, don't buy new' zone -- "
             "matches a custom exit rule like 'exit when rank drops below 18'. Beyond this "
             "rank, a stock is firmly out."
    )
monthly_prices = cached_load_prices(price_col, price_freq)
as_of_date = monthly_prices.index.max()

universe_df = cached_load_current_universe()
name_by_symbol = dict(zip(universe_df["Symbol"], universe_df["Company Name"]))
industry_by_symbol = (
    dict(zip(universe_df["Symbol"], universe_df["Industry"])) if "Industry" in universe_df.columns else {}
)

ranked = compute_momentum_ranking(
    monthly_prices, None, as_of_date, lookback_months, skip_months, min_price, allowed_symbols,
    max_volatility_pct=max_volatility_pct, use_risk_adjusted=use_risk_adjusted,
)
used_fallback_date = False
if (ranked is None or ranked.empty) and len(monthly_prices.index) > 1:
    # The very latest period can have sparse data if only some symbols'
    # price files have been refreshed since the calendar rolled over (each
    # symbol is downloaded independently, not all on the same schedule) --
    # fall back to the most recent earlier period that actually has enough
    # coverage for the selected Universe, rather than showing a blank page.
    for fallback_date in reversed(monthly_prices.index[:-1][-3:]):
        candidate = compute_momentum_ranking(
            monthly_prices, None, fallback_date, lookback_months, skip_months, min_price, allowed_symbols,
            max_volatility_pct=max_volatility_pct, use_risk_adjusted=use_risk_adjusted,
        )
        if candidate is not None and not candidate.empty:
            as_of_date = fallback_date
            ranked = candidate
            used_fallback_date = True
            break

st.subheader(f"Ranking as of {as_of_date.date()}")
if used_fallback_date:
    st.caption(
        f"Note: the latest {period_word} in the dataset didn't have enough price data yet for "
        "this Universe (symbols are refreshed on independent schedules) -- fell back to the "
        f"most recent {period_word} that did."
    )
st.caption(
    f"This is the latest COMPLETE {period_word} in the downloaded data -- matches what your "
    f"next scheduled {period_word}-end rebalance would use. Source price data refreshes "
    "automatically every Saturday morning (see the note at the bottom of the page)."
)

if ranked is None or ranked.empty:
    st.warning("Not enough price history to compute a ranking with these parameters.")
else:
    score_col = "RiskAdjustedScore" if use_risk_adjusted else "TrailingReturn"
    table = ranked.to_frame(score_col).reset_index()
    table.columns = ["Symbol", score_col]
    table.insert(0, "Rank", range(1, len(table) + 1))
    table["CompanyName"] = table["Symbol"].map(name_by_symbol).fillna("-")
    table["Industry"] = table["Symbol"].map(industry_by_symbol).fillna("-")
    table["LastPrice"] = table["Symbol"].map(monthly_prices.loc[as_of_date])
    volatility = compute_trailing_volatility(monthly_prices, as_of_date, lookback_months, skip_months)
    table["Volatility"] = table["Symbol"].map(volatility) if volatility is not None else None

    def zone(rank: int) -> str:
        if rank <= top_n:
            return "Buy zone"
        elif rank <= exit_rank:
            return "Hold zone"
        return "Out"

    table["Zone"] = table["Rank"].apply(zone)

    n_buy = int((table["Zone"] == "Buy zone").sum())
    n_hold = int((table["Zone"] == "Hold zone").sum())
    st.markdown(
        f"**{n_buy}** stocks in the buy zone (rank 1-{top_n})  |  "
        f"**{n_hold}** in the hold zone (rank {top_n + 1}-{exit_rank})  |  "
        f"**{len(table)}** total ranked"
    )

    def highlight_zone(row: pd.Series) -> list:
        if row["Zone"] == "Buy zone":
            return ["background-color: rgba(46, 160, 67, 0.25)"] * len(row)
        elif row["Zone"] == "Hold zone":
            return ["background-color: rgba(210, 153, 34, 0.2)"] * len(row)
        return [""] * len(row)

    score_fmt = "{:.2f}" if use_risk_adjusted else "{:.2%}"
    st.dataframe(
        table[["Rank", "Symbol", "CompanyName", "Industry", score_col, "Volatility", "LastPrice", "Zone"]]
        .style.apply(highlight_zone, axis=1)
        .format({score_col: score_fmt, "Volatility": "{:.2%}", "LastPrice": "{:.2f}"}),
        hide_index=True, height=600, use_container_width=True,
    )

    st.download_button(
        "Download full ranking as CSV",
        table.to_csv(index=False).encode("utf-8"),
        file_name=f"nifty500_ranking_{as_of_date.date()}.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Ranking formula: trailing lookback-period return, skipping the most recent skip periods, "
    "computed from period-end closes -- identical to the momentum backtest's selection rule "
    "(compute_momentum_ranking in backtest_engine.py), so this page and the backtest never "
    "drift out of sync with each other. Volatility (shown as its own column regardless of "
    "ranking method) is the annualized std dev of that SAME lookback window's period returns "
    "(x sqrt(12) for monthly mode) -- see compute_trailing_volatility. Risk-adjusted mode "
    "ranks/sorts by trailing return divided by this volatility instead of the raw return."
)
st.caption(
    "Data freshness: a GitHub Actions workflow (.github/workflows/weekly-data-refresh.yml) "
    "tops up the source price data and NSE universe/F&O lists every Saturday morning (IST) "
    "and updates the release asset the deployed app bootstraps from. The already-running app "
    "still needs a manual Reboot (Manage app -> Reboot app on Streamlit Cloud) to pick up a "
    "refresh, though -- it only re-fetches source data on a fresh start, not automatically."
)
