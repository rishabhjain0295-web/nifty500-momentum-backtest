"""
Live Nifty 500 momentum ranker: ranks all CURRENT Nifty 500 constituents by
trailing return, using the exact same formula as the backtest engine
(compute_momentum_ranking in backtest_engine.py) -- shows what the momentum
backtest's next scheduled rebalance would pick, as of the latest complete
month-end in the downloaded data, and lets you check an existing holding's
current rank against a custom exit threshold.
"""
import pandas as pd
import streamlit as st

from backtest_engine import compute_momentum_ranking, ensure_stock_data
from streamlit_cache import cached_load_current_universe, cached_load_prices

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
    st.header("Ranking parameters")
    lookback_months = st.slider("Trailing return lookback (months)", min_value=1, max_value=24, value=10, step=1)
    skip_months = st.slider(
        "Skip period (months)", min_value=0, max_value=3, value=1, step=1,
        help="Excludes the most recent N months from the lookback, to avoid short-term reversal effects."
    )
    min_price = st.number_input("Minimum price filter (Rs)", min_value=0.0, value=10.0, step=5.0)
    price_col = st.selectbox("Price field", ["Adj Close", "Close"], index=0)
    top_n = st.number_input("Highlight top N (buy zone)", min_value=1, max_value=100, value=10, step=1)
    exit_rank = st.number_input(
        "Exit rank threshold", min_value=int(top_n), max_value=500, value=18, step=1,
        help="Ranks from N+1 up to this are a 'hold if already owned, don't buy new' zone -- "
             "matches a custom exit rule like 'exit when rank drops below 18'. Beyond this "
             "rank, a stock is firmly out."
    )
    use_current_universe = st.checkbox(
        "Restrict to current Nifty 500 only", value=True,
        help="On (recommended): only ranks today's actual Nifty 500 constituents. Off: ranks "
             "all ~970 symbols in the dataset, including historical/delisted names kept for "
             "backtesting -- not investable, shown for reference only."
    )

monthly_prices = cached_load_prices(price_col)
as_of_date = monthly_prices.index.max()

universe_df = cached_load_current_universe()
current_symbols = set(universe_df["Symbol"])
name_by_symbol = dict(zip(universe_df["Symbol"], universe_df["Company Name"]))
industry_by_symbol = (
    dict(zip(universe_df["Symbol"], universe_df["Industry"])) if "Industry" in universe_df.columns else {}
)

if use_current_universe:
    eligible_cols = [c for c in monthly_prices.columns if c in current_symbols]
    ranking_input = monthly_prices[eligible_cols]
else:
    ranking_input = monthly_prices

ranked = compute_momentum_ranking(ranking_input, None, as_of_date, lookback_months, skip_months, min_price)

st.subheader(f"Ranking as of {as_of_date.date()}")
st.caption(
    "This is the latest COMPLETE month-end in the downloaded data -- matches what your next "
    "scheduled month-end rebalance would use. Re-run the data download scripts to refresh."
)

if ranked is None or ranked.empty:
    st.warning("Not enough price history to compute a ranking with these parameters.")
else:
    table = ranked.to_frame("TrailingReturn").reset_index()
    table.columns = ["Symbol", "TrailingReturn"]
    table.insert(0, "Rank", range(1, len(table) + 1))
    table["CompanyName"] = table["Symbol"].map(name_by_symbol).fillna("-")
    table["Industry"] = table["Symbol"].map(industry_by_symbol).fillna("-")
    table["LastPrice"] = table["Symbol"].map(monthly_prices.loc[as_of_date])

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

    st.dataframe(
        table[["Rank", "Symbol", "CompanyName", "Industry", "TrailingReturn", "LastPrice", "Zone"]]
        .style.apply(highlight_zone, axis=1)
        .format({"TrailingReturn": "{:.2%}", "LastPrice": "{:.2f}"}),
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
    "Ranking formula: trailing lookback-month return, skipping the most recent skip months, "
    "computed from month-end closes -- identical to the momentum backtest's selection rule "
    "(compute_momentum_ranking in backtest_engine.py), so this page and the backtest never "
    "drift out of sync with each other."
)
