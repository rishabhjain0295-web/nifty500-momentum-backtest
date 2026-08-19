"""
Fetch the current NSE F&O (futures & options) eligible stock list, used by
the Short Momentum (FnO) swing strategy to restrict its universe to stocks
that actually have tradeable stock futures (see pages/3_Swing_Trading.py's
"Short weak momentum (F&O)" strategy and swing_engine.py's
run_short_ema_crossover_backtest -- the strategy is modeled as shorting the
stock directly since we only have spot price history, which is economically
close to a fully-margined stock future position; see that function's
docstring for the simplification this implies).

NSE blocks plain requests without browser-like headers/cookies, so we first
hit the homepage to pick up cookies, then request the CSV (same pattern as
get_nifty500_list.py). Saves the derivative-eligible equity symbols (index
futures like NIFTY/BANKNIFTY are dropped) to data/fno_stocks.csv.
"""
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent

CSV_URL = "https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv"
HOME_URL = "https://www.nseindia.com"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

OUT_PATH = ROOT / "data" / "fno_stocks.csv"

# Non-equity underlyings that appear in the same NSE F&O lot-size file
# (index futures/options, not single-stock -- irrelevant to this strategy).
INDEX_UNDERLYINGS = {
    "NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50", "NIFTY500", "NIFTYFPI",
}
# The file also embeds a mid-document section-divider row ("Derivatives on
# Individual Securities,Symbol,...") that isn't a real symbol -- its SYMBOL
# column literally reads "Symbol", matching our own header name.
STRAY_ROWS = {"SYMBOL", "Symbol"}


def main():
    session = requests.Session()
    session.headers.update(HEADERS)

    try:
        session.get(HOME_URL, timeout=15)
    except requests.RequestException as e:
        print(f"Warning: could not warm up session against NSE homepage: {e}", file=sys.stderr)

    resp = session.get(CSV_URL, timeout=15)
    resp.raise_for_status()

    raw_path = ROOT / "data" / "_fno_mktlots_raw.csv"
    with open(raw_path, "wb") as f:
        f.write(resp.content)

    df = pd.read_csv(raw_path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    symbol_col = next(c for c in df.columns if c.upper() == "SYMBOL")
    symbols = (
        df[symbol_col].astype(str).str.strip().dropna()
    )
    symbols = symbols[symbols != ""]
    symbols = sorted((set(symbols) - INDEX_UNDERLYINGS) - STRAY_ROWS)

    pd.DataFrame({"Symbol": symbols}).to_csv(OUT_PATH, index=False)
    raw_path.unlink()
    print(f"Downloaded {len(symbols)} F&O-eligible stock symbols to {OUT_PATH}")
    print(symbols[:10])


if __name__ == "__main__":
    main()
