"""
Download price history for symbols that appear in one of the NSE universe
lists (data/universes/*.csv, see get_nse_universes.py) but aren't already in
data/stocks/ -- mostly smaller/more recently listed names pulled in by
Nifty Microcap 250 / Nifty Total Market that weren't part of the original
bulk download. Same download logic as download_historical_stocks.py.
"""
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
STOCKS_DIR = ROOT / "data" / "stocks"
UNIVERSES_DIR = ROOT / "data" / "universes"

START = "2008-01-01"
END = None
MAX_RETRIES = 3
RETRY_SLEEP = 5


def download_one(ticker: str):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=False, threads=False)
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return df
            return None
        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f"  FAILED {ticker}: {e}", file=sys.stderr)
                return None
            time.sleep(RETRY_SLEEP)
    return None


def main():
    all_universe_symbols: set[str] = set()
    for f in UNIVERSES_DIR.glob("*.csv"):
        all_universe_symbols |= set(pd.read_csv(f)["Symbol"])

    have = {f.stem for f in STOCKS_DIR.glob("*.csv")}
    need = sorted(all_universe_symbols - have)
    print(f"{len(need)} symbols to attempt (from {len(all_universe_symbols)} across all universes, {len(have)} already downloaded)")

    failed = []
    for sym in tqdm(need, desc="Downloading missing universe symbols"):
        out = STOCKS_DIR / f"{sym}.csv"
        if out.exists():
            continue
        df = download_one(f"{sym}.NS")
        if df is None or df.empty:
            failed.append(sym)
            continue
        df.to_csv(out)
        time.sleep(0.3)

    print(f"\nDone. {len(need) - len(failed)} newly downloaded, {len(failed)} failed.")
    if failed:
        print("Failed:", failed)


if __name__ == "__main__":
    main()
