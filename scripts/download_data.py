"""
Download daily OHLCV history for all current Nifty 500 constituents plus
the Nifty 500 index itself, from 2008-01-01 to today, via yfinance.

Note on survivorship bias: this uses TODAY's index membership list and
walks each ticker's full available history. Stocks that were removed
from the index (delisted, merged, demoted) between 2008 and now are not
included, and stocks that recently joined will show history predating
their actual index membership. This is a real limitation for momentum
backtests -- see the README note this script prints at the end.
"""
import time
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
LIST_CSV = ROOT / "data" / "nifty500_list.csv"
STOCKS_DIR = ROOT / "data" / "stocks"
INDEX_DIR = ROOT / "data" / "index"

START = "2008-01-01"
END = None  # None = through today

INDEX_TICKERS = {
    "NIFTY500": "^CRSLDX",
    "NIFTY50": "^NSEI",
}

MAX_RETRIES = 3
RETRY_SLEEP = 5


def download_one(ticker: str, start: str, end):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=False,
                threads=False,
            )
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


def download_index():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    for name, ticker in INDEX_TICKERS.items():
        print(f"Downloading index {name} ({ticker})...")
        df = download_one(ticker, START, END)
        if df is None:
            print(f"  Could not download {name}")
            continue
        out = INDEX_DIR / f"{name}.csv"
        df.to_csv(out)
        print(f"  Saved {len(df)} rows to {out}")


def download_stocks():
    STOCKS_DIR.mkdir(parents=True, exist_ok=True)
    constituents = pd.read_csv(LIST_CSV)
    symbols = constituents["Symbol"].tolist()

    failed = []
    skipped = []
    for sym in tqdm(symbols, desc="Downloading stocks"):
        out = STOCKS_DIR / f"{sym}.csv"
        if out.exists():
            skipped.append(sym)
            continue
        ticker = f"{sym}.NS"
        df = download_one(ticker, START, END)
        if df is None or df.empty:
            failed.append(sym)
            continue
        df.to_csv(out)
        time.sleep(0.3)  # be polite to Yahoo's endpoint

    print(f"\nDone. {len(symbols) - len(failed) - len(skipped)} newly downloaded, "
          f"{len(skipped)} already present, {len(failed)} failed.")
    if failed:
        failed_path = ROOT / "data" / "failed_symbols.txt"
        failed_path.write_text("\n".join(failed))
        print(f"Failed symbols written to {failed_path}")
        print("Retry failures by re-running this script (existing files are skipped).")


if __name__ == "__main__":
    download_index()
    download_stocks()
    print(
        "\nNOTE: this list reflects CURRENT Nifty 500 membership. Stocks that "
        "left the index between 2008 and today are absent, and current "
        "members' pre-inclusion history is included even though they weren't "
        "in the index yet. This introduces survivorship bias -- fine for "
        "learning/prototyping a momentum strategy, but be aware of it before "
        "trusting absolute backtest returns."
    )
