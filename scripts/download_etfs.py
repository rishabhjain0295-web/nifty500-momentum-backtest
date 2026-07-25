"""
Download daily OHLCV history for a set of popular India ETFs from 2008-01-01
to today, via yfinance. Uses the same download_one() retry logic as
download_data.py.
"""
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
ETF_DIR = ROOT / "data" / "etfs"

START = "2008-01-01"
END = None

ETFS = {
    "JUNIORBEES": "JUNIORBEES.NS",
    "BANKBEES": "BANKBEES.NS",
    "LIQUIDBEES": "LIQUIDBEES.NS",
    "MIDCAPBEES": "MID150BEES.NS",  # Nippon renamed MIDCAPBEES -> MID150BEES (Nifty Midcap 150)
    "GOLDBEES": "GOLDBEES.NS",
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


def main():
    ETF_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    for name, ticker in ETFS.items():
        print(f"Downloading {name} ({ticker})...")
        df = download_one(ticker, START, END)
        if df is None or df.empty:
            print(f"  FAILED: {name}")
            failed.append(name)
            continue
        out = ETF_DIR / f"{name}.csv"
        df.to_csv(out)
        print(f"  Saved {len(df)} rows ({df.index.min().date()} to {df.index.max().date()}) to {out}")
        time.sleep(0.3)

    if failed:
        print(f"\nFailed: {failed}")
    else:
        print("\nAll ETFs downloaded successfully.")


if __name__ == "__main__":
    main()
