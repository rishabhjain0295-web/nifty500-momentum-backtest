"""
Download price history for symbols found in the point-in-time Nifty 500
membership calendar (data/nifty500_membership_calendar.csv) that aren't
already in data/stocks/. Many of these will fail -- if a company was
delisted, merged, or renamed beyond what the symbol/name-change master
files captured, there is no live Yahoo Finance ticker for it. That's an
expected, real data limitation, not a bug: those companies genuinely have
no tradable price series available for free anywhere.
"""
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
STOCKS_DIR = ROOT / "data" / "stocks"
CALENDAR_CSV = ROOT / "data" / "nifty500_membership_calendar.csv"

START = "2008-01-01"
END = None
MAX_RETRIES = 3
RETRY_SLEEP = 5


def download_one(ticker: str, start: str, end):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = yf.download(
                ticker, start=start, end=end, progress=False, auto_adjust=False, threads=False,
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
    STOCKS_DIR.mkdir(parents=True, exist_ok=True)
    cal = pd.read_csv(CALENDAR_CSV)
    have = {f.stem for f in STOCKS_DIR.glob("*.csv")}
    need = sorted(set(cal["symbol"]) - have)
    print(f"{len(need)} symbols to attempt (from {cal['symbol'].nunique()} in calendar, {len(have)} already downloaded)")

    failed = []
    for sym in tqdm(need, desc="Downloading historical/delisted-candidate symbols"):
        out = STOCKS_DIR / f"{sym}.csv"
        if out.exists():
            continue
        ticker = f"{sym}.NS"
        df = download_one(ticker, START, END)
        if df is None or df.empty:
            failed.append(sym)
            continue
        df.to_csv(out)
        time.sleep(0.3)

    print(f"\nDone. {len(need) - len(failed)} newly downloaded, {len(failed)} failed "
          f"(likely genuinely delisted/no live ticker).")
    if failed:
        failed_path = ROOT / "data" / "failed_historical_symbols.txt"
        failed_path.write_text("\n".join(failed))
        print(f"Failed symbols written to {failed_path}")


if __name__ == "__main__":
    main()
