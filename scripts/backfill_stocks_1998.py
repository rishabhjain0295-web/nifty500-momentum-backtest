"""
Extends every existing data/stocks/<SYMBOL>.csv back to 1998-01-01 (as far
back as Yahoo Finance generally has usable NSE coverage -- see the
conversation that led to this script: individual large/mid-cap stocks
often have data back to 1998, unlike the NIFTY 500/50 index LEVEL itself,
which Yahoo only carries from 2005/2007 -- see
download_nse_benchmark_history.py for that separate fix, sourced directly
from NSE instead).

Backfill, not a full re-download: for each symbol that already has a
file, fetches ONLY the gap between 1998-01-01 and that file's current
earliest date, then prepends it -- much faster than re-pulling the whole
2008-present history again, same reasoning as refresh_weekly_data.py's
topup_one (which does the analogous thing at the RECENT end).

Symbols with no data at all yet are out of scope here -- see
download_historical_stocks.py for those (point-in-time membership
calendar names not yet in data/stocks/). A symbol that fails here
(delisted before Yahoo's coverage window, renamed beyond what the
symbol-change master resolved, or otherwise unavailable) just keeps
its current (2008-starting) history -- not a bug, a real data
limitation like the ones documented elsewhere in this project.
"""
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
STOCKS_DIR = ROOT / "data" / "stocks"

BACKFILL_START = "1998-01-01"
MAX_RETRIES = 3
RETRY_SLEEP = 5


def download_one(ticker: str, start: str, end: str):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False, threads=False)
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


def backfill_one(path: Path, ticker: str) -> str:
    try:
        existing = pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception:
        return "read_failed"
    if existing.empty:
        return "read_failed"
    earliest = existing.index.min()
    if earliest <= pd.Timestamp(BACKFILL_START):
        return "already_covered"

    fresh = download_one(ticker, BACKFILL_START, earliest.date().isoformat())
    if fresh is None or fresh.empty:
        return "no_older_data"

    fresh.index.name = existing.index.name
    combined = pd.concat([fresh[~fresh.index.isin(existing.index)], existing]).sort_index()
    combined.to_csv(path)
    return "backfilled"


def main():
    symbols = sorted(f.stem for f in STOCKS_DIR.glob("*.csv"))
    print(f"Checking {len(symbols)} existing symbols for backfill to {BACKFILL_START}...")

    counts = {"backfilled": 0, "already_covered": 0, "no_older_data": 0, "read_failed": 0}
    no_older = []
    for sym in tqdm(symbols, desc="Backfilling data/stocks to 1998"):
        result = backfill_one(STOCKS_DIR / f"{sym}.csv", f"{sym}.NS")
        counts[result] += 1
        if result == "no_older_data":
            no_older.append(sym)
        if result != "already_covered":
            time.sleep(0.3)  # be polite to Yahoo's endpoint -- only sleep after an actual request

    print(f"\nDone. {counts['backfilled']} backfilled, {counts['already_covered']} already covered, "
          f"{counts['no_older_data']} had no data available before their current start (kept as-is), "
          f"{counts['read_failed']} unreadable (skipped).")
    if no_older:
        out = ROOT / "data" / "no_pre2008_data_symbols.txt"
        out.write_text("\n".join(no_older))
        print(f"Symbols with no pre-2008 Yahoo data written to {out} -- likely IPO'd or were added to "
              f"Yahoo's coverage after 1998, a real limitation, not a bug.")


if __name__ == "__main__":
    main()
