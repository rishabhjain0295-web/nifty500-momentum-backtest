"""
Download 15-minute OHLC for the RSI Oversold Reversal swing strategy
(pages/3_Swing_Trading.py) -- the only strategy in this app needing
sub-hourly bars.

Yahoo Finance caps intervals finer than 1h at a 60-day trailing window
(much shorter than the ~2-3yr available at 1h, let alone the ~18yr of
daily data used elsewhere in this project) -- there is no way around this,
it's a hard Yahoo-side limit, not a download-strategy choice. This makes
the 15-minute (and derived 30-minute, see backtest_engine.load_30min_
full_ohlc) option for this strategy only useful for a short, recent
sample -- not a real multi-year backtest. Downloads for the SAME symbol
set already curated in data/hourly/ (top momentum + F&O, see
download_hourly_data.py / download_fno_hourly_data.py) rather than
recomputing a universe, since a fresh 60-day window makes "which stocks
were in the momentum universe historically" moot -- what matters is
having data for whatever's tradeable now.

Saves to data/15min/<SYMBOL>.csv (Datetime index in IST, Open/High/Low/
Close/Volume columns).
"""
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HOURLY_DIR = ROOT / "data" / "hourly"
FIFTEEN_MIN_DIR = ROOT / "data" / "15min"

MAX_RETRIES = 3
RETRY_SLEEP = 5


def download_one(ticker: str):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = yf.download(ticker, interval="15m", period="60d", progress=False, threads=False)
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
    FIFTEEN_MIN_DIR.mkdir(parents=True, exist_ok=True)
    symbols = sorted(f.stem for f in HOURLY_DIR.glob("*.csv"))
    print(f"{len(symbols)} symbols (reusing the existing hourly universe)")

    failed = []
    for sym in tqdm(symbols, desc="Downloading 15min data"):
        out = FIFTEEN_MIN_DIR / f"{sym}.csv"
        if out.exists():
            continue
        df = download_one(f"{sym}.NS")
        if df is None or df.empty:
            failed.append(sym)
            continue
        df.to_csv(out)
        time.sleep(0.3)

    print(f"\nDone. {len(symbols) - len(failed)} downloaded, {len(failed)} failed.")
    if failed:
        print("Failed:", failed)


if __name__ == "__main__":
    main()
