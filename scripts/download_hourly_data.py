"""
Download hourly OHLC for the stocks needed by the hourly EMA-crossover swing
strategy (pages/3_Swing_Trading.py's EMA 15/50 variant).

Yahoo Finance only serves hourly ("1h") data for roughly the trailing 2-3
years -- there is no way to get hourly history back to 2008 like the rest
of this project's daily data. This script first figures out which stocks
actually appeared in the top-N momentum universe within that available
window (so we don't waste time/requests downloading hourly data for ~970
stocks when only a couple hundred were ever relevant), then downloads
hourly OHLC for exactly that set, using period='730d' -- empirically this
returns more history than period='max' for the 1h interval (observed back
to ~2023-09, not just ~2024-08 as the documented 730-day limit would
suggest -- Yahoo's actual enforcement seems more lenient than documented,
but don't rely on this consistently reaching further back).

Saves to data/hourly/<SYMBOL>.csv (Date/Datetime index in IST, Open/High/
Low/Close/Volume columns).
"""
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest_engine import compute_momentum_ranking, load_membership_matrix, load_prices  # noqa: E402

HOURLY_DIR = ROOT / "data" / "hourly"
UNIVERSE_LOOKBACK_MONTHS = 12
UNIVERSE_SKIP_MONTHS = 1
UNIVERSE_TOP_N = 30  # generous ceiling; UI's n_stocks slider for this strategy should not exceed this
WINDOW_START = pd.Timestamp("2023-08-01")  # a bit before the ~3yr of hourly history we've observed

MAX_RETRIES = 3
RETRY_SLEEP = 5


def find_universe_symbols() -> set[str]:
    monthly_prices = load_prices("Adj Close")
    membership = load_membership_matrix(monthly_prices.index, monthly_prices.columns)
    rebalance_dates = [d for d in monthly_prices.index if d >= WINDOW_START]

    union_symbols: set[str] = set()
    for d in rebalance_dates:
        ranked = compute_momentum_ranking(
            monthly_prices, membership, d, UNIVERSE_LOOKBACK_MONTHS, UNIVERSE_SKIP_MONTHS, min_price=10.0
        )
        if ranked is not None:
            union_symbols |= set(ranked.head(UNIVERSE_TOP_N).index.tolist())
    return union_symbols


def download_one(ticker: str):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = yf.download(ticker, interval="1h", period="730d", progress=False, threads=False)
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
    HOURLY_DIR.mkdir(parents=True, exist_ok=True)
    print("Determining which stocks need hourly data (top-N momentum universe, last ~3 years)...")
    symbols = sorted(find_universe_symbols())
    print(f"  {len(symbols)} distinct symbols")

    failed = []
    for sym in tqdm(symbols, desc="Downloading hourly data"):
        out = HOURLY_DIR / f"{sym}.csv"
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
        (ROOT / "data" / "failed_hourly_symbols.txt").write_text("\n".join(failed))
        print(f"Failed symbols written to data/failed_hourly_symbols.txt")


if __name__ == "__main__":
    main()
