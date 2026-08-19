"""
Extends data/hourly/ to cover every current F&O-eligible stock (see
get_fno_list.py -> data/fno_stocks.csv), needed for the Short Momentum
(F&O) swing strategy's hourly/2h timeframe. The original
download_hourly_data.py only covered stocks that appeared in the TOP-N
momentum universe (for the long EMA crossover strategy) -- large, stable
blue-chips that are common F&O underlyings (RELIANCE, TCS, HDFCBANK, ...)
rarely rank there, so most F&O names were never downloaded. This script
just downloads whatever's missing; safe to rerun (skips existing files).
"""
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from download_hourly_data import HOURLY_DIR, download_one  # noqa: E402


def main():
    HOURLY_DIR.mkdir(parents=True, exist_ok=True)
    fno_symbols = sorted(pd.read_csv(ROOT / "data" / "fno_stocks.csv")["Symbol"])
    missing = [s for s in fno_symbols if not (HOURLY_DIR / f"{s}.csv").exists()]
    print(f"{len(fno_symbols)} F&O symbols, {len(missing)} missing hourly data")

    failed = []
    for i, sym in enumerate(missing, 1):
        print(f"[{i}/{len(missing)}] {sym}...", end=" ", flush=True)
        df = download_one(f"{sym}.NS")
        if df is None or df.empty:
            print("FAILED")
            failed.append(sym)
            continue
        df.to_csv(HOURLY_DIR / f"{sym}.csv")
        print("ok")
        time.sleep(0.3)

    print(f"\nDone. {len(missing) - len(failed)} downloaded, {len(failed)} failed.")
    if failed:
        print("Failed:", failed)


if __name__ == "__main__":
    main()
