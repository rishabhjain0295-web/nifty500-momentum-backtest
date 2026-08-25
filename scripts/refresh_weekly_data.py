"""
Weekly data refresh, run on a schedule (see the "schedule" cloud-agent
routine set up alongside this script) so the Stock Ranker page's live
ranking -- and every other page's most recent month/week -- doesn't go
stale between sessions.

Does four things:
  1. Tops up data/stocks/*.csv and data/index/{NIFTY500,NIFTY50}.csv with
     the last ~45 days of daily OHLCV (yfinance), merged into each existing
     file (overlapping days are overwritten with the fresh fetch, in case
     of a late adjustment/split; new days are appended). Full re-download
     per symbol would also work but is much slower and unnecessary --
     these files already have the 2008-present history, they just need
     topping up.
  2. Re-downloads data/universes/*.csv (get_nse_universes.py) and
     data/fno_stocks.csv (get_fno_list.py) -- current-snapshot lists,
     cheap to just refresh outright rather than top up.
  3. Re-zips data/stocks/ and re-uploads it as the stocks.zip GitHub
     Release asset the deployed app bootstraps from (gh release upload
     --clobber) -- data/stocks/ itself is gitignored (too large for git),
     this is the only way the deployed app sees the refresh.
  4. Commits and pushes data/universes/ and data/fno_stocks.csv (these ARE
     small enough to live in git directly).

Does NOT touch data/hourly/, data/15min/, or data/etfs/ -- out of scope
for this specific request (Stock Ranker only uses data/stocks + the
universe lists), and those have their own separate, shorter-window
refresh cadences that would need their own script if ever needed.

The deployed Streamlit Cloud app still needs a manual Reboot to pick up
a refreshed stocks.zip -- ensure_stock_data() only bootstraps once per
container (no-op if data/stocks/ already has files), it doesn't
periodically re-check the Release asset. This script updates the SOURCE
data; reflecting it in the live app is still a manual step.
"""
import subprocess
import sys
import time
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd
import yfinance as yf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STOCKS_DIR = ROOT / "data" / "stocks"
INDEX_DIR = ROOT / "data" / "index"

TOPUP_PERIOD = "45d"
MAX_RETRIES = 3
RETRY_SLEEP = 5


def download_recent(ticker: str):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = yf.download(ticker, period=TOPUP_PERIOD, progress=False, auto_adjust=False, threads=False)
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


def topup_one(path: Path, ticker: str) -> bool:
    fresh = download_recent(ticker)
    if fresh is None or fresh.empty:
        return False
    try:
        existing = pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception:
        return False
    fresh.index.name = existing.index.name
    combined = pd.concat([existing[~existing.index.isin(fresh.index)], fresh]).sort_index()
    combined.to_csv(path)
    return True


def topup_stocks():
    symbols = sorted(f.stem for f in STOCKS_DIR.glob("*.csv"))
    print(f"Topping up {len(symbols)} symbols in {STOCKS_DIR} (period={TOPUP_PERIOD})...")
    failed = []
    for sym in tqdm(symbols, desc="Topping up data/stocks"):
        ok = topup_one(STOCKS_DIR / f"{sym}.csv", f"{sym}.NS")
        if not ok:
            failed.append(sym)
        time.sleep(0.2)
    print(f"  {len(symbols) - len(failed)} refreshed, {len(failed)} failed (kept as-is).")
    if failed:
        print("  Failed:", failed[:20], "..." if len(failed) > 20 else "")


def topup_indices():
    tickers = {"NIFTY500": "^CRSLDX", "NIFTY50": "^NSEI"}
    print("Topping up index files...")
    for name, ticker in tickers.items():
        path = INDEX_DIR / f"{name}.csv"
        if not path.exists():
            continue
        ok = topup_one(path, ticker)
        print(f"  {name}: {'ok' if ok else 'FAILED'}")


def refresh_universe_lists():
    print("Refreshing NSE universe lists...")
    import get_nse_universes
    get_nse_universes.main()
    print("Refreshing F&O eligible list...")
    import get_fno_list
    get_fno_list.main()


def reupload_stocks_zip():
    print("Re-zipping data/stocks/...")
    zip_path = ROOT / "data" / "stocks.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(STOCKS_DIR.glob("*.csv")):
            zf.write(f, f.name)
    size_mb = zip_path.stat().st_size / 1e6
    print(f"  {size_mb:.1f} MB")
    print("Uploading to GitHub Release data-v1 (stocks.zip)...")
    subprocess.run(
        ["gh", "release", "upload", "data-v1", str(zip_path), "--clobber"],
        check=True, cwd=ROOT,
    )
    zip_path.unlink()


def commit_and_push():
    print("Committing data/universes/ and data/fno_stocks.csv...")
    subprocess.run(["git", "add", "data/universes", "data/fno_stocks.csv"], check=True, cwd=ROOT)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--cached"], check=True, cwd=ROOT, capture_output=True, text=True,
    )
    if not status.stdout.strip():
        print("  No changes to commit.")
        return
    subprocess.run(
        ["git", "commit", "-m", "Weekly data refresh: universe lists + F&O list\n\n"
         "Automated via the scheduled weekly refresh routine "
         "(scripts/refresh_weekly_data.py).\n\n"
         "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"],
        check=True, cwd=ROOT,
    )
    subprocess.run(["git", "push"], check=True, cwd=ROOT)
    print("  Pushed.")


def main():
    topup_stocks()
    topup_indices()
    refresh_universe_lists()
    reupload_stocks_zip()
    commit_and_push()
    print("\nWeekly refresh complete. The deployed Streamlit Cloud app still needs a manual "
          "Reboot to pick up the refreshed stocks.zip.")


if __name__ == "__main__":
    main()
