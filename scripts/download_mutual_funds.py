"""
Download daily NAV history for the curated mutual fund list
(backtest_engine.MUTUAL_FUNDS), for the Mutual Fund Comparison page.

Source: api.mfapi.in -- a free, no-auth, no-rate-limit API backed by AMFI's
own published NAV data (confirmed empirically: GET /mf/{schemeCode} returns
daily NAV back to each scheme's inception, plus fund house/category
metadata). A small pacing delay is used anyway, out of courtesy, even
though the API states no rate limit.

Saves to data/mutual_funds/<scheme_code>.csv with a Date index and one NAV
column, sorted ascending -- see backtest_engine.load_mutual_fund_nav.
"""
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest_engine import MUTUAL_FUNDS  # noqa: E402

OUT_DIR = ROOT / "data" / "mutual_funds"
NAV_URL = "https://api.mfapi.in/mf/{scheme_code}"

MAX_RETRIES = 3
RETRY_SLEEP = 3
REQUEST_PAUSE = 0.3


def fetch_one(session: requests.Session, scheme_code: int) -> pd.DataFrame | None:
    url = NAV_URL.format(scheme_code=scheme_code)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, timeout=20)
            if r.status_code == 200:
                payload = r.json()
                records = payload.get("data", [])
                if not records:
                    return None
                df = pd.DataFrame(records)
                df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
                df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
                df = df.dropna(subset=["nav"]).sort_values("date")
                df = df.rename(columns={"date": "Date", "nav": "NAV"})
                return df.set_index("Date")[["NAV"]]
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP)
                continue
            print(f"  WARNING: status {r.status_code} after {MAX_RETRIES} attempts: {scheme_code}", file=sys.stderr)
            return None
        except requests.RequestException as e:
            if attempt == MAX_RETRIES:
                print(f"  WARNING: {e} after {MAX_RETRIES} attempts: {scheme_code}", file=sys.stderr)
                return None
            time.sleep(RETRY_SLEEP)
    return None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    failed = []
    for name, info in tqdm(MUTUAL_FUNDS.items(), desc="Downloading mutual fund NAV history"):
        scheme_code = info["scheme_code"]
        out = OUT_DIR / f"{scheme_code}.csv"
        if out.exists():
            continue
        df = fetch_one(session, scheme_code)
        if df is None or df.empty:
            failed.append((name, scheme_code))
            continue
        df.to_csv(out)
        time.sleep(REQUEST_PAUSE)

    n_ok = len(MUTUAL_FUNDS) - len(failed)
    print(f"\nDone. {n_ok} downloaded, {len(failed)} failed.")
    if failed:
        print("Failed:", failed)


if __name__ == "__main__":
    main()
