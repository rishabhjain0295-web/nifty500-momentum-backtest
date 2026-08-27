"""
Download 30-minute OHLCV history from Upstox's Historical Candle Data V3
API, for the RSI Oversold Reversal swing strategy's 30-min timeframe --
replaces the ~60-day Yahoo Finance window with real history back to
2022-01-01 (~4.5 years), the earliest date Upstox's V3 minute-candle
endpoint supports (confirmed empirically, not just from docs).

Needs a free Upstox "Analytics Token" (read-only, 1-year validity, includes
Historical Data access -- see https://upstox.com/developer/api-documentation/
analytics-token/ for how to generate one) saved as PLAIN TEXT to
secrets/upstox_token.txt (gitignored).

Downloads for the same symbol set already curated in data/hourly/ (top
momentum + F&O universe, see download_hourly_data.py / download_fno_hourly_
data.py) -- not the full ~1000-symbol data/stocks/ universe, since that's
already the relevant pool for every intraday swing strategy in this project.

Rate limits (per https://upstox.com/developer/api-documentation/
rate-limiting/): 50/sec, 500/min, 2000/30min for this endpoint category --
this script paces well under that, so a run shouldn't need to worry about
throttling in practice.

Saves to data/30min/<SYMBOL>.csv with a naive (IST wall-clock) DatetimeIndex
and Open/High/Low/Close/Volume columns -- same convention as data/hourly/
and data/15min/.
"""
import gzip
import io
import sys
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TOKEN_PATH = ROOT / "secrets" / "upstox_token.txt"
HOURLY_DIR = ROOT / "data" / "hourly"
OUT_DIR = ROOT / "data" / "30min"

INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.csv.gz"
HISTORICAL_URL = "https://api.upstox.com/v3/historical-candle/{instrument_key}/minutes/30/{to_date}/{from_date}"

START_DATE = pd.Timestamp("2022-01-01")  # earliest date the V3 minute endpoint actually returns data for
CHUNK_DAYS = 90  # matches the documented per-request cap for 16-300 min intervals (1 quarter)

MAX_RETRIES = 3
RETRY_SLEEP = 3
REQUEST_PAUSE = 0.15  # ~6-7 req/sec, well under the 50/sec limit


def load_token() -> str:
    if not TOKEN_PATH.exists():
        raise RuntimeError(
            f"Missing {TOKEN_PATH}. Generate a free Analytics Token at "
            "https://upstox.com/developer/apps (Analytics tab) and save it there as plain text."
        )
    token = TOKEN_PATH.read_text(encoding="utf-8-sig").strip()
    if not token:
        raise RuntimeError(f"{TOKEN_PATH} is empty.")
    return token


def load_instrument_map() -> dict[str, str]:
    print("Fetching NSE instrument master...")
    r = requests.get(INSTRUMENTS_URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    data = gzip.decompress(r.content).decode("utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(data))
    df = df[(df["instrument_type"] == "EQUITY") & (df["exchange"] == "NSE_EQ")]
    return dict(zip(df["tradingsymbol"], df["instrument_key"]))


def quarterly_windows(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    windows = []
    cur_to = end
    while cur_to > start:
        cur_from = max(cur_to - pd.Timedelta(days=CHUNK_DAYS), start)
        windows.append((cur_from, cur_to))
        cur_to = cur_from - pd.Timedelta(days=1)
    return windows


def fetch_chunk(
    session: requests.Session, instrument_key: str, from_date: pd.Timestamp, to_date: pd.Timestamp,
    label: str = "",
) -> list:
    """Fetches one window's candles, retrying on failure AND on a
    suspicious empty-but-200 response. The first full download run of
    this script silently accepted a 200 status with an EMPTY candles list
    as "no data in this window" without retrying -- but a multi-week
    window during active trading hours should essentially never be
    legitimately empty, and testing confirmed Upstox actually had real
    data for windows this bug skipped (a transient issue on their side,
    not a real data gap). Now retries empty 200s the same as errors,
    and only gives up after MAX_RETRIES."""
    url = HISTORICAL_URL.format(
        instrument_key=quote(instrument_key, safe=""),
        to_date=to_date.date().isoformat(),
        from_date=from_date.date().isoformat(),
    )
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, timeout=20)
            if r.status_code == 200:
                candles = r.json().get("data", {}).get("candles", [])
                if candles:
                    return candles
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_SLEEP)
                    continue
                print(f"  WARNING: empty candles after {MAX_RETRIES} attempts: "
                      f"{label} {from_date.date()}..{to_date.date()}", file=sys.stderr)
                return []
            if r.status_code == 429:
                time.sleep(RETRY_SLEEP * attempt)
                continue
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP)
                continue
            print(f"  WARNING: status {r.status_code} after {MAX_RETRIES} attempts: "
                  f"{label} {from_date.date()}..{to_date.date()}", file=sys.stderr)
            return []
        except requests.RequestException as e:
            if attempt == MAX_RETRIES:
                print(f"  WARNING: {e} after {MAX_RETRIES} attempts: "
                      f"{label} {from_date.date()}..{to_date.date()}", file=sys.stderr)
                return []
            time.sleep(RETRY_SLEEP)
    return []


def download_one(session: requests.Session, instrument_key: str, label: str = "") -> pd.DataFrame | None:
    end = pd.Timestamp.now().normalize()
    all_candles = []
    for from_date, to_date in quarterly_windows(START_DATE, end):
        all_candles.extend(fetch_chunk(session, instrument_key, from_date, to_date, label=label))
        time.sleep(REQUEST_PAUSE)

    if not all_candles:
        return None

    df = pd.DataFrame(all_candles, columns=["Datetime", "Open", "High", "Low", "Close", "Volume", "OI"])
    df["Datetime"] = pd.to_datetime(df["Datetime"]).dt.tz_localize(None)
    df = df.drop(columns=["OI"]).drop_duplicates(subset="Datetime").sort_values("Datetime")
    return df.set_index("Datetime")


def main():
    token = load_token()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    instrument_map = load_instrument_map()

    symbols = sorted(f.stem for f in HOURLY_DIR.glob("*.csv"))
    print(f"{len(symbols)} symbols to fetch (reusing the existing hourly/15min universe)")

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}", "Accept": "application/json"})

    failed, no_key = [], []
    for sym in tqdm(symbols, desc="Downloading 30-min data from Upstox"):
        out = OUT_DIR / f"{sym}.csv"
        if out.exists():
            continue
        instrument_key = instrument_map.get(sym)
        if instrument_key is None:
            no_key.append(sym)
            continue
        df = download_one(session, instrument_key, label=sym)
        if df is None or df.empty:
            failed.append(sym)
            continue
        df.to_csv(out)

    n_ok = len(symbols) - len(failed) - len(no_key)
    print(f"\nDone. {n_ok} downloaded, {len(failed)} failed, {len(no_key)} had no instrument key match.")
    if failed:
        print("Failed:", failed)
    if no_key:
        print("No instrument key:", no_key)


if __name__ == "__main__":
    main()
