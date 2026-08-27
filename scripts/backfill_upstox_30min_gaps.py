"""
Backfills gaps in the already-downloaded data/30min/*.csv files (see
download_upstox_30min_data.py). The first full download run silently
dropped several windows per symbol because fetch_chunk() accepted an
empty-but-200 API response as "no data" without retrying -- a bug now
fixed in download_upstox_30min_data.py itself, but the ~329 files it
already produced still have the resulting gaps baked in.

Rather than re-running the ~25 minute full download, this scans each
existing CSV for anomalously large gaps between consecutive 30-min bars
(anything wider than GAP_THRESHOLD_DAYS, well beyond a normal weekend/
holiday gap), re-fetches just those windows from Upstox, and merges the
recovered candles back into the file.

Needs the same secrets/upstox_token.txt as the main download script.
"""
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from download_upstox_30min_data import (  # noqa: E402
    CHUNK_DAYS,
    OUT_DIR,
    START_DATE,
    fetch_chunk,
    load_instrument_map,
    load_token,
)

GAP_THRESHOLD_DAYS = 6  # normal weekend/long-holiday gaps stay well under this


def find_gaps(index: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    gaps = []
    diffs = index.to_series().diff()
    for prev, cur, diff in zip(index[:-1], index[1:], diffs[1:]):
        if diff > pd.Timedelta(days=GAP_THRESHOLD_DAYS):
            gaps.append((prev, cur))
    return gaps


def chunk_gap(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    windows = []
    cur_to = end
    while cur_to > start:
        cur_from = max(cur_to - pd.Timedelta(days=CHUNK_DAYS), start)
        windows.append((cur_from, cur_to))
        cur_to = cur_from - pd.Timedelta(days=1)
    return windows


def backfill_one(session: requests.Session, path: Path, instrument_key: str) -> tuple[int, int]:
    """Returns (n_gaps_found, n_candles_recovered)."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    gaps = find_gaps(df.index)
    if not gaps:
        return 0, 0

    recovered = []
    for gap_start, gap_end in gaps:
        # small buffer so we don't miss a bar right at the boundary; dedup handles overlap
        from_date = max(gap_start - pd.Timedelta(days=1), START_DATE)
        to_date = gap_end + pd.Timedelta(days=1)
        for w_from, w_to in chunk_gap(from_date, to_date):
            candles = fetch_chunk(session, instrument_key, w_from, w_to, label=path.stem)
            recovered.extend(candles)
            time.sleep(0.15)

    if not recovered:
        return len(gaps), 0

    new_df = pd.DataFrame(recovered, columns=["Datetime", "Open", "High", "Low", "Close", "Volume", "OI"])
    new_df["Datetime"] = pd.to_datetime(new_df["Datetime"]).dt.tz_localize(None)
    new_df = new_df.drop(columns=["OI"]).set_index("Datetime")

    merged = pd.concat([df, new_df])
    merged = merged[~merged.index.duplicated(keep="first")].sort_index()
    merged.to_csv(path)

    return len(gaps), len(new_df)


def main():
    token = load_token()
    instrument_map = load_instrument_map()

    files = sorted(OUT_DIR.glob("*.csv"))
    print(f"Scanning {len(files)} files for gaps > {GAP_THRESHOLD_DAYS} days...")

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}", "Accept": "application/json"})

    total_gaps, total_recovered, still_gapped = 0, 0, []
    for path in tqdm(files, desc="Backfilling"):
        sym = path.stem
        instrument_key = instrument_map.get(sym)
        if instrument_key is None:
            continue
        n_gaps, n_recovered = backfill_one(session, path, instrument_key)
        if n_gaps:
            total_gaps += n_gaps
            total_recovered += n_recovered
            # re-check after backfill
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            if find_gaps(df.index):
                still_gapped.append(sym)

    print(f"\nDone. {total_gaps} gaps found, {total_recovered} candles recovered.")
    if still_gapped:
        print(f"{len(still_gapped)} symbols still have gaps after backfill: {still_gapped}")
    else:
        print("No symbols have remaining gaps.")


if __name__ == "__main__":
    main()
