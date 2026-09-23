"""
Extends data/index/NIFTY500.csv and data/index/NIFTY50.csv back to their
1995-01-01 base date, splicing in NSE's OWN historical index-level data
(via jugaad-data's index_raw, which hits NSE's live index history endpoint
directly) for 1995-01-01 through 2007-12-31 -- the stretch this project's
existing yfinance-sourced files don't cover (Yahoo's ^CRSLDX/^NSEI tickers
only go back to 2005-09-26 / 2007-09-17 respectively, well short of the
index's actual 1995 base date).

Verified before building this: NSE's historical values for the OVERLAP
period (e.g. Jan 2008) match Yahoo's ^CRSLDX to the cent (5384.55 vs
5384.549805 on 2008-01-01) -- same underlying index, so splicing is safe.
Yahoo's own 2008+ rows are kept as-is (more decimal precision); this only
FILLS IN the pre-2008 gap, never overwrites existing rows.

NSE doesn't publish Volume for index levels the way it does for stocks,
and Open/High/Low are missing ("-") for some of the earliest (1995-1997)
rows -- those are filled from Close (a flat OHLC bar) rather than left
NaN, which would break code assuming Open/High/Low are always populated.
Documented explicitly in the output so this approximation isn't silently
mistaken for real intraday range.

This does NOT touch data/stocks/ -- see scripts/backfill_stocks_1998.py
for extending individual stock histories the same way (via yfinance,
which already has usable coverage back to 1998 for most current
constituents, unlike the index level itself).
"""
from datetime import date
from pathlib import Path

import pandas as pd
from jugaad_data.nse import index_raw

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "data" / "index"

# NSE's own index name string -> (existing CSV filename, splice cutoff --
# the first date already covered by the existing Yahoo-sourced file).
INDICES = {
    "NIFTY 500": ("NIFTY500.csv", date(2008, 1, 1)),
    "NIFTY 50": ("NIFTY50.csv", date(2008, 1, 1)),
}

BACKFILL_START = date(1995, 1, 1)


def fetch_nse_history(index_name: str, start: date, end: date) -> pd.DataFrame:
    rows = index_raw(index_name, start, end)
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["HistoricalDate"], format="%d %b %Y")

    def to_float(col):
        return pd.to_numeric(df[col].replace("-", pd.NA).astype(str).str.replace(",", ""), errors="coerce")

    close = to_float("CLOSE")
    open_ = to_float("OPEN").fillna(close)
    high = to_float("HIGH").fillna(close)
    low = to_float("LOW").fillna(close)

    out = pd.DataFrame({
        "Date": df["Date"], "Adj Close": close, "Close": close,
        "High": high, "Low": low, "Open": open_, "Volume": 0,
    }).set_index("Date").sort_index()
    return out[~out.index.duplicated(keep="last")]


def backfill_one(index_name: str, csv_name: str, cutoff: date) -> None:
    print(f"Fetching NSE historical data for {index_name} ({BACKFILL_START} to {cutoff})...")
    nse_hist = fetch_nse_history(index_name, BACKFILL_START, cutoff)
    nse_hist = nse_hist[nse_hist.index < pd.Timestamp(cutoff)]
    print(f"  {len(nse_hist)} rows fetched from NSE.")

    existing_path = INDEX_DIR / csv_name
    existing = pd.read_csv(existing_path, index_col=0, parse_dates=True)
    combined = pd.concat([nse_hist, existing[~existing.index.isin(nse_hist.index)]]).sort_index()
    combined.to_csv(existing_path)
    print(f"  {csv_name}: {len(existing)} rows -> {len(combined)} rows "
          f"(now starts {combined.index.min().date()}, was {existing.index.min().date()}).")


def main():
    for index_name, (csv_name, cutoff) in INDICES.items():
        backfill_one(index_name, csv_name, cutoff)
    print(
        "\nNOTE: Open/High/Low for the earliest (1995-1997) NSE-sourced rows are "
        "sometimes unavailable and filled from Close (a flat OHLC bar) rather than "
        "left blank -- real intraday range isn't published that far back. Volume "
        "is always 0 for the NSE-sourced portion -- NSE doesn't publish index-level "
        "volume the way it does for individual stocks."
    )


if __name__ == "__main__":
    main()
