"""
Fetch the CURRENT constituent list for every NSE index this app offers as a
"Universe" filter (see backtest_engine.NSE_UNIVERSES). Same pattern as
get_nifty500_list.py / get_fno_list.py: NSE blocks plain requests without
browser-like headers/cookies, so we warm up a session against the homepage
first, then request each CSV.

9 of the 10 are hosted on NSE's own archive (nsearchives.nseindia.com);
Nifty Alpha 50 lives on niftyindices.com instead under a differently-named
file (found via that site's "Nifty Alpha 50" strategy-index page, which
links to ind_nifty_Alpha_Index.csv -- NSE's own archive 404s on every
filename variant tried for this one).

These are CURRENT snapshots, not a point-in-time history like the Nifty 500
membership calendar (build_membership_calendar.py) -- NSE doesn't publish
historical inclusion/exclusion logs for these other indices the way it does
for the (Nifty) 500. Saves each to data/universes/<key>.csv with a plain
Symbol column.
"""
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "universes"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# key -> (warm-up URL, csv URL). Warm-up URL picks up the cookies each CSV
# request needs; nsearchives entries all warm up against the plain NSE
# homepage, the Alpha 50 entry warms up against its own niftyindices.com page.
NSE_HOME = "https://www.nseindia.com"
SOURCES: dict[str, tuple[str, str]] = {
    "nifty50": (NSE_HOME, "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv"),
    "niftynext50": (NSE_HOME, "https://nsearchives.nseindia.com/content/indices/ind_niftynext50list.csv"),
    "niftymidcapselect": (NSE_HOME, "https://nsearchives.nseindia.com/content/indices/ind_niftymidcapselect_list.csv"),
    "nifty100": (NSE_HOME, "https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv"),
    "nifty200": (NSE_HOME, "https://nsearchives.nseindia.com/content/indices/ind_nifty200list.csv"),
    "nifty500": (NSE_HOME, "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"),
    "niftytotalmarket": (NSE_HOME, "https://nsearchives.nseindia.com/content/indices/ind_niftytotalmarket_list.csv"),
    "niftymicrocap250": (NSE_HOME, "https://nsearchives.nseindia.com/content/indices/ind_niftymicrocap250_list.csv"),
    "niftysmallcap250": (NSE_HOME, "https://nsearchives.nseindia.com/content/indices/ind_niftysmallcap250list.csv"),
    "niftyalpha50": (
        "https://www.niftyindices.com/indices/equity/strategy-indices/nifty-alpha-50",
        "https://www.niftyindices.com/IndexConstituent/ind_nifty_Alpha_Index.csv",
    ),
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update(HEADERS)

    warmed = set()
    for key, (warmup_url, csv_url) in SOURCES.items():
        if warmup_url not in warmed:
            try:
                session.get(warmup_url, timeout=15)
            except requests.RequestException as e:
                print(f"Warning: could not warm up session against {warmup_url}: {e}", file=sys.stderr)
            warmed.add(warmup_url)

        resp = session.get(csv_url, timeout=15)
        resp.raise_for_status()
        raw_path = OUT_DIR / f"_{key}_raw.csv"
        raw_path.write_bytes(resp.content)

        df = pd.read_csv(raw_path, skipinitialspace=True)
        df.columns = [c.strip() for c in df.columns]
        symbol_col = next(c for c in df.columns if c.strip().upper() == "SYMBOL")
        symbols = sorted(set(df[symbol_col].astype(str).str.strip().dropna()) - {""})
        pd.DataFrame({"Symbol": symbols}).to_csv(OUT_DIR / f"{key}.csv", index=False)
        raw_path.unlink()
        print(f"{key}: {len(symbols)} symbols -> {OUT_DIR / f'{key}.csv'}")


if __name__ == "__main__":
    main()
