"""
Fetch the current Nifty 500 constituent list from NSE archives.
NSE blocks plain requests without browser-like headers/cookies, so we
first hit the homepage to pick up cookies, then request the CSV.
"""
import sys
from pathlib import Path

import requests
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

CSV_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
HOME_URL = "https://www.nseindia.com"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

OUT_PATH = str(ROOT / "data" / "nifty500_list.csv")


def main():
    session = requests.Session()
    session.headers.update(HEADERS)

    try:
        session.get(HOME_URL, timeout=15)
    except requests.RequestException as e:
        print(f"Warning: could not warm up session against NSE homepage: {e}", file=sys.stderr)

    resp = session.get(CSV_URL, timeout=15)
    resp.raise_for_status()

    with open(OUT_PATH, "wb") as f:
        f.write(resp.content)

    df = pd.read_csv(OUT_PATH)
    print(f"Downloaded {len(df)} constituents to {OUT_PATH}")
    print(df.head())


if __name__ == "__main__":
    main()
