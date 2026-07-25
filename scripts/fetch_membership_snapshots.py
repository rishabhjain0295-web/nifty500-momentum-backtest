"""
Fetch Wayback Machine snapshots of the Nifty 500 constituent CSV at several
points in time (2020-2026), to patch the gap left by NSE's IndexInclExcl.xls
file which stops updating after 2020-09-14. Saves each snapshot as
data/membership_snapshots/<date>.csv
"""
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "membership_snapshots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SNAPSHOTS = [
    ("20200725142023", "https://www1.nseindia.com/content/indices/ind_nifty500list.csv"),
    ("20220504103923", "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"),
    ("20221009160959", "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"),
    ("20230404164710", "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"),
    ("20240226224931", "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"),
    ("20260530123407", "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"),
]

for ts, orig in SNAPSHOTS:
    url = f"https://web.archive.org/web/{ts}id_/{orig}"
    date_str = f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]}"
    try:
        r = requests.get(url, timeout=40)
        r.raise_for_status()
        out = OUT_DIR / f"{date_str}.csv"
        out.write_bytes(r.content)
        # sanity check: should look like csv with 'Symbol' header
        text_head = r.content[:200].decode("utf-8", errors="ignore")
        ok = "Symbol" in text_head or "SYMBOL" in text_head
        print(date_str, "->", len(r.content), "bytes", "OK" if ok else "UNEXPECTED CONTENT")
    except Exception as e:
        print(date_str, "FAILED", e)
