"""
Build a best-effort point-in-time Nifty 500 membership calendar covering
2008-present, to eliminate survivorship bias in the momentum backtest.

Two data sources are combined:

1. NSE's official IndexInclExcl.xls ("Nifty 500" sheet) -- an authoritative
   log of every inclusion/exclusion event, but it stops being updated after
   2020-09-14.
2. Six Wayback Machine snapshots of the full constituent CSV between
   2020-07-25 and 2026-05-30 (data/membership_snapshots/*.csv). Consecutive
   snapshots are diffed to synthesize approximate inclusion/exclusion events
   for the period the official log no longer covers. These synthetic events
   are dated at the snapshot date, so they only bound *when* a change
   happened within a several-month window, not the exact day.

Company/scrip names (source 1) are resolved to tradable NSE symbols using:
  - data/equity_master.csv   (current SYMBOL <-> NAME OF COMPANY)
  - data/namechange.csv      (SYMBOL <-> every historical name it has had)
  - data/symbolchange.csv    (old symbol/name -> new symbol, for renamed tickers)
  - difflib fuzzy match as a last resort

Anything that can't be resolved is logged, not silently dropped -- those are
almost certainly genuinely delisted/merged/liquidated companies with no
current tradable ticker, which is a real, unavoidable data gap (not a bug
in this script).

Outputs:
  data/nifty500_membership_events.csv    -- resolved (date, symbol, action) events
  data/nifty500_membership_calendar.csv  -- collapsed (symbol, start_date, end_date) intervals
  data/nifty500_unresolved_names.csv     -- scrip names that couldn't be mapped to a symbol
"""
import difflib
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SNAP_DIR = DATA / "membership_snapshots"

TODAY = pd.Timestamp("2026-07-25")  # fixed to avoid Date.now()-style nondeterminism


def normalize(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.upper()
    s = s.replace("&", " AND ")
    s = re.sub(r"[.\-,()']", " ", s)
    s = re.sub(r"\bLIMITED\b", "LTD", s)
    s = re.sub(r"\bLTD\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_name_resolver():
    """Returns (normalized_name -> symbol) dict built from all master files,
    plus the set of all normalized names for fuzzy fallback."""
    mapping = {}

    em = pd.read_csv(DATA / "equity_master.csv")
    em.columns = [c.strip() for c in em.columns]
    for _, row in em.iterrows():
        mapping[normalize(row["NAME OF COMPANY"])] = row["SYMBOL"]

    nc = pd.read_csv(DATA / "namechange.csv")
    nc.columns = [c.strip() for c in nc.columns]
    for _, row in nc.iterrows():
        sym = row["NCH_SYMBOL"]
        mapping.setdefault(normalize(row["NCH_PREV_NAME"]), sym)
        mapping.setdefault(normalize(row["NCH_NEW_NAME"]), sym)

    sc = pd.read_csv(DATA / "symbolchange.csv", header=None,
                      names=["company_name", "old_symbol", "new_symbol", "date"])
    # prefer the NEW symbol if it's still an active listing, else the old one
    active_symbols = set(em["SYMBOL"])
    for _, row in sc.iterrows():
        sym = row["new_symbol"] if row["new_symbol"] in active_symbols else row["old_symbol"]
        mapping.setdefault(normalize(row["company_name"]), sym)
        mapping.setdefault(normalize(row["old_symbol"]), sym)

    return mapping


def resolve_name(name: str, mapping: dict, all_names: list) -> tuple[str | None, str]:
    norm = normalize(name)
    if norm in mapping:
        return mapping[norm], "exact"
    close = difflib.get_close_matches(norm, all_names, n=1, cutoff=0.88)
    if close:
        return mapping[close[0]], "fuzzy"
    return None, "unresolved"


def load_xls_events() -> pd.DataFrame:
    df = pd.read_excel(DATA / "IndexInclExcl.xls", sheet_name="Nifty 500")
    df.columns = [c.strip() for c in df.columns]

    def parse_date(v):
        if isinstance(v, str):
            return pd.to_datetime(v, dayfirst=True, errors="coerce")
        return pd.Timestamp(v)

    df["date"] = df["Event Date"].apply(parse_date)
    df["action"] = df["Description"].apply(lambda d: "IN" if "Inclusion" in str(d) else "OUT")
    df["scrip_name"] = df["Scrip Name"].astype(str).str.strip()
    return df[["date", "scrip_name", "action"]]


def load_snapshot_symbols(path: Path) -> set:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    col = "Symbol" if "Symbol" in df.columns else "SYMBOL"
    return set(df[col].astype(str).str.strip())


def build_synthetic_events_from_snapshots() -> pd.DataFrame:
    snaps = sorted(SNAP_DIR.glob("*.csv"))
    # also treat "today" (current live list) as the final checkpoint
    dated_snaps = [(pd.Timestamp(f.stem), load_snapshot_symbols(f)) for f in snaps]
    dated_snaps.append((TODAY, load_snapshot_symbols(DATA / "nifty500_list.csv")))
    dated_snaps.sort(key=lambda x: x[0])

    events = []
    for i in range(1, len(dated_snaps)):
        prev_date, prev_syms = dated_snaps[i - 1]
        cur_date, cur_syms = dated_snaps[i]
        added = cur_syms - prev_syms
        removed = prev_syms - cur_syms
        for sym in added:
            events.append({"date": cur_date, "symbol": sym, "action": "IN", "source": "snapshot_diff"})
        for sym in removed:
            events.append({"date": cur_date, "symbol": sym, "action": "OUT", "source": "snapshot_diff"})
    return pd.DataFrame(events)


def collapse_to_calendar(events: pd.DataFrame) -> pd.DataFrame:
    """events: columns [date, symbol, action], already resolved to symbols.
    Collapses per-symbol IN/OUT event sequences into (start, end) intervals."""
    intervals = []
    for sym, grp in events.sort_values("date").groupby("symbol"):
        grp = grp.reset_index(drop=True)
        open_start = None
        for _, row in grp.iterrows():
            if row["action"] == "IN":
                if open_start is None:
                    open_start = row["date"]
            else:  # OUT
                if open_start is not None:
                    intervals.append({"symbol": sym, "start": open_start, "end": row["date"]})
                    open_start = None
                # OUT with no open interval: ignore (e.g. data starts mid-membership)
        if open_start is not None:
            intervals.append({"symbol": sym, "start": open_start, "end": pd.NaT})  # still a member
    return pd.DataFrame(intervals)


def main():
    print("Loading official inclusion/exclusion log (1998-2020)...")
    xls_events = load_xls_events()
    print(f"  {len(xls_events)} events, {xls_events['scrip_name'].nunique()} distinct scrip names")

    print("Resolving scrip names to tradable symbols...")
    mapping = load_name_resolver()
    all_names = list(mapping.keys())
    resolved_rows = []
    unresolved = []
    method_counts = {"exact": 0, "fuzzy": 0, "unresolved": 0}
    for _, row in xls_events.iterrows():
        sym, method = resolve_name(row["scrip_name"], mapping, all_names)
        method_counts[method] += 1
        if sym is None:
            unresolved.append(row["scrip_name"])
            continue
        resolved_rows.append({"date": row["date"], "symbol": sym, "action": row["action"], "source": "nse_inclexcl"})
    print(f"  resolution: {method_counts}")

    xls_resolved = pd.DataFrame(resolved_rows)

    print("Building synthetic post-2020 events from snapshot diffs...")
    synth_events = build_synthetic_events_from_snapshots()
    print(f"  {len(synth_events)} synthetic events from {len(list(SNAP_DIR.glob('*.csv'))) + 1} checkpoints")

    all_events = pd.concat([xls_resolved, synth_events], ignore_index=True)
    all_events = all_events.sort_values(["symbol", "date"]).reset_index(drop=True)
    all_events.to_csv(DATA / "nifty500_membership_events.csv", index=False)
    print(f"Saved {len(all_events)} total resolved events to nifty500_membership_events.csv")

    print("Collapsing events into membership calendar...")
    calendar = collapse_to_calendar(all_events)
    calendar = calendar.sort_values(["symbol", "start"]).reset_index(drop=True)
    calendar.to_csv(DATA / "nifty500_membership_calendar.csv", index=False)
    print(f"Saved {len(calendar)} membership intervals covering {calendar['symbol'].nunique()} symbols "
          f"to nifty500_membership_calendar.csv")

    pd.Series(sorted(set(unresolved)), name="unresolved_scrip_name").to_csv(
        DATA / "nifty500_unresolved_names.csv", index=False
    )
    print(f"Saved {len(set(unresolved))} unresolved scrip names to nifty500_unresolved_names.csv "
          f"(likely delisted/merged companies with no live ticker)")


if __name__ == "__main__":
    main()
