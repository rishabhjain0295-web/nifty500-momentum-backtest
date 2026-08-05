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

Symbol-rename canonicalization (build_symbol_canonicalizer): Wayback
snapshots record whatever symbol was live on that date (e.g. Motherson
Sumi Systems traded as MOTHERSUMI through 2022-06-09, then MOTHERSON
after). Snapshot comparisons -- both the consecutive-snapshot diff and the
reconcile_against_earliest_snapshot() check below -- canonicalize every
symbol to its current equivalent first via symbolchange.csv's rename chain,
so a rename never looks like an exclusion+new-inclusion or wrongly closes
a still-open interval.

Known residual gap: corporate demergers, where the SPINOFF keeps the
parent's exact legal name while the continuing listed entity is renamed to
something else (e.g. Tata Motors' Nov-2025 commercial-vehicles demerger:
the new "TMCV" inherited the name "Tata Motors Limited", while the
continuing passenger-vehicle business, still trading, was renamed "Tata
Motors Passenger Vehicles Ltd" / symbol TMPV). Exact-name resolution has
no way to know this from company names alone -- it would need pre-2020
ISIN history we don't have. reconcile_against_earliest_snapshot() catches
the worst consequence (an ancient scrip name incorrectly claiming a
multi-decade-old inclusion for a security that didn't exist yet), and the
backtest's own price-data eligibility check (no price history = not
eligible) prevents it from causing an actual wrong stock pick -- but the
calendar interval itself can still show a cosmetically bogus early start
date for the spinoff's symbol in this situation.

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

TODAY = pd.Timestamp("2026-08-05")  # fixed to avoid Date.now()-style nondeterminism -- bump this each time nifty500_list.csv is refreshed and the calendar rebuilt


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


def build_symbol_canonicalizer() -> dict:
    """Maps every symbol a security has EVER traded under to its single
    current/latest symbol, by unioning old_symbol<->new_symbol pairs from
    symbolchange.csv (chained renames handled via union-find). Needed
    because Wayback snapshots record whatever symbol was live on that date
    -- e.g. Motherson Sumi Systems traded as MOTHERSUMI through 2022-06-09,
    then MOTHERSON after. Without canonicalizing, a straight symbol-string
    comparison between an old snapshot and current data wrongly looks like
    an exclusion+new-inclusion instead of one continuous holding, and
    reconcile_against_earliest_snapshot() would wrongly close a still-open
    interval for any such security whose rename happened after the
    snapshot it's being checked against.
    """
    sc = pd.read_csv(DATA / "symbolchange.csv", header=None,
                      names=["company_name", "old_symbol", "new_symbol", "date"])
    em = pd.read_csv(DATA / "equity_master.csv")
    em.columns = [c.strip() for c in em.columns]
    active_symbols = set(em["SYMBOL"])

    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for _, row in sc.iterrows():
        union(row["old_symbol"], row["new_symbol"])

    groups: dict[str, set] = {}
    for s in list(parent.keys()):
        groups.setdefault(find(s), set()).add(s)

    canon = {}
    for members in groups.values():
        active = [m for m in members if m in active_symbols]
        chosen = active[0] if active else sorted(members)[0]
        for m in members:
            canon[m] = chosen
    return canon


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


def load_snapshot_symbols(path: Path, canon: dict | None = None) -> set:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    col = "Symbol" if "Symbol" in df.columns else "SYMBOL"
    syms = set(df[col].astype(str).str.strip())
    if canon:
        syms = {canon.get(s, s) for s in syms}
    return syms


def reconcile_against_earliest_snapshot(xls_resolved: pd.DataFrame, canon: dict) -> pd.DataFrame:
    """The official 1998-2020 log occasionally has an exclusion event whose
    scrip name failed to resolve to a symbol (e.g. an ambiguous/misspelled
    name at the time, or a name from an era load_name_resolver's mapping
    doesn't cover) -- when that happens, the matching INCLUSION event is
    still resolved and the symbol's interval is left permanently "open"
    (end=NaT), even though the security demonstrably isn't a Nifty 500
    member by our first ground-truth checkpoint. Confirmed case: GVPIL
    (GE Power India, née Alstom India) included 2002-03-29 per the xls,
    absent from every Wayback snapshot from 2020-07-25 onward and from the
    live current list, with no resolved exclusion event anywhere.

    For any symbol with a net-positive IN/OUT balance from the xls as of
    the earliest Wayback snapshot date that ISN'T actually a member in that
    snapshot, insert a synthetic OUT event dated the day before -- closing
    the dangling interval using verified ground truth instead of leaving it
    open by default.
    """
    snaps = sorted(SNAP_DIR.glob("*.csv"))
    if not snaps:
        return pd.DataFrame(columns=["date", "symbol", "action", "source"])
    earliest_snap_date = pd.Timestamp(snaps[0].stem)
    earliest_members = load_snapshot_symbols(snaps[0], canon)

    reconciled = []
    pre_cutoff = xls_resolved[xls_resolved["date"] < earliest_snap_date]
    for sym, grp in pre_cutoff.sort_values("date").groupby("symbol"):
        canon_sym = canon.get(sym, sym)
        balance = grp["action"].map({"IN": 1, "OUT": -1}).sum()
        if balance > 0 and canon_sym not in earliest_members:
            reconciled.append({
                "date": earliest_snap_date - pd.Timedelta(days=1),
                "symbol": sym, "action": "OUT", "source": "reconciliation_vs_earliest_snapshot",
            })
    return pd.DataFrame(reconciled, columns=["date", "symbol", "action", "source"])


def build_synthetic_events_from_snapshots(canon: dict) -> pd.DataFrame:
    snaps = sorted(SNAP_DIR.glob("*.csv"))
    # also treat "today" (current live list) as the final checkpoint
    dated_snaps = [(pd.Timestamp(f.stem), load_snapshot_symbols(f, canon)) for f in snaps]
    dated_snaps.append((TODAY, load_snapshot_symbols(DATA / "nifty500_list.csv", canon)))
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

    print("Building symbol-rename canonicalizer (so snapshot comparisons survive symbol changes)...")
    canon = build_symbol_canonicalizer()
    print(f"  {len(canon)} historical symbols mapped to their current equivalent")

    print("Reconciling xls-era open intervals against the earliest verified snapshot...")
    reconciliation_events = reconcile_against_earliest_snapshot(xls_resolved, canon)
    print(f"  {len(reconciliation_events)} dangling-open interval(s) closed "
          f"(exclusion event existed but failed to resolve at the time)")

    print("Building synthetic post-2020 events from snapshot diffs...")
    synth_events = build_synthetic_events_from_snapshots(canon)
    print(f"  {len(synth_events)} synthetic events from {len(list(SNAP_DIR.glob('*.csv'))) + 1} checkpoints")

    all_events = pd.concat([xls_resolved, synth_events, reconciliation_events], ignore_index=True)
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
