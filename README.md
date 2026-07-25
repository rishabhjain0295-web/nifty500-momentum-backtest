# Nifty 500 Momentum Backtest

## Portal

```
streamlit run app.py
```

Opens an interactive backtest UI at http://localhost:8501 with live-adjustable
parameters: number of stocks held, rebalancing period, trailing-return
lookback, skip period, price filter, and a toggle for point-in-time index
membership. Shows equity curve, drawdown, current holdings, and lets you
download the monthly returns.

## Data

- `data/nifty500_list.csv` — current Nifty 500 constituents, pulled from NSE archives.
- `data/stocks/<SYMBOL>.csv` — daily OHLCV, 2008-01-01 to present, for:
  - all 500 current Nifty 500 constituents, and
  - 470 additional symbols that were historical Nifty 500 constituents at some
    point since 1998 but have since been dropped from the index (still
    tradable, so Yahoo Finance has their data) — see "Point-in-time
    membership" below.
- `data/index/NIFTY500.csv` — Nifty 500 index level (`^CRSLDX`), 2008-01-01 to present.
- `data/index/NIFTY50.csv` — Nifty 50 index level (`^NSEI`), for reference.
- `data/etfs/` — daily OHLCV for 5 popular India ETFs: JUNIORBEES, BANKBEES, LIQUIDBEES, GOLDBEES (all from 2009-01-02, Yahoo's earliest available data), and MIDCAPBEES (from 2019-02-04 — Nippon renamed the underlying fund `MID150BEES`, which only launched then; there is no continuous "midcap bees" history back to 2008).
- `data/failed_symbols.txt` / `data/failed_historical_symbols.txt` — symbols yfinance couldn't fetch. Re-run the relevant download script to retry (existing files are skipped).

## Point-in-time membership (survivorship bias fix)

The original dataset used *today's* Nifty 500 list applied backward to 2008,
which is survivorship-biased: stocks that got removed from the index
(delisted, acquired, demoted) were invisible, so the backtest only ever saw
winners. This is now fixed with a point-in-time membership calendar:

- `data/IndexInclExcl.xls` — NSE's official inclusion/exclusion log for every
  index, authoritative from 1998 to **2020-09-14**, after which NSE stopped
  updating this particular file.
- `data/membership_snapshots/*.csv` — 6 Wayback Machine snapshots of the
  constituent list (2020-07 to 2026-05), diffed pairwise to synthesize
  approximate inclusion/exclusion events and patch the 2020-2026 gap. These
  are dated at the snapshot date, so they bound *when* a change happened
  within a several-month window, not the exact day.
- `data/equity_master.csv`, `data/namechange.csv`, `data/symbolchange.csv` —
  NSE master files used to resolve historical scrip names (which change on
  renames/mergers) to current tradable ticker symbols.
- `data/nifty500_membership_calendar.csv` — the result: (symbol, start, end)
  intervals, 1,427 of them across 1,115 symbols. `end` is blank if the stock
  is still a current constituent. Built by `scripts/build_membership_calendar.py`.
- `data/nifty500_membership_events.csv` — the raw resolved (date, symbol,
  action) events before collapsing into intervals.
- `data/nifty500_unresolved_names.csv` — 318 historical scrip names (~25%)
  that couldn't be resolved to a live ticker. Spot-checked: these are almost
  entirely companies that were delisted, merged, or liquidated (e.g. Global
  Trust Bank, Patni Computer Systems, Jaypee Infratech) — a genuine data gap,
  not a matching bug. There is no free tradable price series for them
  anywhere.

**Net effect:** the backtest's eligible universe now averages ~424 stocks/month
(970 stocks loaded total) instead of a fixed 500, and a stock is only
ranked/held during the months it was actually a Nifty 500 member. This is
still best-effort, not a perfect reconstruction — treat absolute returns as
indicative, not precise, especially for 2020-2026 where the calendar is
snapshot-approximated rather than event-exact.

## Scripts

- `scripts/get_nifty500_list.py` — re-fetch the current constituent list.
- `scripts/download_data.py` — download/refresh the current 500 stocks + index price history.
- `scripts/download_etfs.py` — download/refresh the 5 India ETFs.
- `scripts/build_membership_calendar.py` — rebuild the point-in-time membership calendar (see above).
- `scripts/fetch_membership_snapshots.py` — refetch the Wayback Machine snapshot CSVs used to patch the post-2020 gap.
- `scripts/download_historical_stocks.py` — download price history for symbols found in the membership calendar that aren't in `data/stocks/` yet.
- `backtest_engine.py` — shared momentum backtest logic (used by both the CLI script and the portal).
- `scripts/momentum_backtest.py` — CLI runner using fixed parameters set at the top of the file.
- `app.py` — interactive Streamlit portal (see "Portal" above).

## Usage

```
python scripts\download_data.py               # refresh current-500 + index data
python scripts\download_historical_stocks.py  # refresh historical/delisted-constituent data
python scripts\momentum_backtest.py           # run with fixed CLI parameters
streamlit run app.py                          # interactive portal
```

Results are written to `data/momentum_backtest_results.csv` (monthly strategy vs. benchmark returns and cumulative curves).
