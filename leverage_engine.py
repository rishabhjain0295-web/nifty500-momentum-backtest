"""
MTF (Margin Trading Facility) leverage overlay: a two-stage, drawdown-
triggered leverage add-on that can sit on top of any monthly returns
Series -- the momentum strategy's own returns on the Backtest page, or
(per the user's confirmed design) upstream of every SIP mode on the SIP
Simulator page.

Mechanics (generalized from an ad-hoc script verified earlier in this
project's development against the momentum strategy's real returns):
  - Leg 1: a month with return <= -trigger1_pct% adds leg1_pct% extra
    leverage.
  - Leg 2: if, within the 2 months FOLLOWING a leg-1 trigger month, there's
    ANOTHER month with return <= -trigger2_pct%, an additional leg2_pct%
    leverage is added -- fires at most once per leg-1 cycle.
  - Cap: at most one active leg-1 tranche + one active leg-2 tranche at a
    time (not a total-percentage cap, since leg1_pct/leg2_pct are
    independently configurable, not fixed at 50/50) -- a trigger that
    would exceed this is skipped that month.
  - Leverage is sized as a fraction of CURRENT equity each month (a
    rolling leveraged exposure), not a fixed rupee amount pinned to the
    starting capital.
  - Interest: annual_interest_pct/12 per month, charged on the borrowed
    (leveraged) rupee amount for every month a tranche is active,
    deducted from equity regardless of that month's P&L.

Two hold-duration modes for parameter #5:
  - "fixed_months": each tranche independently expires N months after its
    own trigger month (the original ad-hoc script's design).
  - "ath_recovery": a tranche stays active until the UNLEVERAGED
    strategy's own NAV -- (1+strat_rets).cumprod(), not the leveraged NAV,
    which would create a feedback loop -- recovers to recovery_pct% above
    the running all-time-high AS OF the tranche's own trigger month. Same
    "recovery above previous ATH" idea already used by
    sip_engine.simulate_dynamic_sip and the Nifty 50 drawdown-leverage
    exercise done earlier in this project, just made configurable here.

Pure function of a returns Series -- no Streamlit dependency, matching
sip_engine.py's separation between Streamlit-agnostic simulation and the
pages that own the UI.
"""
import pandas as pd


def apply_mtf_leverage(
    strat_rets: pd.Series,
    trigger1_pct: float = 7.0,
    leg1_pct: float = 50.0,
    trigger2_pct: float = 5.0,
    leg2_pct: float = 50.0,
    hold_mode: str = "fixed_months",
    hold_months: int = 6,
    recovery_pct: float = 0.0,
    annual_interest_pct: float = 10.0,
) -> dict:
    dates = strat_rets.index
    n = len(dates)
    returns = strat_rets.values

    nav = (1 + strat_rets).cumprod()
    ath = nav.cummax()

    def recovery_end_idx(trigger_idx: int) -> int:
        """First index j > trigger_idx where nav[j] >= ath[trigger_idx] *
        (1 + recovery_pct/100); the last index if recovery never happens
        before the data ends (an unresolved/ongoing tranche)."""
        target = ath.iloc[trigger_idx] * (1 + recovery_pct / 100.0)
        for j in range(trigger_idx + 1, n):
            if nav.iloc[j] >= target:
                return j
        return n - 1

    tranches: list[dict] = []       # {"start", "end", "leg", "trigger_idx"}
    pending_upgrades: list[dict] = []  # {"trigger_idx", "upgraded"}
    events: list[dict] = []

    def active_leg_count(idx: int, leg: int) -> int:
        return sum(1 for t in tranches if t["leg"] == leg and t["start"] <= idx <= t["end"])

    def tranche_end(trigger_idx: int) -> int:
        if hold_mode == "ath_recovery":
            return recovery_end_idx(trigger_idx)
        return min(trigger_idx + hold_months, n - 1)

    equity = 1.0
    records = []

    for i in range(n):
        r = returns[i]

        lev_frac = sum(t["leg1_pct"] if t["leg"] == 1 else t["leg2_pct"]
                        for t in tranches if t["start"] <= i <= t["end"]) / 100.0
        exposure = equity * (1 + lev_frac)
        gross_pnl = exposure * r
        interest = equity * lev_frac * (annual_interest_pct / 100.0 / 12.0)
        equity_next = equity + gross_pnl - interest

        records.append({
            "date": dates[i], "lev_frac": lev_frac, "interest": interest,
            "leveraged_return": (equity_next - equity) / equity if equity > 0 else 0.0,
        })

        # --- leg 2: upgrade check for cycles within their 2-month window ---
        for p in pending_upgrades:
            if not p["upgraded"] and p["trigger_idx"] < i <= p["trigger_idx"] + 2:
                if r <= -trigger2_pct / 100.0:
                    start = i + 1
                    if start <= n - 1 and active_leg_count(start, 2) == 0:
                        end = tranche_end(i)
                        tranches.append({"start": start, "end": end, "leg": 2,
                                          "trigger_idx": i, "leg1_pct": leg1_pct, "leg2_pct": leg2_pct})
                        events.append({
                            "date": dates[i], "leg": 2, "month_return": r,
                            "start": dates[start], "end": dates[end],
                        })
                    p["upgraded"] = True

        # --- leg 1: new cycle ---
        if r <= -trigger1_pct / 100.0:
            start = i + 1
            if start <= n - 1 and active_leg_count(start, 1) == 0:
                end = tranche_end(i)
                tranches.append({"start": start, "end": end, "leg": 1,
                                  "trigger_idx": i, "leg1_pct": leg1_pct, "leg2_pct": leg2_pct})
                pending_upgrades.append({"trigger_idx": i, "upgraded": False})
                events.append({
                    "date": dates[i], "leg": 1, "month_return": r,
                    "start": dates[start], "end": dates[end],
                })

        equity = equity_next

    rec_df = pd.DataFrame(records).set_index("date")
    leveraged_rets = rec_df["leveraged_return"].rename(strat_rets.name)
    interest_series = rec_df["interest"]
    lev_frac_series = rec_df["lev_frac"]

    return {
        "leveraged_rets": leveraged_rets,
        "events": events,
        # interest_series is in the SAME returns-space units as leveraged_rets
        # (equity starts at 1.0) -- multiply by a capital_base to get rupees,
        # same convention as everywhere else returns-space feeds a rupee
        # display (e.g. app.py's capital_base for cost/tax sizing).
        "interest_series": interest_series,
        "lev_frac_series": lev_frac_series,
        "n_tranches": len(tranches),
    }
