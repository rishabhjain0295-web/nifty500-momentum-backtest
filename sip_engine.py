"""
SIP (Systematic Investment Plan) simulation, layered on top of a momentum
backtest's monthly return series (see backtest_engine.py). Three pieces:

  - Plain SIP: a fixed amount invested into the strategy every month.
  - Dynamic SIP: split between the strategy and a liquid fund (LIQUIDBEES)
    every month, with a tactical rule -- when the strategy's own NAV draws
    down by a threshold from its running all-time high, the ENTIRE liquid
    fund balance is converted into the strategy and 100% of subsequent SIP
    contributions go into the strategy too, until the strategy's NAV climbs
    to a recovery threshold above that PREVIOUS all-time high (the peak
    that was being drawn down from, not any new high made during the
    recovery). At that point contributions revert to the normal split.
  - Lumpsum-on-drawdown (simulate_lumpsum_on_drawdown): an independent,
    separately-tracked sleeve that invests a fixed lumpsum into the
    strategy each time it dips a threshold below its all-time high, armed
    again only after a confirmed recovery. Meant to be combined with
    either SIP mode by the caller, not blended into their totals.

Design assumption (the source spec was ambiguous here): reverting to the
normal split only changes where FUTURE contributions go. Money that moved
into the strategy during the aggressive phase is not force-sold back into
the liquid fund -- there's no active rebalance on exit, only on entry
(the drawdown trigger). If you want an active rebalance-back instead, that
would change simulate_dynamic_sip's "back to normal" branch.

Position sizing throughout uses a NAV/units model (like a real mutual
fund): each contribution buys units at that month's NAV, so total value at
any point is units_held * NAV. The drawdown/ATH tracking for the trigger
uses the strategy's own NAV path only, independent of shares bought --
i.e. it's a pure price/return signal, not muddied by contribution timing.
"""
from __future__ import annotations

import pandas as pd


def simulate_plain_sip(strat_rets: pd.Series, sip_amount: float) -> dict:
    """Fixed sip_amount invested into the strategy at the start of every
    month (before that month's return is realized)."""
    dates = strat_rets.index
    units = 0.0
    nav = 1.0
    total_invested = 0.0

    value_series = pd.Series(index=dates, dtype=float)
    invested_series = pd.Series(index=dates, dtype=float)
    cashflows = []  # (date, amount) -- negative = contribution, for XIRR

    for d in dates:
        units += sip_amount / nav
        total_invested += sip_amount
        cashflows.append((d, -sip_amount))

        r = strat_rets.loc[d]
        r = r if pd.notna(r) else 0.0
        nav *= (1 + r)

        value_series.loc[d] = units * nav
        invested_series.loc[d] = total_invested

    final_value = value_series.iloc[-1] if len(value_series) else 0.0
    xirr_cashflows = cashflows + [(dates[-1], final_value)] if len(dates) else []

    return {
        "value": value_series,
        "invested": invested_series,
        "total_invested": total_invested,
        "final_value": final_value,
        "xirr": compute_xirr(xirr_cashflows),
    }


def simulate_dynamic_sip(
    strat_rets: pd.Series,
    liquid_rets: pd.Series,
    sip_amount: float,
    strategy_alloc_pct: float = 70.0,
    drawdown_trigger_pct: float = 15.0,
    recovery_pct: float = 20.0,
) -> dict:
    """strat_rets and liquid_rets must share the same index (caller aligns
    via .reindex()). See module docstring for the trigger/exit rule and
    the "no active rebalance on exit" design assumption.
    """
    dates = strat_rets.index
    strategy_alloc = strategy_alloc_pct / 100.0
    liquid_alloc = 1 - strategy_alloc

    strategy_units = 0.0
    liquid_units = 0.0
    strategy_nav = 1.0
    liquid_nav = 1.0
    running_ath = 1.0
    trigger_ath = None
    state = "normal"
    total_invested = 0.0

    value_series = pd.Series(index=dates, dtype=float)
    strategy_value_series = pd.Series(index=dates, dtype=float)
    liquid_value_series = pd.Series(index=dates, dtype=float)
    invested_series = pd.Series(index=dates, dtype=float)
    state_series = pd.Series(index=dates, dtype=object)
    transitions: list[tuple[pd.Timestamp, str]] = []
    cashflows = []

    for d in dates:
        # 1. evaluate state transition using last month-end's NAV/ATH (no lookahead)
        if state == "normal":
            if strategy_nav <= running_ath * (1 - drawdown_trigger_pct / 100.0):
                trigger_ath = running_ath
                liquid_value = liquid_units * liquid_nav
                if liquid_value > 0:
                    strategy_units += liquid_value / strategy_nav
                    liquid_units = 0.0
                state = "aggressive"
                transitions.append((d, "triggered: drawdown hit, converted liquid -> strategy"))
        elif state == "aggressive":
            if trigger_ath is not None and strategy_nav >= trigger_ath * (1 + recovery_pct / 100.0):
                state = "normal"
                transitions.append((d, "recovered: back to normal split"))

        # 2. this month's contribution, per current state
        if state == "normal":
            strategy_units += (sip_amount * strategy_alloc) / strategy_nav
            liquid_units += (sip_amount * liquid_alloc) / liquid_nav
        else:
            strategy_units += sip_amount / strategy_nav
        total_invested += sip_amount
        cashflows.append((d, -sip_amount))

        # 3. apply this month's returns, update ATH
        r_s = strat_rets.loc[d]
        r_s = r_s if pd.notna(r_s) else 0.0
        r_l = liquid_rets.loc[d] if d in liquid_rets.index else 0.0
        r_l = r_l if pd.notna(r_l) else 0.0
        strategy_nav *= (1 + r_s)
        liquid_nav *= (1 + r_l)
        running_ath = max(running_ath, strategy_nav)

        strategy_value_series.loc[d] = strategy_units * strategy_nav
        liquid_value_series.loc[d] = liquid_units * liquid_nav
        value_series.loc[d] = strategy_value_series.loc[d] + liquid_value_series.loc[d]
        invested_series.loc[d] = total_invested
        state_series.loc[d] = state

    final_value = value_series.iloc[-1] if len(value_series) else 0.0
    xirr_cashflows = cashflows + [(dates[-1], final_value)] if len(dates) else []

    return {
        "value": value_series,
        "strategy_value": strategy_value_series,
        "liquid_value": liquid_value_series,
        "invested": invested_series,
        "state": state_series,
        "transitions": transitions,
        "total_invested": total_invested,
        "final_value": final_value,
        "xirr": compute_xirr(xirr_cashflows),
    }


def simulate_lumpsum_on_drawdown(
    strat_rets: pd.Series,
    lumpsum_amount: float,
    drawdown_trigger_pct: float,
    reset_recovery_pct: float = 0.0,
) -> dict:
    """Opportunistic lumpsum sleeve, entirely separate from any SIP: invests
    a fixed lumpsum_amount into the strategy the first time its NAV crosses
    drawdown_trigger_pct below its running all-time high, then waits for
    the NAV to climb back to reset_recovery_pct above THAT SAME pre-drawdown
    peak before it's willing to trigger again -- one lumpsum per drawdown
    episode, not one per month spent underwater. reset_recovery_pct=0 means
    "reset once NAV merely reclaims the old high"; set it higher to require
    a confirmed rally past the old peak before re-arming (mirrors
    simulate_dynamic_sip's trigger/recovery pattern).

    This represents money added ON TOP OF, not instead of, a regular
    contribution plan -- it has its own units/NAV tracking and is not
    mixed into simulate_plain_sip or simulate_dynamic_sip's totals. Combine
    the two callers' "value" series yourself if you want a combined total.
    """
    dates = strat_rets.index
    units = 0.0
    nav = 1.0
    running_ath = 1.0
    trigger_ath = None
    state = "watching"
    total_invested = 0.0

    value_series = pd.Series(index=dates, dtype=float)
    invested_series = pd.Series(index=dates, dtype=float)
    triggers: list[tuple[pd.Timestamp, str]] = []
    cashflows = []

    for d in dates:
        if state == "watching":
            if nav <= running_ath * (1 - drawdown_trigger_pct / 100.0):
                trigger_ath = running_ath
                units += lumpsum_amount / nav
                total_invested += lumpsum_amount
                cashflows.append((d, -lumpsum_amount))
                triggers.append((d, f"invested at {(nav / running_ath - 1):.1%} off ATH"))
                state = "triggered"
        elif state == "triggered":
            if trigger_ath is not None and nav >= trigger_ath * (1 + reset_recovery_pct / 100.0):
                state = "watching"

        r = strat_rets.loc[d]
        r = r if pd.notna(r) else 0.0
        nav *= (1 + r)
        running_ath = max(running_ath, nav)

        value_series.loc[d] = units * nav
        invested_series.loc[d] = total_invested

    final_value = value_series.iloc[-1] if len(value_series) else 0.0
    xirr_cashflows = cashflows + [(dates[-1], final_value)] if cashflows else []

    return {
        "value": value_series,
        "invested": invested_series,
        "total_invested": total_invested,
        "final_value": final_value,
        "xirr": compute_xirr(xirr_cashflows),
        "triggers": triggers,
        "n_triggers": len(triggers),
    }


def compute_xirr(cashflows: list[tuple[pd.Timestamp, float]], guess: float = 0.15) -> float:
    """Annualized IRR for a series of (date, amount) cashflows -- negative
    for outflows (contributions), positive for the final inflow (current
    value). Newton-Raphson on Act/365 day-count; falls back to NaN if it
    doesn't converge (e.g. no cashflows, or a pathological input)."""
    if len(cashflows) < 2:
        return float("nan")
    t0 = cashflows[0][0]
    days = [(d - t0).days for d, _ in cashflows]
    amounts = [a for _, a in cashflows]

    def npv(rate):
        return sum(a / (1 + rate) ** (t / 365.0) for a, t in zip(amounts, days))

    def dnpv(rate):
        return sum(-((t / 365.0)) * a / (1 + rate) ** (t / 365.0 + 1) for a, t in zip(amounts, days))

    rate = guess
    for _ in range(100):
        f = npv(rate)
        fp = dnpv(rate)
        if fp == 0:
            break
        new_rate = rate - f / fp
        if abs(new_rate - rate) < 1e-8:
            return new_rate
        rate = new_rate
        if rate <= -0.999:
            rate = -0.999 + 1e-6
    return rate if abs(npv(rate)) < 1.0 else float("nan")
