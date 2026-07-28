"""
India-specific transaction cost and capital-gains tax simulation, layered on
top of a momentum backtest's holdings history (see backtest_engine.py).

This is an approximation for exploring how much of a strategy's edge survives
real-world frictions -- it is NOT tax advice and does not model every nuance
of Indian tax law. Key simplifications:

  - Costs (STT, stamp duty, exchange charges, GST, slippage) are charged only
    when a stock actually enters/exits the held basket at a rebalance, not
    for the implicit monthly equal-weight re-normalization within an
    unchanged basket that the underlying backtest's monthly-mean return
    calculation assumes. Real turnover (and cost) could be modestly higher
    if you literally rebalance to equal weight every month.
  - Capital gains are computed from raw entry/exit prices (STT is not
    deductible from gains under Indian law for STCG/LTCG taxed at special
    rates, so this is directionally correct; brokerage technically is
    deductible but is ignored here for simplicity -- only matters if you set
    brokerage_pct > 0).
  - Short-term (<12 months) vs long-term (>=12 months) classification uses a
    365-day threshold.
  - Loss set-off follows the real ordering (current-year short-term losses
    can offset short-term or long-term gains; long-term losses only offset
    long-term gains) and carries forward net losses indefinitely -- real law
    caps carryforward at 8 assessment years, not modeled here.
  - The annual LTCG exemption (default Rs 1.25 lakh) is applied fresh each
    Indian financial year (April-March), assuming this strategy is the only
    source of LTCG for the investor.
  - Surcharge is not modeled (it's income-level dependent); a flat cess is.
  - Gold defensive-regime rotation (backtest_engine.run_backtest's "GOLD"
    synthetic position): the equity-side cost of liquidating the stock
    portfolio on the way into gold, or rebuilding it on the way back out, IS
    charged. What's NOT modeled is the gold ETF leg's own entry/exit cost
    or any capital-gains tax on gold trades themselves.

Position sizing for tax purposes uses the *actual* post-cost-and-tax capital
path (nav_net), so a realistic amount of capital compounds forward after
each cost/tax deduction, rather than an idealized gross curve.
"""
from dataclasses import dataclass

import pandas as pd


@dataclass
class CostParams:
    slippage_pct: float = 0.15            # one-way market impact / bid-ask, both legs
    stt_buy_pct: float = 0.10             # Securities Transaction Tax, delivery equity buy
    stt_sell_pct: float = 0.10            # STT, delivery equity sell
    stamp_duty_pct: float = 0.015         # stamp duty, buy side only
    exchange_charges_pct: float = 0.0035  # NSE transaction charges + SEBI fee, both legs
    brokerage_pct: float = 0.0            # most discount brokers charge 0 for delivery trades
    gst_pct: float = 18.0                 # GST on (brokerage + exchange charges)

    @property
    def buy_cost_pct(self) -> float:
        base = self.stt_buy_pct + self.stamp_duty_pct + self.exchange_charges_pct + self.brokerage_pct
        gst = self.gst_pct / 100 * (self.exchange_charges_pct + self.brokerage_pct)
        return (base + gst) / 100 + self.slippage_pct / 100

    @property
    def sell_cost_pct(self) -> float:
        base = self.stt_sell_pct + self.exchange_charges_pct + self.brokerage_pct
        gst = self.gst_pct / 100 * (self.exchange_charges_pct + self.brokerage_pct)
        return (base + gst) / 100 + self.slippage_pct / 100


@dataclass
class TaxParams:
    capital_base_rs: float = 1_000_000.0
    stcg_rate_pct: float = 20.0
    ltcg_rate_pct: float = 12.5
    ltcg_exemption_rs: float = 125_000.0
    cess_pct: float = 4.0
    ltcg_holding_days: int = 365


def simulate_costs_and_taxes(
    monthly_prices: pd.DataFrame,
    strat_rets: pd.Series,
    holdings_history: list,
    n_stocks: int,
    cost: CostParams,
    tax: TaxParams,
) -> dict:
    dates = strat_rets.index
    monthly_rets = monthly_prices.pct_change()
    holdings_by_date = dict(holdings_history)
    rebalance_dates = set(holdings_by_date.keys())

    active_holdings = []
    current: list[str] = []
    for d in dates:
        if d in holdings_by_date:
            current = holdings_by_date[d]
        active_holdings.append(list(current))

    buy_cost_pct = cost.buy_cost_pct
    sell_cost_pct = cost.sell_cost_pct

    nav_net = 1.0
    nav_post_cost = 1.0
    nav_net_series = pd.Series(index=dates, dtype=float)
    nav_post_cost_series = pd.Series(index=dates, dtype=float)
    cost_paid_series = pd.Series(0.0, index=dates)
    tax_paid_series = pd.Series(0.0, index=dates)

    open_positions: dict[str, tuple] = {}  # symbol -> (entry_date, entry_price, entry_nav, entry_n_active)
    closed_trades: list[dict] = []
    prev_holdings: set[str] = set()

    stcg_carry = 0.0
    ltcg_carry = 0.0
    taxed_until_idx = 0

    def run_tax_checkpoint() -> float:
        nonlocal stcg_carry, ltcg_carry, taxed_until_idx
        pending = closed_trades[taxed_until_idx:]
        taxed_until_idx = len(closed_trades)
        if not pending:
            return 0.0
        stcg_gain = stcg_carry
        ltcg_gain = ltcg_carry
        for t in pending:
            if t["is_ltcg"]:
                ltcg_gain += t["rupee_gain"]
            else:
                stcg_gain += t["rupee_gain"]
        if stcg_gain < 0 and ltcg_gain > 0:
            offset = min(-stcg_gain, ltcg_gain)
            ltcg_gain -= offset
            stcg_gain += offset
        stcg_carry = min(stcg_gain, 0.0)
        ltcg_carry = min(ltcg_gain, 0.0)
        stcg_taxable = max(stcg_gain, 0.0)
        ltcg_taxable = max(ltcg_gain - tax.ltcg_exemption_rs, 0.0)
        tax_owed = stcg_taxable * tax.stcg_rate_pct / 100 + ltcg_taxable * tax.ltcg_rate_pct / 100
        return tax_owed * (1 + tax.cess_pct / 100)

    for i, d in enumerate(dates):
        holdings = active_holdings[i]
        if holdings:
            r = monthly_rets.loc[d, holdings].mean()
            if pd.isna(r):
                r = 0.0
        else:
            r = 0.0
        nav_net *= (1 + r)
        nav_post_cost *= (1 + r)

        if d in rebalance_dates:
            new_holdings = set(holdings_by_date[d])
            # "GOLD" is a synthetic regime marker (see backtest_engine.run_backtest),
            # not a real monthly_prices symbol -- exclude it from equity turnover/cost
            # accounting. Its own entry/exit costs aren't modeled, but the equity-side
            # cost of liquidating/rebuilding the stock portfolio around a gold rotation
            # still is (dropping to {} on the way into gold, or filling up from {} on
            # the way back).
            equity_prev = prev_holdings - {"GOLD"}
            equity_new = new_holdings - {"GOLD"}
            added = equity_new - equity_prev
            dropped = equity_prev - equity_new
            n_active_buy = len(equity_new) if equity_new else n_stocks
            n_active_sell = len(equity_prev) if equity_prev else n_stocks

            for sym in dropped:
                entry_date, entry_price, entry_nav, entry_n_active = open_positions.pop(sym)
                if sym in monthly_prices.columns and pd.notna(monthly_prices.loc[d, sym]):
                    exit_price = monthly_prices.loc[d, sym]
                    position_value_at_entry = entry_nav / entry_n_active
                    gain_frac = position_value_at_entry * (exit_price / entry_price - 1)
                    hold_days = (d - entry_date).days
                    closed_trades.append({
                        "symbol": sym, "entry_date": entry_date, "exit_date": d,
                        "hold_days": hold_days,
                        "rupee_gain": gain_frac * tax.capital_base_rs,
                        "is_ltcg": hold_days >= tax.ltcg_holding_days,
                    })

            buy_frac = len(added) / n_active_buy if n_active_buy else 0.0
            sell_frac = len(dropped) / n_active_sell if n_active_sell else 0.0
            cost_drag = buy_frac * buy_cost_pct + sell_frac * sell_cost_pct
            cost_paid_series.loc[d] = cost_drag * nav_net * tax.capital_base_rs
            nav_net *= (1 - cost_drag)
            nav_post_cost *= (1 - cost_drag)

            for sym in added:
                if sym in monthly_prices.columns and pd.notna(monthly_prices.loc[d, sym]):
                    entry_price = monthly_prices.loc[d, sym]
                    open_positions[sym] = (d, entry_price, nav_net, n_active_buy)

            prev_holdings = new_holdings

        if d.month == 3 or i == len(dates) - 1:
            tax_owed = run_tax_checkpoint()
            if tax_owed > 0 and nav_net > 0:
                frac = min(tax_owed / (nav_net * tax.capital_base_rs), 1.0)
                nav_net *= (1 - frac)
                tax_paid_series.loc[d] = tax_owed

        nav_net_series.loc[d] = nav_net
        nav_post_cost_series.loc[d] = nav_post_cost

    trades_df = pd.DataFrame(closed_trades)
    return {
        "nav_gross": (1 + strat_rets).cumprod(),
        "nav_post_cost": nav_post_cost_series,
        "nav_net": nav_net_series,
        "cost_paid": cost_paid_series,
        "tax_paid": tax_paid_series,
        "closed_trades": trades_df,
        "total_cost_rs": cost_paid_series.sum(),
        "total_tax_rs": tax_paid_series.sum(),
        "n_stcg_trades": int((~trades_df["is_ltcg"]).sum()) if not trades_df.empty else 0,
        "n_ltcg_trades": int(trades_df["is_ltcg"].sum()) if not trades_df.empty else 0,
    }
