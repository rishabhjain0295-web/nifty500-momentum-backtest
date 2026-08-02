"""
CLI runner for the momentum backtest. See backtest_engine.py for the
strategy logic, point-in-time universe handling, and known limitations.
For an interactive version with adjustable parameters, run:
    streamlit run app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest_engine import build_trade_log, run_full_backtest, yearly_table  # noqa: E402

# ---- strategy parameters ----
PRICE_COL = "Adj Close"       # or "Close"
LOOKBACK_MONTHS = 12
SKIP_MONTHS = 1
HOLD_MONTHS = 1
N_STOCKS = 30
MIN_PRICE = 10.0
USE_MEMBERSHIP_FILTER = True
USE_EXIT_BAND = False    # turnover-reduction: keep a held stock until its rank falls below N*(1+EXIT_BAND_PCT/100)
EXIT_BAND_PCT = 0.0
WEIGHTING_MODE = "equal_monthly"    # or "drift" -- see run_backtest docstring in backtest_engine.py
USE_STOPLOSS = False     # per-stock daily stoploss, see apply_stoploss docstring in backtest_engine.py
STOPLOSS_PCT = 10.0
MAX_REENTRIES = 0


def main():
    print("Running momentum backtest...")
    result = run_full_backtest(
        price_col=PRICE_COL,
        lookback_months=LOOKBACK_MONTHS,
        skip_months=SKIP_MONTHS,
        hold_months=HOLD_MONTHS,
        n_stocks=N_STOCKS,
        min_price=MIN_PRICE,
        use_membership_filter=USE_MEMBERSHIP_FILTER,
        use_exit_band=USE_EXIT_BAND,
        exit_band_pct=EXIT_BAND_PCT,
        weighting_mode=WEIGHTING_MODE,
        use_stoploss=USE_STOPLOSS,
        stoploss_pct=STOPLOSS_PCT,
        max_reentries=MAX_REENTRIES,
    )

    mp = result["monthly_prices"]
    print(f"  {mp.shape[1]} stocks, {mp.shape[0]} months "
          f"({mp.index.min().date()} to {mp.index.max().date()})")
    if result["membership"] is not None:
        avg_eligible = result["membership"].sum(axis=1).replace(0, float("nan")).mean()
        print(f"  avg eligible stocks per month: {avg_eligible:.0f}")

    print("\n=== Momentum strategy ===")
    for k, v in result["strategy_stats"].items():
        print(f"  {k}: {v:.2%}" if k != "Sharpe" else f"  {k}: {v:.2f}")

    print("\n=== Benchmark (Nifty 500) ===")
    for k, v in result["benchmark_stats"].items():
        print(f"  {k}: {v:.2%}" if k != "Sharpe" else f"  {k}: {v:.2f}")

    yt = yearly_table(result["strategy_returns"], result["benchmark_returns"])
    print("\n=== Year-wise returns / max drawdown (strategy | benchmark) ===")
    for year, row in yt.iterrows():
        print(f"  {year}: {row['strategy_return']:+.2%} / {row['strategy_max_drawdown']:.2%}"
              f"   |   {row['benchmark_return']:+.2%} / {row['benchmark_max_drawdown']:.2%}")

    out = result["strategy_returns"].to_frame("strategy_return")
    out["benchmark_return"] = result["benchmark_returns"]
    out["strategy_cum"] = (1 + out["strategy_return"]).cumprod()
    out["benchmark_cum"] = (1 + out["benchmark_return"]).cumprod()
    out_path = ROOT / "data" / "momentum_backtest_results.csv"
    out.to_csv(out_path)
    print(f"\nSaved monthly results to {out_path}")

    yearly_path = ROOT / "data" / "momentum_backtest_yearly.csv"
    yt.to_csv(yearly_path)
    print(f"Saved year-wise results to {yearly_path}")

    trade_log = build_trade_log(
        result["monthly_prices"], result["strategy_returns"], result["holdings_history"], N_STOCKS
    )
    trades_path = ROOT / "data" / "momentum_backtest_trades.csv"
    trade_log.to_csv(trades_path, index=False)
    n_closed = (trade_log["status"] == "closed").sum()
    n_open = (trade_log["status"] == "open").sum()
    print(f"Saved {n_closed} closed + {n_open} open trades to {trades_path}")


if __name__ == "__main__":
    main()
