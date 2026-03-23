# main.py
# Entry point for the Regime-Filtered Momentum Trading System.
# Run: python main.py

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data.fetch_data import fetch_prices, compute_returns, compute_rolling_features
from models.hmm_regime import RegimeHMM
from strategy.momentum_signal import build_strategy_weights
from backtest.engine import BacktestEngine, run_benchmark, run_unfiltered_momentum
from visualization.plots import (
    plot_regime_detection,
    plot_equity_curves,
    plot_drawdowns,
    plot_rolling_sharpe,
    plot_performance_dashboard,
)


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs("data/cache", exist_ok=True)

    print("=" * 60)
    print("  REGIME-FILTERED MOMENTUM TRADING SYSTEM")
    print("=" * 60)

    # ── 1. Fetch Data ─────────────────────────────────────────────────────
    print("\n[Step 1/5] Fetching market data ...")
    prices = fetch_prices(config.TICKERS, config.START_DATE, config.END_DATE)
    prices = prices.dropna(thresh=int(0.8 * len(prices)), axis=1)
    available_tickers = [t for t in config.TICKERS if t in prices.columns]
    prices = prices[available_tickers].dropna()
    returns = compute_returns(prices)

    print(f"  Tickers  : {available_tickers}")
    print(f"  Date range: {prices.index[0].date()} → {prices.index[-1].date()}")
    print(f"  Trading days: {len(prices)}")

    # ── 2. Train HMM Regime Model ─────────────────────────────────────────
    print(f"\n[Step 2/5] Training HMM regime model on {config.HMM_TRAIN_TICKER} ...")
    hmm_features = compute_rolling_features(
        returns, config.HMM_TRAIN_TICKER, config.HMM_FEATURES_WINDOW
    )
    hmm = RegimeHMM(n_regimes=config.N_REGIMES, random_state=config.RANDOM_STATE)
    hmm.fit(hmm_features)
    regime = hmm.predict(hmm_features)

    # Extend regime to full index (forward-fill for early dates)
    regime_full = regime.reindex(returns.index).ffill().fillna(0).astype(int)
    regime_stats = hmm.regime_stats(regime_full, returns[config.HMM_TRAIN_TICKER])
    print("\n  Regime Statistics:")
    print(regime_stats.to_string())

    # ── 3. Build Strategy Weights ─────────────────────────────────────────
    print("\n[Step 3/5] Building strategy weights ...")
    weights = build_strategy_weights(
        prices=prices,
        returns=returns,
        regime=regime_full,
        lookback=config.LOOKBACK_MOMENTUM,
        skip_recent=config.SKIP_RECENT,
        top_n=config.TOP_N,
        vol_window=config.VOL_WINDOW,
        vol_target=config.VOL_TARGET,
        max_leverage=config.MAX_LEVERAGE,
    )
    print(f"  Non-zero weight days: {(weights.abs().sum(axis=1) > 0).sum()}")
    print(f"  Avg daily leverage  : {weights.abs().sum(axis=1).mean():.3f}")

    # ── 4. Run Backtests ──────────────────────────────────────────────────
    print("\n[Step 4/5] Running backtests ...")

    # Regime-filtered strategy
    strategy_bt = BacktestEngine(
        returns=returns,
        weights=weights,
        transaction_cost=config.TRANSACTION_COST,
        initial_capital=config.INITIAL_CAPITAL,
    )

    # Unfiltered momentum (no regime filter, for comparison)
    unfiltered_bt = run_unfiltered_momentum(
        prices=prices,
        returns=returns,
        lookback=config.LOOKBACK_MOMENTUM,
        skip_recent=config.SKIP_RECENT,
        top_n=config.TOP_N,
        vol_window=config.VOL_WINDOW,
        vol_target=config.VOL_TARGET,
        max_leverage=config.MAX_LEVERAGE,
        transaction_cost=config.TRANSACTION_COST,
        initial_capital=config.INITIAL_CAPITAL,
    )

    # Buy-and-hold benchmark
    bm_returns, bm_equity = run_benchmark(returns, config.BENCHMARK, config.INITIAL_CAPITAL)

    # Print comparison
    print("\n" + "=" * 55)
    print(f"  {'Metric':<22} {'Reg.Filter':>10} {'Unfiltered':>10} {'SPY B&H':>10}")
    print("=" * 55)
    s = strategy_bt.summary()
    u = unfiltered_bt.summary()

    bm_bt = BacktestEngine(
        pd.DataFrame({"SPY": bm_returns}),
        pd.DataFrame({"SPY": pd.Series(1.0, index=bm_returns.index)}),
        transaction_cost=0.0,
        initial_capital=config.INITIAL_CAPITAL,
    )
    b = bm_bt.summary()

    for metric in s.index:
        print(f"  {metric:<22} {str(s[metric]):>10} {str(u[metric]):>10} {str(b[metric]):>10}")
    print("=" * 55)

    # ── 5. Generate Charts ────────────────────────────────────────────────
    print("\n[Step 5/5] Generating charts ...")

    plot_regime_detection(
        prices, regime_full, config.HMM_TRAIN_TICKER,
        save_path=os.path.join(config.RESULTS_DIR, "regime_detection.png"),
    )
    plot_equity_curves(
        strategy_bt.equity_curve, unfiltered_bt.equity_curve, bm_equity,
        regime_full,
        save_path=os.path.join(config.RESULTS_DIR, "equity_curves.png"),
    )
    plot_drawdowns(
        strategy_bt.drawdown_series(),
        unfiltered_bt.drawdown_series(),
        bm_bt.drawdown_series(),
        save_path=os.path.join(config.RESULTS_DIR, "drawdowns.png"),
    )
    plot_rolling_sharpe(
        strategy_bt.rolling_sharpe(),
        unfiltered_bt.rolling_sharpe(),
        bm_bt.rolling_sharpe(),
        save_path=os.path.join(config.RESULTS_DIR, "rolling_sharpe.png"),
    )
    plot_performance_dashboard(
        strategy_bt, unfiltered_bt, bm_returns, regime_full, regime_stats,
        save_path=os.path.join(config.RESULTS_DIR, "performance_summary.png"),
    )

    print(f"\n Done! All results saved to ./{config.RESULTS_DIR}/")
    print("\nFiles generated:")
    for f in sorted(os.listdir(config.RESULTS_DIR)):
        print(f"   results/{f}")


if __name__ == "__main__":
    main()
