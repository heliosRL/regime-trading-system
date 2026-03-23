# backtest/engine.py
# Vectorized backtesting engine — no lookahead bias.
# Weights on day t are applied to returns on day t+1.

import numpy as np
import pandas as pd
from typing import Tuple


class BacktestEngine:
    """
    Vectorized backtest engine.

    Given portfolio weights and asset returns, computes:
    - Portfolio returns (after transaction costs)
    - Equity curve
    - Performance metrics
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        weights: pd.DataFrame,
        transaction_cost: float = 0.001,
        initial_capital: float = 100_000,
    ):
        """
        Parameters
        ----------
        returns          : daily asset returns (index=Date, cols=tickers)
        weights          : portfolio weights (same shape as returns)
                           weights[t] are used to compute returns[t+1]
        transaction_cost : one-way cost per unit traded (e.g. 0.001 = 10bps)
        initial_capital  : starting capital in USD
        """
        self.returns = returns
        self.weights = weights
        self.tc = transaction_cost
        self.capital = initial_capital

        self.portfolio_returns: pd.Series = None
        self.equity_curve: pd.Series = None
        self._run()

    def _run(self):
        """Execute the backtest."""
        # Align weights and returns; shift weights by 1 (no lookahead)
        w = self.weights.shift(1).reindex(self.returns.index).fillna(0.0)
        r = self.returns.reindex(self.returns.index)

        # Gross portfolio return = dot(weights, returns) per day
        gross_returns = (w * r).sum(axis=1)

        # Transaction costs = tc * sum of |weight changes| per day
        weight_changes = w.diff().abs().sum(axis=1).fillna(0.0)
        costs = self.tc * weight_changes

        # Net returns
        net_returns = gross_returns - costs
        self.portfolio_returns = net_returns

        # Equity curve
        self.equity_curve = (1 + net_returns).cumprod() * self.capital

    # ── Performance Metrics ────────────────────────────────────────────────

    def sharpe_ratio(self, rf: float = 0.0) -> float:
        """Annualized Sharpe ratio."""
        excess = self.portfolio_returns - rf / 252
        if excess.std() == 0:
            return 0.0
        return float((excess.mean() / excess.std()) * np.sqrt(252))

    def annual_return(self) -> float:
        """Compound annual growth rate (CAGR)."""
        n_years = len(self.portfolio_returns) / 252
        total_return = (1 + self.portfolio_returns).prod()
        return float(total_return ** (1 / n_years) - 1)

    def max_drawdown(self) -> float:
        """Maximum peak-to-trough drawdown."""
        cumulative = (1 + self.portfolio_returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        return float(drawdown.min())

    def drawdown_series(self) -> pd.Series:
        """Full drawdown time series."""
        cumulative = (1 + self.portfolio_returns).cumprod()
        peak = cumulative.cummax()
        return (cumulative - peak) / peak

    def calmar_ratio(self) -> float:
        """Calmar ratio = CAGR / |Max Drawdown|."""
        mdd = abs(self.max_drawdown())
        if mdd == 0:
            return 0.0
        return self.annual_return() / mdd

    def win_rate(self) -> float:
        """Fraction of positive-return days."""
        return float((self.portfolio_returns > 0).mean())

    def rolling_sharpe(self, window: int = 252) -> pd.Series:
        """Rolling annualized Sharpe ratio."""
        r = self.portfolio_returns
        return (r.rolling(window).mean() / r.rolling(window).std()) * np.sqrt(252)

    def summary(self) -> pd.Series:
        """Return a Series of all key performance metrics."""
        return pd.Series({
            "Annual Return (%)": round(self.annual_return() * 100, 2),
            "Sharpe Ratio":      round(self.sharpe_ratio(), 3),
            "Calmar Ratio":      round(self.calmar_ratio(), 3),
            "Max Drawdown (%)":  round(self.max_drawdown() * 100, 2),
            "Win Rate (%)":      round(self.win_rate() * 100, 2),
            "Total Days":        len(self.portfolio_returns),
        })


def run_benchmark(
    returns: pd.DataFrame,
    ticker: str,
    initial_capital: float = 100_000,
) -> Tuple[pd.Series, pd.Series]:
    """
    Buy-and-hold benchmark: 100% allocation to a single ticker.

    Returns
    -------
    (benchmark_returns, equity_curve)
    """
    r = returns[ticker].dropna()
    equity = (1 + r).cumprod() * initial_capital
    return r, equity


def run_unfiltered_momentum(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    lookback: int = 63,
    skip_recent: int = 5,
    top_n: int = 3,
    vol_window: int = 21,
    vol_target: float = 0.15,
    max_leverage: float = 1.5,
    transaction_cost: float = 0.001,
    initial_capital: float = 100_000,
) -> BacktestEngine:
    """
    Run momentum strategy WITHOUT regime filter (for comparison).
    """
    from strategy.momentum_signal import (
        compute_momentum, rank_signal,
        compute_realized_vol, compute_vol_scaled_weights
    )
    momentum = compute_momentum(prices, lookback, skip_recent)
    signal = rank_signal(momentum, top_n)
    realized_vol = compute_realized_vol(returns, vol_window)
    weights = compute_vol_scaled_weights(signal, realized_vol, vol_target, max_leverage)
    return BacktestEngine(returns, weights, transaction_cost, initial_capital)
