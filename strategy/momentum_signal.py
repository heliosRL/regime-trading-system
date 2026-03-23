# strategy/momentum_signal.py
# Cross-sectional momentum signal with regime filtering and volatility scaling.

import numpy as np
import pandas as pd
from typing import Optional


def compute_momentum(
    prices: pd.DataFrame,
    lookback: int = 63,
    skip_recent: int = 5,
) -> pd.DataFrame:
    """
    Cross-sectional momentum signal: past return over [lookback, skip_recent] window.
    Skipping the most recent `skip_recent` days avoids short-term mean reversion.

    Returns
    -------
    momentum : DataFrame of momentum scores (same shape as prices)
    """
    lagged_prices = prices.shift(skip_recent)
    past_prices = prices.shift(lookback)
    momentum = (lagged_prices - past_prices) / past_prices
    return momentum


def rank_signal(momentum: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """
    Convert momentum scores to binary long signals by cross-sectional ranking.
    Long the top_n assets each day, zero for the rest.

    Returns
    -------
    signal : DataFrame of 0/1 signals (1 = long)
    """
    def rank_row(row):
        valid = row.dropna()
        if len(valid) < top_n:
            return pd.Series(0, index=row.index)
        ranked = valid.rank(ascending=False)
        signal = (ranked <= top_n).astype(float)
        signal = signal / signal.sum()   # equal-weight among top_n
        return signal.reindex(row.index, fill_value=0.0)

    return momentum.apply(rank_row, axis=1)


def apply_regime_filter(
    signal: pd.DataFrame,
    regime: pd.Series,
) -> pd.DataFrame:
    """
    Zero out all signals on Risk-Off days (regime == 0).
    This is the core regime-filtering step.

    Parameters
    ----------
    signal : raw momentum signal DataFrame
    regime : pd.Series of 0 (Risk-Off) / 1 (Risk-On) per date

    Returns
    -------
    filtered_signal : signal with Risk-Off days set to 0
    """
    aligned_regime = regime.reindex(signal.index).fillna(0)
    filtered = signal.multiply(aligned_regime, axis=0)
    return filtered


def compute_realized_vol(
    returns: pd.DataFrame,
    window: int = 21,
    annualize: bool = True,
) -> pd.DataFrame:
    """
    Compute rolling realized volatility for each asset.

    Parameters
    ----------
    returns   : daily returns DataFrame
    window    : rolling window (days)
    annualize : if True, annualize by sqrt(252)

    Returns
    -------
    vol : DataFrame of realized volatilities
    """
    vol = returns.rolling(window).std()
    if annualize:
        vol = vol * np.sqrt(252)
    return vol


def compute_vol_scaled_weights(
    signal: pd.DataFrame,
    realized_vol: pd.DataFrame,
    vol_target: float = 0.15,
    max_leverage: float = 1.5,
) -> pd.DataFrame:
    """
    Scale position weights so each asset contributes equally to portfolio volatility.

    Weight for asset i = signal_i * (vol_target / N_assets) / realized_vol_i

    Then cap total portfolio leverage at max_leverage.

    Parameters
    ----------
    signal       : long-only signal weights (sum to 1 or 0)
    realized_vol : annualized realized vol per asset
    vol_target   : target annual portfolio volatility
    max_leverage : maximum sum of absolute weights

    Returns
    -------
    weights : volatility-scaled portfolio weights
    """
    n_assets = signal.shape[1]
    per_asset_vol_target = vol_target / n_assets

    # Scale each asset weight by vol_target / realized_vol
    vol_aligned = realized_vol.reindex(signal.index).replace(0, np.nan)
    raw_weights = signal * (per_asset_vol_target / vol_aligned)

    # Cap leverage
    total_exposure = raw_weights.abs().sum(axis=1)
    scale = (max_leverage / total_exposure).clip(upper=1.0)
    scaled_weights = raw_weights.multiply(scale, axis=0)

    return scaled_weights.fillna(0.0)


def build_strategy_weights(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    regime: pd.Series,
    lookback: int = 63,
    skip_recent: int = 5,
    top_n: int = 3,
    vol_window: int = 21,
    vol_target: float = 0.15,
    max_leverage: float = 1.5,
) -> pd.DataFrame:
    """
    Full strategy pipeline: momentum → rank → regime filter → vol scale.

    Returns
    -------
    weights : final portfolio weights per day
    """
    momentum = compute_momentum(prices, lookback, skip_recent)
    signal = rank_signal(momentum, top_n)
    filtered = apply_regime_filter(signal, regime)
    realized_vol = compute_realized_vol(returns, vol_window)
    weights = compute_vol_scaled_weights(filtered, realized_vol, vol_target, max_leverage)
    return weights
