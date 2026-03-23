# models/hmm_regime.py
# Hidden Markov Model for market regime classification.
# Classifies each trading day as Risk-On (bullish/low-vol) or Risk-Off (bearish/high-vol).

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


class RegimeHMM:
    """
    Gaussian Mixture Model that detects latent market regimes from rolling
    return features. Drop-in replacement for hmmlearn-based HMM.

    After fitting, each regime is labeled as 'risk_on' (1) or 'risk_off' (0)
    based on which state has higher mean returns.
    """

    def __init__(self, n_regimes: int = 2, random_state: int = 42):
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.model = GaussianMixture(
            n_components=n_regimes,
            covariance_type="full",
            n_init=5,
            random_state=random_state,
        )
        self.scaler = StandardScaler()
        self.risk_on_state: int = None   # which HMM state index = Risk-On
        self.is_fitted: bool = False

    def fit(self, features: pd.DataFrame) -> "RegimeHMM":
        """
        Fit the HMM on rolling market features.

        Parameters
        ----------
        features : DataFrame with columns [rolling_mean, rolling_vol, rolling_skew]
        """
        X = self.scaler.fit_transform(features.values)
        self.model.fit(X)

        # Label regimes: the state with higher mean return → Risk-On
        mean_returns = self.model.means_[:, 0]   # rolling_mean is first feature
        self.risk_on_state = int(np.argmax(mean_returns))
        self.is_fitted = True
        print(f"[Regime] Fitted GMM. Risk-On state = {self.risk_on_state} "
              f"(mean={mean_returns[self.risk_on_state]:.5f}), "
              f"Risk-Off state = {1 - self.risk_on_state} "
              f"(mean={mean_returns[1 - self.risk_on_state]:.5f})")
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """
        Predict regimes for each row in features.

        Returns
        -------
        regime : pd.Series of 1 (Risk-On) or 0 (Risk-Off) indexed like features
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict().")
        X = self.scaler.transform(features.values)
        raw_states = self.model.predict(X)
        regime = pd.Series(
            (raw_states == self.risk_on_state).astype(int),
            index=features.index,
            name="regime",
        )
        return regime

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Return posterior probability of each state.

        Returns
        -------
        DataFrame with columns [state_0_prob, state_1_prob, risk_on_prob]
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict_proba().")
        X = self.scaler.transform(features.values)
        posteriors = self.model.predict_proba(X)
        df = pd.DataFrame(
            posteriors,
            index=features.index,
            columns=[f"state_{i}_prob" for i in range(self.n_regimes)],
        )
        df["risk_on_prob"] = df[f"state_{self.risk_on_state}_prob"]
        return df

    def regime_stats(self, regime: pd.Series, returns: pd.Series) -> pd.DataFrame:
        """
        Compute return statistics conditional on each regime.

        Parameters
        ----------
        regime  : pd.Series of 0/1 regime labels
        returns : pd.Series of daily returns (same index)

        Returns
        -------
        DataFrame with annualized mean, vol, Sharpe per regime
        """
        aligned = pd.concat([regime, returns], axis=1).dropna()
        aligned.columns = ["regime", "returns"]
        stats = {}
        for label, name in [(0, "Risk-Off"), (1, "Risk-On")]:
            r = aligned.loc[aligned["regime"] == label, "returns"]
            stats[name] = {
                "days": len(r),
                "ann_return": r.mean() * 252,
                "ann_vol": r.std() * np.sqrt(252),
                "sharpe": (r.mean() / r.std()) * np.sqrt(252) if r.std() > 0 else 0,
            }
        return pd.DataFrame(stats).T
