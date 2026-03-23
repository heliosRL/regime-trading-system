# data/fetch_data.py
# Downloads and caches OHLCV price data using yfinance.

import os
import pandas as pd
import yfinance as yf
from typing import List


def fetch_prices(
    tickers: List[str],
    start: str,
    end: str,
    cache_dir: str = "data/cache",
) -> pd.DataFrame:
    """
    Download adjusted closing prices for a list of tickers.

    Parameters
    ----------
    tickers  : list of ticker symbols
    start    : start date string "YYYY-MM-DD"
    end      : end date string "YYYY-MM-DD"
    cache_dir: directory to cache raw data

    Returns
    -------
    prices : DataFrame (index=Date, columns=tickers) of adjusted close prices
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"prices_{start}_{end}.parquet")

    if os.path.exists(cache_file):
        print(f"[Data] Loading cached prices from {cache_file}")
        return pd.read_parquet(cache_file)

    print(f"[Data] Downloading {tickers} from {start} to {end} ...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.dropna(how="all")
    prices.to_parquet(cache_file)
    print(f"[Data] Saved to cache: {cache_file}")
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns."""
    return prices.pct_change().dropna()


def compute_rolling_features(
    returns: pd.DataFrame,
    ticker: str,
    window: int = 21,
) -> pd.DataFrame:
    """
    Compute rolling features used as HMM inputs for a single ticker.

    Features
    --------
    - rolling_mean   : rolling mean return
    - rolling_vol    : rolling std of returns (realized volatility)
    - rolling_skew   : rolling skewness
    """
    s = returns[ticker]
    features = pd.DataFrame(index=s.index)
    features["rolling_mean"] = s.rolling(window).mean()
    features["rolling_vol"] = s.rolling(window).std()
    features["rolling_skew"] = s.rolling(window).skew()
    return features.dropna()
