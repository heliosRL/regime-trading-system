# config.py
# Central configuration for the Regime-Filtered Momentum Trading System
# Modify parameters here to experiment with different setups.

from datetime import datetime

# ── Data ──────────────────────────────────────────────────────────────────────
TICKERS = ["SPY", "QQQ", "IWM", "GLD", "TLT", "EFA"]   # Universe of assets
BENCHMARK = "SPY"                                         # Benchmark ticker
START_DATE = "2010-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

# ── Regime Model (HMM) ────────────────────────────────────────────────────────
N_REGIMES = 2               # Number of hidden states (2 = Risk-On / Risk-Off)
HMM_FEATURES_WINDOW = 21    # Rolling window for HMM input features (days)
HMM_TRAIN_TICKER = "SPY"    # Ticker used to train the regime model
RANDOM_STATE = 42

# ── Momentum Signal ───────────────────────────────────────────────────────────
LOOKBACK_MOMENTUM = 63      # Momentum lookback in trading days (~3 months)
SKIP_RECENT = 5             # Skip most recent N days (avoid short-term reversal)
TOP_N = 3                   # Number of assets to go long

# ── Volatility Scaling ────────────────────────────────────────────────────────
VOL_WINDOW = 21             # Realized volatility window (days)
VOL_TARGET = 0.15           # Annual portfolio volatility target (15%)
MAX_LEVERAGE = 1.5          # Cap on total portfolio leverage

# ── Backtest ──────────────────────────────────────────────────────────────────
TRANSACTION_COST = 0.001    # One-way cost per trade (10 bps)
INITIAL_CAPITAL = 100_000   # Starting portfolio value (USD)
REBALANCE_FREQ = "daily"    # Rebalancing frequency

# ── Output ────────────────────────────────────────────────────────────────────
RESULTS_DIR = "results"
FIGURE_DPI = 150
