# Regime-Filtered Momentum Trading System

A systematic trading strategy that combines **Hidden Markov Model (HMM) regime detection** with **momentum signals** and **volatility-scaled position sizing** to generate robust risk-adjusted returns.

---

## Core Idea

Raw momentum strategies suffer during market regime transitions — they tend to perform well in trending markets but break down during high-volatility or mean-reverting regimes. This project addresses that by:

1. **Detecting market regimes** using a Hidden Markov Model on returns + volatility features
2. **Filtering momentum trades** — only entering positions when the detected regime is favorable
3. **Scaling position sizes** inversely to realized volatility (volatility targeting) to maintain consistent risk exposure

The hypothesis: regime-filtering significantly improves Sharpe ratio and reduces drawdown compared to naive momentum.

---

## Project Structure

```
regime_trading/
│
├── data/
│   └── fetch_data.py          # Downloads market data via yfinance
│
├── models/
│   └── hmm_regime.py          # HMM-based regime classifier
│
├── strategy/
│   └── momentum_signal.py     # Momentum signal + vol scaling
│
├── backtest/
│   └── engine.py              # Vectorized backtesting engine
│
├── visualization/
│   └── plots.py               # Performance plots and regime charts
│
├── results/                   # Output charts saved here
│
├── main.py                    # Entry point — runs full pipeline
├── config.py                  # All parameters in one place
└── requirements.txt
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline
python main.py
```

Results and charts will be saved to the `results/` folder.

---

## Configuration (`config.py`)

All parameters are centralized — easy to tune:

| Parameter | Default | Description |
|---|---|---|
| `TICKERS` | SPY, QQQ, IWM, GLD | Assets to trade |
| `LOOKBACK_MOMENTUM` | 63 days | Momentum lookback window |
| `VOL_WINDOW` | 21 days | Realized volatility window |
| `VOL_TARGET` | 0.15 | Annual volatility target (15%) |
| `N_REGIMES` | 2 | Number of HMM hidden states |
| `TRANSACTION_COST` | 0.001 | Per-trade cost (10 bps) |

---

## Strategy Logic

### Step 1 — Regime Detection (HMM)
- Features: rolling returns, rolling volatility, rolling skewness
- HMM trained on SPY to classify each day as **Risk-On** or **Risk-Off**
- Risk-Off regime suppresses long signals (protects against trend breakdowns)

### Step 2 — Momentum Signal
- Cross-sectional momentum: rank assets by past 63-day return
- Long top-ranked assets, flat (or short) bottom-ranked assets
- Signal is filtered to zero in Risk-Off regimes

### Step 3 — Volatility Scaling
- Each asset's position is scaled so its contribution to portfolio volatility equals `VOL_TARGET / N_ASSETS`
- Uses 21-day realized volatility as the scaling denominator
- Ensures consistent risk exposure regardless of market conditions

### Step 4 — Execution & Backtesting
- Daily rebalancing with transaction costs (10 bps per trade)
- Vectorized backtest engine — no lookahead bias
- Benchmarked against buy-and-hold SPY

---

## Performance Metrics

The system reports:

- **Sharpe Ratio** (annualized)
- **Calmar Ratio** (return / max drawdown)
- **Maximum Drawdown**
- **Annual Return**
- **Win Rate**
- **Regime-conditional statistics** (performance in each regime)

---

## Results

The strategy is benchmarked against:
- **SPY Buy & Hold**
- **Unfiltered Momentum** (no regime filter)
- **Regime-Filtered Momentum** (this system)

Charts saved to `results/`:
- `regime_detection.png` — HMM regimes overlaid on price
- `equity_curves.png` — Strategy vs benchmarks
- `drawdowns.png` — Drawdown comparison
- `rolling_sharpe.png` — Rolling 252-day Sharpe
- `performance_summary.png` — Full dashboard

---

## Research Extensions

Ideas to extend this work:
- Replace HMM with a **Kalman Filter** for smoother regime estimates
- Add **crypto assets** for cross-asset momentum
- Use **options-implied volatility** (VIX term structure) as an additional regime feature
- Test **short-selling** in Risk-Off regimes instead of going flat
- Incorporate **transaction cost optimization** (round-trip minimization)

---

## References

- Asness, C. et al. (2013). *Value and Momentum Everywhere*. Journal of Finance.
- Ang, A. & Bekaert, G. (2002). *Regime Switches in Interest Rates*. JBES.
- Moreira, A. & Muir, T. (2017). *Volatility-Managed Portfolios*. Journal of Finance.
- Hamilton, J.D. (1989). *A New Approach to the Economic Analysis of Nonstationary Time Series*. Econometrica.
