# visualization/plots.py
# Publication-quality charts for strategy analysis.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter

# ── Style ──────────────────────────────────────────────────────────────────────
COLORS = {
    "strategy":    "#2ECC71",
    "unfiltered":  "#F39C12",
    "benchmark":   "#3498DB",
    "risk_on":     "#2ECC7120",
    "risk_off":    "#E74C3C20",
    "drawdown":    "#E74C3C",
    "neutral":     "#BDC3C7",
}

plt.rcParams.update({
    "figure.facecolor":  "#0D1117",
    "axes.facecolor":    "#161B22",
    "axes.edgecolor":    "#30363D",
    "axes.labelcolor":   "#C9D1D9",
    "axes.titlecolor":   "#F0F6FC",
    "xtick.color":       "#8B949E",
    "ytick.color":       "#8B949E",
    "text.color":        "#C9D1D9",
    "grid.color":        "#21262D",
    "grid.linewidth":    0.8,
    "legend.facecolor":  "#161B22",
    "legend.edgecolor":  "#30363D",
    "font.family":       "monospace",
    "font.size":         10,
})

pct_fmt = FuncFormatter(lambda x, _: f"{x:.0%}")
dollar_fmt = FuncFormatter(lambda x, _: f"${x:,.0f}")


def _shade_regimes(ax, regime: pd.Series):
    """Shade background of axes based on regime (green=on, red=off)."""
    in_regime = None
    start = None
    for date, val in regime.items():
        color = COLORS["risk_on"] if val == 1 else COLORS["risk_off"]
        if in_regime != val:
            if start is not None:
                ax.axvspan(start, date, alpha=0.3,
                           color=COLORS["risk_on"] if in_regime == 1 else COLORS["risk_off"],
                           linewidth=0)
            in_regime = val
            start = date
    if start is not None and in_regime is not None:
        ax.axvspan(start, regime.index[-1], alpha=0.3,
                   color=COLORS["risk_on"] if in_regime == 1 else COLORS["risk_off"],
                   linewidth=0)


def plot_regime_detection(
    prices: pd.DataFrame,
    regime: pd.Series,
    ticker: str,
    save_path: str = None,
):
    """Plot asset price with regime shading."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7),
                                    gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(f"HMM Regime Detection — {ticker}", fontsize=14, fontweight="bold",
                 color="#F0F6FC", y=0.98)

    price_series = prices[ticker].dropna()
    ax1.plot(price_series.index, price_series.values, color=COLORS["benchmark"],
             linewidth=1.2, label=f"{ticker} Price")
    _shade_regimes(ax1, regime.reindex(price_series.index).fillna(0))
    ax1.set_ylabel("Price (USD)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.4)

    # Regime bar
    aligned = regime.reindex(price_series.index).fillna(0)
    ax2.fill_between(aligned.index, aligned.values, step="post",
                     color=COLORS["strategy"], alpha=0.7, label="Risk-On")
    ax2.fill_between(aligned.index, aligned.values, 1, step="post",
                     color=COLORS["drawdown"], alpha=0.3, label="Risk-Off")
    ax2.set_ylabel("Regime")
    ax2.set_ylim(0, 1.2)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Risk-Off", "Risk-On"])
    ax2.legend(loc="upper right")

    risk_on_pct = aligned.mean() * 100
    ax2.set_title(f"Risk-On: {risk_on_pct:.1f}% of days", fontsize=9, color="#8B949E")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved: {save_path}")
    plt.close()


def plot_equity_curves(
    strategy_equity: pd.Series,
    unfiltered_equity: pd.Series,
    benchmark_equity: pd.Series,
    regime: pd.Series,
    save_path: str = None,
):
    """Plot equity curves for all three strategies with regime shading."""
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle("Equity Curves — Strategy Comparison", fontsize=14,
                 fontweight="bold", color="#F0F6FC")

    _shade_regimes(ax, regime.reindex(strategy_equity.index).fillna(0))
    ax.plot(strategy_equity.index, strategy_equity.values,
            color=COLORS["strategy"], lw=2.0, label="Regime-Filtered Momentum")
    ax.plot(unfiltered_equity.index, unfiltered_equity.values,
            color=COLORS["unfiltered"], lw=1.5, linestyle="--", label="Unfiltered Momentum")
    ax.plot(benchmark_equity.index, benchmark_equity.values,
            color=COLORS["benchmark"], lw=1.5, linestyle=":", label="SPY Buy & Hold")

    ax.yaxis.set_major_formatter(dollar_fmt)
    ax.set_ylabel("Portfolio Value (USD)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.4)

    on_patch = mpatches.Patch(color=COLORS["risk_on"], alpha=0.6, label="Risk-On Regime")
    off_patch = mpatches.Patch(color=COLORS["risk_off"], alpha=0.6, label="Risk-Off Regime")
    ax.legend(handles=ax.get_legend_handles_labels()[0] + [on_patch, off_patch],
              labels=ax.get_legend_handles_labels()[1] + ["Risk-On Regime", "Risk-Off Regime"],
              loc="upper left", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved: {save_path}")
    plt.close()


def plot_drawdowns(
    strategy_dd: pd.Series,
    unfiltered_dd: pd.Series,
    benchmark_dd: pd.Series,
    save_path: str = None,
):
    """Plot drawdown comparison."""
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle("Drawdown Analysis", fontsize=14, fontweight="bold", color="#F0F6FC")

    ax.fill_between(strategy_dd.index, strategy_dd.values * 100, 0,
                    alpha=0.6, color=COLORS["strategy"], label="Regime-Filtered Momentum")
    ax.fill_between(unfiltered_dd.index, unfiltered_dd.values * 100, 0,
                    alpha=0.3, color=COLORS["unfiltered"], label="Unfiltered Momentum")
    ax.plot(benchmark_dd.index, benchmark_dd.values * 100,
            color=COLORS["benchmark"], lw=1.2, linestyle=":", label="SPY Buy & Hold")

    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved: {save_path}")
    plt.close()


def plot_rolling_sharpe(
    strategy_rs: pd.Series,
    unfiltered_rs: pd.Series,
    benchmark_rs: pd.Series,
    save_path: str = None,
):
    """Plot rolling 252-day Sharpe ratio."""
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle("Rolling 252-Day Sharpe Ratio", fontsize=14,
                 fontweight="bold", color="#F0F6FC")

    ax.plot(strategy_rs.index, strategy_rs.values,
            color=COLORS["strategy"], lw=1.8, label="Regime-Filtered Momentum")
    ax.plot(unfiltered_rs.index, unfiltered_rs.values,
            color=COLORS["unfiltered"], lw=1.4, linestyle="--", label="Unfiltered Momentum")
    ax.plot(benchmark_rs.index, benchmark_rs.values,
            color=COLORS["benchmark"], lw=1.2, linestyle=":", label="SPY Buy & Hold")
    ax.axhline(0, color=COLORS["neutral"], lw=0.8, linestyle="-")
    ax.axhline(1, color=COLORS["neutral"], lw=0.6, linestyle="--", alpha=0.5)

    ax.set_ylabel("Sharpe Ratio (252-day)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved: {save_path}")
    plt.close()


def plot_performance_dashboard(
    strategy_bt,
    unfiltered_bt,
    benchmark_returns: pd.Series,
    regime: pd.Series,
    regime_stats: pd.DataFrame,
    save_path: str = None,
):
    """Full performance dashboard — the hero chart."""
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#0D1117")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Equity Curve ──────────────────────────────────────────────────────
    ax_eq = fig.add_subplot(gs[0, :])
    from backtest.engine import BacktestEngine
    bm_equity = (1 + benchmark_returns).cumprod() * 100_000

    _shade_regimes(ax_eq, regime.reindex(strategy_bt.equity_curve.index).fillna(0))
    ax_eq.plot(strategy_bt.equity_curve, color=COLORS["strategy"], lw=2,
               label="Regime-Filtered Momentum")
    ax_eq.plot(unfiltered_bt.equity_curve, color=COLORS["unfiltered"], lw=1.5,
               linestyle="--", label="Unfiltered Momentum")
    ax_eq.plot(bm_equity, color=COLORS["benchmark"], lw=1.5,
               linestyle=":", label="SPY Buy & Hold")
    ax_eq.yaxis.set_major_formatter(dollar_fmt)
    ax_eq.set_title("Portfolio Equity Curve", color="#F0F6FC", fontweight="bold")
    ax_eq.legend(loc="upper left", fontsize=9)
    ax_eq.grid(True, alpha=0.4)

    # ── Drawdowns ─────────────────────────────────────────────────────────
    ax_dd = fig.add_subplot(gs[1, :2])
    ax_dd.fill_between(strategy_bt.drawdown_series().index,
                       strategy_bt.drawdown_series().values * 100, 0,
                       alpha=0.7, color=COLORS["strategy"])
    ax_dd.plot(unfiltered_bt.drawdown_series().index,
               unfiltered_bt.drawdown_series().values * 100,
               color=COLORS["unfiltered"], lw=1, linestyle="--")
    ax_dd.set_title("Drawdowns (%)", color="#F0F6FC", fontweight="bold")
    ax_dd.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax_dd.grid(True, alpha=0.4)

    # ── Rolling Sharpe ────────────────────────────────────────────────────
    ax_rs = fig.add_subplot(gs[1, 2])
    rs = strategy_bt.rolling_sharpe()
    rs_unf = unfiltered_bt.rolling_sharpe()
    ax_rs.plot(rs.index, rs.values, color=COLORS["strategy"], lw=1.5)
    ax_rs.plot(rs_unf.index, rs_unf.values, color=COLORS["unfiltered"],
               lw=1, linestyle="--")
    ax_rs.axhline(0, color=COLORS["neutral"], lw=0.8)
    ax_rs.set_title("Rolling Sharpe (252d)", color="#F0F6FC", fontweight="bold")
    ax_rs.grid(True, alpha=0.4)

    # ── Metrics Table ─────────────────────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[2, :2])
    ax_tbl.axis("off")

    bm_bt = type("BM", (), {
        "portfolio_returns": benchmark_returns,
        "equity_curve": bm_equity,
    })()

    from backtest.engine import BacktestEngine as _BE
    bm_engine = _BE(
        pd.DataFrame({"SPY": benchmark_returns}),
        pd.DataFrame({"SPY": pd.Series(1.0, index=benchmark_returns.index)}),
        transaction_cost=0.0,
    )

    rows = ["Annual Return (%)", "Sharpe Ratio", "Calmar Ratio",
            "Max Drawdown (%)", "Win Rate (%)"]
    s_vals = strategy_bt.summary()
    u_vals = unfiltered_bt.summary()
    bm_vals = bm_engine.summary()

    cell_data = [[s_vals[r], u_vals[r], bm_vals[r]] for r in rows]
    tbl = ax_tbl.table(
        cellText=cell_data,
        rowLabels=rows,
        colLabels=["Regime-Filtered", "Unfiltered", "SPY B&H"],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.1, 1.8)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor("#161B22")
        cell.set_edgecolor("#30363D")
        cell.set_text_props(color="#C9D1D9")
        if row == 0:
            cell.set_facecolor("#21262D")
            cell.set_text_props(color="#F0F6FC", fontweight="bold")

    ax_tbl.set_title("Performance Summary", color="#F0F6FC",
                     fontweight="bold", pad=10)

    # ── Regime Stats ──────────────────────────────────────────────────────
    ax_rg = fig.add_subplot(gs[2, 2])
    ax_rg.axis("off")
    regime_data = [[f"{regime_stats.loc[r, c]:.3f}" for c in regime_stats.columns]
                   for r in regime_stats.index]
    tbl2 = ax_rg.table(
        cellText=regime_data,
        rowLabels=list(regime_stats.index),
        colLabels=list(regime_stats.columns),
        loc="center",
        cellLoc="center",
    )
    tbl2.auto_set_font_size(False)
    tbl2.set_fontsize(9)
    tbl2.scale(1.0, 1.8)
    for (row, col), cell in tbl2.get_celld().items():
        cell.set_facecolor("#161B22")
        cell.set_edgecolor("#30363D")
        cell.set_text_props(color="#C9D1D9")
        if row == 0:
            cell.set_facecolor("#21262D")
            cell.set_text_props(color="#F0F6FC", fontweight="bold")
    ax_rg.set_title("Regime Statistics", color="#F0F6FC",
                    fontweight="bold", pad=10)

    fig.suptitle("Regime-Filtered Momentum System — Full Dashboard",
                 fontsize=16, fontweight="bold", color="#F0F6FC", y=1.01)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[Plot] Saved: {save_path}")
    plt.close()
