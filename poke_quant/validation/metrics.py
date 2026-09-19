"""
poke_quant/validation/metrics.py — Metriche di performance standard (CAGR, Sharpe, Max Drawdown, Calmar, Sortino).
Adattate dal framework quantitativo istituzionale di ApexConvex per serie periodiche (mensili).
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def cagr(returns: pd.Series, periods_per_year: int = 12) -> float:
    """Rendimento annuo composto (Compound Annual Growth Rate)."""
    if returns is None or len(returns) == 0:
        return float("nan")
    total_growth = float((1 + returns).prod())
    years = len(returns) / periods_per_year
    if years <= 0 or total_growth <= 0:
        return float("nan")
    return total_growth ** (1 / years) - 1


def sharpe(returns: pd.Series, rf_annual: float = 0.0, periods_per_year: int = 12) -> float:
    """Sharpe Ratio annualizzato con tasso privo di rischio rf_annual."""
    if returns is None or len(returns) < 2:
        return 0.0
    excess = returns - rf_annual / periods_per_year
    std = excess.std(ddof=1)
    if std < 1e-12 or np.isnan(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """Massimo drawdown storico del portafoglio (numero negativo o zero)."""
    if returns is None or len(returns) == 0:
        return 0.0
    nav = (1 + returns).cumprod()
    peak = nav.cummax()
    dd = nav / peak - 1
    return float(dd.min())


def calmar(returns: pd.Series, periods_per_year: int = 12) -> float:
    """Calmar Ratio = CAGR / |MaxDD|."""
    dd = max_drawdown(returns)
    if abs(dd) < 1e-12:
        return float("nan")
    c = cagr(returns, periods_per_year=periods_per_year)
    return c / abs(dd)


def sortino_ratio(returns: pd.Series, rf_annual: float = 0.0, periods_per_year: int = 12) -> float:
    """Sortino Ratio annualizzato considerando solo la deviazione al ribasso (downside deviation)."""
    if returns is None or len(returns) < 2:
        return float("nan")
    excess = returns - rf_annual / periods_per_year
    downside = excess[excess < 0]
    if len(downside) < 2 or downside.std(ddof=1) < 1e-12:
        return float("nan")
    return float(excess.mean() / downside.std(ddof=1) * np.sqrt(periods_per_year))


def ulcer_index(returns: pd.Series) -> float:
    """Ulcer Index: radice quadrata della media dei drawdown al quadrato."""
    if returns is None or len(returns) == 0:
        return 0.0
    nav = (1 + returns).cumprod()
    dd_pct = (nav / nav.cummax() - 1.0) * 100.0
    return float(np.sqrt((dd_pct ** 2).mean()))


def compute_trade_metrics(trades_df: pd.DataFrame) -> dict:
    """Calcola statistiche dettagliate a livello di singolo trade chiuso."""
    if trades_df is None or trades_df.empty:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": float("nan"),
            "avg_net_return": 0.0,
            "avg_holding_months": 0.0,
            "total_net_pnl": 0.0,
            "total_fees_paid": 0.0
        }

    total_trades = len(trades_df)
    wins = trades_df[trades_df["net_pnl"] > 0]
    losses = trades_df[trades_df["net_pnl"] <= 0]
    
    win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
    gross_profit = wins["net_pnl"].sum()
    gross_loss = abs(losses["net_pnl"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    
    return {
        "total_trades": total_trades,
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "avg_net_return": float(trades_df["net_roi"].mean()),
        "avg_holding_months": float(trades_df["holding_months"].mean()) if "holding_months" in trades_df.columns else 0.0,
        "total_net_pnl": float(trades_df["net_pnl"].sum()),
        "total_fees_paid": float(trades_df["fees_paid"].sum()) if "fees_paid" in trades_df.columns else 0.0
    }
