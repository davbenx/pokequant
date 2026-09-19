"""
tests/test_metrics.py — Unit test per le metriche di performance quantitative.
"""

import numpy as np
import pandas as pd
import pytest
from poke_quant.validation.metrics import (
    cagr, sharpe, max_drawdown, calmar, sortino_ratio, ulcer_index, compute_trade_metrics
)


def test_cagr_calculation():
    # 10% costante ogni anno per 3 anni (12 mesi/anno = 36 mesi)
    # rendimento mensile = 1.10^(1/12) - 1
    r_month = 1.10 ** (1.0 / 12.0) - 1.0
    series = pd.Series([r_month] * 36)
    val = cagr(series, periods_per_year=12)
    assert pytest.approx(val, rel=1e-3) == 0.10


def test_sharpe_zero_variance():
    series = pd.Series([0.01] * 24)
    val = sharpe(series, rf_annual=0.0, periods_per_year=12)
    assert val == 0.0


def test_max_drawdown():
    # Crescita 100 -> 150 -> 75 -> 120 (max peak 150, trough 75 = -50% drawdown)
    prices = pd.Series([100.0, 150.0, 75.0, 120.0])
    returns = prices.pct_change().dropna()
    dd = max_drawdown(returns)
    assert pytest.approx(dd, abs=1e-4) == -0.50


def test_calmar_ratio():
    r_month = 1.20 ** (1.0 / 12.0) - 1.0  # ~20% CAGR
    returns = pd.Series([r_month] * 24)
    returns.iloc[10] = -0.10  # inserisce drawdown
    dd = max_drawdown(returns)
    c = cagr(returns, periods_per_year=12)
    val = calmar(returns, periods_per_year=12)
    assert pytest.approx(val, abs=1e-3) == c / abs(dd)


def test_compute_trade_metrics():
    df = pd.DataFrame([
        {"net_pnl": 100.0, "net_roi": 0.50, "fees_paid": 10.0, "holding_months": 12.0},
        {"net_pnl": 200.0, "net_roi": 1.00, "fees_paid": 15.0, "holding_months": 24.0},
        {"net_pnl": -50.0, "net_roi": -0.20, "fees_paid": 5.0, "holding_months": 6.0},
    ])
    metrics = compute_trade_metrics(df)
    assert metrics["total_trades"] == 3
    assert pytest.approx(metrics["win_rate"], abs=1e-3) == 2 / 3
    assert pytest.approx(metrics["profit_factor"], abs=1e-3) == 300.0 / 50.0  # 6.0
    assert metrics["total_net_pnl"] == 250.0
    assert metrics["total_fees_paid"] == 30.0
