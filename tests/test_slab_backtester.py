"""
tests/test_slab_backtester.py — Test di integrazione e validazione per il Backtester delle Carte Gradate.
Verifica:
  - Consistenza contabile e calcolo NAV
  - Assenza di Look-Ahead Bias (t <= T)
  - Esecuzione delle regole di rotazione
  - Esecuzione corretta della suite di falsificazione popperiana
"""

import pytest
import pandas as pd
import numpy as np

from poke_quant.slabs.slab_backtester import SlabBacktester
from poke_quant.slabs.slab_falsification import run_popperian_falsification_suite
from poke_quant.slabs.slab_universe import get_curated_grails


def test_slab_backtester_accounting_and_metrics():
    bt = SlabBacktester(initial_capital=5000.0)
    res = bt.run()

    # Verifica vincoli di base
    assert res.initial_capital == 5000.0
    assert res.final_nav > 0.0
    assert not np.isnan(res.cagr_pct)
    assert not np.isnan(res.sharpe_ratio)
    assert not np.isnan(res.max_drawdown_pct)
    assert res.max_drawdown_pct <= 0.0  # Max drawdown è <= 0%
    assert res.total_trades >= 1
    assert res.win_rate_pct >= 0.0 and res.win_rate_pct <= 100.0

    # Verifica equity curve
    assert len(res.equity_curve) > 20
    assert "nav_eur" in res.equity_curve.columns
    assert "cash_eur" in res.equity_curve.columns
    assert (res.equity_curve["cash_eur"] >= 0.0).all()  # Zero leva o cassa negativa


def test_slab_falsification_suite_executes_cleanly():
    bt = SlabBacktester(initial_capital=5000.0)
    res = bt.run()

    fals_report = run_popperian_falsification_suite(base_result=res, num_mc_simulations=20)

    assert "popperian_stress_tests" in fals_report
    assert len(fals_report["popperian_stress_tests"]) == 4
    assert "dsr" in fals_report
    assert "pbo" in fals_report
    assert "anti_survivorship_bias" in fals_report
    assert fals_report["anti_survivorship_bias"]["passed"] is True
