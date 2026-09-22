"""
tests/test_bootstrap.py — Unit test per il block bootstrap.
"""

import numpy as np
import pandas as pd
import pytest

from poke_quant.validation.bootstrap import block_bootstrap_metrics


def test_bootstrap_on_positive_drift_series_mostly_positive():
    idx = pd.date_range("2021-01-01", periods=36, freq="MS")
    returns = pd.Series(np.full(36, 0.02), index=idx)  # +2%/mese costante
    sims = block_bootstrap_metrics(returns, n_sims=200, block_size=6, seed=1)
    assert (sims["cagr"] > 0).mean() > 0.95
    assert sims["cagr"].shape == (200,)


def test_bootstrap_too_short_series_raises():
    idx = pd.date_range("2021-01-01", periods=4, freq="MS")
    returns = pd.Series([0.01, 0.02, -0.01, 0.03], index=idx)
    with pytest.raises(ValueError):
        block_bootstrap_metrics(returns, block_size=6)


def test_bootstrap_zero_variance_series_gives_zero_sharpe_variance():
    idx = pd.date_range("2021-01-01", periods=24, freq="MS")
    returns = pd.Series(np.full(24, 0.01), index=idx)
    sims = block_bootstrap_metrics(returns, n_sims=50, block_size=4, seed=2)
    assert np.allclose(sims["sharpe"], sims["sharpe"][0], atol=1e-6)
