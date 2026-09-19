"""
tests/test_statistical_validation.py — Unit test per DSR e PBO.
"""

import numpy as np
import pytest
from poke_quant.validation.statistical_validation import deflated_sharpe_ratio, pbo_cscv


def test_dsr_single_trial_vs_multiple():
    # Per uno Sharpe di 1.5 con 60 mesi di storico:
    # Con 1 solo trial la confidenza deve essere molto alta (>0.90)
    # Con 50 trial indipendenti la confidenza deve scendere (deflated)
    dsr_1 = deflated_sharpe_ratio(observed_sr=1.5, n_trials=1, n_obs=60)
    dsr_50 = deflated_sharpe_ratio(observed_sr=1.5, n_trials=50, n_obs=60)
    assert dsr_1 > 0.95
    assert dsr_50 < dsr_1


def test_pbo_cscv_random_noise():
    # Se le strategie sono puro rumore bianco (media 0, dev std 1),
    # il PBO deve oscillare attorno al 50% (scelta in-sample non predittiva OOS)
    rng = np.random.default_rng(42)
    # T=64 mesi (divisibile per 8), N=10 varianti casuali
    noise_matrix = rng.normal(0.0, 0.05, size=(64, 10))
    pbo = pbo_cscv(noise_matrix, n_splits=8)
    assert 0.25 <= pbo <= 0.75
