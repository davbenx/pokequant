"""
poke_quant/validation/statistical_validation.py — Validazione statistica istituzionale anti-overfitting.
Implementa Deflated Sharpe Ratio (DSR) e Probability of Backtest Overfitting (PBO via CSCV).
Adattato da ApexConvex e basato sui lavori di Bailey, Borwein, Lopez de Prado & Zhu.
"""

from __future__ import annotations
import itertools
import math
from typing import List, Tuple
import numpy as np


def _norm_cdf(x: float) -> float:
    """CDF della normale standard via math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inversa della CDF normale standard (Acklam rational approximation)."""
    if p <= 0.0:
        return -np.inf
    if p >= 1.0:
        return np.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


EULER_MASCHERONI = 0.5772156649015329


def deflated_sharpe_ratio(
    observed_sr: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """
    Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).
    Restituisce la probabilità che lo Sharpe osservato sia genuinamente positivo,
    corretta per il numero di varianti/strategie testate (n_trials).
    """
    if n_obs < 2:
        raise ValueError("servono almeno 2 osservazioni per stimare la varianza dello Sharpe")

    sr_var = (1.0 - skew * observed_sr + ((kurtosis - 1.0) / 4.0) * observed_sr**2) / (n_obs - 1)
    sr_std = math.sqrt(max(sr_var, 1e-12))

    if n_trials <= 1:
        expected_max_sr_null = 0.0
    else:
        z1 = _norm_ppf(1.0 - 1.0 / n_trials)
        z2 = _norm_ppf(1.0 - 1.0 / (n_trials * math.e))
        expected_max_sr_null = sr_std * ((1 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)

    dsr = _norm_cdf((observed_sr - expected_max_sr_null) / sr_std)
    return float(dsr)


def pbo_cscv(performance_matrix: np.ndarray, n_splits: int = 8) -> float:
    """
    Probability of Backtest Overfitting via Combinatorially Symmetric
    Cross-Validation (Bailey, Borwein, Lopez de Prado, Zhu 2015).
    """
    T, N = performance_matrix.shape
    if N < 2:
        raise ValueError("servono almeno 2 varianti da confrontare per calcolare il PBO")
    if n_splits < 2 or n_splits % 2 != 0:
        raise ValueError("n_splits deve essere un numero pari >= 2")
    if T < n_splits:
        raise ValueError(f"T={T} periodi insufficienti per n_splits={n_splits}")

    # Se T non è un multiplo esatto di n_splits, scartiamo i residui iniziali
    rem = T % n_splits
    if rem > 0:
        performance_matrix = performance_matrix[rem:, :]
        T = performance_matrix.shape[0]

    block_size = T // n_splits
    blocks = [performance_matrix[i*block_size:(i+1)*block_size, :] for i in range(n_splits)]

    def sharpe(returns_2d: np.ndarray) -> np.ndarray:
        mu = returns_2d.mean(axis=0)
        sd = returns_2d.std(axis=0, ddof=1)
        sd = np.where(sd < 1e-12, 1e-12, sd)
        return mu / sd

    half = n_splits // 2
    below_median_count = 0
    total_combos = 0

    for is_idx in itertools.combinations(range(n_splits), half):
        oos_idx = [i for i in range(n_splits) if i not in is_idx]
        is_data = np.vstack([blocks[i] for i in is_idx])
        oos_data = np.vstack([blocks[i] for i in oos_idx])

        is_sharpe = sharpe(is_data)
        oos_sharpe = sharpe(oos_data)

        winner = int(np.argmax(is_sharpe))
        rank = float(np.mean(oos_sharpe <= oos_sharpe[winner]))
        if rank < 0.5:
            below_median_count += 1
        total_combos += 1

    return float(below_median_count / total_combos)
