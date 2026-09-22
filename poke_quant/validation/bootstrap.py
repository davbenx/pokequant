"""
poke_quant/validation/bootstrap.py — Block bootstrap sui rendimenti mensili per
stimare un intervallo di confidenza su CAGR/Sharpe, invece di fidarsi di un
singolo numero puntuale stimato su 69 mesi.
"""

from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd


def block_bootstrap_metrics(
    monthly_returns: pd.Series,
    n_sims: int = 500,
    block_size: int = 6,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Ricampiona a blocchi (per preservare l'autocorrelazione mensile) i rendimenti
    osservati e ricalcola CAGR/Sharpe su ciascun ricampionamento. Ritorna gli array
    completi di simulazioni, non solo un riassunto, così il chiamante può scegliere
    i percentili che vuole riportare."""
    returns = monthly_returns.dropna().values
    n = len(returns)
    if n < block_size * 2:
        raise ValueError(f"serie troppo corta ({n} mesi) per block bootstrap con block_size={block_size}")

    rng = np.random.default_rng(seed)
    n_blocks_needed = int(np.ceil(n / block_size))

    cagrs, sharpes = [], []
    for _ in range(n_sims):
        start_idxs = rng.integers(0, n - block_size + 1, size=n_blocks_needed)
        sample = np.concatenate([returns[s:s + block_size] for s in start_idxs])[:n]
        nav = np.cumprod(1 + sample)
        years = n / 12.0
        cagr = nav[-1] ** (1 / years) - 1 if nav[-1] > 0 else -1.0
        sd = sample.std(ddof=1)
        sharpe = (sample.mean() / sd * np.sqrt(12)) if sd > 1e-12 else 0.0
        cagrs.append(cagr)
        sharpes.append(sharpe)

    return {"cagr": np.array(cagrs), "sharpe": np.array(sharpes)}


def summarize_bootstrap(sims: Dict[str, np.ndarray]) -> str:
    lines = []
    for metric, arr in sims.items():
        p5, p50, p95 = np.percentile(arr, [5, 50, 95])
        pct_positive = (arr > 0).mean() * 100.0
        lines.append(f"{metric}: mediana={p50:.3f} [5%={p5:.3f}, 95%={p95:.3f}] | P(>0)={pct_positive:.0f}%")
    return "\n".join(lines)
