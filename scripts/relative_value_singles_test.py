#!/usr/bin/env python3
"""
scripts/relative_value_singles_test.py — Testa RelativeValueFactorStrategy
(regressione cross-sezionale log-prezzo ~ rarita'+franchise+lingua+eta', compra
il quantile col residuo piu' negativo) sull'universo pieno chase+controllo
(864 singole, nessuna pre-selezione sull'esito). Stesso standard di rigore di
ogni altro test in questa ricerca: griglia -> PBO -> DSR -> bootstrap ->
walk-forward H1/H2.

ESITO: il piu' promettente di tutta la ricerca sulle singole - vedi il
docstring completo in poke_quant/engine/strategies/relative_value_factor.py
per i numeri e la verifica decisiva (l'effetto tiene anche dentro il solo
campione di controllo casuale, non contaminato da survivorship bias). Non
ancora da produzione - serve un altro ciclo di validazione dedicato.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.engine.strategies.relative_value_factor import RelativeValueFactorStrategy
from poke_quant.validation.statistical_validation import deflated_sharpe_ratio, pbo_cscv
from poke_quant.validation.bootstrap import block_bootstrap_metrics, summarize_bootstrap
from scripts.optimize_and_falsify import run_bt


def main():
    metadata = load_metadata()
    prices_full = load_price_matrix("historical_prices_graded_singles_grade9.csv")
    singles_ids = [
        k for k, v in metadata.items()
        if v.get("type") == "single" and v.get("data_quality") != "thin_unreliable" and k in prices_full.columns
    ]
    meta_sub = {k: metadata[k] for k in singles_ids}
    prices_sub = prices_full[singles_ids]
    print(f"Universo: {len(singles_ids)} singole (chase + controllo, nessuna pre-selezione)")

    configs = {
        "VALUE rebal=6 q=0.20": dict(rebalance_every_months=6, top_quantile=0.20, min_age_months=6),
        "VALUE rebal=3 q=0.20": dict(rebalance_every_months=3, top_quantile=0.20, min_age_months=6),
        "VALUE rebal=6 q=0.10": dict(rebalance_every_months=6, top_quantile=0.10, min_age_months=6),
        "VALUE rebal=6 q=0.30": dict(rebalance_every_months=6, top_quantile=0.30, min_age_months=6),
        "VALUE rebal=12 q=0.20": dict(rebalance_every_months=12, top_quantile=0.20, min_age_months=6),
    }
    results = {}
    for name, kw in configs.items():
        strat = RelativeValueFactorStrategy(**kw)
        res = run_bt(strat, prices_sub, meta_sub)
        results[name] = res
        print(f"  {name:24s} | CAGR {res.cagr*100:+6.2f}% | Sharpe {res.sharpe:5.2f} | "
              f"MaxDD {res.max_drawdown*100:6.2f}% | Trade {res.total_trades:4d}")

    common_idx = None
    for res in results.values():
        common_idx = res.monthly_returns.index if common_idx is None else common_idx.intersection(res.monthly_returns.index)
    perf_matrix = np.column_stack([results[k].monthly_returns.loc[common_idx].values for k in configs])
    t_len = len(perf_matrix)
    splits = 8 if t_len >= 32 else 4
    rem = t_len % splits
    pbo = pbo_cscv(perf_matrix[rem:, :] if rem else perf_matrix, n_splits=splits)
    print(f"\nPBO ({len(configs)} candidati, {splits} split): {pbo:.3f}")

    best_name = max(results, key=lambda k: results[k].sharpe)
    best = results[best_name]
    dsr = deflated_sharpe_ratio(observed_sr=best.sharpe / np.sqrt(12), n_trials=len(configs), n_obs=len(best.monthly_returns))
    print(f"Migliore: '{best_name}' Sharpe {best.sharpe:.2f} | DSR: {dsr:.3f}")

    sims = block_bootstrap_metrics(best.monthly_returns, n_sims=500, block_size=6)
    print(summarize_bootstrap(sims))

    mid = len(prices_sub) // 2
    h1_dates, h2_dates = prices_sub.index[:mid], prices_sub.index[mid:]
    print(f"\nWalk-forward H1/H2 sul vincitore '{best_name}':")
    for label, dates in [("H1", h1_dates), ("H2", h2_dates)]:
        sub = prices_sub.loc[dates]
        strat = RelativeValueFactorStrategy(**configs[best_name])
        res = run_bt(strat, sub, meta_sub)
        print(f"  {label} ({dates[0].strftime('%Y-%m')}->{dates[-1].strftime('%Y-%m')}) | Sharpe {res.sharpe:5.2f} | CAGR {res.cagr*100:+6.2f}%")


if __name__ == "__main__":
    main()
