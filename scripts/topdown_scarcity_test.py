#!/usr/bin/env python3
"""
scripts/topdown_scarcity_test.py — Testa la selezione Top-Down (Mercato -> Set
-> Singola): il fattore scarsita' validato (scarcity_value_factor.py), ma
applicato SOLO alle singole di set il cui box ha momentum positivo oggi,
invece che a tutto l'universo. Confronto diretto contro il fattore scarsita'
non filtrato sullo stesso periodo, stesso standard di rigore.

ESITO: NON VALIDATO - la selezione top-down peggiora, non migliora, rispetto
al fattore scarsita' non filtrato. Migliore candidato ('rebal=3 q=0.20'):
Sharpe 0,63 (contro 1,74 del non filtrato), PBO 38,6% (contro 1,4%), DSR
0,613 sulla griglia propria -> 0,200 corretto per 56 trial totali - fallisce
decisamente. Walk-forward H1 quasi piatto (+0,03), H2 debole (+0,75) contro
il +3,16 del non filtrato. Il meccanismo: restringere alle sole singole di
set con box in momentum riduce la diversificazione (trade 15-53 contro 272)
senza aggiungere selettivita' di qualita' sufficiente a compensare. Dato che
box e fattore scarsita' hanno correlazione bassa (0,19 - vedi conversazione),
il valore sta nel trattarli come DUE POSIZIONI INDIPENDENTI (un blend, che da
solo porta Sharpe 1,28 -> 2,01 con MaxDD -13,4% -> -5,7%), non nel farne
dipendere una dall'altra come filtro a cascata - la nidificazione butta via
esattamente la diversificazione che li rende complementari.
"""

import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.engine.strategies.scarcity_value_factor import ScarcityValueFactorStrategy
from poke_quant.engine.strategies.topdown_scarcity_factor import TopDownScarcityFactorStrategy
from poke_quant.validation.statistical_validation import deflated_sharpe_ratio, pbo_cscv
from poke_quant.validation.bootstrap import block_bootstrap_metrics, summarize_bootstrap
from scripts.optimize_and_falsify import run_bt, MODERN_ERA_CUTOFF

PRIOR_SINGLES_TRIALS = 51  # 46 + i 5 della griglia scarsita' non filtrata


def main():
    metadata = load_metadata()
    prices_singles_full = load_price_matrix("historical_prices_graded_singles_grade9.csv")
    prices_sealed_full = load_price_matrix()
    single_to_box = json.load(open("data_cache/single_to_box_map.json"))

    singles_ids = [
        k for k, v in metadata.items()
        if v.get("type") == "single" and v.get("data_quality") != "thin_unreliable" and k in prices_singles_full.columns
    ]
    meta_sub = {k: metadata[k] for k in singles_ids}
    prices_sub = prices_singles_full[singles_ids]

    sealed_ids = [
        k for k, v in metadata.items()
        if v.get("type") == "sealed" and v.get("data_quality") != "thin_unreliable"
        and k in prices_sealed_full.columns and v.get("release_date") and v["release_date"] >= MODERN_ERA_CUTOFF
    ]
    box_prices = prices_sealed_full[sealed_ids]

    print(f"Universo singole: {len(singles_ids)} | Universo box per il filtro momentum: {len(sealed_ids)}")

    configs = {
        "TOPDOWN rebal=6 q=0.20": dict(rebalance_every_months=6, top_quantile=0.20, min_age_months=6),
        "TOPDOWN rebal=3 q=0.20": dict(rebalance_every_months=3, top_quantile=0.20, min_age_months=6),
        "TOPDOWN rebal=6 q=0.10": dict(rebalance_every_months=6, top_quantile=0.10, min_age_months=6),
        "TOPDOWN rebal=6 q=0.30": dict(rebalance_every_months=6, top_quantile=0.30, min_age_months=6),
        "TOPDOWN rebal=12 q=0.20": dict(rebalance_every_months=12, top_quantile=0.20, min_age_months=6),
    }
    results = {}
    for name, kw in configs.items():
        strat = TopDownScarcityFactorStrategy(box_prices, single_to_box, **kw)
        res = run_bt(strat, prices_sub, meta_sub)
        results[name] = res
        print(f"  {name:26s} | CAGR {res.cagr*100:+6.2f}% | Sharpe {res.sharpe:5.2f} | "
              f"MaxDD {res.max_drawdown*100:6.2f}% | Trade {res.total_trades:4d}")

    # Confronto diretto: fattore scarsita' NON filtrato, stesso periodo/universo
    strat_baseline = ScarcityValueFactorStrategy(rebalance_every_months=3, top_quantile=0.20, min_age_months=6)
    res_baseline = run_bt(strat_baseline, prices_sub, meta_sub)
    print(f"\n  [confronto] SCARSITA' non filtrata rebal=3 q=0.20 | CAGR {res_baseline.cagr*100:+6.2f}% | "
          f"Sharpe {res_baseline.sharpe:5.2f} | MaxDD {res_baseline.max_drawdown*100:6.2f}% | Trade {res_baseline.total_trades:4d}")

    common_idx = None
    for res in results.values():
        common_idx = res.monthly_returns.index if common_idx is None else common_idx.intersection(res.monthly_returns.index)
    perf_matrix = np.column_stack([results[k].monthly_returns.loc[common_idx].values for k in configs])
    t_len = len(perf_matrix)
    splits = 8 if t_len >= 32 else 4
    rem = t_len % splits
    pbo = pbo_cscv(perf_matrix[rem:, :] if rem else perf_matrix, n_splits=splits)
    print(f"\nPBO (5 candidati, {splits} split): {pbo:.3f}")

    best_name = max(results, key=lambda k: results[k].sharpe)
    best = results[best_name]
    total_trials = PRIOR_SINGLES_TRIALS + len(configs)
    dsr_own = deflated_sharpe_ratio(observed_sr=best.sharpe / np.sqrt(12), n_trials=len(configs), n_obs=len(best.monthly_returns))
    dsr_full = deflated_sharpe_ratio(observed_sr=best.sharpe / np.sqrt(12), n_trials=total_trials, n_obs=len(best.monthly_returns))
    print(f"Migliore: '{best_name}' Sharpe {best.sharpe:.2f} | Trade {best.total_trades}")
    print(f"DSR (solo griglia propria, n_trials=5): {dsr_own:.3f}")
    print(f"DSR (intera ricerca sulle singole, n_trials={total_trials}): {dsr_full:.3f}")

    sims = block_bootstrap_metrics(best.monthly_returns, n_sims=500, block_size=6)
    print(summarize_bootstrap(sims))

    mid = len(prices_sub) // 2
    h1_dates, h2_dates = prices_sub.index[:mid], prices_sub.index[mid:]
    print(f"\nWalk-forward H1/H2 sul vincitore '{best_name}':")
    for label, dates in [("H1", h1_dates), ("H2", h2_dates)]:
        sub_singles = prices_sub.loc[dates]
        sub_box = box_prices.loc[box_prices.index.intersection(dates)]
        strat = TopDownScarcityFactorStrategy(sub_box, single_to_box, **configs[best_name])
        res = run_bt(strat, sub_singles, meta_sub)
        print(f"  {label} ({dates[0].strftime('%Y-%m')}->{dates[-1].strftime('%Y-%m')}) | Sharpe {res.sharpe:5.2f} | CAGR {res.cagr*100:+6.2f}% | Trade {res.total_trades}")


if __name__ == "__main__":
    main()
