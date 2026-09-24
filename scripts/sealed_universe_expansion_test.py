#!/usr/bin/env python3
"""
scripts/sealed_universe_expansion_test.py — Testa se allargare l'universo
sealed oltre il taglio per anno (MODERN_ERA_CUTOFF=2019) con un criterio di
liquidita' basato sul rapporto prezzo/MSRP REALE (poke_quant/data/liquidity_filter.py::is_liquid_sealed)
migliora o comunque non peggiora la strategia in produzione.

Motivazione: MODERN_ERA_CUTOFF e' un taglio per anno, non per liquidita' -
alcuni set 2016-2018 (Crimson Invasion, Burning Shadows, Ultra Prism, Lost
Thunder) hanno un rapporto prezzo-attuale/MSRP dentro lo stesso range gia'
osservato nell'universo moderno validato (1,5x-21,6x), quindi si sono
apprezzati come un box "normale", non come un pezzo da museo a scambio quasi
nullo (es. Team Rocket Returns a 485x). La soglia (21,6x, il MAX osservato
nell'universo moderno) e' stata fissata GUARDANDO SOLO l'universo moderno
esistente, PRIMA di controllare l'effetto sul backtest - altrimenti sarebbe
overfitting della definizione di universo, non delle sue regole. Nessun MSRP
e' stato inventato per i set senza questo dato (restano esclusi, stessa regola
di scripts/discover_sealed_universe.py).

Stesso standard di rigore delle altre ricerche su questa strategia (DSR
corretto per tutti i trial sui sealed, block bootstrap, walk-forward H1/H2),
NESSUN parametro della strategia cambiato - solo l'universo di asset eleggibili.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.data.liquidity_filter import liquid_sealed_ids
from poke_quant.engine.strategies.time_series_momentum import TimeSeriesMomentumStrategy
from poke_quant.validation.statistical_validation import deflated_sharpe_ratio, pbo_cscv
from poke_quant.validation.bootstrap import block_bootstrap_metrics, summarize_bootstrap
from scripts.optimize_and_falsify import run_bt, MODERN_ERA_CUTOFF

# Trial gia' tentati sui sealed in questa sessione (32 dell'audit originale +
# 7 di sealed_momentum_confirmation_search.py + 7 di sealed_hysteresis_search.py,
# entrambi respinti prima di questo) + questo (1 solo candidato: non e' una
# griglia di parametri, e' un cambio di universo con un criterio fissato a
# priori, ma resta un trial nel senso della DSR - va comunque contato).
PRIOR_SEALED_TRIALS = 32 + 7 + 7


def main():
    metadata = load_metadata()
    prices_full = load_price_matrix()

    old_ids = [
        k for k, v in metadata.items()
        if v.get("type") == "sealed" and v.get("data_quality") != "thin_unreliable"
        and k in prices_full.columns and v.get("release_date") and v["release_date"] >= MODERN_ERA_CUTOFF
    ]
    new_ids = liquid_sealed_ids(metadata, prices_full)
    added = sorted(set(new_ids) - set(old_ids))
    print(f"Universo attuale (solo cutoff anno): {len(old_ids)}")
    print(f"Universo allargato (cutoff anno + MSRP reale dentro range 1,5x-21,6x): {len(new_ids)} (+{len(added)}: {added})")

    meta_old = {k: metadata[k] for k in old_ids}
    meta_new = {k: metadata[k] for k in new_ids}
    prices_old, prices_new = prices_full[old_ids], prices_full[new_ids]

    baseline = TimeSeriesMomentumStrategy(prices_old, lookback_months=12)
    expanded = TimeSeriesMomentumStrategy(prices_new, lookback_months=12)
    res_old = run_bt(baseline, prices_old, meta_old)
    res_new = run_bt(expanded, prices_new, meta_new)

    print(f"\n{'Universo':45s} {'CAGR':>8s} {'Sharpe':>7s} {'MaxDD':>8s} {'Trade':>6s}")
    print(f"{'ATTUALE (36, cutoff anno)':45s} {res_old.cagr*100:+7.2f}% {res_old.sharpe:7.2f} {res_old.max_drawdown*100:7.2f}% {res_old.total_trades:6d}")
    print(f"{'ALLARGATO (40, + MSRP reale)':45s} {res_new.cagr*100:+7.2f}% {res_new.sharpe:7.2f} {res_new.max_drawdown*100:7.2f}% {res_new.total_trades:6d}")

    common_idx = res_old.monthly_returns.index.intersection(res_new.monthly_returns.index)
    perf_matrix = np.column_stack([res_old.monthly_returns.loc[common_idx].values, res_new.monthly_returns.loc[common_idx].values])
    t_len = len(perf_matrix)
    splits = 8 if t_len >= 32 else 4
    rem = t_len % splits
    pbo = pbo_cscv(perf_matrix[rem:, :] if rem else perf_matrix, n_splits=splits)
    print(f"\nPBO (2 candidati, {splits} split): {pbo:.3f}")

    total_trials = PRIOR_SEALED_TRIALS + 1
    dsr_full = deflated_sharpe_ratio(observed_sr=res_new.sharpe / np.sqrt(12), n_trials=total_trials, n_obs=len(res_new.monthly_returns))
    print(f"DSR universo allargato (intera ricerca sui sealed, n_trials={total_trials}): {dsr_full:.3f}")

    sims = block_bootstrap_metrics(res_new.monthly_returns, n_sims=500, block_size=6)
    print(f"\nBlock bootstrap (500 sim, blocchi 6m) sull'universo allargato:")
    print(summarize_bootstrap(sims))

    print(f"\nWalk-forward H1/H2, allargato vs attuale:")
    for label_strat, prices_sub, meta_sub in [("ATTUALE (36)", prices_old, meta_old), ("ALLARGATO (40)", prices_new, meta_new)]:
        mid = len(prices_sub) // 2
        for label, dates in [("H1", prices_sub.index[:mid]), ("H2", prices_sub.index[mid:])]:
            sub = prices_sub.loc[dates]
            strat = TimeSeriesMomentumStrategy(sub, lookback_months=12)
            r = run_bt(strat, sub, meta_sub)
            print(f"  {label_strat:20s} {label} ({dates[0].strftime('%Y-%m')}->{dates[-1].strftime('%Y-%m')}) | "
                  f"Sharpe {r.sharpe:5.2f} | CAGR {r.cagr*100:+6.2f}% | Trade {r.total_trades:3d}")


if __name__ == "__main__":
    main()
