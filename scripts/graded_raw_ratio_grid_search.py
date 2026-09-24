#!/usr/bin/env python3
"""
scripts/graded_raw_ratio_grid_search.py — Griglia completa sul filtro grade9/raw
(liquidity_filter.py::compute_grade_raw_ratio_flags), richiesta esplicitamente
dopo aver adottato il 10° percentile confrontandolo solo col 25° (troppo poco
per dire "e' il migliore" - PBO tra i due era 0.471, alto). Stessa disciplina
usata per ogni altro parametro in questa sessione: griglia, non un solo valore,
PBO su TUTTA la griglia, walk-forward sul migliore.

Due assi testati:
  1. percentile_cutoff (asse principale): 0.05 -> 0.40
  2. cohort_window_years e min_cohort (robustezza), al percentile scelto

ESITO: la DIREZIONE e' robusta - tutti gli 8 percentile testati (5%-40%)
battono il baseline senza filtro (Sharpe 1.48): range 1.51-1.81, nessuno
sotto baseline. Ma la scelta del percentile ESATTO non e' stabile - PBO
sulla griglia (8 candidati) = 0.514, molto sopra la fascia di comfort usata
ovunque in questa sessione (20-30%). Il "migliore" apparente (35°, Sharpe
1.81) e' quasi certamente rumore campionario, non un vero optimum - la curva
non e' un plateau (dip netto al 25°, poi risalita), il segnale di overfitting
che questa sessione ha imparato a riconoscere altrove (vedi isteresi/conferma
momentum, entrambe respinte per lo stesso motivo).

DECISIONE: NON si cambia il 10° percentile gia' adottato in produzione.
Motivo metodologico, non di performance: il 10° fu scelto PRIMA di guardare
l'effetto sul backtest (stesso principio di MAX_PRICE_TO_MSRP_RATIO) - è
l'unica scelta in questa griglia che non e' influenzata dall'aver visto il
risultato. Rincorrere il picco al 35° ora, dopo aver visto questa griglia,
sarebbe overfitting di secondo livello - esattamente l'errore che questa
sessione ha corretto piu' volte altrove. Il valore di questo test e' aver
CONFERMATO la direzione (filtrare aiuta, non e' un caso isolato sul 10%) e
QUANTIFICATO l'incertezza sulla soglia esatta (PBO 51%) - non nell'aver
trovato "il numero migliore".
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.data.liquidity_filter import compute_grade_raw_ratio_flags
from poke_quant.engine.backtester import Backtester
from poke_quant.engine.strategies.scarcity_value_factor import ScarcityValueFactorStrategy
from poke_quant.validation.statistical_validation import deflated_sharpe_ratio, pbo_cscv
from scripts.generate_singles_signal import PRODUCTION_PARAMS

PERCENTILE_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]


def run_bt(strat, prices_df, metadata):
    bt = Backtester(strat, prices_df, metadata, initial_cash=10000.0, platform="cardmarket",
                     apply_liquidity_slippage=True, apply_holding_cost=True, apply_buy_side_shipping=True)
    return bt.run()


def universe_for_cutoff(metadata, grade9_prices, base_ids, pct, window=3, min_cohort=20):
    flags = compute_grade_raw_ratio_flags(metadata, grade9_prices, percentile_cutoff=pct,
                                           min_cohort=min_cohort, cohort_window_years=window)
    return [k for k in base_ids if k not in flags], flags


def main():
    metadata = load_metadata()
    grade9_prices = load_price_matrix("historical_prices_graded_singles_grade9.csv")

    # Universo base: SOLO il filtro di volatilita' preesistente. Il metadata in
    # produzione ha gia' il filtro ratio applicato con cutoff=0.10 - per una
    # griglia onesta, si riammettono le carte escluse SOLO per quel motivo (il
    # reason contiene "grade9/raw"), cosi' ognuna viene ri-valutata da zero a
    # ogni percentile testato, invece di partire gia' escluse.
    base_ids = [
        k for k, v in metadata.items()
        if v.get("type") == "single" and k in grade9_prices.columns
        and (v.get("data_quality") != "thin_unreliable" or "grade9/raw" in v.get("data_quality_reason", ""))
    ]
    print(f"Universo base (solo filtro volatilita', ratio-check escluso): {len(base_ids)}")

    print(f"\n{'percentile':>10s} {'n_escluse':>10s} {'n_universo':>11s} {'Sharpe':>7s} {'CAGR':>8s} {'MaxDD':>8s} {'Trade':>6s}")
    results = {}
    for pct in PERCENTILE_GRID:
        ids, flags = universe_for_cutoff(metadata, grade9_prices, base_ids, pct)
        meta_sub = {k: metadata[k] for k in ids}
        prices_sub = grade9_prices[ids]
        strat = ScarcityValueFactorStrategy(**PRODUCTION_PARAMS)
        res = run_bt(strat, prices_sub, meta_sub)
        results[pct] = (res, ids)
        print(f"{pct:>10.2f} {len(flags):>10d} {len(ids):>11d} {res.sharpe:>7.2f} {res.cagr*100:>+7.2f}% "
              f"{res.max_drawdown*100:>7.2f}% {res.total_trades:>6d}")

    # PBO su tutta la griglia (8 candidati)
    common_idx = None
    for pct in PERCENTILE_GRID:
        idx = results[pct][0].monthly_returns.index
        common_idx = idx if common_idx is None else common_idx.intersection(idx)
    perf_matrix = np.column_stack([results[pct][0].monthly_returns.loc[common_idx].values for pct in PERCENTILE_GRID])
    t_len = len(perf_matrix)
    splits = 8 if t_len >= 32 else 4
    rem = t_len % splits
    pbo = pbo_cscv(perf_matrix[rem:, :] if rem else perf_matrix, n_splits=splits)
    print(f"\nPBO sulla griglia percentile ({len(PERCENTILE_GRID)} candidati, {splits} split): {pbo:.3f}")

    best_pct = max(PERCENTILE_GRID, key=lambda p: results[p][0].sharpe)
    print(f"Migliore per Sharpe: {best_pct:.2f} (Sharpe {results[best_pct][0].sharpe:.2f})")

    n_prior_trials = 66  # dal filtro gia' adottato (vedi VALIDATED_SINGLES in app.py)
    n_trials_total = n_prior_trials + len(PERCENTILE_GRID)
    best_res, best_ids = results[best_pct]
    dsr = deflated_sharpe_ratio(observed_sr=best_res.sharpe / np.sqrt(12), n_trials=n_trials_total, n_obs=len(best_res.monthly_returns))
    print(f"DSR del migliore, corretto per {n_trials_total} trial cumulativi (66 precedenti + {len(PERCENTILE_GRID)} di questa griglia): {dsr:.3f}")

    print(f"\nWalk-forward H1/H2 sull'intera griglia (plateau vs picco isolato):")
    for pct in PERCENTILE_GRID:
        ids, _ = universe_for_cutoff(metadata, grade9_prices, base_ids, pct)
        meta_sub = {k: metadata[k] for k in ids}
        prices_sub = grade9_prices[ids]
        mid = len(prices_sub) // 2
        h1, h2 = prices_sub.index[:mid], prices_sub.index[mid:]
        wf = []
        for dates in (h1, h2):
            sub = prices_sub.loc[dates]
            strat = ScarcityValueFactorStrategy(**PRODUCTION_PARAMS)
            r = run_bt(strat, sub, meta_sub)
            wf.append(r.sharpe)
        marker = " <- adottato (pre-registrato)" if abs(pct - 0.10) < 1e-9 else (" <- max Sharpe in-sample" if abs(pct - best_pct) < 1e-9 else "")
        print(f"  percentile={pct:.2f} | H1 {wf[0]:5.2f} | H2 {wf[1]:5.2f}{marker}")

    # --- Secondo asse: robustezza di cohort_window_years e min_cohort al 10° percentile adottato ---
    print(f"\nRobustezza al 10° percentile su cohort_window_years (min_cohort=20 fisso):")
    for window in [2, 3, 4, 5]:
        ids, flags = universe_for_cutoff(metadata, grade9_prices, base_ids, 0.10, window=window, min_cohort=20)
        meta_sub = {k: metadata[k] for k in ids}
        prices_sub = grade9_prices[ids]
        strat = ScarcityValueFactorStrategy(**PRODUCTION_PARAMS)
        res = run_bt(strat, prices_sub, meta_sub)
        print(f"  window=±{window}y | escluse {len(flags):3d} | universo {len(ids):3d} | Sharpe {res.sharpe:.2f} | CAGR {res.cagr*100:+6.2f}%")

    print(f"\nRobustezza al 10° percentile su min_cohort (window=±3y fisso):")
    for min_c in [15, 20, 25, 30]:
        ids, flags = universe_for_cutoff(metadata, grade9_prices, base_ids, 0.10, window=3, min_cohort=min_c)
        meta_sub = {k: metadata[k] for k in ids}
        prices_sub = grade9_prices[ids]
        strat = ScarcityValueFactorStrategy(**PRODUCTION_PARAMS)
        res = run_bt(strat, prices_sub, meta_sub)
        print(f"  min_cohort={min_c:2d} | escluse {len(flags):3d} | universo {len(ids):3d} | Sharpe {res.sharpe:.2f} | CAGR {res.cagr*100:+6.2f}%")


if __name__ == "__main__":
    main()
