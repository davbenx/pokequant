#!/usr/bin/env python3
"""
scripts/sealed_time_stop_search.py — Testa se un TIME STOP (vendita forzata dopo
N mesi di possesso, indipendente dal segnale di momentum) migliora la strategia
sealed in produzione. Motivazione per testarlo: la distribuzione dei tempi di
possesso in produzione e' fortemente asimmetrica (24 trade chiusi: mediana 4
mesi, ma 10/24 durano esattamente 1 mese e la coda arriva a 29/29/35/45 mesi) -
un'ipotesi ragionevole e' che rotare piu' spesso il capitale sui trade brevi
migliori l'efficienza. Stesso standard di rigore delle altre ricerche su questa
strategia (griglia -> PBO/CSCV -> DSR corretto per tutti i candidati -> block
bootstrap -> walk-forward H1/H2), entrata fissa a lookback=12m (validata).

ESITO: NON MIGLIORA, e per il motivo temuto in anticipo. Su 36 sealed (era
2019+): BASELINE Sharpe 1.10/CAGR +23.5%/MaxDD -13.4%/24 trade resta il
migliore della griglia (PBO 1.4% su 7 candidati - ranking molto stabile, non
fortuna). Ogni time stop provato (6/9/12/18/24/36 mesi) peggiora sia Sharpe
sia CAGR, monotonicamente meno quanto piu' lungo e' lo stop (6m: Sharpe 0.48,
MaxDD -23.9%, 102 trade forzati; 36m: Sharpe 1.07, il piu' vicino alla
baseline ma ancora sotto). Il meccanismo e' esattamente quello temuto: la
distribuzione di possesso in produzione e' asimmetrica (mediana 4 mesi, ma
10/24 trade durano 1 mese e la coda arriva a 29-45 mesi) - un time stop
tronca proprio quella coda destra, i pochi trade lunghi che nel
trend-following generano la maggior parte del rendimento composto, e in
piu' forza rotazioni premature che pagano frizioni (fee+spedizione) senza
necessita'. Nessuna modifica alla logica di uscita in produzione.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.engine.strategies.time_series_momentum import TimeSeriesMomentumStrategy
from poke_quant.validation.statistical_validation import deflated_sharpe_ratio, pbo_cscv
from poke_quant.validation.bootstrap import block_bootstrap_metrics, summarize_bootstrap
from scripts.optimize_and_falsify import run_bt, MODERN_ERA_CUTOFF


def main():
    metadata = load_metadata()
    prices_full = load_price_matrix()
    sealed_ids = [
        k for k, v in metadata.items()
        if v.get("type") == "sealed" and v.get("data_quality") != "thin_unreliable"
        and k in prices_full.columns and v.get("release_date") and v["release_date"] >= MODERN_ERA_CUTOFF
    ]
    meta_sub = {k: v for k, v in metadata.items() if k in sealed_ids}
    prices_sub = prices_full[sealed_ids]
    print(f"Universo: {len(sealed_ids)} sealed (era moderna 2019+)")

    baseline_factory = lambda p: TimeSeriesMomentumStrategy(p, lookback_months=12)
    candidate_factories = {"BASELINE (nessun time stop, in produzione)": baseline_factory}
    for ts in [6, 9, 12, 18, 24, 36]:
        candidate_factories[f"time_stop={ts}m"] = (
            lambda p, t=ts: TimeSeriesMomentumStrategy(p, lookback_months=12, max_holding_months=t)
        )

    results = {}
    print("\nEntrata fissa a lookback=12m (validata) - solo il time stop varia:")
    for name, factory in candidate_factories.items():
        res = run_bt(factory(prices_sub), prices_sub, meta_sub)
        results[name] = res
        print(f"  {name:45s} | CAGR {res.cagr*100:+6.2f}% | Sharpe {res.sharpe:5.2f} | "
              f"MaxDD {res.max_drawdown*100:6.2f}% | Trade {res.total_trades:3d}")

    common_idx = None
    for res in results.values():
        common_idx = res.monthly_returns.index if common_idx is None else common_idx.intersection(res.monthly_returns.index)
    perf_matrix = np.column_stack([results[k].monthly_returns.loc[common_idx].values for k in candidate_factories])
    t_len = len(perf_matrix)
    splits = 8 if t_len >= 32 else 4
    rem = t_len % splits
    pbo = pbo_cscv(perf_matrix[rem:, :] if rem else perf_matrix, n_splits=splits)
    print(f"\nPBO sulla griglia completa ({len(candidate_factories)} candidati, {splits} split): {pbo:.3f} ({pbo*100:.1f}%)")

    baseline_sharpe = results["BASELINE (nessun time stop, in produzione)"].sharpe
    best_name = max(results, key=lambda k: results[k].sharpe)
    best = results[best_name]
    print(f"\nBaseline Sharpe: {baseline_sharpe:.2f} | Migliore della griglia: '{best_name}' (Sharpe {best.sharpe:.2f})")

    dsr = deflated_sharpe_ratio(observed_sr=best.sharpe / np.sqrt(12), n_trials=len(candidate_factories), n_obs=len(best.monthly_returns))
    print(f"DSR del vincitore (n_trials={len(candidate_factories)}): {dsr:.3f}")

    if best_name == "BASELINE (nessun time stop, in produzione)":
        print("\nLa baseline resta la migliore - nessuna modifica alla logica di uscita in produzione.")
        sims = block_bootstrap_metrics(best.monthly_returns, n_sims=500, block_size=6)
        print(summarize_bootstrap(sims))
        return

    sims = block_bootstrap_metrics(best.monthly_returns, n_sims=500, block_size=6)
    print(f"\nBlock bootstrap (500 sim, blocchi 6m) sul vincitore:")
    print(summarize_bootstrap(sims))

    print(f"\nWalk-forward H1/H2 sul vincitore '{best_name}' vs baseline:")
    mid = len(prices_sub) // 2
    h1_dates, h2_dates = prices_sub.index[:mid], prices_sub.index[mid:]
    for label_strat, factory in [("BASELINE", baseline_factory), (best_name, candidate_factories[best_name])]:
        for label, dates in [("H1", h1_dates), ("H2", h2_dates)]:
            sub_prices = prices_sub.loc[dates]
            res = run_bt(factory(sub_prices), sub_prices, meta_sub)
            print(f"  {label_strat:45s} {label} ({dates[0].strftime('%Y-%m')}->{dates[-1].strftime('%Y-%m')}) | "
                  f"Sharpe {res.sharpe:5.2f} | MaxDD {res.max_drawdown*100:6.2f}%")


if __name__ == "__main__":
    main()
