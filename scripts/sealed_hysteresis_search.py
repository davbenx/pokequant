#!/usr/bin/env python3
"""
scripts/sealed_hysteresis_search.py — Testa exit_threshold (banda di isteresi:
vendi solo se il momentum scende sotto una soglia negativa, non appena tocca
zero) sui box sealed. Diverso da scripts/sealed_time_stop_search.py (limite
temporale) e da scripts/optimize_and_falsify.py::section_sealed_exit_logic_search
(exit_lookback_months piu' corto, trailing_stop_pct) - quella ricerca non aveva
mai testato exit_threshold negativo, il parametro pensato apposta per questo
(vedi docstring di TimeSeriesMomentumStrategy).

Motivazione: un momentum che oscilla appena sopra/sotto zero puo' generare
entrate/uscite ripetute (whipsaw) che pagano frizioni (fee+spedizione) senza
un vero cambio di trend dietro. Una banda di isteresi (es. vendi solo se
mom<=-10%, non mom<=0%) dovrebbe ridurre il turnover senza perdere i trend
reali, SE il rumore attorno allo zero e' davvero rumore.

Stesso standard di rigore delle altre ricerche su questa strategia (griglia ->
PBO/CSCV -> DSR corretto per tutti i candidati -> block bootstrap -> walk-forward
H1/H2), entrata fissa a lookback=12m (validata), NESSUN ALTRO PARAMETRO CAMBIATO.

ESITO: NON VALIDATO, NESSUNA MODIFICA alla produzione (exit_threshold resta 0.0).
Il "vincitore" per Sharpe (exit_threshold=-30%, Sharpe 1,26, DSR full-session
0,747 su 46 trial cumulativi - sopra pure il baseline 0,675) ha lo STESSO
problema gia' trovato e respinto in sealed_momentum_confirmation_search.py:
solo 2 trade totali, 0 nella seconda meta' del periodo (H2 trade=0) - il
"rendimento" e' semplicemente aver tenuto UNA posizione comprata in H1
attraverso tutto il rally 2023-2026, non un effetto di isteresi
generalizzabile. Il PBO qui e' basso (18,6%, diversamente dall'87% del
confirm-months) ma non salva la conclusione: PBO/DSR misurano
sull'autocorrelazione dei rendimenti MENSILI, non "vedono" che quei rendimenti
vengono da 1-2 trade soli.

Anche i candidati piu' moderati (che mantengono un conteggio trade paragonabile
al baseline) non passano: exit_threshold=-15% (13 trade, circa meta' del
baseline) ha Sharpe 1,13 (appena sopra 1,10) ma DSR full-session-corretto
0,643 su 46 trial cumulativi - PEGGIORE del baseline 0,675, perche' il piccolo
guadagno di Sharpe non compensa il costo di essere un trial aggiuntivo. E
anche qui H2 ha 0 trade - stesso singolo mancato-stop del 2023 dietro il
numero, solo diluito. Nessuna banda di isteresi testata in questa griglia
batte la regola simmetrica originale in modo che regga sia il conteggio trade
sia la correzione DSR insieme.
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

# Trial gia' tentati sui sealed in questa sessione (32 dell'audit originale, vedi
# scripts/dsr_session_audit.py, + 7 di scripts/sealed_momentum_confirmation_search.py
# gia' respinto prima di questo) + questa griglia.
PRIOR_SEALED_TRIALS = 32 + 7


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
    candidate_factories = {"BASELINE (exit_threshold=0, in produzione)": baseline_factory}
    for th in [-0.05, -0.10, -0.15, -0.20, -0.25, -0.30]:
        candidate_factories[f"exit_threshold={th*100:.0f}%"] = (
            lambda p, t=th: TimeSeriesMomentumStrategy(p, lookback_months=12, exit_threshold=t)
        )

    results = {}
    print(f"\n{'Candidato':42s} {'CAGR':>8s} {'Sharpe':>7s} {'MaxDD':>8s} {'Trade':>6s}")
    for name, factory in candidate_factories.items():
        res = run_bt(factory(prices_sub), prices_sub, meta_sub)
        results[name] = res
        print(f"{name:42s} {res.cagr*100:+7.2f}% {res.sharpe:7.2f} {res.max_drawdown*100:7.2f}% {res.total_trades:6d}")

    common_idx = None
    for res in results.values():
        common_idx = res.monthly_returns.index if common_idx is None else common_idx.intersection(res.monthly_returns.index)
    perf_matrix = np.column_stack([results[k].monthly_returns.loc[common_idx].values for k in candidate_factories])
    t_len = len(perf_matrix)
    splits = 8 if t_len >= 32 else 4
    rem = t_len % splits
    pbo = pbo_cscv(perf_matrix[rem:, :] if rem else perf_matrix, n_splits=splits)
    print(f"\nPBO sulla griglia completa ({len(candidate_factories)} candidati, {splits} split): {pbo:.3f} ({pbo*100:.1f}%)")

    baseline_sharpe = results["BASELINE (exit_threshold=0, in produzione)"].sharpe
    best_name = max(results, key=lambda k: results[k].sharpe)
    best = results[best_name]
    print(f"\nBaseline Sharpe: {baseline_sharpe:.2f} | Migliore della griglia: '{best_name}' (Sharpe {best.sharpe:.2f})")

    total_trials = PRIOR_SEALED_TRIALS + len(candidate_factories)
    dsr_own = deflated_sharpe_ratio(observed_sr=best.sharpe / np.sqrt(12), n_trials=len(candidate_factories), n_obs=len(best.monthly_returns))
    dsr_full = deflated_sharpe_ratio(observed_sr=best.sharpe / np.sqrt(12), n_trials=total_trials, n_obs=len(best.monthly_returns))
    print(f"DSR vincitore (solo griglia propria, n_trials={len(candidate_factories)}): {dsr_own:.3f}")
    print(f"DSR vincitore (intera ricerca sui sealed, n_trials={total_trials}): {dsr_full:.3f}")

    if best_name == "BASELINE (exit_threshold=0, in produzione)":
        print("\nLa baseline resta la migliore - nessuna modifica alla logica di uscita in produzione.")
        return

    sims = block_bootstrap_metrics(best.monthly_returns, n_sims=500, block_size=6)
    print(f"\nBlock bootstrap (500 sim, blocchi 6m) sul vincitore:")
    print(summarize_bootstrap(sims))

    print(f"\nWalk-forward H1/H2, vincitore vs baseline:")
    mid = len(prices_sub) // 2
    h1_dates, h2_dates = prices_sub.index[:mid], prices_sub.index[mid:]
    for label_strat, factory in [("BASELINE", baseline_factory), (best_name, candidate_factories[best_name])]:
        for label, dates in [("H1", h1_dates), ("H2", h2_dates)]:
            sub_prices = prices_sub.loc[dates]
            res = run_bt(factory(sub_prices), sub_prices, meta_sub)
            print(f"  {label_strat:42s} {label} ({dates[0].strftime('%Y-%m')}->{dates[-1].strftime('%Y-%m')}) | "
                  f"Sharpe {res.sharpe:5.2f} | CAGR {res.cagr*100:+6.2f}% | Trade {res.total_trades:3d}")


if __name__ == "__main__":
    main()
