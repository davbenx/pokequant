#!/usr/bin/env python3
"""
scripts/singles_hysteresis_search.py — Testa exit_quantile (banda di isteresi:
tieni una posizione finche' non esce da un quantile PIU' LARGO di quello usato
per l'ingresso, invece di vendere appena esce dal quantile stretto) sul fattore
scarsita' delle singole (poke_quant/engine/strategies/scarcity_value_factor.py).

Motivazione: con top_quantile=0.20 e un residuo che oscilla di poco attorno al
taglio esatto, una carta puo' essere venduta e ricomprata a ogni ribilanciamento
senza che sia cambiato nulla nella tesi - paga frizioni (Cardmarket 5%+0,60€,
spedizione) per rumore, non per un vero segnale di uscita.

Stesso standard di rigore delle altre ricerche su questo fattore (griglia ->
PBO/CSCV -> DSR corretto per TUTTI i trial sulle singole -> block bootstrap ->
walk-forward H1/H2), entrata fissa a top_quantile=0.20/rebal=3m (validata),
NESSUN ALTRO PARAMETRO CAMBIATO.

ESITO: NON VALIDATO, NESSUNA MODIFICA alla produzione (exit_quantile resta None).
La baseline (nessuna isteresi) vince per Sharpe su TUTTI i candidati e PBO=0,0%
sulla griglia completa - qui non c'e' nemmeno l'ambiguita' vista sui box, il
ranking e' netto. Il ROI medio per trade sale vistosamente allargando la banda
(+39,8% -> +185,4%) ma e' lo stesso miraggio gia' visto altrove in questa
sessione: il conteggio trade crolla (294 -> 20) e il MaxDD peggiora
drasticamente (-8,5% -> -23,6%, Sharpe 2,02 -> 1,26) - allargare la banda di
uscita non riduce whipsaw innocuo, tiene posizioni che stanno gia' girando
contro finche' il taglio largo non scatta, cavalcandole in perdita per piu'
tempo prima di vendere. Il quantile stretto e simmetrico (entrata=uscita)
resta la scelta migliore anche guardando il rendimento per trade, non solo lo
Sharpe di portafoglio.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.engine.strategies.scarcity_value_factor import ScarcityValueFactorStrategy
from poke_quant.validation.statistical_validation import deflated_sharpe_ratio, pbo_cscv
from poke_quant.validation.bootstrap import block_bootstrap_metrics, summarize_bootstrap
from scripts.optimize_and_falsify import run_bt

# Trial gia' tentati sulle singole in questa sessione (46 originali + 5 griglia
# scarsita' + 5 topdown + 6 varianti di specifica = 62) + questa griglia.
PRIOR_SINGLES_TRIALS = 62

PRODUCTION_PARAMS = dict(rebalance_every_months=3, top_quantile=0.20, min_age_months=6,
                          max_positions=60, min_cross_section=20)


def per_trade_stats(res):
    td = res.trades_df
    if td.empty:
        return 0.0, 0.0, 0
    return float(td["net_roi"].mean()), float(td["net_roi"].median()), len(td)


def main():
    metadata = load_metadata()
    prices_full = load_price_matrix("historical_prices_graded_singles_grade9.csv")
    singles_ids = [
        k for k, v in metadata.items()
        if v.get("type") == "single" and v.get("data_quality") != "thin_unreliable" and k in prices_full.columns
    ]
    meta_sub = {k: metadata[k] for k in singles_ids}
    prices_sub = prices_full[singles_ids]
    print(f"Universo: {len(singles_ids)} singole (chase + controllo)")

    baseline_factory = lambda: ScarcityValueFactorStrategy(**PRODUCTION_PARAMS)
    candidate_factories = {"BASELINE (exit_quantile=None, in produzione)": baseline_factory}
    for eq in [0.25, 0.30, 0.35, 0.40, 0.50]:
        candidate_factories[f"exit_quantile={eq:.2f}"] = (
            lambda eq=eq: ScarcityValueFactorStrategy(**PRODUCTION_PARAMS, exit_quantile=eq)
        )

    results = {}
    print(f"\n{'Candidato':42s} {'CAGR':>8s} {'Sharpe':>7s} {'MaxDD':>8s} {'Trade':>6s} {'ROI medio/trade':>16s}")
    for name, factory in candidate_factories.items():
        res = run_bt(factory(), prices_sub, meta_sub)
        results[name] = res
        roi_mean, _, n_trades = per_trade_stats(res)
        print(f"{name:42s} {res.cagr*100:+7.2f}% {res.sharpe:7.2f} {res.max_drawdown*100:7.2f}% {n_trades:6d} {roi_mean*100:+15.2f}%")

    common_idx = None
    for res in results.values():
        common_idx = res.monthly_returns.index if common_idx is None else common_idx.intersection(res.monthly_returns.index)
    perf_matrix = np.column_stack([results[k].monthly_returns.loc[common_idx].values for k in candidate_factories])
    t_len = len(perf_matrix)
    splits = 8 if t_len >= 32 else 4
    rem = t_len % splits
    pbo = pbo_cscv(perf_matrix[rem:, :] if rem else perf_matrix, n_splits=splits)
    print(f"\nPBO sulla griglia completa ({len(candidate_factories)} candidati, {splits} split): {pbo:.3f} ({pbo*100:.1f}%)")

    baseline_sharpe = results["BASELINE (exit_quantile=None, in produzione)"].sharpe
    best_name = max(results, key=lambda k: results[k].sharpe)
    best = results[best_name]
    print(f"\nBaseline Sharpe: {baseline_sharpe:.2f} | Migliore della griglia: '{best_name}' (Sharpe {best.sharpe:.2f})")

    total_trials = PRIOR_SINGLES_TRIALS + len(candidate_factories)
    dsr_own = deflated_sharpe_ratio(observed_sr=best.sharpe / np.sqrt(12), n_trials=len(candidate_factories), n_obs=len(best.monthly_returns))
    dsr_full = deflated_sharpe_ratio(observed_sr=best.sharpe / np.sqrt(12), n_trials=total_trials, n_obs=len(best.monthly_returns))
    print(f"DSR vincitore (solo griglia propria, n_trials={len(candidate_factories)}): {dsr_own:.3f}")
    print(f"DSR vincitore (intera ricerca sulle singole, n_trials={total_trials}): {dsr_full:.3f}")

    if best_name == "BASELINE (exit_quantile=None, in produzione)":
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
            res = run_bt(factory(), sub_prices, meta_sub)
            roi_mean, _, n_t = per_trade_stats(res)
            print(f"  {label_strat:42s} {label} ({dates[0].strftime('%Y-%m')}->{dates[-1].strftime('%Y-%m')}) | "
                  f"Sharpe {res.sharpe:5.2f} | CAGR {res.cagr*100:+6.2f}% | Trade {n_t:4d} | ROI medio/trade {roi_mean*100:+6.2f}%")


if __name__ == "__main__":
    main()
