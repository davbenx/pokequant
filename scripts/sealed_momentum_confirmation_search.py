#!/usr/bin/env python3
"""
scripts/sealed_momentum_confirmation_search.py — Testa se richiedere N mesi di
CONFERMA del momentum positivo prima di comprare (min_confirm_months, vedi
poke_quant/engine/strategies/time_series_momentum.py) migliora il rendimento
PER TRADE sui box sigillati — non lo Sharpe di portafoglio: richiesta esplicita
dell'utente ("il breadth non mi interessa, tieni il rendimento per box come
obiettivo").

Motivazione: e' l'opposto del filtro di freschezza sulle singole. Un check
diretto (vedi sotto DIAGNOSI) mostra che per il momentum dei box la PERSISTENZA
del segnale predice rendimenti forward MIGLIORI, non peggiori (l'opposto delle
singole, dove un residuo di valore persistente e' un value trap) - un trend
confermato da piu' mesi e' un trend piu' solido, non uno stantio.

DIAGNOSI (block bootstrap 6m, 500 sim, orizzonte forward 6m, split streak<=3
vs streak>3 sui box con momentum positivo): spread mediano fresco-persistente
= -8,0 punti percentuali (il PERSISTENTE vince), 27% dei mesi con fresco
migliore, CI bootstrap 90% [-22,4%, -2,7%] - il segno e' opposto e robusto in
entrambe le meta' del periodo.

Stesso standard di rigore delle altre ricerche su questa strategia (griglia ->
PBO/CSCV -> DSR corretto per tutti i candidati -> block bootstrap -> walk-forward
H1/H2), entrata a lookback=12m (validata), NESSUN ALTRO PARAMETRO CAMBIATO.

ESITO: NON VALIDATO. NESSUNA MODIFICA alla produzione (min_confirm_months resta
1, comportamento originale). Il "vincitore" per ROI medio/trade (min_confirm=9m,
+75,1%/trade contro +23,6% baseline) e' un artefatto statistico, non un edge:
PBO 87,1% su 7 candidati (quasi il peggiore possibile - il ranking in-sample e'
essenzialmente ANTI-informativo, sotto il caso), DSR corretto per l'intera
ricerca sui sealed 0,634 (sotto anche il baseline gia' marginale, 0,675), e il
walk-forward lo conferma nel modo piu' diretto: il "vincitore" ha SOLO 1 trade
in H1 (2020-12->2023-10) - una media "per trade" calcolata su un singolo trade
non e' una stima, e' un numero a caso con un'incertezza enorme.

Il meccanismo e' esattamente il rischio segnalato PRIMA di lanciare la griglia
(non dopo, per bias di conferma): alzare la soglia di conferma riduce i trade
totali (24 -> 15 -> 14 -> 13 -> 9 -> 7 -> 5) e la media per trade, con pochi
trade, e' dominata da 1-2 vincite eccezionali - non misura la qualita' della
regola, misura la fortuna del piccolo campione residuo. "Il breadth non conta"
e' una preferenza legittima per la costruzione del portafoglio, ma non rende
piu' affidabile una stima fatta su n=1-7 osservazioni: quella e' una questione
di potenza statistica, non di diversificazione, e vale a prescindere da quante
posizioni si vogliono davvero tenere in portafoglio.
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

# Trial gia' tentati sui sealed in questa sessione (vedi scripts/dsr_session_audit.py,
# box_ev_theory_test.py, sealed_time_stop_search.py, sealed_age_window_search.py e la
# ricerca originale in optimize_and_falsify.py) + questa griglia.
PRIOR_SEALED_TRIALS = 32


def per_trade_stats(res):
    td = res.trades_df
    if td.empty:
        return 0.0, 0.0, 0
    return float(td["net_roi"].mean()), float(td["net_roi"].median()), len(td)


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
    candidate_factories = {"BASELINE (min_confirm=1, in produzione)": baseline_factory}
    for c in [2, 3, 4, 6, 9, 12]:
        candidate_factories[f"min_confirm={c}m"] = (
            lambda p, c=c: TimeSeriesMomentumStrategy(p, lookback_months=12, min_confirm_months=c)
        )

    results = {}
    print(f"\n{'Candidato':42s} {'CAGR':>8s} {'Sharpe':>7s} {'MaxDD':>8s} {'Trade':>6s} {'ROI medio/trade':>16s} {'ROI mediano/trade':>18s}")
    for name, factory in candidate_factories.items():
        res = run_bt(factory(prices_sub), prices_sub, meta_sub)
        results[name] = res
        roi_mean, roi_med, n_trades = per_trade_stats(res)
        print(f"{name:42s} {res.cagr*100:+7.2f}% {res.sharpe:7.2f} {res.max_drawdown*100:7.2f}% {n_trades:6d} "
              f"{roi_mean*100:+15.2f}% {roi_med*100:+17.2f}%")

    common_idx = None
    for res in results.values():
        common_idx = res.monthly_returns.index if common_idx is None else common_idx.intersection(res.monthly_returns.index)
    perf_matrix = np.column_stack([results[k].monthly_returns.loc[common_idx].values for k in candidate_factories])
    t_len = len(perf_matrix)
    splits = 8 if t_len >= 32 else 4
    rem = t_len % splits
    pbo = pbo_cscv(perf_matrix[rem:, :] if rem else perf_matrix, n_splits=splits)
    print(f"\nPBO sulla griglia completa ({len(candidate_factories)} candidati, {splits} split): {pbo:.3f} ({pbo*100:.1f}%)")

    # Vincitore per OBIETTIVO RICHIESTO: rendimento medio per trade (non Sharpe).
    best_name = max(results, key=lambda k: per_trade_stats(results[k])[0])
    best = results[best_name]
    roi_mean_best, roi_med_best, n_trades_best = per_trade_stats(best)
    baseline_roi_mean, _, n_trades_base = per_trade_stats(results["BASELINE (min_confirm=1, in produzione)"])
    print(f"\nVincitore per ROI MEDIO PER TRADE: '{best_name}' ({roi_mean_best*100:+.2f}%/trade, n={n_trades_best}) "
          f"vs baseline ({baseline_roi_mean*100:+.2f}%/trade, n={n_trades_base})")

    total_trials = PRIOR_SEALED_TRIALS + len(candidate_factories)
    dsr_own = deflated_sharpe_ratio(observed_sr=best.sharpe / np.sqrt(12), n_trials=len(candidate_factories), n_obs=len(best.monthly_returns))
    dsr_full = deflated_sharpe_ratio(observed_sr=best.sharpe / np.sqrt(12), n_trials=total_trials, n_obs=len(best.monthly_returns))
    print(f"DSR vincitore (solo griglia propria, n_trials={len(candidate_factories)}): {dsr_own:.3f}")
    print(f"DSR vincitore (intera ricerca sui sealed, n_trials={total_trials} = {PRIOR_SEALED_TRIALS} precedenti + "
          f"{len(candidate_factories)} di qui): {dsr_full:.3f}")

    sims = block_bootstrap_metrics(best.monthly_returns, n_sims=500, block_size=6)
    print(f"\nBlock bootstrap (500 sim, blocchi 6m) sul vincitore:")
    print(summarize_bootstrap(sims))

    print(f"\nWalk-forward H1/H2, ROI medio per trade, vincitore vs baseline:")
    mid = len(prices_sub) // 2
    h1_dates, h2_dates = prices_sub.index[:mid], prices_sub.index[mid:]
    for label_strat, factory in [("BASELINE", baseline_factory), (best_name, candidate_factories[best_name])]:
        for label, dates in [("H1", h1_dates), ("H2", h2_dates)]:
            sub_prices = prices_sub.loc[dates]
            res = run_bt(factory(sub_prices), sub_prices, meta_sub)
            roi_mean_h, roi_med_h, n_h = per_trade_stats(res)
            print(f"  {label_strat:42s} {label} ({dates[0].strftime('%Y-%m')}->{dates[-1].strftime('%Y-%m')}) | "
                  f"ROI medio/trade {roi_mean_h*100:+7.2f}% | n={n_h:3d} | Sharpe {res.sharpe:5.2f}")


if __name__ == "__main__":
    main()
