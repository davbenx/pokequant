#!/usr/bin/env python3
"""
scripts/grade_spread_test.py — Testa l'idea "spread tra gradazioni della stessa
carta" su un campione piccolo e casuale (40 carte chase, seed fisso, vedi
scripts/fetch_grade_ladder.py) usando l'infrastruttura gia' validata
(CrossSectionalFactorStrategy + zscore_factor, mai wired prima d'ora):
il segnale si calcola sul RAPPORTO tra due gradi (es. PSA10/Grade9), l'azione
reale (buy/sell/size) e' sempre sull'asset del grado piu' basso della coppia -
comprare quello che oggi e' storicamente CARO rispetto al grado superiore
(z-score alto sul rapporto), aspettando che la differenza si comprima.

Stesso standard di rigore di tutta questa ricerca, PIU' due controlli chiesti
esplicitamente per non cadere in bias di conferma:
  1. Placebo: rimescola le etichette carta<->rapporto e riesegue lo stesso
     identico test - se il placebo rende quasi uguale, il "segnale" e' un
     artefatto della meccanica di backtest (frizioni, deriva generale del
     mercato), non un vero mispricing carta-per-carta.
  2. Conteggio trade esplicito ad ogni riga - un risultato con troppi pochi
     trade non e' statisticamente significativo, va segnalato come tale
     invece di essere presentato come un edge.

ESITO: NON VALIDATO. Su 40 carte chase campionate a caso (seed=42), 4 coppie
di grado x 2 frequenze di ribilanciamento: PBO sulla griglia 78,6% (stesso
livello del test sealed age-window, gia' fallito) - il ranking tra candidati
e' essenzialmente rumore. Il "vincitore" (grade9/grade8, rebal=6m) ha DSR
0,850 (sotto soglia), bootstrap con 5* percentile che attraversa lo zero
(-0,026), e soprattutto un walk-forward che si inverte di segno: H1 Sharpe
-0,47 (solo 7 trade - troppo pochi per significare qualcosa), H2 +2,59 (15
trade) - lo stesso schema boom/bust di ogni altro fattore fallito su singole
in questa ricerca. Il placebo (rimescolo le etichette carta<->rapporto:
Sharpe 0,37 contro 1,08 del vincitore) esclude che sia un puro artefatto di
frizioni/drift generico, ma non basta a salvarla dato tutto il resto - e' una
condizione necessaria, non sufficiente. Aumentare il campione oltre le 40
carte probabilmente non cambia il verdetto: il problema e' la struttura del
periodo (un solo super-ciclo 2021-2026), non la dimensione campionaria.
"""

import json
import sys
from pathlib import Path
import random
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata
from poke_quant.engine.strategies.cross_sectional_factor import CrossSectionalFactorStrategy, zscore_factor
from poke_quant.validation.statistical_validation import deflated_sharpe_ratio, pbo_cscv
from poke_quant.validation.bootstrap import block_bootstrap_metrics, summarize_bootstrap
from scripts.optimize_and_falsify import run_bt

LADDER_FILE = Path(__file__).resolve().parent.parent / "data_cache" / "grade_ladder_prices.json"
MIN_MONTHS = 24


def build_tier_panel(ladder: dict, tier: str) -> pd.DataFrame:
    cols = {}
    for item_id, tiers in ladder.items():
        if tier in tiers and len(tiers[tier]) >= MIN_MONTHS:
            s = pd.Series(tiers[tier])
            s.index = pd.to_datetime(s.index)
            cols[item_id] = s.sort_index()
    return pd.DataFrame(cols)


def main():
    ladder = json.loads(LADDER_FILE.read_text())
    metadata = load_metadata()
    print(f"Campione: {len(ladder)} carte")

    panels = {t: build_tier_panel(ladder, t) for t in ["grade7", "grade8", "grade9", "grade9_5", "psa10"]}
    for t, p in panels.items():
        print(f"  {t}: {p.shape[1]} carte, {p.shape[0]} mesi")

    grade_pairs = [
        ("psa10", "grade9"), ("grade9_5", "grade9"),
        ("grade9", "grade8"), ("grade8", "grade7"),
    ]

    all_results = {}
    for high, low in grade_pairs:
        common_ids = [c for c in panels[low].columns if c in panels[high].columns]
        if len(common_ids) < 10:
            print(f"\n{high}/{low}: solo {len(common_ids)} carte in comune, salto (troppo poco per un test).")
            continue
        low_df = panels[low][common_ids].reindex(panels[low].index.union(panels[high].index)).sort_index()
        high_df = panels[high][common_ids].reindex(low_df.index)
        ratio_df = (high_df / low_df).replace([np.inf, -np.inf], np.nan)
        meta_sub = {c: {**metadata.get(c, {}), "type": "single"} for c in common_ids}

        print(f"\n=== {high}/{low} (n={len(common_ids)} carte) ===")
        for rebal in [3, 6]:
            strat = CrossSectionalFactorStrategy(
                ratio_df, zscore_factor, lookback_months=12, top_quantile=0.30,
                ascending=False, rebalance_every_months=rebal, item_type_filter="single", min_age_months=0,
            )
            res = run_bt(strat, low_df, meta_sub)
            name = f"{high}/{low} rebal={rebal}m"
            all_results[name] = (res, low_df, meta_sub, ratio_df)
            print(f"  rebal={rebal}m | CAGR {res.cagr*100:+6.2f}% | Sharpe {res.sharpe:5.2f} | "
                  f"MaxDD {res.max_drawdown*100:6.2f}% | Trade {res.total_trades:4d}")

    if not all_results:
        print("\nNessun candidato con dati sufficienti - impossibile procedere.")
        return

    common_idx = None
    for res, _, _, _ in all_results.values():
        common_idx = res.monthly_returns.index if common_idx is None else common_idx.intersection(res.monthly_returns.index)
    perf_matrix = np.column_stack([all_results[k][0].monthly_returns.loc[common_idx].values for k in all_results])
    t_len = len(perf_matrix)
    splits = 8 if t_len >= 32 else 4
    rem = t_len % splits
    pbo = pbo_cscv(perf_matrix[rem:, :] if rem else perf_matrix, n_splits=splits)
    print(f"\nPBO ({len(all_results)} candidati, {splits} split): {pbo:.3f}")

    best_name = max(all_results, key=lambda k: all_results[k][0].sharpe)
    best, best_low_df, best_meta, best_ratio_df = all_results[best_name]
    dsr = deflated_sharpe_ratio(observed_sr=best.sharpe / np.sqrt(12), n_trials=len(all_results), n_obs=len(best.monthly_returns))
    print(f"Migliore: '{best_name}' Sharpe {best.sharpe:.2f} | Trade {best.total_trades} | DSR: {dsr:.3f}")

    if best.total_trades < 15:
        print(f"ATTENZIONE: solo {best.total_trades} trade totali - troppo pochi per un giudizio statistico solido, "
              "qualunque sia il Sharpe.")

    sims = block_bootstrap_metrics(best.monthly_returns, n_sims=500, block_size=6)
    print(summarize_bootstrap(sims))

    mid = len(best_low_df) // 2
    h1_dates, h2_dates = best_low_df.index[:mid], best_low_df.index[mid:]
    print(f"\nWalk-forward H1/H2 sul vincitore '{best_name}':")
    for label, dates in [("H1", h1_dates), ("H2", h2_dates)]:
        sub_low = best_low_df.loc[dates]
        sub_ratio = best_ratio_df.loc[best_ratio_df.index.intersection(dates)]
        rebal = int(best_name.split("rebal=")[1].rstrip("m"))
        strat = CrossSectionalFactorStrategy(sub_ratio, zscore_factor, lookback_months=12, top_quantile=0.30,
                                              ascending=False, rebalance_every_months=rebal, item_type_filter="single")
        r = run_bt(strat, sub_low, best_meta)
        print(f"  {label} ({dates[0].strftime('%Y-%m')}->{dates[-1].strftime('%Y-%m')}) | "
              f"Sharpe {r.sharpe:5.2f} | CAGR {r.cagr*100:+6.2f}% | Trade {r.total_trades}")

    print(f"\n--- PLACEBO: rimescolo le etichette carta<->rapporto sul vincitore '{best_name}' ---")
    rng = random.Random(7)
    shuffled_cols = list(best_ratio_df.columns)
    rng.shuffle(shuffled_cols)
    placebo_ratio_df = best_ratio_df.copy()
    placebo_ratio_df.columns = shuffled_cols  # il rapporto della carta X ora e' etichettato come carta Y
    rebal = int(best_name.split("rebal=")[1].rstrip("m"))
    placebo_strat = CrossSectionalFactorStrategy(placebo_ratio_df, zscore_factor, lookback_months=12, top_quantile=0.30,
                                                  ascending=False, rebalance_every_months=rebal, item_type_filter="single")
    placebo_res = run_bt(placebo_strat, best_low_df, best_meta)
    print(f"  Placebo | CAGR {placebo_res.cagr*100:+6.2f}% | Sharpe {placebo_res.sharpe:5.2f} | "
          f"MaxDD {placebo_res.max_drawdown*100:6.2f}% | Trade {placebo_res.total_trades:4d}")
    print(f"  Vincitore reale | CAGR {best.cagr*100:+6.2f}% | Sharpe {best.sharpe:5.2f}")
    if placebo_res.sharpe >= best.sharpe * 0.7:
        print("  ATTENZIONE: il placebo rende quasi come il vincitore reale - il risultato e' probabilmente "
              "un artefatto della meccanica di backtest, non un vero segnale carta-per-carta.")
    else:
        print("  Il placebo rende molto peggio del vincitore reale - il segnale non e' solo un artefatto di frizioni/drift generale.")


if __name__ == "__main__":
    main()
