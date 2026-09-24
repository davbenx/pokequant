#!/usr/bin/env python3
"""
scripts/grade_specific_scarcity_test.py — Testa se il fattore scarsita'
(poke_quant/engine/strategies/scarcity_value_factor.py, validato SOLO su
Grado 9) mostra un edge A SE STANTE quando rifittato INDIPENDENTEMENTE sui
dati di Grado 7, Grado 8 e PSA10 - non se il segnale di Grado 9 si trasferisce
ad altri gradi (gia' testato e falsificato in scripts/grade_correlation_test.py:
correlazione mensile per-carta 0,13-0,24, troppo debole per un trasferimento).
Qui la domanda e' diversa: ogni grado potrebbe avere una propria dinamica di
scarsita' scollegata da quella di Grado 9.

ATTENZIONE METODOLOGICA GRAVE, dichiarata subito: il campione (data_cache/
grade_ladder_prices.json, scripts/fetch_grade_ladder.py) e' composto AL 100%
da carte "chase" (selection_method=chase_price_filter_survivorship_biased) -
NESSUNA carta di controllo casuale. Questo e' esattamente il bias che il
fattore scarsita' su Grado 9 aveva dovuto superare con un campione di controllo
dedicato (864 carte, chase+random_control) prima di essere preso sul serio.
Qualunque risultato positivo qui e' PRELIMINARE per costruzione - andrebbe
rifatto su un campione con controllo prima di essere considerato un edge reale,
non solo un artefatto di "abbiamo scelto carte che sappiamo essere salite".

Universo piccolo (max ~130 carte con storico sufficiente, contro 864 per
Grado 9) - risultati direzionali, non statisticamente definitivi (n troppo
piccolo per DSR/PBO affidabili, ma sufficiente per invalidare l'ipotesi se il
segnale non regge nemmeno qui).

ESITO: NON VALIDATO su nessuno dei tre gradi, e per il motivo che conta di
piu' in questa ricerca - lo stesso schema visto ripetutamente sui fattori
falliti sulle singole:
  - Grado 7: PBO 40,0% (griglia di 3 candidati - il ranking e' quasi rumore),
    e il vincitore SI INVERTE DI SEGNO tra le meta': H1 Sharpe -0,39, H2 +1,08.
  - Grado 8: PBO 34,3%, H1 Sharpe debole (+0,29) contro H2 forte (+1,68) -
    non un'inversione netta, ma fortemente concentrato in H2.
  - PSA10: PBO 48,6% (il peggiore, quasi un lancio di moneta), Sharpe massimo
    solo 0,99 anche sulla sua griglia migliore, H1 debole (+0,33) contro H2
    (+1,22).
In tutti e tre i casi il "rendimento" e' concentrato nel rally 2023-2026 delle
chase cards in generale (beta di mercato), non in una correzione di prezzo
sistematica - lo stesso identikit di ogni altro fattore sulle singole scartato
in questa sessione prima di scarcity_value_factor (che invece regge H1 E H2).
Aggravato dal campione 100% chase (nessun controllo casuale) - anche se uno di
questi avesse superato i controlli, sarebbe stato comunque da riverificare su
un campione non biased prima di crederci. Nessuna modifica alla produzione: il
fattore scarsita' resta validato SOLO su Grado 9.
"""

import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata
from poke_quant.engine.strategies.scarcity_value_factor import ScarcityValueFactorStrategy
from poke_quant.validation.statistical_validation import pbo_cscv
from scripts.optimize_and_falsify import run_bt

LADDER_FILE = Path(__file__).resolve().parent.parent / "data_cache" / "grade_ladder_prices.json"
TIERS = {"grade7": "Grado 7", "grade8": "Grado 8", "psa10": "PSA10"}
MIN_MONTHS = 24


def build_tier_panel_and_meta(ladder: dict, metadata: dict, tier: str):
    cols = {}
    for item_id, tiers in ladder.items():
        if tier in tiers and len(tiers[tier]) >= MIN_MONTHS and item_id in metadata:
            s = pd.Series(tiers[tier])
            s.index = pd.to_datetime(s.index)
            cols[item_id] = s.sort_index()
    prices_df = pd.DataFrame(cols)
    meta_sub = {c: metadata[c] for c in prices_df.columns}
    return prices_df, meta_sub


def main():
    ladder = json.loads(LADDER_FILE.read_text())
    metadata = load_metadata()
    print(f"Campione (tutto chase, nessun controllo casuale): {len(ladder)} carte")

    for tier_key, tier_label in TIERS.items():
        prices_df, meta_sub = build_tier_panel_and_meta(ladder, metadata, tier_key)
        n = len(prices_df.columns)
        print(f"\n=== {tier_label}: {n} carte con >= {MIN_MONTHS} mesi di storico ===")
        if n < 20:
            print(f"  Universo troppo piccolo (min_cross_section=20 richiesto) - non testabile con questo campione.")
            continue

        configs = {
            "rebal=3 q=0.20": dict(rebalance_every_months=3, top_quantile=0.20, min_age_months=0, min_cross_section=15),
            "rebal=6 q=0.20": dict(rebalance_every_months=6, top_quantile=0.20, min_age_months=0, min_cross_section=15),
            "rebal=6 q=0.30": dict(rebalance_every_months=6, top_quantile=0.30, min_age_months=0, min_cross_section=15),
        }
        results = {}
        for name, kw in configs.items():
            strat = ScarcityValueFactorStrategy(**kw)
            res = run_bt(strat, prices_df, meta_sub)
            results[name] = res
            print(f"  {name:16s} | CAGR {res.cagr*100:+7.2f}% | Sharpe {res.sharpe:5.2f} | "
                  f"MaxDD {res.max_drawdown*100:7.2f}% | Trade {res.total_trades:3d}")

        common_idx = None
        for res in results.values():
            common_idx = res.monthly_returns.index if common_idx is None else common_idx.intersection(res.monthly_returns.index)
        if common_idx is not None and len(common_idx) >= 8:
            perf_matrix = np.column_stack([results[k].monthly_returns.loc[common_idx].values for k in configs])
            t_len = len(perf_matrix)
            splits = 8 if t_len >= 32 else 4
            rem = t_len % splits
            pbo = pbo_cscv(perf_matrix[rem:, :] if rem else perf_matrix, n_splits=splits)
            print(f"  PBO ({len(configs)} candidati, {splits} split): {pbo:.3f}")

        best_name = max(results, key=lambda k: results[k].sharpe)
        best = results[best_name]
        if best.total_trades < 15:
            print(f"  ATTENZIONE: il vincitore ha solo {best.total_trades} trade - troppo pochi per un giudizio, "
                  "qualunque sia lo Sharpe.")
        mid = len(prices_df) // 2
        if mid >= 12:
            h1_dates, h2_dates = prices_df.index[:mid], prices_df.index[mid:]
            print(f"  Walk-forward del vincitore '{best_name}':")
            for label, dates in [("H1", h1_dates), ("H2", h2_dates)]:
                sub = prices_df.loc[dates]
                strat = ScarcityValueFactorStrategy(**configs[best_name])
                r = run_bt(strat, sub, meta_sub)
                print(f"    {label} ({dates[0].strftime('%Y-%m')}->{dates[-1].strftime('%Y-%m')}) | "
                      f"Sharpe {r.sharpe:5.2f} | CAGR {r.cagr*100:+6.2f}% | Trade {r.total_trades:3d}")


if __name__ == "__main__":
    main()
