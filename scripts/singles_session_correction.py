#!/usr/bin/env python3
"""
scripts/singles_session_correction.py — Chiusura della ricerca sistematica su
singole gradate di questa sessione: corregge il DSR del candidato piu'
promettente (RelativeValueFactorStrategy) per TUTTI i trial indipendenti
tentati sulle singole in questa sessione, non solo per la griglia interna del
suo stesso test - richiesto esplicitamente dall'utente ("procedi" dopo aver
elencato il piano di validazione).

Conteggio dei trial (ricostruito dagli script/sezioni effettivamente eseguiti):
  9  section_singles_factor_search       (TSMOM x3, XSMOM x2, Dip x3, RarityTier x1)
  4  section_singles_technical_fundamental_search (MOM skip1, 52w-HIGH, LOW-VOL, RATIO-MOM)
  4  griglia illustratore (rarity_tier_factor field=artist)
  3  promo_factor_grid.py
  3  chase_singles_tsmom_test.py          (chase / random_control / full)
  2  box_vs_singles_basket_test.py        (box reale vs paniere di singole)
  5  relative_value_singles_test.py       (griglia con is_chase, il candidato stesso)
  8  grade_spread_test.py                 (4 coppie di grado x 2 frequenze)
  8  grade_lead_lag_test.py               (4 coppie di grado x 2 direzioni)
 --
 46  TOTALE

ESITO: DSR 0,895 (solo griglia propria) -> 0,581 (intera sessione). Sotto la
soglia 0,90-0,95 usata per ogni altro candidato in questa ricerca. Vedi il
docstring completo (con i controlli di stabilita' per segmento e robustezza
della specifica) in poke_quant/engine/strategies/relative_value_factor.py.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.engine.strategies.relative_value_factor import RelativeValueFactorStrategy
from poke_quant.validation.statistical_validation import deflated_sharpe_ratio
from scripts.optimize_and_falsify import run_bt

TRIAL_COUNTS = {
    "section_singles_factor_search (TSMOM x3, XSMOM x2, Dip x3, RarityTier x1)": 9,
    "section_singles_technical_fundamental_search (MOM/52w/LOWVOL/RATIO-MOM)": 4,
    "griglia illustratore (rarity_tier_factor field=artist)": 4,
    "promo_factor_grid.py": 3,
    "chase_singles_tsmom_test.py (chase/random_control/full)": 3,
    "box_vs_singles_basket_test.py (box vs paniere)": 2,
    "relative_value_singles_test.py (candidato stesso)": 5,
    "grade_spread_test.py (4 coppie x 2 rebal)": 8,
    "grade_lead_lag_test.py (4 coppie x 2 direzioni)": 8,
}


def main():
    metadata = load_metadata()
    prices_full = load_price_matrix("historical_prices_graded_singles_grade9.csv")
    singles_ids = [
        k for k, v in metadata.items()
        if v.get("type") == "single" and v.get("data_quality") != "thin_unreliable" and k in prices_full.columns
    ]
    meta_sub = {k: metadata[k] for k in singles_ids}
    prices_sub = prices_full[singles_ids]

    strat = RelativeValueFactorStrategy(rebalance_every_months=3, top_quantile=0.20, min_age_months=6)
    res = run_bt(strat, prices_sub, meta_sub)

    total_trials = sum(TRIAL_COUNTS.values())
    print("Trial indipendenti tentati sulla ricerca singole in questa sessione:")
    for name, n in TRIAL_COUNTS.items():
        print(f"  {n:3d}  {name}")
    print(f"TOTALE: {total_trials}\n")

    dsr_own_grid = deflated_sharpe_ratio(observed_sr=res.sharpe / np.sqrt(12), n_trials=5, n_obs=len(res.monthly_returns))
    dsr_full_session = deflated_sharpe_ratio(observed_sr=res.sharpe / np.sqrt(12), n_trials=total_trials, n_obs=len(res.monthly_returns))
    print(f"Sharpe osservato: {res.sharpe:.2f} (n_obs={len(res.monthly_returns)} mesi)")
    print(f"DSR corretto solo per la griglia propria (n_trials=5):        {dsr_own_grid:.3f}")
    print(f"DSR corretto per l'intera sessione (n_trials={total_trials}):            {dsr_full_session:.3f}")
    print(f"\nSoglia di comfort usata in questa ricerca: 0.90-0.95. "
          f"{'PASSA' if dsr_full_session >= 0.90 else 'NON PASSA'} con la correzione onesta.")


if __name__ == "__main__":
    main()
