#!/usr/bin/env python3
"""
scripts/dsr_session_audit.py — Ricalcola il DSR della strategia in produzione
(TS Momentum sealed, lookback=12m) corretto per TUTTI i trial indipendenti
tentati sul lato sealed in questa sessione, non solo per la griglia di
lookback originale (n_trials=5) con cui fu scelta.

Richiesto dopo aver scoperto lo stesso problema sul fattore di valore
relativo delle singole (che passava con n_trials=5 propri, DSR 0,895, e
falliva con n_trials=46 dell'intera ricerca sulle singole, DSR 0,581) e sulla
teoria EV del box (DSR 0,910 propri -> 0,616 sessione intera). Per coerenza,
lo stesso audit va fatto sulla strategia che sta in produzione, non solo sui
candidati falliti - altrimenti si applica un doppio standard.

Conteggio dei trial (ricostruito dagli script/sezioni effettivamente eseguiti
sul lato sealed in questa sessione):
  5  section_tsmom_sealed          (griglia lookback: 6/9/12/15/18m - qui la
                                     strategia fu scelta e il DSR originale
                                     0,913 calcolato)
  9  section_sealed_exit_logic_search (baseline + 3 exit_lookback + 5 trailing_stop)
  7  sealed_age_window_search      (baseline + 6 finestre d'eta')
  7  sealed_time_stop_search       (baseline + 6 time stop)
  4  box_ev_theory_test            (2 lookback x 2 direzioni)
 --
 32  TOTALE

ESITO: DSR 0,913 (griglia propria) -> 0,675 (intera sessione). Sotto la
soglia di comfort 0,90-0,95 usata ovunque in questa ricerca, ma resta il piu'
alto tra tutti i candidati testati (valore relativo singole 0,581, teoria EV
box 0,616). Aggiornato in app.py - la dashboard mostra ora entrambi i numeri.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.engine.strategies.time_series_momentum import TimeSeriesMomentumStrategy
from poke_quant.validation.statistical_validation import deflated_sharpe_ratio
from scripts.optimize_and_falsify import run_bt, MODERN_ERA_CUTOFF

TRIAL_COUNTS = {
    "section_tsmom_sealed (griglia lookback, dove la strategia fu scelta)": 5,
    "section_sealed_exit_logic_search": 9,
    "sealed_age_window_search": 7,
    "sealed_time_stop_search": 7,
    "box_ev_theory_test": 4,
}


def main():
    metadata = load_metadata()
    prices_full = load_price_matrix()
    sealed_ids = [
        k for k, v in metadata.items()
        if v.get("type") == "sealed" and v.get("data_quality") != "thin_unreliable"
        and k in prices_full.columns and v.get("release_date") and v["release_date"] >= MODERN_ERA_CUTOFF
    ]
    meta_sub = {k: metadata[k] for k in sealed_ids}
    prices_sub = prices_full[sealed_ids]

    strat = TimeSeriesMomentumStrategy(prices_sub, lookback_months=12)
    res = run_bt(strat, prices_sub, meta_sub)

    total_trials = sum(TRIAL_COUNTS.values())
    print(f"Sharpe osservato: {res.sharpe:.2f} (n_obs={len(res.monthly_returns)} mesi)\n")
    print("Trial indipendenti tentati sul lato sealed in questa sessione:")
    for name, n in TRIAL_COUNTS.items():
        print(f"  {n:3d}  {name}")
    print(f"TOTALE: {total_trials}\n")

    dsr_own_grid = deflated_sharpe_ratio(observed_sr=res.sharpe / np.sqrt(12), n_trials=5, n_obs=len(res.monthly_returns))
    dsr_full_session = deflated_sharpe_ratio(observed_sr=res.sharpe / np.sqrt(12), n_trials=total_trials, n_obs=len(res.monthly_returns))
    print(f"DSR corretto solo per la griglia con cui fu scelta (n_trials=5): {dsr_own_grid:.3f}")
    print(f"DSR corretto per l'intera sessione (n_trials={total_trials}):            {dsr_full_session:.3f}")
    print(f"\nSoglia di comfort usata in questa ricerca: 0.90-0.95. "
          f"{'PASSA' if dsr_full_session >= 0.90 else 'NON PASSA'} con la correzione onesta.")


if __name__ == "__main__":
    main()
