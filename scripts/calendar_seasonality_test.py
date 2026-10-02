#!/usr/bin/env python3
"""
scripts/calendar_seasonality_test.py — Stagionalita' di calendario su box
sigillati e singole gradate: ipotesi MAI testata finora in questo repo
(verificato: nessuno script esistente contiene "season"/"calendar"/
"holiday"/"month dummy"/"december"/"november").

RICHIESTA DELL'UTENTE: "Cerca online tutte le informazioni e strategie su
investimenti in slabs e sealed. Valuta cosa ancora non abbiamo testato e
prova a testare ed invalidarlo." Ricerca web (2026-10-02) su strategie di
investimento in slab/sealed: la maggior parte dei fattori discussi online
(crack-and-regrade, spread tra case di gradazione, popolazione come segnale,
momentum su storico prezzi, scarsita' tipografica, lead-lag tra gradi) risulta
GIA' testata in questo repo (vedi grading_company_crossover_arbitrage_test.py,
same_grade_crossover_arbitrage_test.py, population_scarcity_factor_test.py,
grade_lead_lag_test.py, graded_raw_ratio_grid_search.py, ecc.). Il claim
ricorrente mai verificato qui e' la STAGIONALITA': piu' fonti (forum
elitefourum.com/t/pokemon-cards-price-analysis-september-to-december/19163,
guide di settore pokemonpricetracker.com/pokefolio.eu) affermano che la
domanda (e quindi il prezzo) sale verso Ottobre-Dicembre (regali di Natale) e
si affloscia dopo, un pattern da "effetto calendario" distinto dal momentum
puro gia' nel motore di produzione.

METODO (disciplina identica al resto della sessione - calibrare/guardare la
diagnostica PRIMA di costruire qualunque regola di trading, mai il contrario):
1. Indice equal-weighted mensile (stessa costruzione di
   scripts/update_monthly_cache.py::compute_market_indices_data,
   sub.pct_change().mean(axis=1)) sull'universo liquido BOX e SINGOLE
   separatamente - non sulle singole carte (pseudo-replicazione: carte nello
   stesso mese sono fortemente co-mosse, l'unita' statisticamente onesta e'
   il mese-indice, non la carta).
2. Raggruppa i rendimenti mensili dell'indice per MESE DI CALENDARIO (1-12).
   Con ~70 mesi di storico (dic-2020/gen-2021 -> set-2026) questo da SOLO
   5-6 osservazioni indipendenti per mese - dichiarato esplicitamente come
   potenza statistica bassa, PRIMA di guardare i risultati.
3. Test pre-registrato (un solo confronto, non 12, per evitare il
   p-hacking multiplo che questa sessione ha gia' imparato a evitare altrove
   - vedi graded_raw_ratio_grid_search.py sul PBO della griglia): raggruppa
   in due soli bucket - "finestra regali" (Ott-Dic) vs "resto dell'anno" -
   il claim esatto trovato online, non un cherry-pick a posteriori del mese
   migliore. T-test Welch a due code + Mann-Whitney U (robusto a outlier/non
   normalita' con n piccolo) su quel SOLO confronto.
4. Solo se il confronto pre-registrato risulta significativo, si procede a
   costruire e backtestare una regola di trading stagionale (griglia, PBO,
   walk-forward, stesso standard di ogni altro fattore in produzione) -
   altrimenti si riporta l'invalidazione e ci si ferma qui.

ESITO: vedi output.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.data.liquidity_filter import liquid_sealed_ids, liquid_singles_ids


def build_index_returns(prices_full: pd.DataFrame, ids: list) -> pd.Series:
    sub = prices_full[ids]
    return sub.pct_change().mean(axis=1, skipna=True).dropna()


def month_name(m: int) -> str:
    return ["Gen", "Feb", "Mar", "Apr", "Mag", "Giu", "Lug", "Ago", "Set", "Ott", "Nov", "Dic"][m - 1]


def report_seasonality(label: str, returns: pd.Series):
    print(f"\n{'='*78}\n{label}\n{'='*78}")
    n_years = len(returns) / 12.0
    print(f"Storico: {len(returns)} mesi (~{n_years:.1f} anni) -> {len(returns)//12}-{-(-len(returns)//12)} "
          f"osservazioni indipendenti per mese di calendario. Potenza statistica BASSA per costruzione.")

    by_month = {}
    for i, r in returns.items():
        by_month.setdefault(i.month, []).append(r)

    print(f"\n1. Diagnostica pura - rendimento medio per mese di calendario (NON un test, solo descrittivo):")
    for m in range(1, 13):
        vals = by_month.get(m, [])
        if vals:
            print(f"   {month_name(m):4s}  n={len(vals)}  media={np.mean(vals)*100:+6.2f}%  "
                  f"mediana={np.median(vals)*100:+6.2f}%")
        else:
            print(f"   {month_name(m):4s}  n=0 (nessuna osservazione)")

    holiday_months = {10, 11, 12}
    holiday_vals = [r for i, r in returns.items() if i.month in holiday_months]
    rest_vals = [r for i, r in returns.items() if i.month not in holiday_months]

    print(f"\n2. Test PRE-REGISTRATO (Ott-Dic vs resto dell'anno, l'unico confronto preso in esame):")
    print(f"   Ott-Dic:        n={len(holiday_vals)}  media={np.mean(holiday_vals)*100:+.2f}%  "
          f"dev.std={np.std(holiday_vals, ddof=1)*100:.2f}%")
    print(f"   Resto dell'anno: n={len(rest_vals)}  media={np.mean(rest_vals)*100:+.2f}%  "
          f"dev.std={np.std(rest_vals, ddof=1)*100:.2f}%")

    t_stat, t_p = stats.ttest_ind(holiday_vals, rest_vals, equal_var=False)
    u_stat, u_p = stats.mannwhitneyu(holiday_vals, rest_vals, alternative="two-sided")
    print(f"   Welch t-test:     t={t_stat:+.3f}  p={t_p:.3f}")
    print(f"   Mann-Whitney U:   U={u_stat:.1f}  p={u_p:.3f}")

    alpha = 0.05
    validated = t_p < alpha and u_p < alpha and np.mean(holiday_vals) > np.mean(rest_vals)
    verdict = "VALIDATO (entrambi i test p<0.05, direzione coerente col claim)" if validated else \
        "NON VALIDATO (claim di stagionalita' Ott-Dic respinto a soglia 5%)"
    print(f"\n   -> {verdict}")
    return validated


def main():
    metadata = load_metadata()

    print("Verifica claim: 'la domanda/i prezzi dei collezionabili Pokemon salgono verso")
    print("Ott-Dic per la stagione dei regali' (fonti web: elitefourum, pokemonpricetracker,")
    print("pokefolio.eu) - MAI testato prima in questo repo.")

    box_prices = load_price_matrix()
    sealed_ids = liquid_sealed_ids(metadata, box_prices)
    box_idx_returns = build_index_returns(box_prices, sealed_ids)
    v_box = report_seasonality(f"BOX SIGILLATI (indice equal-weighted, {len(sealed_ids)} prodotti liquidi)", box_idx_returns)

    singles_prices = load_price_matrix("historical_prices_graded_singles_grade9.csv")
    metadata_s, singles_prices_s = metadata, singles_prices
    singles_ids = liquid_singles_ids(metadata_s, singles_prices_s)
    singles_idx_returns = build_index_returns(singles_prices_s, singles_ids)
    v_singles = report_seasonality(f"SINGOLE GRADATE (indice equal-weighted, {len(singles_ids)} carte liquide)", singles_idx_returns)

    print(f"\n{'='*78}\nCONCLUSIONE\n{'='*78}")
    if not v_box and not v_singles:
        print("Claim di stagionalita' Ott-Dic INVALIDATO su entrambi gli universi (box e singole).")
        print("Nessuna regola di trading stagionale verra' costruita - si ferma qui per disciplina")
        print("anti-overfitting (non si cerca un altro mese/finestra a posteriori sugli stessi dati).")
    else:
        print("Claim VALIDATO su almeno un universo - procedere a backtest completo (griglia/PBO/")
        print("walk-forward) prima di qualunque adozione in produzione.")


if __name__ == "__main__":
    main()
