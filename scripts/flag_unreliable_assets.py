#!/usr/bin/env python3
"""
scripts/flag_unreliable_assets.py — Applica il filtro di liquidità/attendibilità
(poke_quant/data/liquidity_filter.py), scrivendo un campo "data_quality" in
items_metadata.json per ogni asset che non lo supera. Non cancella nessun dato:
li rende visibili e facilmente esclusi dagli script di validazione.

Sealed: valutati su historical_prices.csv (comportamento originale, invariato).

Singole: valutate sul pannello historical_prices_graded_singles_grade9.csv -
NON su historical_prices.csv. FIX (richiesto indagando un "prezzo massimo/
falsi positivi da mercato sottile" per la dashboard): le due serie hanno
valori DIVERSI per la stessa carta (es. umbreon_vmax_215: legacy 1929€->1615€
in 2 mesi, grade9 1821€->1806€ nello stesso periodo - non sono lo stesso dato).
Il fattore scarsita' (scarcity_value_factor.py) legge SOLO il pannello grade9:
valutare l'attendibilita' sulla serie legacy non protegge affatto la serie
che la strategia usa davvero. Impatto misurato prima di applicare il fix:
62 carte che oggi passano il filtro (legacy pulito) hanno in realta' un
grade9 inaffidabile - nessuna protezione reale le intercettava. Al contrario,
133 carte oggi escluse (legacy sporco) hanno un grade9 pulito - erano escluse
per un motivo che non le riguarda. Netto: universo singole 864 -> ~935 asset,
PIU' pulito E PIU' grande, non un compromesso tra i due.

AGGIORNAMENTO (trovato verificando un prezzo reale - vedi
scripts/graded_raw_ratio_reliability_test.py): il filtro sopra vede solo
salti/volatilita'. Una carta puo' avere una serie grade9 liscia ma
PERSISTENTEMENTE troppo bassa (raichu_14: 107,60€ mostrati, 250€ reali
trovati dall'utente, tutto compreso) - tipico di carte vintage poco liquide
con storico PSA9 sottile che PriceCharting non riesce a prezzare bene.
Aggiunto compute_grade_raw_ratio_flags() come check indipendente, usando
cardmarket_ref_price_eur (raw, gia' in metadata, mai usato finora) come
ancora: 69 carte in piu' flaggate, e testato che escluderle MIGLIORA il
backtest (Sharpe 1,48->1,57) - falsi positivi da dato sottile, non alfa reale.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, save_metadata, load_price_matrix
from poke_quant.data.liquidity_filter import compute_reliability_flags, compute_grade_raw_ratio_flags
from poke_quant.data.unified_singles_panel import build_unified_singles_price_panel


def main():
    metadata = load_metadata()
    sealed_prices = load_price_matrix("historical_prices.csv")
    grade9_prices = load_price_matrix("historical_prices_graded_singles_grade9.csv")
    raw_prices = load_price_matrix("historical_prices_graded_singles_raw.csv")

    sealed_ids = {k for k, v in metadata.items() if v.get("type") == "sealed"}
    single_ids = {k for k, v in metadata.items() if v.get("type") == "single"}

    sealed_flags = compute_reliability_flags(sealed_prices[[c for c in sealed_prices.columns if c in sealed_ids]])
    # BUG TROVATO (progetto pilota MTG, 2026-09-26): questo controllo valutava
    # SOLO il pannello grade9 per le singole - le carte MTG (prezzo RAW, nessun
    # dato gradato) erano semplicemente ASSENTI dalle sue colonne, quindi non
    # venivano ne' promosse ne' flaggate: zero protezione dai salti di prezzo
    # estremi (lo stesso problema che ha gia' colpito il campione di controllo
    # Pokemon ampliato - vedi commit precedente). Corretto usando lo stesso
    # pannello UNIFICATO (grade9 con fallback su raw) che il fattore di scarsita'
    # userebbe davvero, cosi' ogni singola - qualunque sia la sua base di
    # prezzo - viene valutata sulla serie che conta.
    unified_prices = build_unified_singles_price_panel(grade9_prices, raw_prices)
    single_flags = compute_reliability_flags(unified_prices[[c for c in unified_prices.columns if c in single_ids]])
    ratio_flags = compute_grade_raw_ratio_flags(metadata, grade9_prices)  # solo le non-ok, vedi docstring
    flags = {**sealed_flags, **single_flags, **ratio_flags}

    n_flagged = 0
    for item_id, (ok, reason) in flags.items():
        if item_id not in metadata:
            continue
        if ok:
            metadata[item_id].pop("data_quality", None)
            metadata[item_id].pop("data_quality_reason", None)
        else:
            metadata[item_id]["data_quality"] = "thin_unreliable"
            metadata[item_id]["data_quality_reason"] = reason
            n_flagged += 1

    save_metadata(metadata)
    print(f"Asset valutati: {len(flags)} | flaggati come thin_unreliable: {n_flagged}")
    n_sealed = sum(1 for k, (ok, _) in flags.items() if not ok and metadata.get(k, {}).get("type") == "sealed")
    n_single = sum(1 for k, (ok, _) in flags.items() if not ok and metadata.get(k, {}).get("type") == "single")
    print(f"  di cui sealed: {n_sealed} | single: {n_single} (valutate sul pannello grade9, non su historical_prices.csv)")


if __name__ == "__main__":
    main()
