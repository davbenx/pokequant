#!/usr/bin/env python3
"""
scripts/psa_blend_premium_calibration.py — Documenta il ragionamento dietro
PSA_BLEND_PREMIUM_FACTOR (poke_quant/slabs/grading_multipliers.py), introdotto
dopo che l'utente ha segnalato: "trovo molte slab ben sopra il prezzo max Edge
sul mercato europeo... valuta se questo è calcolato correttamente."

PROBLEMA: PriceCharting NON separa il prezzo per casa di gradazione sotto il
Grado 10 (verificato dal vivo in grading_company_crossover_arbitrage_test.py,
2026-09-25, fetch reale su 3 carte: "Ungraded/Grade 1.../Grade 9/Grade 9.5/
PSA 10/CGC 10/SGC 10/BGS 10/..." - solo il Grado 10 ha split per compagnia).
Il modello legge quella colonna "graded"/Grado 9 e la tratta come il prezzo
PSA (is_grade_benchmark_price=True, comp_rel(PSA)=1.0 esatto per definizione)
- ma e' un BLEND cross-company, non un prezzo PSA puro.

EVIDENZA REALE RACCOLTA (2026-10-02, 5 inserzioni indipendenti per lo stesso
identico grado/compagnia, Dragonite-EX #106 PSA 9):
  eBay (venditore 189 feedback, 100% positivi): 94,30€ (solo carta, no sped.)
  Vinted: 110€ / 125€ / 160€ / 200€
Benchmark modello PRIMA del fix: 86,07€ (Fair Value) - anche l'inserzione
PIU' economica (eBay, 94,30€) era +9,6% sopra, nonostante grado e compagnia
corretti (verificato: non e' il bug Raichu, 1st Edition vs Unlimited - qui
non c'e' discrepanza di stampa/variante).

FONTE CITATA (same_grade_crossover_arbitrage_test.py, pokeprice.gg
2026-05-21): "CGC slabs trade at a small discount to PSA in the resale
market right now — typically 10-20% less for the same grade — purely
because PSA has the bigger buyer pool."

RAGIONAMENTO (STIMA, non calibrazione statistica - dichiarato esplicitamente,
un solo caso reale non e' un campione): se PSA e' la maggioranza (non la
totalita') dei volumi di sottomissione per una chase card tipica, e le altre
case scambiano ~10-20% sotto PSA, il blend PriceCharting risulta diluito
verso il basso di circa 6-12 punti percentuali rispetto al vero prezzo PSA.
PSA_BLEND_PREMIUM_FACTOR = 1.08 e' una stima CONSERVATIVA dentro quel range,
coerente con (non adattata a forza su) l'unico riscontro reale disponibile.

COME AGGIORNARE QUESTA STIMA: se emergono altri comp reali (altre carte,
altre piattaforme, idealmente prezzi di VENDITA realizzata non solo ask),
ricalcolare qui sotto il rapporto osservato e confrontarlo con 1.08 - se piu'
di 2-3 casi reali convergono stabilmente su un valore diverso, aggiornare la
costante in grading_multipliers.py con lo stesso standard di questo file.

ESITO: vedi output (impatto su Fair Value/Tetto Max Edge per il caso reale
raccolto, PRIMA e DOPO il fix).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.slabs.grading_multipliers import adjust_price_for_grading, PSA_BLEND_PREMIUM_FACTOR, Era

REAL_LISTINGS_DRAGONITE_EX_106_PSA9 = [
    ("eBay (tcgcollectibles-nightspear89, 189 feedback)", 94.30),
    ("Vinted #1", 110.00),
    ("Vinted #2", 125.00),
    ("Vinted #3", 160.00),
    ("Vinted #4", 200.00),
]

BLEND_PRICE_DRAGONITE_EX_106 = 86.07  # PriceCharting Grado 9 reale (blend), verificato dal vivo 2026-10-02


def main():
    print(f"PSA_BLEND_PREMIUM_FACTOR adottato: {PSA_BLEND_PREMIUM_FACTOR:.2f}x\n")

    print("Fair Value PRIMA del fix (blend grezzo trattato come prezzo PSA):")
    fv_before = BLEND_PRICE_DRAGONITE_EX_106
    print(f"  {fv_before:.2f} €\n")

    fv_after, sc_after, _ = adjust_price_for_grading(
        base_psa_price_eur=BLEND_PRICE_DRAGONITE_EX_106, company="PSA", grade="9", era=Era.MID_ERA,
        is_grade_benchmark_price=True,
    )
    print(f"Fair Value DOPO il fix: {fv_after:.2f} € (Tetto Max Edge: {sc_after:.2f} €)\n")

    print("Confronto con le 5 inserzioni reali indipendenti (Dragonite-EX #106 PSA 9):")
    print(f"  {'Venditore':45s} {'Prezzo':>10s} {'Gap vs FV pre-fix':>20s} {'Gap vs FV post-fix':>20s}")
    for label, price in REAL_LISTINGS_DRAGONITE_EX_106_PSA9:
        gap_before = (price / fv_before - 1.0) * 100.0
        gap_after = (price / fv_after - 1.0) * 100.0
        print(f"  {label:45s} {price:10.2f} {gap_before:19.1f}% {gap_after:19.1f}%")

    print(f"\nGap minimo (eBay, inserzione piu' economica e piu' affidabile - venditore con storico):")
    min_price = min(p for _, p in REAL_LISTINGS_DRAGONITE_EX_106_PSA9)
    print(f"  Prima del fix: {(min_price/fv_before-1.0)*100:+.1f}%  |  Dopo il fix: {(min_price/fv_after-1.0)*100:+.1f}%")
    print("\nNOTA: il fix non elimina il gap (non e' questo l'obiettivo - resta normale che la")
    print("maggioranza delle inserzioni superi il tetto, per design del filtro edge-preserving),")
    print("ma lo riduce dal +9,6% misurato sull'unico comp piu' affidabile a un residuo coerente")
    print("con la normale dispersione ask-price su un mercato sottile, non con un errore sistematico.")


if __name__ == "__main__":
    main()
