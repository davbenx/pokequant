#!/usr/bin/env python3
"""
scripts/grade_correlation_test.py — Testa l'ipotesi "il trend tra RAW e diversi
gradi e' molto simile, quindi a parita' di capitale comprare 2x Grado 7 invece
di 1x Grado 9 e' equivalente" (richiesta esplicita dell'utente).

Distingue esplicitamente due domande diverse, spesso confuse:
  1. La media cross-sezionale (l'INDICE del mercato Grado7 vs l'INDICE Grado9)
     si muove insieme? SI', moderatamente (0,3-0,7) - gia' visto in sessione.
  2. PER LA STESSA CARTA, il suo prezzo Grado7 mese-per-mese segue il suo
     prezzo Grado9? Questa e' la domanda che conta per l'ipotesi "equivalenza
     a parita' di capitale" - e la risposta e' NO.

Dati: data_cache/grade_ladder_prices.json, 151 carte campionate a caso (seed
fisso, scripts/fetch_grade_ladder.py, SAMPLE_SIZE=150 - ampliato da 40 su
richiesta esplicita "procedi" per un test piu' potente).

ESITO: IPOTESI FALSIFICATA. Su 489 coppie carta-grado con >=12 mesi comuni:
  - Grado7 vs Grado9 (stessa carta): correlazione mensile MEDIANA 0,13
    (Q1=0,01, Q3=0,26), 22% delle carte con correlazione NEGATIVA. Rapporto di
    volatilita' 1,02x (non piu' stabile).
  - Grado8 vs Grado9: mediana 0,24. Grado9,5 vs Grado9: 0,63 (piu' vicini,
    fasce adiacenti). PSA10 vs Grado9: mediana 0,17, volatilita' 2,3x piu' alta.

0,13 di correlazione non e' "molto simile" - e' quasi rumore. Il motivo:
ogni fascia di grado ha un mercato sottile a se' per ciascuna carta (poche
vendite/mese in QUELLA fascia specifica) - il "trend simile" visibile nel
grafico aggregato e' una media su tante carte nel tempo, non il comportamento
mese-per-mese di una singola carta. Coerente con l'esito di
scripts/grade_spread_test.py (arbitraggio sullo spread tra gradi, PBO 78,6%,
stesso meccanismo di fondo).

CONSEGUENZA PRATICA: comprare Grado7/8/PSA10 non e' "la stessa scommessa a
buon mercato" - e' una scommessa diversa e debolmente relazionata con quella
su cui il fattore scarsita' e' validato (solo su dati Grado9 mensili).
Estendere il segnale BUY ad altri gradi senza validarlo separatamente
significherebbe estrapolare un risultato validato a condizioni mai testate.

Nessuna modifica alla produzione: il fattore scarsita' resta unicamente su
Grado9.
"""

import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LADDER_FILE = Path(__file__).resolve().parent.parent / "data_cache" / "grade_ladder_prices.json"
TIERS = ["grade7", "grade8", "grade9_5", "psa10"]
MIN_MONTHS = 12


def main():
    d = json.loads(LADDER_FILE.read_text())
    print(f"Carte nel campione: {len(d)}")

    records = []
    for item_id, td in d.items():
        if "grade9" not in td or len(td["grade9"]) < 24:
            continue
        s9 = pd.Series(td["grade9"]); s9.index = pd.to_datetime(s9.index); s9 = s9.sort_index()
        ret9 = s9.pct_change().dropna()
        for t in TIERS:
            if t not in td or len(td[t]) < 24:
                continue
            st = pd.Series(td[t]); st.index = pd.to_datetime(st.index); st = st.sort_index()
            rett = st.pct_change().dropna()
            common = ret9.index.intersection(rett.index)
            if len(common) < MIN_MONTHS:
                continue
            corr = ret9.loc[common].corr(rett.loc[common])
            vol_ratio = rett.loc[common].std() / ret9.loc[common].std() if ret9.loc[common].std() > 0 else np.nan
            records.append({"item_id": item_id, "tier": t, "n_months": len(common),
                             "corr_vs_grade9": corr, "vol_ratio_vs_grade9": vol_ratio})

    df = pd.DataFrame(records)
    print(f"Coppie carta-grado con almeno {MIN_MONTHS} mesi comuni: {len(df)}\n")

    print(f"{'tier':10s} {'n':>4s} {'corr_mediana':>12s} {'corr_Q1':>8s} {'corr_Q3':>8s} {'%negativa':>10s} {'vol_ratio_mediana':>18s}")
    for t in TIERS:
        sub = df[df.tier == t]
        if sub.empty:
            continue
        c = sub["corr_vs_grade9"]
        print(f"{t:10s} {len(sub):4d} {c.median():12.2f} {c.quantile(.25):8.2f} {c.quantile(.75):8.2f} "
              f"{100*(c<0).mean():9.0f}% {sub['vol_ratio_vs_grade9'].median():18.2f}")


if __name__ == "__main__":
    main()
