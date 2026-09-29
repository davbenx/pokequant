#!/usr/bin/env python3
"""
scripts/thin_market_drift_window_sensitivity_test.py — Verifica se il controllo
"mercato sottile" gia' in produzione (compute_thin_market_drift_flags, finestra
fissa a 6 mesi, soglia 3,0x - vedi scripts/thin_market_ratio_drift_test.py)
intercetta davvero i DUE casi che l'hanno motivato.

DOMANDA (dopo aver flaggato manualmente Azumarill #114 [Delta Species] a
90,25EUR come "da verificare a mano" durante la valutazione di 7 acquisti
reali): il controllo appena aggiunto flagga quella carta? RISPOSTA: NO.
azumarill_114 -> non flaggato (drift 6 mesi fisso = 1,72x, sotto soglia 3,0x).
Nemmeno clefable_1 (Jungle #1, l'altra carta della stessa lista, valutata
"BUY CONSIGLIATO") viene flaggato (drift 1,84x). Il "Clefable #1" che appariva
nell'output di thin_market_ratio_drift_test.py era in realta' un item_id
DIVERSO (clefable_1_call_of_legends_ctrl, stampa Call of Legends - drift 6,7x,
correttamente flaggato) - i 7 nomi "clefable" in metadata si sovrappongono nel
printout per nome, non per item_id.

CAUSA: il rapporto grade9/raw di Azumarill e' salito da 1,81x (feb) a 3,25x
(mar) - un salto quasi completo GIA' AVVENUTO nel primo mese della finestra di
6 mesi. Confrontare "oggi" con "esattamente 6 mesi fa" (mar, gia' post-salto)
sottostima il drift totale rispetto al confronto originale fatto a mano
(feb->set, 7 mesi, drift ~3,1x, sopra soglia). Il controllo NON e' un bug di
codice: e' un limite del design a "singolo punto fisso" quando il salto e'
concentrato nel primo mese della finestra.

TENTATIVI DI RIDISEGNO (testati PRIMA di decidere se adottarli, stesso
principio "calibra sull'universo prima di guardare l'effetto"):

1) baseline = MIN del rapporto sui 12 mesi precedenti (invece di un punto
   fisso a 6 mesi): cattura azumarill_114 (drift ora 3,10x, molto vicino al
   numero originale calcolato a mano) - MA la distribuzione sull'universo
   liquido esplode per rumore (carte raw sotto 1-2EUR: denominatore quasi
   zero -> rapporti a 3 cifre). p50 1,77x, p97,5 17,5x, p99 31,8x, max 465x
   (Ferroseed #79). Una soglia che cattura Azumarill (3,0x) sarebbe intorno
   al p75-p80 di QUESTA distribuzione - troppo permissiva, avrebbe flaggato
   >20% dell'universo. SCARTATO.

2) baseline = MEDIANA del rapporto sui 12 mesi precedenti, con un pavimento
   di prezzo raw (>=3EUR) per escludere il rumore da carte a pochi centesimi:
   distribuzione molto piu' pulita (p97,5 2,68x, p99 3,34x - vicina al design
   attuale) - MA azumarill_114 ora ha drift 1,55x (ANCORA sotto qualunque
   soglia ragionevole): usando la mediana su 12 mesi, il salto di Azumarill
   viene "diluito" da altrettanti mesi PRE-salto quanto POST-salto nella
   stessa finestra. SCARTATO (non risolve il problema che doveva risolvere).

ESITO: nessun redesign provato batte insieme (a) intercettare in modo
robusto Azumarill/Clefable Jungle E (b) restare pulito sull'universo
generale, senza costruire la soglia apposta su questi due casi (violazione
esplicita della disciplina di questa sessione: mai calibrare un parametro
guardando quale valore fa scattare il caso che l'ha motivato). Conclusione
onesta: il drift di Azumarill (1,7-3,1x secondo la definizione) e Clefable
Jungle (1,4-1,8x) sono CASI DI CONFINE, non anomalie schiaccianti nella coda
della distribuzione - coerente con l'averli gia' segnalati come "da
verificare a mano" invece che come "falso positivo certo". Il filtro
compute_thin_market_drift_flags (finestra fissa 6 mesi/3,0x) RESTA in
produzione com'e' (utile per le altre 148 carte, validato su backtest) - non
viene ne' sostituito ne' allargato. Azumarill #114 e Clefable #1 [Jungle]
restano segnalati per revisione manuale nel referto all'utente, non dal
filtro automatico.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix


def _drift_universe(grade9, raw, lookback_months, baseline_fn, raw_floor=0.0):
    metadata = load_metadata()
    drifts = {}
    for item_id, info in metadata.items():
        if info.get("type") != "single":
            continue
        if item_id not in grade9.columns or item_id not in raw.columns:
            continue
        g = grade9[item_id].dropna()
        r = raw[item_id].dropna()
        common = g.index.intersection(r.index)
        if len(common) < lookback_months + 1:
            continue
        g, r = g.loc[common], r.loc[common]
        baseline_r = r.iloc[-(lookback_months + 1):-1]
        if r.iloc[-1] < raw_floor or baseline_r.min() < raw_floor:
            continue
        ratio = (g / r).replace([np.inf, -np.inf], np.nan).dropna()
        if len(ratio) < lookback_months + 1:
            continue
        now = ratio.iloc[-1]
        baseline = baseline_fn(ratio.iloc[-(lookback_months + 1):-1])
        if baseline <= 0 or now <= 0:
            continue
        drifts[item_id] = now / baseline
    return drifts


def main():
    grade9 = load_price_matrix("historical_prices_graded_singles_grade9.csv")
    raw = load_price_matrix("historical_prices_graded_singles_raw.csv")

    configs = [
        ("MIN su 12 mesi, nessun pavimento", 12, lambda s: s.min(), 0.0),
        ("MEDIANA su 12 mesi, pavimento raw >=3EUR", 12, lambda s: s.median(), 3.0),
    ]
    for label, lookback, fn, floor in configs:
        drifts = _drift_universe(grade9, raw, lookback, fn, floor)
        vals = np.array(list(drifts.values()))
        print(f"\n=== {label} (n={len(vals)}) ===")
        for p in (50, 90, 95, 97.5, 99):
            print(f"  p{p}: {np.percentile(vals, p):.2f}")
        print(f"  max: {vals.max():.2f}")
        print(f"  azumarill_114: {drifts.get('azumarill_114')}")
        print(f"  clefable_1 (Jungle): {drifts.get('clefable_1')}")
        print(f"  clefable_1_call_of_legends_ctrl: {drifts.get('clefable_1_call_of_legends_ctrl')}")


if __name__ == "__main__":
    main()
