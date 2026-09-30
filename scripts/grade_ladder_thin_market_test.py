#!/usr/bin/env python3
"""
scripts/grade_ladder_thin_market_test.py — Calibrazione e verifica del
controllo "fermo dopo un salto" per le serie storiche per-grado di
data_cache/grade_ladder_prices.json (poke_quant/data/liquidity_filter.py::
compute_grade_ladder_reliability_flags).

BUCO STRUTTURALE TROVATO (l'utente, dopo aver segnalato "mercato sottile" su
una CGC 9.5 valutata col Valutatore Slab: "tappa il buco del filtro mercato
sottile"): compute_thin_market_drift_flags() guarda SOLO il pannello
grade9(PSA9)/raw - quando l'evaluator trova un dato REALE per un grado
specifico (9.5/10/8/7) sulla pagina PriceCharting dedicata (data_cache/
grade_ladder_prices.json, storico mensile per tier), quel numero non passava
da NESSUN controllo di attendibilita'. Caso reale: Charizard & Braixen-GX
#212 [Cosmic Eclipse], tier grade9_5 -> 92,21€(mag) 207,06€(giu, +124% in un
mese) 193,74€(lug) 203,55€(ago) 203,55€(set, IDENTICO al mese prima).
Ne' compute_reliability_flags (soglia salto singolo 200%) ne'
compute_thin_market_drift_flags (soglia drift vs raw 3,0x su finestra fissa
a 6 mesi -> qui solo 1,90x) l'avrebbero preso.

CALIBRAZIONE (PRIMA di guardare l'effetto sul caso che ha motivato il
controllo): vedi il blocco di commenti sopra check_grade_ladder_tier_reliable
in liquidity_filter.py per il ragionamento completo. Sintesi: "salto massimo"
e "fermo da N mesi" presi singolarmente sono troppo comuni per essere
segnali utilizzabili su queste serie (mediana salto singolo mensile gia'
1,6-2,0x; 23-34% dell'universo fermo da 2+ mesi a seconda del tier) - il
segnale specifico e' la LORO COMBINAZIONE (fermo DOPO un salto, non ancora
confermato ne' smentito). Soglie adottate al 90° percentile per tier
(dichiarato esplicitamente: qui la coda e' meno estrema che nel check
grade9/raw, un taglio al 97,5° lascerebbe passare la classe di caso che ha
motivato il controllo).

ESITO: vedi output.
"""

import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.liquidity_filter import (
    compute_grade_ladder_reliability_flags, GRADE_LADDER_TIERS_CHECKED,
    GRADE_LADDER_FROZEN_JUMP_CUTOFF, GRADE_LADDER_FROZEN_MIN_MONTHS, GRADE_LADDER_JUMP_WINDOW_MONTHS,
)
from poke_quant.data.storage import load_metadata

GRADE_LADDER_FILE = Path(__file__).resolve().parent.parent / "data_cache" / "grade_ladder_prices.json"


def _frozen_jump_signal(series: dict, window: int) -> tuple:
    dates = sorted(series.keys())
    vals = [series[d] for d in dates]
    if len(vals) < window + 1:
        return None
    recent = vals[-(window + 1):]
    run = 1
    i = len(recent) - 1
    while i > 0 and recent[i] == recent[i - 1]:
        run += 1
        i -= 1
    if run < GRADE_LADDER_FROZEN_MIN_MONTHS:
        return None
    max_jump = 1.0
    for j in range(1, len(recent)):
        if recent[j - 1] > 0:
            max_jump = max(max_jump, recent[j] / recent[j - 1], recent[j - 1] / recent[j])
    return run, max_jump


def main():
    grade_ladder = json.loads(GRADE_LADDER_FILE.read_text(encoding="utf-8"))
    metadata = load_metadata()

    print("1. Calibrazione per-tier (mediana e code della distribuzione 'fermo dopo salto')")
    print("   sull'intero universo, PRIMA di guardare l'effetto sul caso motivante:\n")
    for tier in GRADE_LADDER_TIERS_CHECKED:
        signals = []
        for item_id, tiers in grade_ladder.items():
            series = tiers.get(tier)
            if not series:
                continue
            r = _frozen_jump_signal(series, GRADE_LADDER_JUMP_WINDOW_MONTHS)
            if r:
                signals.append(r[1])
        if not signals:
            continue
        arr = np.array(signals)
        p50 = np.percentile(arr, 50)
        p90 = np.percentile(arr, 90)
        p975 = np.percentile(arr, 97.5)
        print(f"   {tier:10s} n_fermi={len(signals):4d}  mediana={p50:.2f}x  p90={p90:.2f}x  "
              f"p97.5={p975:.2f}x  soglia_adottata={GRADE_LADDER_FROZEN_JUMP_CUTOFF[tier]:.2f}x")

    print("\n2. Applicazione del controllo calibrato a tutto l'universo:")
    flags = compute_grade_ladder_reliability_flags(grade_ladder)
    print(f"   Combinazioni (carta, grado) flaggate: {len(flags)} su "
          f"{sum(len(v) for v in grade_ladder.values())} serie totali")
    by_tier = {}
    for key in flags:
        tier = key.rsplit(":", 1)[1]
        by_tier[tier] = by_tier.get(tier, 0) + 1
    print(f"   Per tier: {by_tier}")

    print("\n3. Verifica sul caso motivante (Charizard & Braixen-GX #212, grade9_5):")
    key = "charizard_braixen_gx_212:grade9_5"
    if key in flags:
        print(f"   -> FLAGGATO: {flags[key][1]}")
    else:
        print("   -> NON flaggato (verificare la calibrazione)")

    print("\n4. Prime 15 combinazioni flaggate (campione, non esaustivo):")
    for key, (_, reason) in sorted(flags.items())[:15]:
        item_id = key.rsplit(":", 1)[0]
        name = metadata.get(item_id, {}).get("name", item_id)
        print(f"   {name:35s} [{key.rsplit(':', 1)[1]:9s}] {reason}")


if __name__ == "__main__":
    main()
