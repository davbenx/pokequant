#!/usr/bin/env python3
"""
scripts/residual_forward_return_test.py — L'utente ha proposto: "se in Italia
trovo tutto al prezzo sdoganato o superiore, probabilmente lo compro comunque
e lo rivendo a prezzo più alto" - cioè comprare anche quando il residuo del
fattore scarsità non è più negativo (sconto vs pari), assumendo che il prezzo
salga comunque. Testato in scripts/usa_landed_cost_edge_test.py che pagare
sistematicamente il costo sdoganato distrugge l'edge - ma quel test simula un
COSTO aggiunto, non verifica direttamente se il MECCANISMO del fattore (comprare
sconto, aspettarsi recupero) regge ancora o si inverte quando non c'è sconto.

Metodo: per ogni mese disponibile nel pannello grade9, calcola il residuo
cross-sezionale (stessa regressione di ScarcityValueFactorStrategy: log-prezzo
~ scarsità continua + controlli) per OGNI carta, poi il rendimento REALE nei
6 e 12 mesi successivi. Raggruppa per decile di residuo (decile 0 = più
scontate vs pari, decile 9 = più sovrapprezzate vs pari) e riporta il
rendimento forward medio/mediano per decile - non un costo simulato, il
rapporto vero tra "quanto sei sopra/sotto il fair value del modello" e "cosa
succede dopo" nella storia reale.

ESITO: la relazione e' negativa e in gran parte monotona (correlazione -0,10
a 6m, -0,06 a 12m) - MA con una coda: il decile piu' sovrapprezzato (9) torna
a salire. Verificato chi ci sta dentro: Gold Star (Mudkip★, Vaporeon★...),
Charizard #146, Lugia #149, Umbreon Crown Zenith - carte "grail" iconiche e
ultra-rare, non carte qualsiasi. Il premio di queste carte viene dalla fama/
desiderabilita' che il modello (rarita'/eta'/franchise) non cattura, non da
un errore di stima - coerente con la nota gia' in app.py sulla sezione
"Singole da evitare". Per una carta ORDINARIA (non un grail riconosciuto) i
deciles centrali (5-8, leggermente-moderatamente sovrapprezzata) mostrano
rendimento forward MEDIANO negativo (-0,007 a -0,038 a 12m): la carta
"tipica" pagata sopra il fair value del modello NON recupera, va sotto o
resta ferma piu' spesso di quanto salga - la media e' positiva solo perche'
trainata da pochi outlier enormi (bucket 9), non perche' la maggioranza vinca.
Non e' quindi vero in generale che "se tutti la prezzano alta, salira'
ancora" - vale solo per un sottoinsieme riconoscibile di carte iconiche, non
per il caso generico ("trovo tutto sopra al prezzo sdoganato") descritto
dall'utente.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.engine.strategies.scarcity_value_factor import ScarcityValueFactorStrategy

FORWARD_MONTHS = [6, 12]
N_BUCKETS = 10


def _snapshot_for_date(prices_full: pd.DataFrame, metadata: dict, date) -> dict:
    price_row = prices_full.loc[date]
    snap = {}
    for item_id, info in metadata.items():
        if item_id in price_row.index and price_row[item_id] > 0 and not pd.isna(price_row[item_id]):
            s = dict(info)
            s["current_price"] = float(price_row[item_id])
            snap[item_id] = s
    return snap


def main():
    metadata = load_metadata()
    prices_full = load_price_matrix("historical_prices_graded_singles_grade9.csv")
    singles_ids = [k for k, v in metadata.items() if v.get("type") == "single"
                   and v.get("data_quality") != "thin_unreliable" and k in prices_full.columns]
    meta_sub = {k: metadata[k] for k in singles_ids}
    prices_sub = prices_full[singles_ids]

    strat = ScarcityValueFactorStrategy(min_age_months=6, min_cross_section=20)

    dates = list(prices_sub.index)
    max_fwd = max(FORWARD_MONTHS)
    usable_dates = dates[:-max_fwd] if len(dates) > max_fwd else []
    # Campiona ogni 2 mesi per non sovra-pesare lo stesso "regime" con mesi
    # quasi identici e ridurre il tempo di calcolo, senza perdere copertura
    # temporale sull'intero storico disponibile.
    sample_dates = usable_dates[::2]
    print(f"Mesi campionati: {len(sample_dates)} (ogni 2 mesi, su {len(usable_dates)} disponibili)\n")

    rows = []
    for d in sample_dates:
        snap = _snapshot_for_date(prices_sub, meta_sub, d)
        residuals = strat._fit_residuals(pd.to_datetime(d), snap)
        if not residuals:
            continue
        d_pos = dates.index(d)
        for item_id, resid in residuals.items():
            cur_price = snap[item_id]["current_price"]
            for fwd in FORWARD_MONTHS:
                fwd_pos = d_pos + fwd
                if fwd_pos >= len(dates):
                    continue
                fwd_date = dates[fwd_pos]
                if item_id not in prices_sub.columns:
                    continue
                fwd_price = prices_sub.loc[fwd_date, item_id]
                if pd.isna(fwd_price) or fwd_price <= 0:
                    continue
                fwd_ret = (fwd_price - cur_price) / cur_price
                rows.append({"date": d, "item_id": item_id, "residual": resid, "fwd_months": fwd, "fwd_return": fwd_ret})

    df = pd.DataFrame(rows)
    print(f"Osservazioni totali (carta x mese x orizzonte): {len(df)}\n")

    for fwd in FORWARD_MONTHS:
        sub = df[df["fwd_months"] == fwd].copy()
        sub["bucket"] = pd.qcut(sub["residual"], N_BUCKETS, labels=False, duplicates="drop")
        agg = sub.groupby("bucket").agg(
            n=("fwd_return", "count"),
            residual_min=("residual", "min"),
            residual_max=("residual", "max"),
            fwd_return_mean=("fwd_return", "mean"),
            fwd_return_median=("fwd_return", "median"),
        )
        print(f"=== Rendimento forward a {fwd} mesi, per decile di residuo (0=più scontate, {agg.index.max()}=più sovrapprezzate) ===")
        print(agg.to_string(float_format=lambda x: f"{x:.3f}"))
        corr = sub["residual"].corr(sub["fwd_return"])
        print(f"Correlazione residuo <-> rendimento forward a {fwd}m: {corr:+.3f}\n")


if __name__ == "__main__":
    main()
