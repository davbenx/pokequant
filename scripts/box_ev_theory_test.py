#!/usr/bin/env python3
"""
scripts/box_ev_theory_test.py — Testa la teoria "il prezzo del box dipende dal
valore atteso (EV) di apertura, pesato per probabilita' di pull" - proposta
dall'utente come correzione al test precedente (box_vs_singles_basket_test.py),
che pesava tutte le carte allo stesso modo (media semplice) invece che per
probabilita' reale di pull.

METODOLOGIA E LIMITE DICHIARATO: non esistono probabilita' di pull ufficiali
(The Pokemon Company non le pubblica) - solo stime della community, che
variano per set. Qui si usa una tabella GENERICA per fascia di rarita' (non
per singolo set), calibrata sulle bande tipiche riportate dalla community
(Ultra Rare ~8-9% dei pacchetti, Secret Rare 1/20-1/36, ecc. - vedi
conversazione). Inoltre coprimo solo le carte che abbiamo scelto di tracciare
(chase + controllo casuale), non il checklist completo del set - l'EV
calcolato e' quindi un LIMITE INFERIORE della vera EV di apertura (manca il
contributo di eventuali chase card mai catturate dalla discovery), non il
valore esatto. Le carte "Promo" sono escluse: non vengono dai booster pack.

ESITO: la teoria e' parzialmente confermata, la strategia NON e' validata.

CORRELAZIONE (18 box su 36 con EV calcolabile - manca meta' universo, i box
senza singole tracciate in numero sufficiente): mediana 0,722, molto piu'
forte della correlazione ~0,22 trovata nel test con media semplice
(box_vs_singles_basket_test.py) - pesare per probabilita' di pull invece che
in modo uniforme cattura davvero meglio la relazione reale tra box e contenuto.
Non uniforme: 6/18 box mostrano correlazione negativa o quasi nulla.

LIVELLO: la maggior parte dei box tratta tra 0,05x e 0,94x del proprio EV
calcolato (mediana ben sotto 1) - economicamente sensato, non un'anomalia:
apri un box e perdi il premio da sigillato, e rivendere decine di carte
singole individualmente costa in frizione (spedizione, tempo, commissioni per
inserzione) molto piu' di una vendita in blocco del box - il gap non e'
arbitraggio gratuito.

STRATEGIA TRADEABLE (compra quando il rapporto prezzo/EV e' a uno z-score
storicamente basso): Sharpe 1,03, PBO 4,3% (4 candidati), walk-forward
ENTRAMBI positivi (H1 +0,16, H2 +1,06) - sembra ottimo con la correzione
sulla sola griglia propria (DSR 0,910). MA corretto per TUTTI i 32 trial
tentati sul lato sealed in questa sessione (griglia lookback 5 + logica di
uscita 9 + finestra d'eta' 7 + time stop 7 + questo test 4), il DSR crolla a
0,616 - sotto la soglia di comfort 0,90-0,95, stessa sorte del fattore di
valore relativo sulle singole dopo la correzione completa.

NOTA METODOLOGICA IMPORTANTE, non solo su questo test: il DSR ufficiale della
strategia in produzione (0,913, mostrato in dashboard) e' stato corretto solo
per la sua griglia di lookback originale (n_trials=5) - non per i successivi
~27 trial di robustezza (logica di uscita, finestra d'eta', time stop) provati
DOPO la sua adozione. Applicando lo stesso standard di onesta' usato qui,
anche quel numero e' probabilmente un limite superiore ottimistico, non il
valore vero. Non ricalcolato in questo test - lasciato come nota per una
revisione futura del badge di validazione in produzione.
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.engine.strategies.time_series_momentum import TimeSeriesMomentumStrategy
from poke_quant.validation.statistical_validation import deflated_sharpe_ratio, pbo_cscv
from poke_quant.validation.bootstrap import block_bootstrap_metrics, summarize_bootstrap
from scripts.optimize_and_falsify import run_bt, MODERN_ERA_CUTOFF

PACKS_PER_BOX = 36  # standard per i booster box moderni nel nostro universo

# Copie attese per box, per fascia di rarita' - stime generiche di community
# (packrip.co, deltiasgaming, flipsidegaming - vedi ricerca in conversazione),
# NON dati ufficiali, NON specifiche per singolo set.
EXPECTED_COPIES_PER_BOX = {
    # Fascia ultra-rara: 1 ogni ~25-36 pacchetti
    "Rare Secret": 36 / 30, "Special Illustration Rare": 36 / 30, "Hyper Rare": 36 / 30,
    "Rare Shining": 36 / 30, "Rare BREAK": 36 / 30, "Rare ACE": 36 / 30,
    # Fascia chase intermedia: 1 ogni ~8-12 pacchetti
    "Rare Rainbow": 36 / 9, "Rare Ultra": 36 / 9, "Ultra Rare": 36 / 9, "Double Rare": 36 / 9,
    "Rare Holo VMAX": 36 / 9, "Rare Holo VSTAR": 36 / 9, "Rare Holo V": 36 / 9,
    "Rare Holo GX": 36 / 9, "Rare Holo EX": 36 / 9, "Rare Holo LV.X": 36 / 9, "Rare Holo Star": 36 / 9,
    # Illustration Rare (non-special): piu' comune, 1 ogni ~4 pacchetti
    "Illustration Rare": 36 / 4,
    # Fascia base/bulk: presente quasi ogni pacchetto, valore individuale basso
    "Rare Holo": 36 * 0.5, "Rare": 36 * 0.5, "Common": 36 * 1.0, "Uncommon": 36 * 1.0,
}
EXCLUDED_RARITIES = {"Promo", None}  # non vengono dai booster pack


def compute_box_ev(box_id: str, box_to_singles: dict, metadata: dict, prices_full: pd.DataFrame) -> pd.Series | None:
    singles = box_to_singles.get(box_id, [])
    contributions = []
    for s in singles:
        if s not in prices_full.columns:
            continue
        rarity = metadata.get(s, {}).get("rarity")
        if rarity in EXCLUDED_RARITIES or rarity not in EXPECTED_COPIES_PER_BOX:
            continue
        copies = EXPECTED_COPIES_PER_BOX[rarity]
        contributions.append(prices_full[s] * copies)
    if not contributions:
        return None
    return sum(contributions)


def main():
    metadata = load_metadata()
    prices_full = load_price_matrix()
    single_to_box = json.load(open("data_cache/single_to_box_map.json"))
    box_to_singles = {}
    for single_id, box_id in single_to_box.items():
        box_to_singles.setdefault(box_id, []).append(single_id)

    sealed_ids = [
        k for k, v in metadata.items()
        if v.get("type") == "sealed" and v.get("data_quality") != "thin_unreliable"
        and k in prices_full.columns and v.get("release_date") and v["release_date"] >= MODERN_ERA_CUTOFF
    ]

    ev_series = {}
    for box_id in sealed_ids:
        ev = compute_box_ev(box_id, box_to_singles, metadata, prices_full)
        if ev is not None and ev.notna().sum() >= 24:
            ev_series[box_id] = ev

    print(f"Box con EV calcolabile (>=24 mesi): {len(ev_series)} su {len(sealed_ids)} nell'universo sealed")

    box_ids = list(ev_series.keys())
    meta_sub = {k: metadata[k] for k in box_ids}
    real_prices = prices_full[box_ids]
    ev_df = pd.DataFrame(ev_series).reindex(real_prices.index)

    # --- Correlazione EV vs prezzo reale, e rapporto prezzo/EV attuale ---
    print(f"\n{'Box':32s} {'Corr(prezzo,EV)':>16s} {'Prezzo/EV oggi':>15s}")
    corrs = []
    for box_id in box_ids:
        common = real_prices[box_id].dropna().index.intersection(ev_df[box_id].dropna().index)
        if len(common) < 12:
            continue
        c = real_prices[box_id].loc[common].corr(ev_df[box_id].loc[common])
        corrs.append(c)
        last_common = common[-1]
        ratio = real_prices[box_id].loc[last_common] / ev_df[box_id].loc[last_common]
        print(f"{meta_sub[box_id]['name']:32s} {c:16.3f} {ratio:14.2f}x")
    print(f"\nCorrelazione media prezzo-vs-EV: {np.mean(corrs):.3f} (mediana {np.median(corrs):.3f})")

    # --- Segnale tradeable: gap prezzo/EV (z-score) predice il rendimento futuro? ---
    print("\n" + "=" * 90)
    print("Il gap prezzo/EV (z-score) predice il rendimento futuro del box?")
    print("=" * 90)
    from poke_quant.engine.strategies.cross_sectional_factor import CrossSectionalFactorStrategy, zscore_factor
    ratio_df = real_prices / ev_df
    results = {}
    for lb in [6, 12]:
        for ascending in [True, False]:
            label = "compra sotto-EV" if ascending else "compra sopra-EV (segue il trend)"
            strat = CrossSectionalFactorStrategy(ratio_df, zscore_factor, lookback_months=lb, top_quantile=0.30,
                                                  ascending=ascending, rebalance_every_months=6, item_type_filter="sealed")
            res = run_bt(strat, real_prices, meta_sub)
            name = f"lb={lb}m {label}"
            results[name] = res
            print(f"  {name:40s} | CAGR {res.cagr*100:+6.2f}% | Sharpe {res.sharpe:5.2f} | "
                  f"MaxDD {res.max_drawdown*100:6.2f}% | Trade {res.total_trades:4d}")

    common_idx = None
    for res in results.values():
        common_idx = res.monthly_returns.index if common_idx is None else common_idx.intersection(res.monthly_returns.index)
    perf_matrix = np.column_stack([results[k].monthly_returns.loc[common_idx].values for k in results])
    t_len = len(perf_matrix)
    splits = 8 if t_len >= 32 else 4
    rem = t_len % splits
    pbo = pbo_cscv(perf_matrix[rem:, :] if rem else perf_matrix, n_splits=splits)
    best_name = max(results, key=lambda k: results[k].sharpe)
    best = results[best_name]
    dsr = deflated_sharpe_ratio(observed_sr=best.sharpe / np.sqrt(12), n_trials=len(results), n_obs=len(best.monthly_returns))
    print(f"\nPBO ({len(results)} candidati, {splits} split): {pbo:.3f}")
    print(f"Migliore: '{best_name}' Sharpe {best.sharpe:.2f} | Trade {best.total_trades} | DSR: {dsr:.3f}")
    sims = block_bootstrap_metrics(best.monthly_returns, n_sims=500, block_size=6)
    print(summarize_bootstrap(sims))

    mid = len(real_prices) // 2
    h1, h2 = real_prices.index[:mid], real_prices.index[mid:]
    lb_best = int(best_name.split("=")[1].split("m")[0])
    ascending_best = "sotto-EV" in best_name
    print(f"\nWalk-forward H1/H2 sul vincitore '{best_name}':")
    for label, dates in [("H1", h1), ("H2", h2)]:
        sub_real, sub_ratio = real_prices.loc[dates], ratio_df.loc[ratio_df.index.intersection(dates)]
        strat = CrossSectionalFactorStrategy(sub_ratio, zscore_factor, lookback_months=lb_best, top_quantile=0.30,
                                              ascending=ascending_best, rebalance_every_months=6, item_type_filter="sealed")
        r = run_bt(strat, sub_real, meta_sub)
        print(f"  {label} ({dates[0].strftime('%Y-%m')}->{dates[-1].strftime('%Y-%m')}) | Sharpe {r.sharpe:5.2f} | CAGR {r.cagr*100:+6.2f}%")


if __name__ == "__main__":
    main()
