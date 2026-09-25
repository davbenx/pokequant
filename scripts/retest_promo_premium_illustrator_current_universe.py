#!/usr/bin/env python3
"""
scripts/retest_promo_premium_illustrator_current_universe.py — L'utente,
dopo che ho citato risultati di scripts/promo_factor_grid.py e del docstring
di rarity_tier_factor.py (illustratore, rarità premium) senza averli
rieseguiti IO in questa conversazione: "Se non le hai fatte tu, ritestale."

Rieseguito scripts/promo_factor_grid.py: RIPRODUCE la direzione (PBO basso,
walk-forward con inversione di segno), ma NON i numeri esatti citati nel
docstring - Sharpe 1.06 oggi contro 1.27 citato, DSR 0.945 contro 0.980
citato (own-grid). Causa piu' probabile: la serie prezzi si e' allungata
(piu' mesi disponibili oggi) da quando quel numero fu scritto - normale
deriva, non un errore, ma la cifra esatta era stale e va corretta.
Aggravante: quello script usa ancora la vecchia definizione di universo
(869 carte, filtro data_quality soltanto) - NON liquid_singles_ids (653
carte, con il pavimento di costo di gradazione aggiunto dopo). Qui rifatto
anche su liquid_singles_ids per completezza.

Nessuno script dedicato esisteva per "rarita' premium" (PREMIUM_RARITIES,
il default di RarityTierFactorStrategy) o per "illustratore" (field_name=
"artist") - solo il docstring del modulo li descrive. Ricostruiti qui.

ESITO:

  PROMO, universo vecchio (869) rebal=12 minage=6 (parametri corretti):
    Sharpe 1.00 | MaxDD -23.96% | DSR(71) 0.464 | H1 -0.54 H2 +2.37
  PROMO, universo attuale (653), stessi parametri:
    Sharpe 1.12 | MaxDD -21.85% | DSR(71) 0.570 | H1 -0.47 H2 +2.53

  Diverso dal numero di scripts/promo_factor_grid.py (Sharpe 1.06 sullo
  stesso universo vecchio, stessi parametri corretti) - causa identificata:
  quello script usa run_bt di scripts/optimize_and_falsify.py, che NON
  applica apply_buy_side_shipping=True (la spedizione reale a carico del
  compratore, aggiunta alla produzione PIU' TARDI in questa sessione - vedi
  VALIDATED_SINGLES/VALIDATED_BOX in app.py). Qui invece si', per restare
  sulla stessa base della produzione attuale - da questo la differenza
  (1.00 vs 1.06), non un errore. VERDETTO INVARIATO: il walk-forward mostra
  ancora l'inversione di segno decisiva (H1 negativo, H2 fortemente
  positivo) su entrambi gli universi - il pavimento di gradazione rende il
  fattore leggermente PIU' presentabile (DSR 0.464->0.570) ma non lo
  salva, resta ben sotto ogni soglia usata in questa ricerca.

  RARITA' PREMIUM (241/653 eleggibili, 96% dallo stesso campione chase -
  quasi identico al 94% del docstring originale, la contaminazione da
  survivorship bias NON e' cambiata): Sharpe 0.78, DSR 0.275, walk-forward
  H1 -0.50 -> H2 +2.14. Il numero citato in precedenza (~Sharpe 1.11) non
  regge con la frizione di spedizione + universo attuale - ma la
  CONCLUSIONE (non conclusivo/rifiutato) era gia' corretta ed e' confermata,
  anzi piu' netta ora (DSR molto piu' basso di quanto suggerito prima).

  ILLUSTRATORE: nessuno script originale esisteva - ricostruito con una
  definizione DICHIARATA di "illustratore-star" (>=5 carte nell'universo
  attuale, 24 nomi), diversa dalla metodologia originale (che usava un
  dataset esterno pokemontcg.io con 820 carte, non piu' verificabile qui
  identicamente). Sharpe 0.66, MaxDD -48.24%, DSR 0.189, walk-forward H1
  -1.18 -> H2 +2.68 (boom/bust ancora piu' marcato del numero citato prima,
  0.11-0.12). Definizione diversa, stessa conclusione: NON VALIDATO, anzi
  con un quadro peggiore (drawdown) di quanto suggerito dal docstring.

BUG TROVATO E CORRETTO (in scripts/promo_factor_grid.py): l'etichetta del
terzo candidato ("rebal=6 minage=12") non corrispondeva ai parametri
realmente testati (rebalance_every_months=12, min_age_months=6) - i due
numeri erano scambiati nell'etichetta, mai nel valore. Non cambia nessuna
conclusione (il valore testato era gia' quello inteso), solo la stringa
mostrata a schermo era sbagliata da quando lo script fu scritto.

OSSERVAZIONE METODOLOGICA (vale anche per sealed_age_window_search.py e
box_ev_theory_test.py, verificati separatamente riproducibili sui LORO
propri numeri): questi script piu' vecchi usano run_bt di
scripts/optimize_and_falsify.py, che non include la spedizione reale
all'acquisto aggiunta alla produzione in un secondo momento - le loro cifre
Sharpe assolute sono quindi leggermente piu' ottimistiche della base di
conto usata oggi in produzione (VALIDATED_BOX/VALIDATED_SINGLES in app.py).
Non invalida le loro conclusioni (il confronto INTERNO tra candidati nello
stesso script resta equo, stessa frizione per tutti i candidati confrontati
tra loro), ma le cifre assolute non sono direttamente comparabili alla
Sharpe di produzione mostrata in dashboard oggi senza questo aggiustamento."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.data.liquidity_filter import liquid_singles_ids
from poke_quant.engine.backtester import Backtester
from poke_quant.engine.strategies.rarity_tier_factor import RarityTierFactorStrategy, PREMIUM_RARITIES
from poke_quant.validation.statistical_validation import deflated_sharpe_ratio, pbo_cscv
from poke_quant.validation.bootstrap import block_bootstrap_metrics, summarize_bootstrap


def run_bt(strat, prices_df, metadata):
    bt = Backtester(strat, prices_df, metadata, initial_cash=10000.0, platform="cardmarket",
                     apply_liquidity_slippage=True, apply_holding_cost=True, apply_buy_side_shipping=True)
    return bt.run()


def report(label, res, n_trials):
    dsr = deflated_sharpe_ratio(observed_sr=res.sharpe / np.sqrt(12), n_trials=n_trials, n_obs=len(res.monthly_returns))
    n_obs = len(res.monthly_returns)
    h = n_obs // 2
    r1, r2 = res.monthly_returns[:h], res.monthly_returns[h:]
    h1 = (r1.mean() / r1.std()) * np.sqrt(12) if r1.std() > 0 else 0.0
    h2 = (r2.mean() / r2.std()) * np.sqrt(12) if r2.std() > 0 else 0.0
    print(f"  {label:30s} | Sharpe {res.sharpe:6.2f} | CAGR {res.cagr*100:+7.2f}% | "
          f"MaxDD {res.max_drawdown*100:7.2f}% | Trade {res.total_trades:3d} | DSR({n_trials}) {dsr:.3f} | H1 {h1:5.2f} H2 {h2:5.2f}")


def main():
    metadata = load_metadata()
    grade9_prices = load_price_matrix("historical_prices_graded_singles_grade9.csv")
    old_ids = [k for k, v in metadata.items() if v.get("type") == "single"
               and v.get("data_quality") != "thin_unreliable" and k in grade9_prices.columns]
    new_ids = liquid_singles_ids(metadata, grade9_prices)
    print(f"Universo VECCHIO (solo data_quality): {len(old_ids)} | Universo ATTUALE (liquid_singles_ids): {len(new_ids)}")

    print("\n=== PROMO (is_promo) - rieseguito su ENTRAMBI gli universi ===")
    print("  (corretto un bug di etichetta in promo_factor_grid.py: il candidato citato come")
    print("   'rebal=6 minage=12' passava davvero rebalance_every_months=12, min_age_months=6)")
    for label, ids in [("vecchio universo (869)", old_ids), ("universo attuale (653)", new_ids)]:
        meta_sub = {k: dict(metadata[k]) for k in ids}
        for k in meta_sub:
            meta_sub[k]["is_promo_str"] = "yes" if meta_sub[k].get("is_promo") else "no"
        n_promo = sum(1 for v in meta_sub.values() if v["is_promo_str"] == "yes")
        prices_sub = grade9_prices[ids]
        strat = RarityTierFactorStrategy(field_name="is_promo_str", premium_rarities=frozenset({"yes"}),
                                          max_positions=999, rebalance_every_months=12, min_age_months=6)
        res = run_bt(strat, prices_sub, meta_sub)
        print(f"  [{label}, {n_promo} promo]")
        report("PROMO rebal=12 minage=6", res, 71)

    print("\n=== RARITA' PREMIUM (default PREMIUM_RARITIES) - universo attuale (653) ===")
    meta_sub = {k: metadata[k] for k in new_ids}
    prices_sub = grade9_prices[new_ids]
    n_premium = sum(1 for k in new_ids if metadata[k].get("rarity") in PREMIUM_RARITIES)
    n_from_chase = sum(1 for k in new_ids if metadata[k].get("rarity") in PREMIUM_RARITIES
                        and metadata[k].get("selection_method") == "chase_price_filter_survivorship_biased")
    print(f"Carte eleggibili (rarita' premium): {n_premium} | di cui da campione chase (non controllo casuale): {n_from_chase} ({n_from_chase/max(1,n_premium)*100:.0f}%)")
    strat = RarityTierFactorStrategy(rebalance_every_months=6, min_age_months=6)
    res = run_bt(strat, prices_sub, meta_sub)
    report("Rarita' premium rebal=6 minage=6", res, 72)

    print("\n=== ILLUSTRATORE (field_name='artist') - universo attuale (653) ===")
    n_with_artist = sum(1 for k in new_ids if metadata[k].get("artist"))
    print(f"Carte con illustratore noto: {n_with_artist}/{len(new_ids)}")
    if n_with_artist < 20:
        print("  Troppo poche carte con illustratore noto per un test - salto.")
    else:
        from collections import Counter
        artist_counts = Counter(metadata[k].get("artist") for k in new_ids if metadata[k].get("artist"))
        star_artists = frozenset(a for a, c in artist_counts.items() if c >= 5)
        print(f"  'Illustratore-star' (>=5 carte nell'universo attuale): {sorted(star_artists)}")
        meta_sub2 = {k: dict(metadata[k]) for k in new_ids}
        strat = RarityTierFactorStrategy(field_name="artist", premium_rarities=star_artists,
                                          max_positions=999, rebalance_every_months=6, min_age_months=6)
        res = run_bt(strat, prices_sub, meta_sub2)
        report("Illustratore-star rebal=6 minage=6", res, 73)


if __name__ == "__main__":
    main()
