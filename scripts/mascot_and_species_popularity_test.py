#!/usr/bin/env python3
"""
scripts/mascot_and_species_popularity_test.py — L'utente ha chiesto se sono
stati testati: (1) box filtrati per fine stampa o eta' dal lancio, (2)
presenza di Pokemon/chase card specifici sui box, (3) popolarita' della
specie Pokemon sulle singole, (4) vantaggio di holo/promo/carte speciali.

INVENTARIO DI CIO' CHE ERA GIA' STATO TESTATO PRIMA DI QUESTO SCRIPT (nessuna
nuova esecuzione necessaria, solo verificato nei docstring esistenti):

  - Eta' dal lancio (finestra min/max mesi) - scripts/sealed_age_window_search.py:
    NON MIGLIORA. Baseline (nessun tetto d'eta') Sharpe 1.10 batte ogni
    finestra testata (incl. 4-14m richiesta esplicitamente in precedenza:
    Sharpe 1.02). PBO 78.6% sulla griglia - restringere l'eta' toglie
    diversificazione (24 trade -> 3-8), non aggiunge segnale.

  - EV di apertura del box (valore atteso pesato per probabilita' di pull,
    non "quale personaggio" ma "quanto vale il contenuto") -
    scripts/box_ev_theory_test.py: parzialmente confermata la correlazione
    (mediana 0.722), ma la strategia tradeable fallisce dopo la correzione
    per l'intera sessione (DSR 0.910 propria -> 0.616 corretto, sotto soglia).

  - Fattore "carte speciali" (rarita' premium: Rare Secret, Rare Rainbow,
    Rare Ultra, Special/Illustration Rare, Hyper Rare, Rare Holo VMAX/VSTAR) -
    poke_quant/engine/strategies/rarity_tier_factor.py (PREMIUM_RARITIES):
    NON CONCLUSIVO - 94% delle carte eleggibili vengono dal campione
    selezionato sul prezzo (survivorship bias), solo 6% da controllo
    casuale - il risultato (Sharpe 1.11) e' quasi certamente lo stesso bias
    di discover_chase_cards.py sotto altra etichetta, non un fattore
    indipendente verificato.

  - Fattore illustratore (field_name="artist") - stesso file: NON VALIDATO.
    Sharpe 0.11-0.12 full-sample (debole), MaxDD -60%, PBO 24.3%, DSR 0.221.
    Walk-forward H1 -1.92 -> H2 +2.51: stesso schema boom/bust di ogni altro
    fattore rotazionale su questo universo.

  - Fattore promo (field_name="is_promo_str") - stesso file: NON VALIDATO,
    il caso piu' insidioso - numeri full-sample MIGLIORI di tutta la ricerca
    (Sharpe 1.27, PBO 0.000, DSR 0.980), ma walk-forward H1 -0.95 -> H2 +1.93:
    stesso super-ciclo boom/bust, solo amplificato dalla minore liquidita' del
    sottomercato promo/SVP. Respinto per coerenza con lo standard usato ovunque.

  - "Holo" come dummy INDIPENDENTE dalla fascia di rarita': MAI testato -
    "Rare Holo" e' gia' un tier della tabella EXPECTED_COPIES_PER_BOX (stesso
    punteggio di scarsita' di "Rare" non-holo), quindi il fattore scarsita'
    lo tratta gia' come equivalente, non testato come premio ORTOGONALE.

  - "Fine della stampa" (out of print): NESSUN campo dati esiste per questo
    (verificato: game_slug/release_date/msrp/set_tier sono gli unici campi
    box, nessuna data di fine produzione) - non testabile senza raccogliere
    prima questo dato, che TPCi non pubblica in modo strutturato.

DUE CANDIDATI NUOVI, TESTATI QUI PER LA PRIMA VOLTA:

  A) Box: presenza di Charizard come chase mascot (il piu' ricorrente
     nell'universo attuale - 9/40 box, l'unico con campione abbastanza
     grande da testare senza n troppo piccolo) - filtro sull'universo della
     strategia TS Momentum validata, stesso trattamento del test sull'eta'.

  B) Singole: "specie Pokemon popolare" (Pikachu, Charizard, Mewtwo, Mew,
     Eevee/Eeveelutions, Umbreon, Sylveon, Lugia, Rayquaza, Gyarados,
     Dragonite, Gengar, Blastoise, Venusaur, Snorlax, Gardevoir - i classici
     piu' noti trasversalmente, non solo scarsi/costosi) come fattore
     categorico con lo stesso motore usato per promo/illustratore
     (RarityTierFactorStrategy, field_name generico).

ESITO:

  A) BOX - Charizard come chase mascot (n_trials 69->70, un solo esperimento):
     Baseline (tutti, 40 box)      | Sharpe 1.18 | MaxDD -11.12% | DSR 0.678 | H1 +0.84 H2 +1.84
     Solo Charizard (9 box)        | Sharpe 0.85 | MaxDD -18.11% | DSR 0.388 | H1 -0.86 H2 +1.76
     Solo NON-Charizard (31 box)   | Sharpe 1.20 | MaxDD -11.55% | DSR 0.691 | H1 +0.86 H2 +1.81

     I box con Charizard come chase mascot NON vanno meglio - vanno PEGGIO
     (Sharpe 0.85 vs 1.20, drawdown quasi il doppio, e nella prima meta' del
     periodo il momentum si INVERTE, Sharpe negativo). Ipotesi plausibile:
     Charizard e' talmente iconico che la domanda (e il prezzo) e' gia'
     strutturalmente alta ancora prima che il box salga di prezzo - il
     fattore che la strategia sfrutta (momentum, prezzo che sta SALENDO
     rispetto a 12 mesi prima) ha meno spazio su un asset gia' costoso in
     partenza per la sua fama, non per la sua scarsita' relativa. Campione
     piccolo (9 box, quasi tutti con lo stesso "franchise" Charizard - poca
     indipendenza reale tra le osservazioni) quindi DSR debole (0.388) non
     e' una prova forte in se', ma la direzione (peggio, non meglio) e
     l'inversione nella prima meta' bastano per NON raccomandare "cerca i
     box con personaggi famosi" come euristica - se qualcosa, il contrario.

  B) SINGOLE - specie Pokemon popolare (n_trials 70->71):
     rebal=6m  | Sharpe 0.71 | MaxDD -46.47% | DSR 0.224 | H1 -1.02 H2 +2.39
     rebal=3m  | Sharpe 0.70 | MaxDD -46.78% | DSR 0.219 | H1 -1.02 H2 +2.39
     rebal=12m | Sharpe 0.74 | MaxDD -45.23% | DSR 0.248 | H1 -1.02 H2 +2.46

     "Trade=0" in tutte le configurazioni non e' un bug: la specie popolare
     e' un attributo FISSO (una carta non "esce" mai dall'essere Pikachu),
     quindi nessuna posizione viene mai chiusa per uscita dal fattore -
     tutto il rendimento e' buy&hold non realizzato, mark-to-market
     (Sharpe/CAGR/MaxDD restano validi, calcolati sul NAV mensile).
     NON VALIDATO: Sharpe debole (0.70-0.74), drawdown severissimo (-45/-46%,
     il peggiore di qualunque fattore testato in questa sessione), DSR
     0.22-0.25 (ben sotto soglia), e lo STESSO identico schema boom/bust
     visto in illustratore e promo (H1 fortemente negativo -1.02, H2
     fortemente positivo +2.4) - le carte di specie iconiche hanno vissuto
     lo stesso super-ciclo 2021-2023 (crollo) / 2023-2026 (recupero) del
     resto del mercato collezionabile, amplificato (drawdown peggiore),
     non un vantaggio strutturale legato alla popolarita' del personaggio.

CONCLUSIONE: nessuno dei due candidati adottato. Charizard/mascotte famoso
sui box: la direzione e' addirittura opposta a quella intuita (peggio, non
meglio). Specie Pokemon popolare sulle singole: stesso schema boom/bust
gia' visto e respinto per promo e illustratore - conferma, non eccezione,
del pattern "niente fattore categorico rotazionale regge il walk-forward su
questo universo" trovato in questa sessione."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.data.liquidity_filter import liquid_sealed_ids, liquid_singles_ids, MAX_PRICE_TO_MSRP_RATIO
from poke_quant.engine.backtester import Backtester
from poke_quant.engine.strategies.time_series_momentum import TimeSeriesMomentumStrategy
from poke_quant.engine.strategies.rarity_tier_factor import RarityTierFactorStrategy
from poke_quant.validation.statistical_validation import deflated_sharpe_ratio

POPULAR_SPECIES = {
    "pikachu", "charizard", "mewtwo", "mew", "eevee", "vaporeon", "jolteon", "flareon",
    "espeon", "umbreon", "leafeon", "glaceon", "sylveon", "lugia", "rayquaza", "gyarados",
    "dragonite", "gengar", "blastoise", "venusaur", "snorlax", "gardevoir", "greninja",
    "lucario", "garchomp", "tyranitar", "alakazam", "arcanine", "ho-oh", "kyogre", "groudon",
}


def run_bt_box(strat, prices_df, metadata):
    bt = Backtester(strat, prices_df, metadata, initial_cash=10000.0, platform="cardmarket",
                     apply_liquidity_slippage=True, apply_holding_cost=True, apply_buy_side_shipping=True)
    return bt.run()


def report_box(label, ids, prices_full, metadata, n_trials):
    if len(ids) < 3:
        print(f"  {label:28s} | universo troppo piccolo ({len(ids)}), non testabile")
        return
    meta_sub = {k: metadata[k] for k in ids}
    prices_sub = prices_full[ids]
    strat = TimeSeriesMomentumStrategy(prices_sub, lookback_months=12, max_price_msrp_ratio=MAX_PRICE_TO_MSRP_RATIO)
    res = run_bt_box(strat, prices_sub, meta_sub)
    dsr = deflated_sharpe_ratio(observed_sr=res.sharpe / np.sqrt(12), n_trials=n_trials, n_obs=len(res.monthly_returns))
    n_obs = len(res.monthly_returns)
    h = n_obs // 2
    r1, r2 = res.monthly_returns[:h], res.monthly_returns[h:]
    h1 = (r1.mean() / r1.std()) * np.sqrt(12) if r1.std() > 0 else 0.0
    h2 = (r2.mean() / r2.std()) * np.sqrt(12) if r2.std() > 0 else 0.0
    print(f"  {label:28s} | universo {len(ids):3d} | Sharpe {res.sharpe:6.2f} | CAGR {res.cagr*100:+7.2f}% | "
          f"MaxDD {res.max_drawdown*100:7.2f}% | Trade {res.total_trades:3d} | DSR({n_trials}) {dsr:.3f} | H1 {h1:5.2f} H2 {h2:5.2f}")


def main():
    metadata = load_metadata()
    prices_full = load_price_matrix()
    box_ids = liquid_sealed_ids(metadata, prices_full)

    print("=== A) Box: presenza di Charizard come chase mascot (universo attuale, TS Momentum validato) ===")
    charizard_ids = [k for k in box_ids if "charizard" in (metadata[k].get("chase_mascot") or "").lower()]
    other_ids = [k for k in box_ids if k not in charizard_ids]
    report_box("Baseline (produzione, tutti)", box_ids, prices_full, metadata, 50)
    report_box("Solo box con Charizard", charizard_ids, prices_full, metadata, 50)
    report_box("Solo box SENZA Charizard", other_ids, prices_full, metadata, 50)

    print("\n=== B) Singole: fattore 'specie Pokemon popolare' (Pikachu, Charizard, Mewtwo, Eevee...) ===")
    grade9_prices = load_price_matrix("historical_prices_graded_singles_grade9.csv")
    singles_ids = liquid_singles_ids(metadata, grade9_prices)
    meta_sub = {k: dict(metadata[k]) for k in singles_ids}
    n_popular = 0
    for k in meta_sub:
        name_lower = meta_sub[k].get("name", "").lower()
        is_popular = any(sp in name_lower for sp in POPULAR_SPECIES)
        meta_sub[k]["is_popular_species_str"] = "yes" if is_popular else "no"
        n_popular += is_popular
    print(f"Universo: {len(singles_ids)} singole, specie popolari tra queste: {n_popular}")
    prices_sub = grade9_prices[singles_ids]

    for rebal, minage in [(6, 6), (3, 6), (12, 6)]:
        strat = RarityTierFactorStrategy(field_name="is_popular_species_str", premium_rarities=frozenset({"yes"}),
                                          max_positions=999, rebalance_every_months=rebal, min_age_months=minage,
                                          item_type_filter="single")
        bt = Backtester(strat, prices_sub, meta_sub, initial_cash=10000.0, platform="cardmarket",
                         apply_liquidity_slippage=True, apply_holding_cost=True, apply_buy_side_shipping=True)
        res = bt.run()
        dsr = deflated_sharpe_ratio(observed_sr=res.sharpe / np.sqrt(12), n_trials=71, n_obs=len(res.monthly_returns))
        n_obs = len(res.monthly_returns)
        h = n_obs // 2
        r1, r2 = res.monthly_returns[:h], res.monthly_returns[h:]
        h1 = (r1.mean() / r1.std()) * np.sqrt(12) if r1.std() > 0 else 0.0
        h2 = (r2.mean() / r2.std()) * np.sqrt(12) if r2.std() > 0 else 0.0
        print(f"  rebal={rebal:2d}m minage={minage:2d}m | Sharpe {res.sharpe:6.2f} | CAGR {res.cagr*100:+7.2f}% | "
              f"MaxDD {res.max_drawdown*100:7.2f}% | Trade {res.total_trades:3d} | DSR(71) {dsr:.3f} | H1 {h1:5.2f} H2 {h2:5.2f}")


if __name__ == "__main__":
    main()
