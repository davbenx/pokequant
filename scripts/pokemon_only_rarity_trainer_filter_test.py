#!/usr/bin/env python3
"""
scripts/pokemon_only_rarity_trainer_filter_test.py — L'utente ha chiesto:
"Verifica anche se tenere solo le carte Pokemon da rarita' rara in su o
ignorare le carte allenatore puo' migliorare la strategia".

Due candidati distinti testati separatamente (nessuno dei due era mai stato
provato prima in questa sessione):

  A) "Solo Pokemon, rarita' Rare+": esclude le carte One Piece TCG
     (franchise=one_piece, 72/653 nell'universo attuale) E le Pokemon comuni/
     non comuni (rarity in {Common, Uncommon}, 26/653) - la tesi e' che
     comuni/non comuni sono "rumore" categoriale (gia' in gran parte tolto
     dal pavimento di costo di gradazione, ma non tutto) e che One Piece
     potrebbe avere una dinamica di prezzo diversa dal Pokemon TCG (fan base
     piu' piccola, meno storico).

  B) "Ignora le carte Allenatore/Energia": esclude Supporter/Item/Stadium/
     Energy (Trainer supertype) via un elenco CURATO A MANO (nessun campo
     "supertype" esiste nei metadati o nella pagina PriceCharting scrappata -
     verificato: la pagina di un Trainer come "Professor Birch #82" ha lo
     stesso breadcrumb generico "Pokemon Cards" di una carta Pokemon, nessun
     segnale distintivo). L'elenco e' stato compilato ispezionando a mano i
     646 nomi distinti dell'universo attuale (non e' un classificatore
     automatico generale, riconosce solo i Trainer REALMENTE presenti oggi -
     va aggiornato manualmente se l'universo cambia). La tesi e' che il
     fattore scarsita' (rarita'/eta'/franchise) misura male le carte
     Trainer: la loro "rarita'" di stampa non riflette la stessa scarsita'
     collezionistica delle carte Pokemon (un Supporter Ultra Rare non e'
     "difficile da ottenere" nello stesso senso di uno Pokemon VMAX/EX).

ESITO (nessun nuovo trial DSR separato per A/B/combinato - un solo esperimento,
due varianti della stessa domanda, n_trials 69->70):

  Baseline (produzione)              | universo 653 | Sharpe 1.58 | DSR 0.877 | H1 0.61 H2 3.04
  A) Solo Pokemon, rarita' Rare+     | universo 555 | Sharpe 1.51 | DSR 0.842 | H1 0.46 H2 3.10
  B) No Allenatore/Energia           | universo 605 | Sharpe 1.43 | DSR 0.802 | H1 0.22 H2 3.15
  A+B combinati                      | universo 514 | Sharpe 1.38 | DSR 0.771 | H1 0.15 H2 3.09

ENTRAMBI I CANDIDATI PEGGIORANO LA STRATEGIA, IN MODO MONOTONO E COERENTE -
non e' rumore isolato: rarita' floor (A) perde 0,07 di Sharpe, escludere gli
Allenatori (B) perde ANCORA PIU' (0,15), e insieme (A+B) perdono 0,20 -
peggiora ogni volta che si restringe di piu' l'universo. Walk-forward resta
positivo in entrambe le meta' in tutti i candidati (nessuna inversione di
segno), ma H1 si affievolisce progressivamente (0,61->0,46->0,22->0,15),
segno che l'edge nella prima meta' del periodo dipende in parte proprio
dalle carte escluse.

Spiegazione plausibile: il modello controlla GIA' per franchise (is_op) e
rarita' (rango continuo) dentro la regressione stessa - non servono
esclusioni categoriali extra, la regressione le gestisce gia' come
covariate. Escludere intere categorie (One Piece, common/uncommon,
Allenatori) non rimuove rumore: rimuove trade che il modello prezzava
correttamente in relazione alle loro stesse pari, riducendo semplicemente
il campione e l'edge misurato.

CONCLUSIONE: NESSUNO dei due filtri adottato. Nessun cambio ai parametri di
produzione (VALIDATED_SINGLES resta come da ultimo aggiornamento) - l'ipotesi
dell'utente era plausibile a priori (categorie diverse potrebbero avere
dinamiche diverse) ma i dati dicono il contrario in modo pulito e coerente,
non ambiguo."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.data.liquidity_filter import liquid_singles_ids
from poke_quant.engine.backtester import Backtester
from poke_quant.engine.strategies.scarcity_value_factor import ScarcityValueFactorStrategy
from poke_quant.validation.statistical_validation import deflated_sharpe_ratio
from scripts.generate_singles_signal import PRODUCTION_PARAMS

# Compilato ispezionando a mano i nomi distinti dell'universo attuale (post
# pavimento di gradazione) - Supporter/Item/Stadium/Energy, MAI carte Pokemon
# con un nome di Supporter come prefisso (es. "Cynthia's Garchomp ex" resta
# una carta Pokemon, non va escluso).
TRAINER_OR_ENERGY_NAMES = {
    "Boss's Orders #189", "Brock's Grit #172", "Colress #135", "Copycat #73",
    "Cyclone Energy #143", "Cynthia #148", "Cynthia & Caitlin #228",
    "Delinquent #98b", "Double Full Heal #77", "Energy Recycler #143",
    "Erika #16", "Erika's Hospitality #174", "Ethan's Adventure #236",
    "Fairy Energy #169", "Fighting Energy #127", "Fire Energy #89",
    "Green's Exploration #209", "Jasmine #177", "Judge Whistle #194",
    "Lillie #151", "Lillie's Full Force #230", "Lisia #164",
    "Lucky Stadium #41 (Promo)", "Mallow & Lana #231", "Mars #154",
    "Misty's Favor #235", "Morty's Conviction #211", "N's Resolve #232",
    "Paradise Resort #45 (Promo)", "Perrin #220",
    "Professor Elm's Lecture #213", "Professor Oak's Setup #233",
    "Rainbow Brush #182", "Rare Candy #105", "Red & Blue #234", "Rosa #236",
    "Sabrina #20", "Sabrina's Suggestion #181", "Skyla #122",
    "Super Scoop Up #100", "Tate & Liza #166", "Team Rocket's Giovanni #238",
    "Tropical Wind #26 (Promo)", "Victory Cup #BW29 (Promo)", "Whitney #214",
    "Winona #108", "Booster Pack",  # artefatto dati, non una carta - escluso qui per sicurezza
}

NON_POKEMON_RARITY_FLOOR = {"Common", "Uncommon"}


def run_bt(strat, prices_df, metadata):
    bt = Backtester(strat, prices_df, metadata, initial_cash=10000.0, platform="cardmarket",
                     apply_liquidity_slippage=True, apply_holding_cost=True, apply_buy_side_shipping=True)
    return bt.run()


def report(label, ids, metadata, grade9_prices, n_trials):
    meta_sub = {k: metadata[k] for k in ids}
    prices_sub = grade9_prices[ids]
    strat = ScarcityValueFactorStrategy(**PRODUCTION_PARAMS)
    res = run_bt(strat, prices_sub, meta_sub)
    dsr = deflated_sharpe_ratio(observed_sr=res.sharpe / np.sqrt(12), n_trials=n_trials, n_obs=len(res.monthly_returns))
    n_obs = len(res.monthly_returns)
    h = n_obs // 2
    r1, r2 = res.monthly_returns[:h], res.monthly_returns[h:]
    h1 = (r1.mean() / r1.std()) * np.sqrt(12) if r1.std() > 0 else 0.0
    h2 = (r2.mean() / r2.std()) * np.sqrt(12) if r2.std() > 0 else 0.0
    print(f"  {label:32s} | universo {len(ids):4d} | Sharpe {res.sharpe:6.2f} | CAGR {res.cagr*100:+7.2f}% | "
          f"MaxDD {res.max_drawdown*100:7.2f}% | Trade {res.total_trades:3d} | DSR({n_trials}) {dsr:.3f} | "
          f"H1 {h1:5.2f} H2 {h2:5.2f}")


def main():
    metadata = load_metadata()
    grade9_prices = load_price_matrix("historical_prices_graded_singles_grade9.csv")
    base_ids = liquid_singles_ids(metadata, grade9_prices)

    n_trials = 70  # +1 dal 69 corrente: due candidati testati in un solo esperimento (stessa domanda, due varianti)

    print("=== Baseline (produzione attuale, post pavimento di gradazione) ===")
    report("nessun filtro aggiuntivo", base_ids, metadata, grade9_prices, n_trials)

    print("\n=== Candidato A: solo Pokemon, rarita' Rare+ (no One Piece, no Common/Uncommon) ===")
    ids_a = [k for k in base_ids if metadata[k].get("franchise") == "pokemon"
             and metadata[k].get("rarity") not in NON_POKEMON_RARITY_FLOOR]
    report("solo Pokemon rarita' Rare+", ids_a, metadata, grade9_prices, n_trials)

    print("\n=== Candidato B: ignora Allenatore/Energia (elenco curato) ===")
    ids_b = [k for k in base_ids if metadata[k].get("name") not in TRAINER_OR_ENERGY_NAMES]
    report("no Allenatore/Energia", ids_b, metadata, grade9_prices, n_trials)

    print("\n=== Controllo: A + B combinati ===")
    ids_ab = [k for k in ids_a if metadata[k].get("name") not in TRAINER_OR_ENERGY_NAMES]
    report("A+B combinati", ids_ab, metadata, grade9_prices, n_trials)


if __name__ == "__main__":
    main()
