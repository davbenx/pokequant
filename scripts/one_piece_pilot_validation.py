#!/usr/bin/env python3
"""
scripts/one_piece_pilot_validation.py — Valutazione quantitativa di One Piece
TCG in isolamento, stesso rigore del pilota MTG (scripts/mtg_pilot_validation.py):
DSR/PBO/bootstrap/walk-forward dove il campione lo permette.

Richiesto dall'utente: "valutiamo quantitativamente di eliminare franchise
deboli tra mtg, Pokemon Jap, One piece". MTG gia' fatto (rigettato). Qui One
Piece, mai testato in isolamento prima d'ora - appariva solo come indice di
prezzo aggregato (scripts/update_monthly_cache.py) o come regressore binario
in un test OLS separato (scripts/population_scarcity_factor_test.py), non in
un vero backtest con Sharpe/DSR proprio.

BOX: 5 box liquidi (VSTAR... no, One Piece: Romance Dawn/Paramount War/
Pillars of Strength/Awakening of the New Era/Wings of the Captain - vedi
scripts/discover_one_piece_singles.py::OP_GAME_SLUGS che elenca anche i box
gia' confermati). Campione piccolo ma testabile.

SINGOLE: TROVATO PRIMA DI QUALUNQUE BACKTEST (verificato via metadata, non
assunto) - le 119 singole One Piece hanno TUTTE rarity=None (scripts/
discover_one_piece_singles.py non ha mai catturato la rarita' - PriceCharting
non la esporta sulla pagina console usata per la scoperta). ScarcityValueFactorStrategy
esclude rarity=None per costruzione (EXCLUDED_RARITIES, poke_quant/engine/
strategies/scarcity_value_factor.py:102-170): significa che One Piece NON ha
MAI contribuito nulla al fattore scarsita' sulle singole, in tutta questa
ricerca - ogni "Singole UNIFICATE Pokemon+OP" citata nei mesi precedenti era
in realta' Pokemon (+MTG dove testato), zero P&L attribuibile a One Piece.
Non e' una scoperta "One Piece singole sono deboli" - e' "One Piece singole
non sono mai state nel modello", un problema di dato mancante (rarita'), non
di edge. Non costruito un backfill di rarita' in questo pilota (119 carte,
richiederebbe una nuova fonte dati/mappatura manuale rarita'-per-carta,
sproporzionato rispetto al resto del piano) - riportato come limite esplicito,
non come esito negativo della strategia.

ESITO (2026-09-29):

  Box SOLO One Piece:  Sharpe 0,63 | DSR(50) 0,217 | 4 box liquidi | **1 trade totale**
                        Bootstrap: CAGR mediana +11,2% [5%=0,1%, 95%=33,7%], P(>0)=95%
  Box UNIFICATO:        Sharpe 1,23 | DSR(50) 0,711
  Box SOLO Pokemon (senza OP): Sharpe 1,22 | DSR(50) 0,708
  -> Il campione (4 box, 1 solo trade in tutto il periodo) e' troppo piccolo per
     supportare QUALUNQUE conclusione statistica affidabile - DSR 0,217 e' ben
     sotto la soglia di comfort, ma e' calcolato su un singolo trade, non su un
     campione ripetuto. Il confronto diretto (box unificato 1,23 vs box solo
     Pokemon 1,22) mostra che tenere o togliere One Piece dal box UNIFICATO
     cambia lo Sharpe di 0,01 - un effetto NEUTRO, non ne aiuta ne' lo danneggia
     in modo misurabile. Diverso da MTG (che peggiorava attivamente il blend).

  DECISIONE: mantenere One Piece nel box per ora (nessuna evidenza di danno,
  effetto trascurabile sul blend) MA con confidenza bassa dichiarata - non e'
  una validazione positiva come Pokemon EN, e' "nessuna prova sufficiente per
  eliminarlo". Rivalutare quando il campione sara' piu' grande (nuovi set OP).

  Singole One Piece: STRUTTURALMENTE ASSENTI dal fattore scarsita' (0/72 carte
  liquide hanno una rarita' valida - tutte rarity=None, mai raccolta alla
  scoperta). Non e' una domanda "tenere o eliminare" - il dato per farle
  partecipare non esiste. Backlog (non eseguito in questo pilota, sproporzionato
  per 119 carte): costruire una mappatura rarita'-per-carta reale (es. da Bandai
  TCG wiki o API dedicata) se si vuole dare a One Piece una possibilita' reale
  sul lato singole.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.data.liquidity_filter import liquid_sealed_ids, liquid_singles_ids
from poke_quant.engine.backtester import Backtester
from poke_quant.engine.strategies.time_series_momentum import TimeSeriesMomentumStrategy
from poke_quant.engine.strategies.scarcity_value_factor import ScarcityValueFactorStrategy, EXCLUDED_RARITIES
from poke_quant.validation.statistical_validation import deflated_sharpe_ratio
from poke_quant.validation.bootstrap import block_bootstrap_metrics, summarize_bootstrap
from scripts.generate_singles_signal import PRODUCTION_PARAMS

# Cumulativi di sessione - vedi scripts/mtg_pilot_validation.py per la stessa
# convenzione. One Piece isolato e' un candidato NUOVO (mai testato con DSR
# prima), quindi +1 su entrambi rispetto all'ultimo valore noto.
N_TRIALS_BOX = 50
N_TRIALS_SINGLES = 70


def run_box_bt(prices_df, metadata):
    strat = TimeSeriesMomentumStrategy(prices_df, lookback_months=12)
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
    print(f"  {label:42s} | Sharpe {res.sharpe:6.2f} | CAGR {res.cagr*100:+7.2f}% | "
          f"MaxDD {res.max_drawdown*100:7.2f}% | Trade {res.total_trades:3d} | DSR({n_trials}) {dsr:.3f} | "
          f"H1 {h1:5.2f} H2 {h2:5.2f}")
    return dsr


def main():
    metadata = load_metadata()
    sealed_prices = load_price_matrix("historical_prices.csv")
    grade9_prices = load_price_matrix("historical_prices_graded_singles_grade9.csv")

    print("=" * 100)
    print("1. BOX One Piece (TS Momentum) - isolato vs unificato")
    print("=" * 100)
    sealed_ids = liquid_sealed_ids(metadata, sealed_prices, exclude_franchises=frozenset())
    op_box_ids = [k for k in sealed_ids if metadata[k].get("franchise") == "one_piece"]
    print(f"Box One Piece liquidi: {len(op_box_ids)} | Box totali (Pokemon+OP): {len(sealed_ids)}")

    if len(op_box_ids) >= 3:
        meta_op = {k: metadata[k] for k in op_box_ids}
        res_op_box = run_box_bt(sealed_prices[op_box_ids], meta_op)
        report("Box SOLO One Piece", res_op_box, N_TRIALS_BOX)
        print("\n  Bootstrap a blocchi (n piccolo, block_size ridotto a 3):")
        sims = block_bootstrap_metrics(res_op_box.monthly_returns, n_sims=500, block_size=3)
        print(f"  {summarize_bootstrap(sims)}")
    else:
        print(f"  Troppo pochi box One Piece liquidi ({len(op_box_ids)}) per un backtest a se' stante.")

    meta_all = {k: metadata[k] for k in sealed_ids}
    res_unified_box = run_box_bt(sealed_prices[sealed_ids], meta_all)
    report("Box UNIFICATO (Pokemon+OP)", res_unified_box, N_TRIALS_BOX)

    # Confronto diretto: quanto cambia il box UNIFICATO se One Piece viene tolto?
    pokemon_only_ids = [k for k in sealed_ids if metadata[k].get("franchise") != "one_piece"]
    meta_pokemon_only = {k: metadata[k] for k in pokemon_only_ids}
    res_pokemon_only_box = run_box_bt(sealed_prices[pokemon_only_ids], meta_pokemon_only)
    report("Box SOLO Pokemon (senza OP, per confronto)", res_pokemon_only_box, N_TRIALS_BOX)

    print("\n" + "=" * 100)
    print("2. SINGOLE One Piece - verifica strutturale (non un backtest)")
    print("=" * 100)
    singles_ids = liquid_singles_ids(metadata, grade9_prices, exclude_franchises=frozenset())
    op_singles_ids = [k for k in singles_ids if metadata[k].get("franchise") == "one_piece"]
    op_singles_with_rarity = [k for k in op_singles_ids if metadata[k].get("rarity") not in EXCLUDED_RARITIES]
    print(f"Singole One Piece nell'universo liquido: {len(op_singles_ids)}")
    print(f"Di cui con una rarita' valida (non esclusa dal fattore scarsita'): {len(op_singles_with_rarity)}")
    if len(op_singles_with_rarity) == 0:
        print("  -> CONFERMATO: zero singole One Piece hanno mai potuto generare un residuo nel fattore\n"
              "     scarsita' (tutte rarity=None, EXCLUDED_RARITIES le esclude per costruzione). Nessun\n"
              "     backtest possibile finche' non esiste una fonte reale di rarita' per carta - non e'\n"
              "     un esito 'debole', e' un dato mancante. Le singole One Piece contribuiscono ESATTAMENTE\n"
              "     zero P&L alla strategia singole oggi, silenziosamente, da quando sono state scoperte.")


if __name__ == "__main__":
    main()
