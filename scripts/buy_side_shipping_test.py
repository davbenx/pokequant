#!/usr/bin/env python3
"""
scripts/buy_side_shipping_test.py — Quanto cambia il backtest validato se si
include la spedizione REALE a carico del compratore (Backtester.buy_side_shipping,
poke_quant.config.SHIPPING_COSTS) ad ogni acquisto, non solo alla vendita.

Trovato indagando la richiesta di un "prezzo massimo che non rompa l'edge":
Portfolio.buy() non ha MAI applicato nessuna frizione all'acquisto (solo
unit_price*quantity) - ogni numero validato finora (Sharpe 1,83/2,02 sulle
singole, 1,31 sui box) assume implicitamente un acquisto a costo di transazione
zero. Per le singole (prezzo medio spesso 50-300EUR) una spedizione fissa di
7EUR e' un peso proporzionalmente enorme su un trade piccolo - controllato
prima di ipotizzare: l'11 dei 15 candidati BUY di oggi perdono l'INTERO
margine teorico solo per questo (vedi generate_singles_signal.py::max_edge_price_eur).

ESITO: impatto reale, misurato non ipotizzato -
  BOX (universo a 40, con guardia prezzo/MSRP): Sharpe 1,31->1,18, CAGR
    +23,82%->+22,80%, MaxDD -10,62%->-11,12%, DSR sessione intera 0,778->0,684
    (poi 0,681 con 1 trial in piu' per la guardia prezzo/MSRP - vedi
    scripts/box_max_price_ratio_test.py). Ancora sotto soglia 0,90-0,95, come
    prima.
  SINGOLE (universo corretto 935): Sharpe 1,83->1,48, CAGR +29,28%->+25,28%,
    MaxDD -8,96%->-14,06%, DSR sessione intera 0,954->0,836 - SOTTO la soglia
    0,90-0,95 per la prima volta da quando questo fattore l'ha superata.
    Impatto molto piu' grande che sui box: un trade tipico da 50-300EUR regge
    molto meno bene una spedizione fissa di quanto regga un box da centinaia
    di euro.
ADOTTATO in produzione (Backtester.apply_buy_side_shipping=True in app.py,
entrambe le meta') - nessun acquisto e' mai stato gratis nella realta', e
l'utente ha chiesto esplicitamente che il prezzo massimo mostrato in
dashboard sia il prezzo FINITO, spedizione inclusa: non si puo' mostrare quel
numero onestamente se il backtest sottostante ignora lo stesso costo.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.data.liquidity_filter import liquid_sealed_ids
from poke_quant.engine.backtester import Backtester
from poke_quant.engine.strategies.time_series_momentum import TimeSeriesMomentumStrategy
from poke_quant.engine.strategies.scarcity_value_factor import ScarcityValueFactorStrategy


def run(strat, prices, meta, apply_shipping):
    bt = Backtester(strat, prices, meta, initial_cash=10000.0, platform="cardmarket",
                     apply_liquidity_slippage=True, apply_holding_cost=True,
                     apply_buy_side_shipping=apply_shipping)
    return bt.run()


def main():
    metadata = load_metadata()

    print("=== BOX (TS Momentum, universo a 40) ===")
    sealed_prices = load_price_matrix()
    sealed_ids = liquid_sealed_ids(metadata, sealed_prices)
    meta_box = {k: metadata[k] for k in sealed_ids}
    prices_box = sealed_prices[sealed_ids]
    strat_box = TimeSeriesMomentumStrategy(prices_box, lookback_months=12)
    for label, flag in [("SENZA spedizione all'acquisto (attuale)", False), ("CON spedizione all'acquisto (10EUR/box)", True)]:
        res = run(strat_box, prices_box, meta_box, flag)
        print(f"  {label:42s} | Sharpe {res.sharpe:5.2f} | CAGR {res.cagr*100:+7.2f}% | Trade {res.total_trades:3d}")

    print("\n=== SINGOLE (Fattore Scarsita', universo corretto) ===")
    grade9_prices = load_price_matrix("historical_prices_graded_singles_grade9.csv")
    singles_ids = [k for k, v in metadata.items() if v.get("type") == "single"
                   and v.get("data_quality") != "thin_unreliable" and k in grade9_prices.columns]
    meta_singles = {k: metadata[k] for k in singles_ids}
    prices_singles = grade9_prices[singles_ids]
    strat_singles = ScarcityValueFactorStrategy(rebalance_every_months=3, top_quantile=0.20,
                                                 min_age_months=6, max_positions=60, min_cross_section=20)
    for label, flag in [("SENZA spedizione all'acquisto (attuale)", False), ("CON spedizione all'acquisto (7EUR/carta)", True)]:
        res = run(strat_singles, prices_singles, meta_singles, flag)
        print(f"  {label:42s} | Sharpe {res.sharpe:5.2f} | CAGR {res.cagr*100:+7.2f}% | Trade {res.total_trades:3d}")


if __name__ == "__main__":
    main()
