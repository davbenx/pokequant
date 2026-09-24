#!/usr/bin/env python3
"""
scripts/eu_resale_premium_breakeven_test.py — Correzione di un errore
metodologico segnalato dall'utente in scripts/usa_landed_cost_edge_test.py:
quel test gonfiava SOLO il lato acquisto (costo sdoganato da USA/JP) ma
vendeva sulla stessa serie storica USA/PriceCharting grezza, non rialzata -
corretto SOLO se in EU (dove l'utente rivende sempre, anche quando compra
importando) non esiste nessun premio persistente alla rivendita rispetto al
tracciato USA. Non lo sappiamo (1 sola osservazione reale finora, Raichu,
spiegata da un artefatto di prodotto - 1st Edition vs Unlimited - non da un
premio generale) - quindi invece di assumere un numero, si trasforma
l'incognita in una domanda diretta: A CHE PREMIO EU ALLA RIVENDITA la
strategia "compra a costo sdoganato, vendi in EU" torna in pareggio con la
produzione attuale (comprata e venduta in EU, nessun import)?

Metodo: Backtester.sell_side_eu_premium (backtester.py) moltiplica il
prezzo di vendita per un fattore costante, con buy_at_usa_landed_cost=True.
Griglia di premi 1.0 -> 2.0, per box e singole separatamente.

ESITO: vedi output.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.data.liquidity_filter import liquid_sealed_ids, MAX_PRICE_TO_MSRP_RATIO
from poke_quant.engine.backtester import Backtester
from poke_quant.engine.strategies.time_series_momentum import TimeSeriesMomentumStrategy
from poke_quant.engine.strategies.scarcity_value_factor import ScarcityValueFactorStrategy
from scripts.generate_singles_signal import PRODUCTION_PARAMS

PREMIUM_GRID = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0]


def run_bt(strat, prices_df, metadata, premium):
    bt = Backtester(strat, prices_df, metadata, initial_cash=10000.0, platform="cardmarket",
                     apply_liquidity_slippage=True, apply_holding_cost=True,
                     buy_at_usa_landed_cost=True, sell_side_eu_premium=premium)
    return bt.run()


def main():
    metadata = load_metadata()

    print("=== BOX (TS Momentum) — target: Sharpe prod. EU-only 1.18, MaxDD -11.1% ===")
    sealed_prices = load_price_matrix()
    sealed_ids = liquid_sealed_ids(metadata, sealed_prices)
    meta_box = {k: metadata[k] for k in sealed_ids}
    prices_box = sealed_prices[sealed_ids]
    for premium in PREMIUM_GRID:
        strat = TimeSeriesMomentumStrategy(prices_box, lookback_months=12, max_price_msrp_ratio=MAX_PRICE_TO_MSRP_RATIO)
        res = run_bt(strat, prices_box, meta_box, premium)
        print(f"  premio EU rivendita {premium:.1f}x | Sharpe {res.sharpe:6.2f} | CAGR {res.cagr*100:+7.2f}% | "
              f"MaxDD {res.max_drawdown*100:7.2f}% | Trade {res.total_trades:3d}")

    print("\n=== SINGOLE (Fattore Scarsita') — target: Sharpe prod. EU-only 1.57, MaxDD -12.0% ===")
    grade9_prices = load_price_matrix("historical_prices_graded_singles_grade9.csv")
    singles_ids = [k for k, v in metadata.items() if v.get("type") == "single"
                   and v.get("data_quality") != "thin_unreliable" and k in grade9_prices.columns]
    meta_singles = {k: metadata[k] for k in singles_ids}
    prices_singles = grade9_prices[singles_ids]
    for premium in PREMIUM_GRID:
        strat = ScarcityValueFactorStrategy(**PRODUCTION_PARAMS)
        res = run_bt(strat, prices_singles, meta_singles, premium)
        print(f"  premio EU rivendita {premium:.1f}x | Sharpe {res.sharpe:6.2f} | CAGR {res.cagr*100:+7.2f}% | "
              f"MaxDD {res.max_drawdown*100:7.2f}% | Trade {res.total_trades:3d}")


if __name__ == "__main__":
    main()
