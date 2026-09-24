#!/usr/bin/env python3
"""
scripts/eu_price_level_invariance_test.py — L'utente ha fatto un'osservazione
piu' profonda di quanto sembrasse: "tu assumi che in Europa ci siano gli
stessi prezzi USA, ma non e' cosi'". Non riguarda solo lo scenario di
importazione (scripts/eu_resale_premium_breakeven_test.py) - riguarda il
BASELINE "produzione" stesso (Sharpe 1,57 singole / 1,18 box), che gira
sulla serie PriceCharting (USA) trattandola come se fosse il prezzo EU reale.

Ho risposto in chat che un riscalaggio uniforme (prezzi EU = K x prezzi USA,
stesso K per acquisto e vendita) si "annulla" nel calcolo del rendimento %
- vero in teoria (Sharpe/CAGR dipendono da rapporti, non da livelli assoluti),
MA falso in pratica in questo motore, perche' diversi controlli usano un
riferimento ASSOLUTO che NON scala con K:
  - Box: MAX_PRICE_TO_MSRP_RATIO confronta il prezzo (che scalerebbe con K)
    con l'MSRP (fisso, reale, non scala) - riscalare i prezzi cambia quali
    box passano il filtro di liquidita' e quali carte restano bloccate da
    "prezzo eccessivo".
  - Singole: il filtro grade9/raw (liquidity_filter.py::compute_grade_raw_ratio_flags)
    confronta il grade9 (scalerebbe con K) con cardmarket_ref_price_eur
    (fisso, fonte separata) - riscalare cambia quali carte sono "thin_unreliable".
  - Entrambi: SHIPPING_COSTS (7/10EUR) e' un costo FISSO assoluto (spedire
    fisicamente una carta non costa piu' solo perche' vale piu') - a K piu'
    alto pesa proporzionalmente MENO, quindi NON e' neutro.
Il resto (commissione Cardmarket 5%, costo di custodia annuo %) scala
correttamente con K, nessuna distorsione lì.

Metodo: rigira il backtest di produzione (nessun import, compra/vendi sempre
in EU) con TUTTI i prezzi storici moltiplicati per K, ricalcolando ENTRAMBI
i filtri di liquidita' sulla serie scalata (msrp e cardmarket_ref_price_eur
NON scalati, sono dati reali indipendenti da PriceCharting) - non solo
assumendo che si annulli, verificandolo.

ESITO: vedi output.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.data.liquidity_filter import (
    liquid_sealed_ids, compute_reliability_flags, compute_grade_raw_ratio_flags,
)
from poke_quant.engine.backtester import Backtester
from poke_quant.engine.strategies.time_series_momentum import TimeSeriesMomentumStrategy
from poke_quant.engine.strategies.scarcity_value_factor import ScarcityValueFactorStrategy
from poke_quant.data.liquidity_filter import MAX_PRICE_TO_MSRP_RATIO
from scripts.generate_singles_signal import PRODUCTION_PARAMS

K_GRID = [1.0, 1.5, 2.0, 3.0]


def run_bt(strat, prices_df, metadata):
    bt = Backtester(strat, prices_df, metadata, initial_cash=10000.0, platform="cardmarket",
                     apply_liquidity_slippage=True, apply_holding_cost=True, apply_buy_side_shipping=True)
    return bt.run()


def main():
    metadata = load_metadata()

    print("=== BOX — livello di prezzo EU = K x PriceCharting (MSRP NON scalato) ===")
    sealed_prices_base = load_price_matrix()
    for K in K_GRID:
        scaled_prices = sealed_prices_base * K
        sealed_ids = liquid_sealed_ids(metadata, scaled_prices, max_price_msrp_ratio=MAX_PRICE_TO_MSRP_RATIO)
        meta_box = {k: metadata[k] for k in sealed_ids}
        prices_box = scaled_prices[sealed_ids]
        strat = TimeSeriesMomentumStrategy(prices_box, lookback_months=12, max_price_msrp_ratio=MAX_PRICE_TO_MSRP_RATIO)
        res = run_bt(strat, prices_box, meta_box)
        print(f"  K={K:.1f} | universo {len(sealed_ids):3d} | Sharpe {res.sharpe:6.2f} | CAGR {res.cagr*100:+7.2f}% | "
              f"MaxDD {res.max_drawdown*100:7.2f}% | Trade {res.total_trades:3d}")

    print("\n=== SINGOLE — livello di prezzo EU = K x PriceCharting (cardmarket_ref_price_eur NON scalato) ===")
    grade9_prices_base = load_price_matrix("historical_prices_graded_singles_grade9.csv")
    for K in K_GRID:
        scaled_grade9 = grade9_prices_base * K
        single_ids_all = {k for k, v in metadata.items() if v.get("type") == "single"}
        vol_flags = compute_reliability_flags(scaled_grade9[[c for c in scaled_grade9.columns if c in single_ids_all]])
        ratio_flags = compute_grade_raw_ratio_flags(metadata, scaled_grade9)  # cardmarket_ref_price_eur non scalato
        bad_ids = {k for k, (ok, _) in vol_flags.items() if not ok} | set(ratio_flags.keys())
        singles_ids = [k for k in single_ids_all if k in scaled_grade9.columns and k not in bad_ids]
        meta_singles = {k: metadata[k] for k in singles_ids}
        prices_singles = scaled_grade9[singles_ids]
        strat = ScarcityValueFactorStrategy(**PRODUCTION_PARAMS)
        res = run_bt(strat, prices_singles, meta_singles)
        print(f"  K={K:.1f} | universo {len(singles_ids):3d} | Sharpe {res.sharpe:6.2f} | CAGR {res.cagr*100:+7.2f}% | "
              f"MaxDD {res.max_drawdown*100:7.2f}% | Trade {res.total_trades:3d}")


if __name__ == "__main__":
    main()
