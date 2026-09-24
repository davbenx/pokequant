#!/usr/bin/env python3
"""
scripts/eu_landed_cost_equivalent_test.py — L'utente ha osservato: "comprare
in Europa ha prezzi simili a comprare dall'estero e sdoganare". Non e' la
stessa cosa del test precedente (sell_side_eu_premium, un moltiplicatore
COSTANTE): il costo sdoganato (config.py::estimate_usa_import_landed_cost)
NON e' un multiplo uniforme - ha una parte fissa (IVA 22% + dazio 3EUR +
corriere ~15EUR) enorme in proporzione su una carta economica, piccola su
un box costoso. Se il prezzo EU converge davvero a quel livello, questo vale
per ENTRAMBI i lati del trade (compro E vendo nello stesso mercato EU
"gonfiato"), non solo per l'acquisto.

Metodo: Backtester.sell_at_usa_landed_cost_equivalent (backtester.py) applica
la STESSA formula del costo sdoganato anche alla vendita, non solo
all'acquisto - simula un mercato EU dove sia il prezzo pagato che il prezzo
di realizzo sono al livello "sdoganato-equivalente", non il tracciato
PriceCharting grezzo.

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
from poke_quant.validation.statistical_validation import deflated_sharpe_ratio
from scripts.generate_singles_signal import PRODUCTION_PARAMS


def run_variant(strat, prices_df, metadata, label, **kwargs):
    bt = Backtester(strat, prices_df, metadata, initial_cash=10000.0, platform="cardmarket",
                     apply_liquidity_slippage=True, apply_holding_cost=True, **kwargs)
    res = bt.run()
    dsr = deflated_sharpe_ratio(observed_sr=res.sharpe / np.sqrt(12), n_trials=70, n_obs=len(res.monthly_returns))
    print(f"  {label:55s} | Sharpe {res.sharpe:6.2f} | CAGR {res.cagr*100:+7.2f}% | "
          f"MaxDD {res.max_drawdown*100:7.2f}% | Trade {res.total_trades:3d} | DSR(70) {dsr:.3f}")
    return res


def main():
    metadata = load_metadata()

    print("=== BOX (TS Momentum) ===")
    sealed_prices = load_price_matrix()
    sealed_ids = liquid_sealed_ids(metadata, sealed_prices)
    meta_box = {k: metadata[k] for k in sealed_ids}
    prices_box = sealed_prices[sealed_ids]

    strat_a = TimeSeriesMomentumStrategy(prices_box, lookback_months=12, max_price_msrp_ratio=MAX_PRICE_TO_MSRP_RATIO)
    run_variant(strat_a, prices_box, meta_box, "Produzione (compra/vendi a prezzo grezzo EU)", apply_buy_side_shipping=True)

    strat_b = TimeSeriesMomentumStrategy(prices_box, lookback_months=12, max_price_msrp_ratio=MAX_PRICE_TO_MSRP_RATIO)
    run_variant(strat_b, prices_box, meta_box, "Mercato EU sempre a livello sdoganato-equivalente",
                buy_at_usa_landed_cost=True, sell_at_usa_landed_cost_equivalent=True)

    print("\n=== SINGOLE (Fattore Scarsita') ===")
    grade9_prices = load_price_matrix("historical_prices_graded_singles_grade9.csv")
    singles_ids = [k for k, v in metadata.items() if v.get("type") == "single"
                   and v.get("data_quality") != "thin_unreliable" and k in grade9_prices.columns]
    meta_singles = {k: metadata[k] for k in singles_ids}
    prices_singles = grade9_prices[singles_ids]

    strat_c = ScarcityValueFactorStrategy(**PRODUCTION_PARAMS)
    run_variant(strat_c, prices_singles, meta_singles, "Produzione (compra/vendi a prezzo grezzo EU)", apply_buy_side_shipping=True)

    strat_d = ScarcityValueFactorStrategy(**PRODUCTION_PARAMS)
    res_landed = run_variant(strat_d, prices_singles, meta_singles, "Mercato EU sempre a livello sdoganato-equivalente",
                              buy_at_usa_landed_cost=True, sell_at_usa_landed_cost_equivalent=True)

    mid = len(prices_singles) // 2
    h1, h2 = prices_singles.index[:mid], prices_singles.index[mid:]
    print("\nWalk-forward H1/H2 (singole, mercato sdoganato-equivalente):")
    for label, dates in [("H1", h1), ("H2", h2)]:
        sub = prices_singles.loc[dates]
        strat = ScarcityValueFactorStrategy(**PRODUCTION_PARAMS)
        res = run_variant(strat, sub, meta_singles, f"  {label} ({dates[0].strftime('%Y-%m')}->{dates[-1].strftime('%Y-%m')})",
                           buy_at_usa_landed_cost=True, sell_at_usa_landed_cost_equivalent=True)


if __name__ == "__main__":
    main()
