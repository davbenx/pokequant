#!/usr/bin/env python3
"""
scripts/usa_landed_cost_edge_test.py — L'utente ha chiesto di verificare se
comprare al "prezzo sdoganato da USA" (config.py::estimate_usa_import_landed_cost,
costruito su richiesta esplicita per il problema di importazione dall'Italia)
possa comunque preservare l'edge statisticamente testato, non solo essere il
canale piu' economico tra EU e USA.

Metodo: simula il CASO PEGGIORE in cui OGNI acquisto avviene al costo
sdoganato pieno (oggetto + spedizione internazionale + IVA 22% + dazio UE 3€
+ commissione corriere ~15€), invece del prezzo dashboard + spedizione EU
gia' validato (Backtester.buy_at_usa_landed_cost=True, vedi backtester.py).
Confrontato con la produzione attuale (spedizione EU, non USA).

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


def run_bt(strat, prices_df, metadata, usa_landed):
    bt = Backtester(strat, prices_df, metadata, initial_cash=10000.0, platform="cardmarket",
                     apply_liquidity_slippage=True, apply_holding_cost=True,
                     apply_buy_side_shipping=not usa_landed, buy_at_usa_landed_cost=usa_landed)
    return bt.run()


def report(label, res, n_trials, n_obs):
    dsr = deflated_sharpe_ratio(observed_sr=res.sharpe / np.sqrt(12), n_trials=n_trials, n_obs=n_obs)
    print(f"  {label:42s} | Sharpe {res.sharpe:6.2f} | CAGR {res.cagr*100:+7.2f}% | "
          f"MaxDD {res.max_drawdown*100:7.2f}% | Trade {res.total_trades:3d} | DSR(n={n_trials}) {dsr:.3f}")
    return dsr


def main():
    metadata = load_metadata()

    print("=== BOX (TS Momentum, universo a 40) ===")
    sealed_prices = load_price_matrix()
    sealed_ids = liquid_sealed_ids(metadata, sealed_prices)
    meta_box = {k: metadata[k] for k in sealed_ids}
    prices_box = sealed_prices[sealed_ids]
    strat_box_a = TimeSeriesMomentumStrategy(prices_box, lookback_months=12, max_price_msrp_ratio=MAX_PRICE_TO_MSRP_RATIO)
    res_box_eu = run_bt(strat_box_a, prices_box, meta_box, usa_landed=False)
    report("Spedizione EU (produzione attuale)", res_box_eu, 67, len(res_box_eu.monthly_returns))
    strat_box_b = TimeSeriesMomentumStrategy(prices_box, lookback_months=12, max_price_msrp_ratio=MAX_PRICE_TO_MSRP_RATIO)
    res_box_usa = run_bt(strat_box_b, prices_box, meta_box, usa_landed=True)
    report("Costo sdoganato pieno da USA (caso peggiore)", res_box_usa, 67, len(res_box_usa.monthly_returns))

    print("\n=== SINGOLE (Fattore Scarsita', universo 869) ===")
    grade9_prices = load_price_matrix("historical_prices_graded_singles_grade9.csv")
    singles_ids = [k for k, v in metadata.items() if v.get("type") == "single"
                   and v.get("data_quality") != "thin_unreliable" and k in grade9_prices.columns]
    meta_singles = {k: metadata[k] for k in singles_ids}
    prices_singles = grade9_prices[singles_ids]
    strat_singles_a = ScarcityValueFactorStrategy(**PRODUCTION_PARAMS)
    res_singles_eu = run_bt(strat_singles_a, prices_singles, meta_singles, usa_landed=False)
    report("Spedizione EU (produzione attuale)", res_singles_eu, 67, len(res_singles_eu.monthly_returns))
    strat_singles_b = ScarcityValueFactorStrategy(**PRODUCTION_PARAMS)
    res_singles_usa = run_bt(strat_singles_b, prices_singles, meta_singles, usa_landed=True)
    report("Costo sdoganato pieno da USA (caso peggiore)", res_singles_usa, 67, len(res_singles_usa.monthly_returns))

    print("\nWalk-forward H1/H2 al costo sdoganato pieno da USA:")
    for label, prices_df, meta, strat_cls, kwargs in [
        ("BOX", prices_box, meta_box, TimeSeriesMomentumStrategy,
         dict(lookback_months=12, max_price_msrp_ratio=MAX_PRICE_TO_MSRP_RATIO)),
        ("SINGOLE", prices_singles, meta_singles, ScarcityValueFactorStrategy, PRODUCTION_PARAMS),
    ]:
        mid = len(prices_df) // 2
        for h_label, dates in [("H1", prices_df.index[:mid]), ("H2", prices_df.index[mid:])]:
            sub = prices_df.loc[dates]
            strat = strat_cls(sub, **kwargs) if label == "BOX" else strat_cls(**kwargs)
            res = run_bt(strat, sub, meta, usa_landed=True)
            print(f"  {label} {h_label} ({dates[0].strftime('%Y-%m')}->{dates[-1].strftime('%Y-%m')}) | Sharpe {res.sharpe:5.2f} | CAGR {res.cagr*100:+6.2f}%")


if __name__ == "__main__":
    main()
