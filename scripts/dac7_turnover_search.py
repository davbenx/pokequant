#!/usr/bin/env python3
"""
scripts/dac7_turnover_search.py — Ricerca vincolata per la modalita' "resta
sotto 2.000EUR/30 vendite annue" (DAC7, direttiva UE 2021/514: le piattaforme
devono segnalare un venditore alle autorita' fiscali se supera OGNI ANNO 30
transazioni O 2.000EUR di incasso totale - basta superare UNA delle due soglie).

BUG TROVATO verificando la richiesta dell'utente "verifica che [DAC7 vs
produzione] sia la scelta ottimale": DAC7_SINGLES_PARAMS (generate_singles_
signal.py) e il commento che lo giustifica ("Sharpe 1,49 vs 2,02", scripts/
dac7_turnover_search.py) citavano un file che non esiste nel repo ne' nella
history di git - non riproducibile, un singolo punto scelto a mano, non un
optimum verificato sotto vincolo. Questo script e' la ricerca reale, mai
fatta prima.

VINCOLO CONTROLLABILE DAL MODELLO: solo il numero di transazioni/anno (le
vendite chiuse, portfolio.closed_trades) dipende dalla cadenza di
ribilanciamento e dal numero di posizioni - la soglia dei 2.000EUR/anno
dipende dal CAPITALE e dalla taglia di posizione scelti dall'utente, non da
un parametro di questa strategia, quindi non e' nel perimetro di questa
ricerca (l'utente deve comunque restare sotto quella soglia scegliendo un
capitale/allocazione compatibile in sidebar).

METODO: grid search su (rebalance_every_months, max_positions), stessi altri
parametri di PRODUCTION_PARAMS (top_quantile=0.20, min_age_months=6,
min_cross_section=20) - nessun parametro scelto guardando il risultato.
Per ogni config: Sharpe/DSR pieno E vendite/anno reali dal backtest
(len(trades_df) / anni coperti, non una stima). Tra le config con
vendite/anno <= 30, si sceglie quella con Sharpe migliore = optimum
vincolato reale.

Uso: python scripts/dac7_turnover_search.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.data.liquidity_filter import liquid_singles_ids
from poke_quant.engine.strategies.scarcity_value_factor import ScarcityValueFactorStrategy
from poke_quant.engine.backtester import Backtester
from poke_quant.validation.statistical_validation import deflated_sharpe_ratio

DAC7_MAX_SALES_PER_YEAR = 30

REBALANCE_GRID = [3, 6, 9, 12]
MAX_POSITIONS_GRID = [10, 15, 20, 30, 40, 60]
FIXED_PARAMS = dict(top_quantile=0.20, min_age_months=6, min_cross_section=20)


def run_config(prices_sub, meta_sub, rebalance_every_months: int, max_positions: int):
    strat = ScarcityValueFactorStrategy(rebalance_every_months=rebalance_every_months,
                                         max_positions=max_positions, **FIXED_PARAMS)
    bt = Backtester(strat, prices_sub, meta_sub, initial_cash=10000.0, platform="cardmarket",
                     apply_liquidity_slippage=True, apply_holding_cost=True, apply_buy_side_shipping=True)
    res = bt.run()
    n_years = len(res.monthly_returns) / 12.0
    n_sales = len(res.trades_df) if res.trades_df is not None else 0
    sales_per_year = n_sales / n_years if n_years > 0 else float("inf")
    dsr = deflated_sharpe_ratio(observed_sr=res.sharpe / np.sqrt(12),
                                 n_trials=len(REBALANCE_GRID) * len(MAX_POSITIONS_GRID),
                                 n_obs=len(res.monthly_returns))
    return res, sales_per_year, dsr


def main():
    metadata = load_metadata()
    prices_full = load_price_matrix("historical_prices_graded_singles_grade9.csv")
    singles_ids = liquid_singles_ids(metadata, prices_full)
    meta_sub = {k: v for k, v in metadata.items() if k in singles_ids}
    prices_sub = prices_full[singles_ids]

    print(f"{'rebalance':>10} {'max_pos':>8} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8} "
          f"{'vendite/anno':>13} {'DAC7 ok':>8} {'DSR':>6}")
    rows = []
    for rb in REBALANCE_GRID:
        for mp in MAX_POSITIONS_GRID:
            res, sales_yr, dsr = run_config(prices_sub, meta_sub, rb, mp)
            ok = sales_yr <= DAC7_MAX_SALES_PER_YEAR
            rows.append((rb, mp, res.sharpe, res.cagr, res.max_drawdown, sales_yr, ok, dsr))
            print(f"{rb:>10} {mp:>8} {res.sharpe:>8.3f} {res.cagr*100:>7.2f}% {res.max_drawdown*100:>7.2f}% "
                  f"{sales_yr:>13.1f} {'si' if ok else 'no':>8} {dsr:>6.3f}")

    compliant = [r for r in rows if r[6]]
    if compliant:
        best = max(compliant, key=lambda r: r[2])
        print(f"\nOptimum vincolato (vendite/anno <= {DAC7_MAX_SALES_PER_YEAR}): "
              f"rebalance_every_months={best[0]}, max_positions={best[1]} | "
              f"Sharpe {best[2]:.3f} | CAGR {best[3]*100:+.2f}% | vendite/anno {best[5]:.1f} | DSR {best[7]:.3f}")
    else:
        print("\nNessuna config nella grid resta sotto la soglia DAC7 - ampliare la grid.")

    prod = next((r for r in rows if r[0] == 3 and r[1] == 60), None)
    if prod and compliant:
        best = max(compliant, key=lambda r: r[2])
        print(f"\nConfronto: produzione (3, 60) Sharpe {prod[2]:.3f} vs optimum DAC7 ({best[0]}, {best[1]}) "
              f"Sharpe {best[2]:.3f} - costo della compliance: {(prod[2]-best[2]):.3f} Sharpe.")


if __name__ == "__main__":
    main()
