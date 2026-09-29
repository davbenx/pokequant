#!/usr/bin/env python3
"""
scripts/thin_market_ratio_drift_test.py — Test empirico del nuovo controllo
"mercato sottile" (poke_quant/data/liquidity_filter.py::compute_thin_market_drift_flags).

Richiesto dall'utente ("e' possibile aggiungere un controllo per mercato
sottile?") dopo aver valutato un acquisto reale: Azumarill #114 [Delta
Species] a 90,25EUR dava "COLPACCIO -52%" (fair value 189,68EUR), ma il
prezzo Grade 9 era passato da 47,50EUR a 257,37EUR in 7 mesi (+442%) mentre
il RAW della stessa carta, nello stesso periodo, saliva solo da 26,23EUR a
45,91EUR (+75%) - il rapporto grade9/raw e' triplicato mentre il mercato raw
(molto piu' liquido) confermava solo una crescita modesta.

CALIBRAZIONE (PRIMA di guardare l'effetto sul backtest, stesso principio di
graded_raw_ratio_reliability_test.py): calcolato il drift del rapporto
grade9/raw su una finestra di 6 mesi per TUTTO l'universo liquido (1.077
carte con storico raw+grade9 comune sufficiente). Distribuzione: mediana
1,17x, p90 2,12x, p95 2,56x, p97,5 3,09x, p99 3,79x, max 10,28x - un salto
netto tra il grosso della distribuzione e la coda estrema attorno al
97,5-99* percentile. Soglia adottata: 3,0x (vicino al p97,5 reale, non
scelta per far scattare Azumarill in particolare - che con drift ~3,1x non
e' nemmeno il caso piu' estremo dell'universo, vedi Eternatus #141 a 10,28x).

ESITO: vedi output.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.data.liquidity_filter import liquid_singles_ids, compute_thin_market_drift_flags
from poke_quant.engine.backtester import Backtester
from poke_quant.engine.strategies.scarcity_value_factor import ScarcityValueFactorStrategy
from poke_quant.validation.statistical_validation import deflated_sharpe_ratio
from scripts.generate_singles_signal import PRODUCTION_PARAMS

# Conteggio cumulativo onesto: 86 trial gia' spesi sulla linea di ricerca
# singole (vedi scripts/dac7_turnover_search.py::PRIOR_SINGLES_TRIALS, corretto
# 2026-09-29) + questo trial.
N_TRIALS_TOTAL = 87


def run_bt(strat, prices_df, metadata):
    bt = Backtester(strat, prices_df, metadata, initial_cash=10000.0, platform="cardmarket",
                     apply_liquidity_slippage=True, apply_holding_cost=True, apply_buy_side_shipping=True)
    return bt.run()


def main():
    metadata = load_metadata()
    grade9_prices = load_price_matrix("historical_prices_graded_singles_grade9.csv")
    raw_prices = load_price_matrix("historical_prices_graded_singles_raw.csv")

    flags = compute_thin_market_drift_flags(metadata, grade9_prices, raw_prices)
    print(f"Carte flaggate (drift grade9/raw >= 3,0x in 6 mesi): {len(flags)}")
    for item_id, (_, reason) in sorted(flags.items(), key=lambda x: x[0])[:20]:
        print(f"  {metadata[item_id].get('name', item_id):35s} {reason}")

    base_ids = liquid_singles_ids(metadata, grade9_prices)
    strict_ids = [k for k in base_ids if k not in flags]
    print(f"\nUniverso attuale: {len(base_ids)} carte | con il nuovo filtro: {len(strict_ids)} "
          f"({len(base_ids) - len(strict_ids)} escluse in piu')")

    for label, ids in [("SENZA filtro mercato sottile (attuale)", base_ids),
                        ("CON filtro mercato sottile", strict_ids)]:
        meta_sub = {k: metadata[k] for k in ids}
        prices_sub = grade9_prices[ids]
        strat = ScarcityValueFactorStrategy(**PRODUCTION_PARAMS)
        res = run_bt(strat, prices_sub, meta_sub)
        dsr = deflated_sharpe_ratio(observed_sr=res.sharpe / np.sqrt(12), n_trials=N_TRIALS_TOTAL, n_obs=len(res.monthly_returns))
        mid = len(prices_sub) // 2
        h1, h2 = prices_sub.index[:mid], prices_sub.index[mid:]
        wf = []
        for dates in (h1, h2):
            sub = prices_sub.loc[dates]
            s2 = ScarcityValueFactorStrategy(**PRODUCTION_PARAMS)
            r2 = run_bt(s2, sub, meta_sub)
            wf.append(r2.sharpe)
        print(f"\n  {label}")
        print(f"    Sharpe {res.sharpe:5.2f} | CAGR {res.cagr*100:+7.2f}% | MaxDD {res.max_drawdown*100:7.2f}% | "
              f"Trade {res.total_trades:3d} | DSR(n={N_TRIALS_TOTAL}) {dsr:.3f} | WF H1/H2 {wf[0]:.2f}/{wf[1]:.2f}")


if __name__ == "__main__":
    main()
