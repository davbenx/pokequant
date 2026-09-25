#!/usr/bin/env python3
"""
scripts/max_quantity_retest_expanded_universe.py — Riverifica il tetto di quantita'
per trade (scripts/max_quantity_per_trade_test.py) sull'universo singole ESPANSO
(discover_random_control_singles.py --per-set 30, rebuild_prices_with_real_fx.py,
flag_unreliable_assets.py rieseguiti su richiesta di rendere il motore piu' completo
e verificare le strategie sui dati aggiornati).

TROVATO PRIMA DI QUESTO SCRIPT (vedi scripts/production_singles_revalidation_
expanded_universe.py): la produzione (Sharpe 1,52 -> 2,05, CAGR 27,52% -> 43,71%
sull'universo espanso) sembrava un miglioramento, ma ispezionando i trade con PnL
piu' alto sono quasi tutti carte COMUNI ECONOMICHE comprate in decine di copie
(es. Lure Ball #128: 83 copie a 4,92EUR, Growlithe #10: 62 copie a 9,72EUR) - la
STESSA assunzione irrealistica gia' documentata da max_quantity_per_trade_test.py,
ma AMPLIFICATA: il campione di controllo casuale 6x piu' grande pesca in modo
uniforme su TUTTE le carte di un set, la maggioranza delle quali sono comuni
economiche - un budget fisso per posizione compra molte piu' copie di una carta da
5EUR che di una da 50EUR, quindi ogni mossa di prezzo genuina su una comune viene
amplificata in modo sproporzionato nel backtest. Questo script quantifica quanto
il gap tra Sharpe teorico (nessun tetto) e Sharpe realistico (1-2 copie) si sia
allargato con l'universo espanso.

ESITO: il gap si e' allargato, non ridotto - l'universo piu' grande NON migliora la
strategia in modo realizzabile, gonfia solo il numero teorico.

  tetto=1        | Sharpe 0.21 | CAGR  +4.24% | MaxDD -14.11% | DSR(69) 0.028
  tetto=2        | Sharpe 0.86 | CAGR +11.01% | MaxDD -10.69% | DSR(69) 0.346
  tetto=3        | Sharpe 1.14 | CAGR +15.43% | MaxDD -12.20% | DSR(69) 0.588
  tetto=5        | Sharpe 1.35 | CAGR +20.80% | MaxDD -15.10% | DSR(69) 0.751
  tetto=10       | Sharpe 1.66 | CAGR +28.70% | MaxDD -13.91% | DSR(69) 0.906
  nessuno        | Sharpe 2.05 | CAGR +43.71% | MaxDD -15.11% | DSR(69) 0.981 (headline)

Confronto con l'universo precedente (869 carte, max_quantity_per_trade_test.py):
tetto=1 era Sharpe 0.30/DSR 0.044, ora 0.21/0.028 - PEGGIORE, non migliore. tetto=2
era 0.89/0.371, ora 0.86/0.346 - sostanzialmente invariato. Il salto del numero
"headline" (1.57->2.05 nella metodologia precedente, poi ricontrollato qui) viene
interamente dalla maggiore capacita' di comprare in blocco carte comuni economiche
aggiunte dal campione di controllo ampliato (vedi il docstring sopra) - non da un
miglioramento reale dell'edge catturabile.

CONCLUSIONE PRATICA: a QUALSIASI tetto di quantita' realistico per carte gradate
poco liquide (1 o 2 copie - il caso normale, un'inserzione = una copia), il DSR
(0.028-0.346) resta ben sotto ogni soglia usata in questa ricerca (0.90-0.95) -
territorio statisticamente indistinguibile dal rumore, lo stesso di ogni fattore
categoriale gia' respinto. Il numero "headline" senza tetto resta quello che questa
ricerca ha sempre mostrato in dashboard (per coerenza con la metodologia usata
ovunque in questo progetto), ma il caveat sul suo carattere teorico va reso ancora
piu' esplicito ora che il divario si e' allargato, non ridotto, con piu' dati.
"""
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

N_TRIALS_FULL_SESSION = 69  # stress-test di realismo su strategia gia' adottata, non un nuovo trial
CAP_GRID = [1, 2, 3, 5, 10, None]


def run_bt(strat, prices_df, metadata):
    bt = Backtester(strat, prices_df, metadata, initial_cash=10000.0, platform="cardmarket",
                     apply_liquidity_slippage=True, apply_holding_cost=True, apply_buy_side_shipping=True)
    return bt.run()


def main():
    metadata = load_metadata()
    prices = load_price_matrix("historical_prices_graded_singles_grade9.csv")
    singles_ids = liquid_singles_ids(metadata, prices)
    meta_sub = {k: metadata[k] for k in singles_ids}
    prices_sub = prices[singles_ids]
    print(f"Universo (liquid_singles_ids, espanso e pulito): {len(singles_ids)} carte\n")

    for cap in CAP_GRID:
        strat = ScarcityValueFactorStrategy(max_quantity_per_trade=cap, **PRODUCTION_PARAMS)
        res = run_bt(strat, prices_sub, meta_sub)
        dsr = deflated_sharpe_ratio(observed_sr=res.sharpe / np.sqrt(12), n_trials=N_TRIALS_FULL_SESSION, n_obs=len(res.monthly_returns))
        avg_qty = res.trades_df["quantity"].mean() if not res.trades_df.empty else 0
        max_qty = res.trades_df["quantity"].max() if not res.trades_df.empty else 0
        cap_label = "nessuno (headline)" if cap is None else str(cap)
        print(f"  tetto={cap_label:18s} | Sharpe {res.sharpe:6.2f} | CAGR {res.cagr*100:+7.2f}% | "
              f"MaxDD {res.max_drawdown*100:7.2f}% | Trade {res.total_trades:3d} | qty media {avg_qty:5.1f} "
              f"(max {max_qty:.0f}) | DSR({N_TRIALS_FULL_SESSION}) {dsr:.3f}")


if __name__ == "__main__":
    main()
