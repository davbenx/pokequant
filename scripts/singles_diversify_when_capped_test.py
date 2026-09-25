#!/usr/bin/env python3
"""
scripts/singles_diversify_when_capped_test.py — L'utente ha chiesto: "se non
trovo tutte le copie consigliate di una carta, cosa mi conviene comprare al
suo posto?" Il meccanismo di fondo e' quello gia' trovato in
max_quantity_per_trade_test.py: con un tetto realistico di 1 copia per
acquisto, lo Sharpe delle singole crolla 1.57->0.30 (DSR 0.044, rumore).
Domanda distinta: quel capitale "liberato" (che con 1 sola copia resta in
parte inutilizzato, perche' max_positions=60 tronca la selezione ben prima
di aver speso tutto il budget) e' meglio lasciarlo fermo, o usarlo per
comprare PIU' carte DIVERSE (1 copia ciascuna) dal resto del pool GIA'
qualificato (stesso quantile 20% piu' sottovalutato, solo fuori dai primi
60 per rank)?

Leva testata: max_positions (quante delle ~173 carte nel quantile 20% piu'
sottovalutato vengono davvero comprate, non solo le prime 60 per rank) -
NON il top_quantile stesso (quello resta 0.20, il fattore già validato:
allargarlo vorrebbe dire comprare carte meno sottovalutate, un cambio di
tesi diverso, testato separatamente sotto come controllo).

Nessuna scelta post-hoc: max_positions=60 e' il default di produzione
(pre-registrato), la griglia [60,100,173] e' valutata per intero, non
si sceglie il picco - se il pattern e' monotono e coerente (non un singolo
punto isolato) e regge a PBO/walk-forward, si adotta il piu' conservativo
che recupera la maggior parte dell'edge perso.

ESITO:

  Baseline (nessun tetto, produzione)     | Sharpe 1.57 | CAGR +26.73% | MaxDD -12.03% | DSR 0.870
  tetto=1, max_positions=60  (produzione) | Sharpe 0.30 | CAGR  +4.91% | MaxDD -13.22% | DSR 0.044
  tetto=1, max_positions=100              | Sharpe 0.57 | CAGR  +8.08% | MaxDD -19.03% | DSR 0.142
  tetto=1, max_positions=173 (intero 20%) | Sharpe 0.80 | CAGR +11.80% | MaxDD -20.89% | DSR 0.293

  Controllo (allarga anche il quantile, non solo max_positions):
  tetto=1, quantile=0.30, max_positions=300 | Sharpe 0.95 | MaxDD -23.29% | DSR 0.417
  tetto=1, quantile=0.40, max_positions=300 | Sharpe 1.10 | MaxDD -22.36% | DSR 0.555

  Diversificare su PIU' carte diverse (stesso quantile 20%, solo oltre le
  prime 60 per rank) quando il tetto di 1 copia lascia capitale inutilizzato
  RECUPERA parte dell'edge perso (Sharpe 0.30->0.80), ma con un costo reale:
  il MaxDD quasi raddoppia (-13%->-21%) e il DSR resta ben sotto la soglia
  usata per validare le altre strategie di questa dashboard (0.29 contro
  0.87-0.95) - non torna "validato", solo meno peggio che lasciare i soldi
  fermi. Il controllo (allargare anche il quantile stesso, non solo quante
  delle carte gia' qualificate si comprano) continua a "migliorare" lo
  Sharpe in modo monotono (0.80->0.95->1.10) diluendo la soglia del
  fattore - pattern sospetto (assomiglia a un in-sample tuning che sembra
  buono), MaxDD comunque pessimo, NON adottato.

CONCLUSIONE PRATICA: se non trovi le copie/carte consigliate, usa il
budget liberato per comprare ALTRE carte gia' nel quantile 20% piu'
sottovalutato (oltre le prime 60 per rank) invece di lasciarlo fermo - e'
empiricamente meglio, ma resta un ripiego, non una strategia a se' testata.
Non allargare la soglia del fattore stesso (carte fuori dal quantile 20%,
grado diverso da 9, lingua diversa dall'inglese) - territorio gia' battuto
e respinto in questa sessione. Implementato in app.py come lista
informativa separata ("alternative"), non come secondo elenco BUY.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.engine.backtester import Backtester
from poke_quant.engine.strategies.scarcity_value_factor import ScarcityValueFactorStrategy
from poke_quant.validation.statistical_validation import deflated_sharpe_ratio

MAX_POSITIONS_GRID = [60, 100, 173]  # 173 = intero quantile 20% dell'universo (869 carte)
TOP_QUANTILE_CONTROL = [0.20, 0.30, 0.40]  # controllo separato: allargare il quantile stesso


def run_bt(strat, prices_df, metadata):
    bt = Backtester(strat, prices_df, metadata, initial_cash=10000.0, platform="cardmarket",
                     apply_liquidity_slippage=True, apply_holding_cost=True, apply_buy_side_shipping=True)
    return bt.run()


def report(label, res, n_names_series=None):
    dsr = deflated_sharpe_ratio(observed_sr=res.sharpe / np.sqrt(12), n_trials=71, n_obs=len(res.monthly_returns))
    avg_qty = res.trades_df["quantity"].mean() if not res.trades_df.empty else 0
    print(f"  {label:28s} | Sharpe {res.sharpe:6.2f} | CAGR {res.cagr*100:+7.2f}% | "
          f"MaxDD {res.max_drawdown*100:7.2f}% | Trade {res.total_trades:3d} | qty media {avg_qty:4.2f} | DSR(71) {dsr:.3f}")


def main():
    metadata = load_metadata()
    grade9_prices = load_price_matrix("historical_prices_graded_singles_grade9.csv")
    singles_ids = [k for k, v in metadata.items() if v.get("type") == "single"
                   and v.get("data_quality") != "thin_unreliable" and k in grade9_prices.columns]
    meta_singles = {k: metadata[k] for k in singles_ids}
    prices_singles = grade9_prices[singles_ids]

    print("=== Baseline: nessun tetto quantita', max_positions=60 (produzione validata) ===")
    strat = ScarcityValueFactorStrategy(rebalance_every_months=3, top_quantile=0.20, min_age_months=6,
                                         max_positions=60, min_cross_section=20)
    report("nessun tetto (rif.)", run_bt(strat, prices_singles, meta_singles))

    print("\n=== Tetto realistico 1 copia/acquisto, poi allarga SOLO max_positions (stesso quantile 20%) ===")
    for mp in MAX_POSITIONS_GRID:
        strat = ScarcityValueFactorStrategy(rebalance_every_months=3, top_quantile=0.20, min_age_months=6,
                                             max_positions=mp, min_cross_section=20, max_quantity_per_trade=1)
        res = run_bt(strat, prices_singles, meta_singles)
        n_months = (res.trades_df["date"].nunique() if not res.trades_df.empty and "date" in res.trades_df.columns else None)
        report(f"tetto=1, max_positions={mp}", res)

    print("\n=== Controllo separato: tetto=1 copia, max_positions=173, ma ALLARGA anche il quantile stesso ===")
    for tq in TOP_QUANTILE_CONTROL:
        strat = ScarcityValueFactorStrategy(rebalance_every_months=3, top_quantile=tq, min_age_months=6,
                                             max_positions=300, min_cross_section=20, max_quantity_per_trade=1)
        res = run_bt(strat, prices_singles, meta_singles)
        report(f"tetto=1, quantile={tq:.2f}", res)


if __name__ == "__main__":
    main()
