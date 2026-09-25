#!/usr/bin/env python3
"""
scripts/max_quantity_per_trade_test.py — L'utente ha chiesto "e se compro
più copie di box o carte?". Verificando la domanda ho trovato che il
backtest VALIDATO (Sharpe 1,57 singole / 1,18 box) già assume di comprare
molte copie identiche per trade - senza nessun tetto, qty = budget_posizione
// prezzo, e su una carta economica questo arriva a 95 copie in un solo mese
(Aegislash Grade9 a 2,20EUR, luglio 2023) o 12 box identici in un mese.
Nessun mercato reale ha mai 95 slab IDENTICI (stessa carta, stesso grado)
disponibili insieme allo stesso prezzo tracciato - il backtest assume
liquidità infinita per costruzione, mai verificato finora.

Metodo: Strategy.max_quantity_per_trade (time_series_momentum.py e
scarcity_value_factor.py) applica un tetto realistico invece di nessuno.
Griglia 1/2/3/5/10/nessun tetto (comportamento validato attuale). Nessun
nuovo trial DSR: e' uno stress-test di realismo sulla strategia GIA'
adottata, non una selezione del "miglior tetto" tra candidati nuovi (stesso
trattamento di execution_dropout_montecarlo_test.py) - n_trials resta 70.

ESITO:

BOX (TS Momentum):
  tetto=1  | Sharpe 1.22 | CAGR +14.13% | MaxDD  -4.88% | qty media 1.0 | DSR 0.659
  tetto=2  | Sharpe 1.25 | CAGR +19.98% | MaxDD  -9.79% | qty media 2.0 | DSR 0.684
  tetto=3  | Sharpe 1.34 | CAGR +23.07% | MaxDD -10.45% | qty media 2.8 | DSR 0.751
  tetto=5  | Sharpe 1.35 | CAGR +23.31% | MaxDD -10.26% | qty media 4.2 | DSR 0.756
  tetto=10 | Sharpe 1.15 | CAGR +22.25% | MaxDD -11.40% | qty media 5.2 | DSR 0.602
  nessuno  | Sharpe 1.18 | CAGR +22.80% | MaxDD -11.12% | qty media 5.1 | DSR 0.632 (validato)

  L'assunzione implicita di "piu' copie" e' quasi innocua per i box: anche
  al tetto piu' severo (1 box per trade) lo Sharpe resta 1.22, in linea col
  validato. Comprare piu' box sigillati IDENTICI (stesso set) e' anche
  realistico (distributori/rivenditori spesso ne hanno diversi) - nessuna
  azione richiesta, nessun cambio ai parametri di produzione.

SINGOLE (Fattore Scarsita'):
  tetto=1  | Sharpe 0.30 | CAGR  +4.91% | MaxDD -13.22% | qty media 1.0 | DSR 0.044
  tetto=2  | Sharpe 0.89 | CAGR +11.31% | MaxDD -10.07% | qty media 1.8 | DSR 0.371
  tetto=3  | Sharpe 1.10 | CAGR +15.28% | MaxDD -10.56% | qty media 2.6 | DSR 0.553
  tetto=5  | Sharpe 1.22 | CAGR +18.90% | MaxDD -13.52% | qty media 3.6 | DSR 0.653
  tetto=10 | Sharpe 1.39 | CAGR +23.24% | MaxDD -13.25% | qty media 5.3 | DSR 0.779
  nessuno  | Sharpe 1.57 | CAGR +26.73% | MaxDD -12.03% | qty media 7.8 | DSR 0.871 (validato)

  Risultato opposto e MOLTO piu' serio: gran parte dello Sharpe 1.57
  validato dipende dal comprare diverse copie IDENTICHE (stessa carta,
  stesso grado PSA/CGC, stessa lingua) allo stesso prezzo tracciato, su
  quasi tutti i 299 trade del backtest - non un caso isolato (l'Aegislash
  a 95 copie era solo l'esempio piu' estremo). Con il tetto piu'
  conservativo e realistico (1 slab per acquisto - un'inserzione = una
  copia, il caso normale per carte gradate poco liquide), lo Sharpe crolla
  a 0.30 e il DSR a 0.04: statisticamente indistinguibile dal rumore, nello
  stesso territorio di ogni fattore singole gia' scartato in questa sessione
  (grade7/grade8/PSA10/altre lingue). Con 2 copie il DSR resta debole (0.37).
  Serve solo dal tetto=5 in su per riportare il DSR sopra ~0.65.

CONCLUSIONE PRATICA: per i box, nessun cambio - l'assunzione di produzione
e' gia' realistica. Per le singole, l'headline "Sharpe 1.57" e' un tetto
teorico che vale SOLO se si riesce a comprare sistematicamente piu' slab
identici per acquisto - non verificato quanto sia vero nel mercato reale
(nessun dato di profondita' di liquidita' per-carta disponibile). Aggiunta
in dashboard (app.py) una stima "-> N pz." per ogni carta/box in BUY, con
avviso quando assume piu' di una copia identica disponibile insieme -
sostituisce il gap segnalato dall'utente ("non vedo quanto comprare").
Nessun cambio ai parametri VALIDATED_SINGLES/VALIDATED_BOX in produzione:
la decisione su quale tetto trattare come "vero" default richiede la
conoscenza di mercato dell'utente (quante copie identiche trova davvero),
non deducibile dai dati storici disponibili.
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

CAP_GRID = [1, 2, 3, 5, 10, None]


def run_bt(strat, prices_df, metadata):
    bt = Backtester(strat, prices_df, metadata, initial_cash=10000.0, platform="cardmarket",
                     apply_liquidity_slippage=True, apply_holding_cost=True, apply_buy_side_shipping=True)
    return bt.run()


def main():
    metadata = load_metadata()

    print("=== BOX (TS Momentum) — validato SENZA tetto: Sharpe 1,18, qty media 5,1, max 12 ===")
    sealed_prices = load_price_matrix()
    sealed_ids = liquid_sealed_ids(metadata, sealed_prices)
    meta_box = {k: metadata[k] for k in sealed_ids}
    prices_box = sealed_prices[sealed_ids]
    for cap in CAP_GRID:
        strat = TimeSeriesMomentumStrategy(prices_box, lookback_months=12, max_price_msrp_ratio=MAX_PRICE_TO_MSRP_RATIO,
                                            max_quantity_per_trade=cap)
        res = run_bt(strat, prices_box, meta_box)
        dsr = deflated_sharpe_ratio(observed_sr=res.sharpe / np.sqrt(12), n_trials=70, n_obs=len(res.monthly_returns))
        avg_qty = res.trades_df["quantity"].mean() if not res.trades_df.empty else 0
        cap_label = "nessuno (validato)" if cap is None else str(cap)
        print(f"  tetto={cap_label:18s} | Sharpe {res.sharpe:6.2f} | CAGR {res.cagr*100:+7.2f}% | "
              f"MaxDD {res.max_drawdown*100:7.2f}% | Trade {res.total_trades:3d} | qty media {avg_qty:5.1f} | DSR(70) {dsr:.3f}")

    print("\n=== SINGOLE (Fattore Scarsita') — validato SENZA tetto: Sharpe 1,57, qty media 7,8, max 95 ===")
    grade9_prices = load_price_matrix("historical_prices_graded_singles_grade9.csv")
    singles_ids = [k for k, v in metadata.items() if v.get("type") == "single"
                   and v.get("data_quality") != "thin_unreliable" and k in grade9_prices.columns]
    meta_singles = {k: metadata[k] for k in singles_ids}
    prices_singles = grade9_prices[singles_ids]
    for cap in CAP_GRID:
        strat = ScarcityValueFactorStrategy(max_quantity_per_trade=cap, **PRODUCTION_PARAMS)
        res = run_bt(strat, prices_singles, meta_singles)
        dsr = deflated_sharpe_ratio(observed_sr=res.sharpe / np.sqrt(12), n_trials=70, n_obs=len(res.monthly_returns))
        avg_qty = res.trades_df["quantity"].mean() if not res.trades_df.empty else 0
        cap_label = "nessuno (validato)" if cap is None else str(cap)
        print(f"  tetto={cap_label:18s} | Sharpe {res.sharpe:6.2f} | CAGR {res.cagr*100:+7.2f}% | "
              f"MaxDD {res.max_drawdown*100:7.2f}% | Trade {res.total_trades:3d} | qty media {avg_qty:5.1f} | DSR(70) {dsr:.3f}")


if __name__ == "__main__":
    main()
