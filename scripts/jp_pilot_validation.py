#!/usr/bin/env python3
"""
scripts/jp_pilot_validation.py — Valutazione quantitativa di Pokemon JP
(box sigillati - nessuna singola JP esiste nell'universo, verificato via
metadata) in isolamento, con la stessa metodologia delle altre valutazioni
franchise di questo pilota (One Piece, Magic: The Gathering).

Riprende e aggiorna scripts/box_language_test.py (2026, sessione precedente):
quel test uso' scripts/optimize_and_falsify.py::run_bt (frizioni/parametri
potenzialmente diversi da quelli usati oggi in mtg_pilot_validation.py e
one_piece_pilot_validation.py) e concluse "Sharpe JP-only 0,03, 4/4 trade in
perdita, campione troppo piccolo per DSR/PBO" senza pero' un intervallo di
confidenza (bootstrap). Qui: stesso identico setup di Backtester usato in
tutto il resto di questo pilota (per comparabilita' diretta) + bootstrap a
blocchi, che funziona anche su pochi trade (a differenza di DSR/PBO che
richiedono un numero minimo di osservazioni per essere significativi).

ESITO (2026-09-29): RIGETTATO - raccomandata l'eliminazione dalla produzione.

  Box SOLO JP:          Sharpe 0,04 | DSR(51) 0,015 | 5 box, 4 trade
                        Bootstrap: Sharpe mediana 0,43 [5%=-0,57, 95%=1,13], P(>0)=78%
                        TUTTI i 4 trade completati sono in perdita: -19,2%/-33,7%/-29,8%/-18,0%
  Box SOLO EN/altro:    Sharpe 1,23 | DSR(51) 0,710
  Box UNIFICATO:        Sharpe 1,23 | DSR(51) 0,709 (quasi identico a EN-only: la
                        diluizione su 5/55 asset rende l'effetto sul blend trascurabile
                        in entrambe le direzioni - JP non sta "salvando" né "affondando"
                        il portafoglio pooled)

  Diverso dal caso One Piece (1 solo trade, dato insufficiente per concludere): qui
  ci sono 4 trade REALI, tutti negativi, un pattern consistente non un rumore isolato
  - il bootstrap conferma che anche nella coda ottimistica (95%) lo Sharpe resta
  modesto (1,13), e nella coda pessimistica diventa negativo (-0,57). DSR 0,015 e'
  ordini di grandezza sotto la soglia di comfort (0,90-0,95).

  DECISIONE: eliminare Pokemon JP dalla produzione. Non perche' danneggi
  misurabilmente il blend pooled (l'effetto e' quasi nullo, essendo diluito su
  soli 5/55 asset) - ma perche' non esiste alcuna giustificazione statistica per
  tenerlo come componente "validata": 4/4 trade in perdita e DSR vicino a zero
  sono un pattern negativo consistente, non un campione insufficiente come One
  Piece. Nessuna singola JP esiste nell'universo (verificato via metadata) - la
  decisione riguarda solo il lato box.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.data.liquidity_filter import liquid_sealed_ids
from poke_quant.engine.backtester import Backtester
from poke_quant.engine.strategies.time_series_momentum import TimeSeriesMomentumStrategy
from poke_quant.validation.statistical_validation import deflated_sharpe_ratio
from poke_quant.validation.bootstrap import block_bootstrap_metrics, summarize_bootstrap

N_TRIALS_BOX = 51  # +1 su one_piece_pilot_validation.py: nuovo candidato isolato


def run_box_bt(prices_df, metadata):
    strat = TimeSeriesMomentumStrategy(prices_df, lookback_months=12)
    bt = Backtester(strat, prices_df, metadata, initial_cash=10000.0, platform="cardmarket",
                     apply_liquidity_slippage=True, apply_holding_cost=True, apply_buy_side_shipping=True)
    return bt.run()


def report(label, res, n_trials):
    dsr = deflated_sharpe_ratio(observed_sr=res.sharpe / np.sqrt(12), n_trials=n_trials, n_obs=len(res.monthly_returns))
    print(f"  {label:42s} | Sharpe {res.sharpe:6.2f} | CAGR {res.cagr*100:+7.2f}% | "
          f"MaxDD {res.max_drawdown*100:7.2f}% | Trade {res.total_trades:3d} | DSR({n_trials}) {dsr:.3f}")
    return dsr


def main():
    metadata = load_metadata()
    prices = load_price_matrix()
    # exclude_languages=frozenset(): dopo l'ESITO di questo stesso script, "jp" e'
    # diventato il default escluso (DEFAULT_EXCLUDED_LANGUAGES) - questo script
    # esiste apposta per testarlo isolato, deve includerlo esplicitamente.
    liquid = liquid_sealed_ids(metadata, prices, exclude_franchises=frozenset(), exclude_languages=frozenset())
    jp_ids = [k for k in liquid if metadata[k].get("language") == "jp"]
    en_ids = [k for k in liquid if metadata[k].get("language") != "jp"]

    print("=" * 100)
    print("BOX Pokemon JP (TS Momentum) - isolato vs EN/altro vs unificato")
    print("=" * 100)
    print(f"Universo liquido: {len(liquid)} | JP: {len(jp_ids)} | EN/altro: {len(en_ids)}\n")

    res_jp = run_box_bt(prices[jp_ids], {k: metadata[k] for k in jp_ids})
    report("Box SOLO JP", res_jp, N_TRIALS_BOX)
    print("\n  Bootstrap a blocchi (n piccolo, block_size=3):")
    sims = block_bootstrap_metrics(res_jp.monthly_returns, n_sims=500, block_size=3)
    print(f"  {summarize_bootstrap(sims)}")

    res_en = run_box_bt(prices[en_ids], {k: metadata[k] for k in en_ids})
    report("\nBox SOLO EN/altro (senza JP)", res_en, N_TRIALS_BOX)

    res_all = run_box_bt(prices[liquid], {k: metadata[k] for k in liquid})
    report("Box UNIFICATO (produzione attuale)", res_all, N_TRIALS_BOX)

    print("\nDettaglio trade JP:")
    td = res_jp.trades_df
    if td.empty:
        print("  Nessun trade.")
    else:
        for _, r in td.iterrows():
            print(f"  {r['item_id']:24s} {r['buy_date']} -> {r['sell_date']} | "
                  f"{r['buy_price_unit']:7.2f}€ -> {r['sell_price_unit']:7.2f}€ | ROI netto {r['net_roi']*100:+.1f}%")


if __name__ == "__main__":
    main()
