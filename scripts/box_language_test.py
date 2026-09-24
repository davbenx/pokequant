#!/usr/bin/env python3
"""
scripts/box_language_test.py — Testa se la strategia TS Momentum sui box
sigillati regge separatamente sui box GIAPPONESI, non solo sull'universo
combinato EN+JP (la produzione attuale). A differenza del test sulle singole
in lingua diversa (nessun dato reale disponibile, verificato in una fase
precedente), qui i dati SONO reali e gia' in produzione: 5 box JP
(VSTAR Universe, VMAX Climax, Shiny Star V, Tag All Stars, Shiny Treasure ex)
sono gia' nell'universo liquido a 40 asset (poke_quant/data/liquidity_filter.py)
con storico reale PriceCharting (34-49 mesi ciascuno).

Campione minuscolo (n=5 asset, solo 4 trade totali nel periodo) - il risultato
e' direzionale/di attenzione, non una prova statistica definitiva (troppo
pochi trade per DSR/PBO). Ma e' un segnale concreto da monitorare, non
un'ipotesi.

ESITO: i box JP, isolati, NON mostrano l'edge della strategia - anzi, le
uniche 4 operazioni completate sono TUTTE in perdita (-18,5%, -33,2%, -29,1%,
-17,2%). Sharpe JP-only 0,03 (CAGR +3,1%) contro Sharpe 1,29 (CAGR +24,0%)
dell'universo EN/altro - la performance combinata a produzione (Sharpe 1,31)
e' quasi identica a quella EN-only perche' i JP pesano poco (5/40 asset), non
perche' contribuiscano positivamente.

Nessun problema di qualita' dati (nessuno dei 5 flaggato thin_unreliable dal
filtro di attendibilita' esistente) - il momentum a 12m ha semplicemente
identificato falsi positivi su questo piccolo campione. Le 4 date di acquisto
cadono tutte tra il 2023-09 e il 2025-04 - non escluso che sia in parte un
effetto di timing comune (il mercato dei box in generale ha attraversato fasi
di correzione in quel periodo) piuttosto che una caratteristica specifica del
mercato giapponese, ma con solo 4 trade non si puo' scorporare le due cose.

Nessuna modifica alla produzione: i box JP restano nell'universo (rimuoverli
richiederebbe evidenza piu' solida di 4 trade, e comunque contribuiscono in
modo quasi neutro, non catastrofico, al portafoglio combinato). Da rivalutare
quando ci sara' piu' storico.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.data.liquidity_filter import liquid_sealed_ids
from poke_quant.engine.strategies.time_series_momentum import TimeSeriesMomentumStrategy
from scripts.optimize_and_falsify import run_bt


def main():
    metadata = load_metadata()
    prices = load_price_matrix()
    liquid = liquid_sealed_ids(metadata, prices)
    jp_ids = [k for k in liquid if metadata[k].get("language") == "jp"]
    en_ids = [k for k in liquid if metadata[k].get("language") != "jp"]

    print(f"Universo liquido: {len(liquid)} | JP: {len(jp_ids)} | EN/altro: {len(en_ids)}\n")

    for label, ids in [("JP", jp_ids), ("EN/altro", en_ids), ("Combinato (produzione)", liquid)]:
        meta_sub = {k: metadata[k] for k in ids}
        prices_sub = prices[ids]
        strat = TimeSeriesMomentumStrategy(prices_sub, lookback_months=12)
        res = run_bt(strat, prices_sub, meta_sub)
        print(f"{label:24s} (n={len(ids):2d}) | CAGR {res.cagr*100:+7.2f}% | Sharpe {res.sharpe:5.2f} | "
              f"MaxDD {res.max_drawdown*100:7.2f}% | Trade {res.total_trades:3d}")

    print("\nDettaglio trade JP:")
    strat_jp = TimeSeriesMomentumStrategy(prices[jp_ids], lookback_months=12)
    res_jp = run_bt(strat_jp, prices[jp_ids], {k: metadata[k] for k in jp_ids})
    td = res_jp.trades_df
    if td.empty:
        print("  Nessun trade.")
    else:
        for _, r in td.iterrows():
            print(f"  {r['item_id']:24s} {r['buy_date']} -> {r['sell_date']} | "
                  f"{r['buy_price_unit']:7.2f}€ -> {r['sell_price_unit']:7.2f}€ | ROI netto {r['net_roi']*100:+.1f}%")


if __name__ == "__main__":
    main()
