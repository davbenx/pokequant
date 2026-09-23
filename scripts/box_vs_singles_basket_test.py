#!/usr/bin/env python3
"""
scripts/box_vs_singles_basket_test.py — Verifica se l'edge del TS Momentum
sui box sigillati è "trainato" dalle singole che contengono, o se è un
fenomeno specifico del box (premio da sigillato/scarsità, non dalle carte).

Metodo: per ogni box del universo validato, costruisce un paniere equal-weight
delle sue singole note (data_cache/single_to_box_map.json, da
scripts/build_ratio_matrix.py) come indice sintetico (cumulato dai rendimenti
mensili medi cross-sezionali), poi fa girare la STESSA TimeSeriesMomentumStrategy
su quell'indice al posto del prezzo del box. Se l'aggregazione delle singole
recupera un Sharpe comparabile al box reale, l'edge "viene dalle carte" e i
test sui fattori-singola erano solo troppo rumorosi al livello di singola
carta. Se non lo recupera, il box ha una dinamica propria (premio da
sigillato) che le sue stesse carte non spiegano.

ESITO: il legame e' reale ma parziale, non un'illusione e non una spiegazione
completa. Su 22 box con >=3 singole mappate: BOX REALE Sharpe 0.99/CAGR
+20.6%/MaxDD -12.4%; PANIERE SINGOLE (stessi set) Sharpe 0.51/CAGR +12.8%/MaxDD
-40.0%. Correlazione dei rendimenti mensili di STRATEGIA (non dei prezzi
grezzi) 0.452 - positiva e non banale, ma lontana da 1. Il paniere aggregato
recupera piu' Sharpe di qualsiasi fattore-singola individuale testato in questa
sessione (0.11-0.35), confermando che parte del problema nei test precedenti
era rumore idiosincratico a livello di singola carta - ma non chiude il gap
col box (0.51 contro 0.99), e il drawdown resta 3x peggiore. Walk-forward sul
paniere: H1 Sharpe -0.32, H2 Sharpe +0.11 - entrambi debolissimi, molto lontani
dalla persistenza vista sui box reali (H1 -0.10, H2 +1.29). Conclusione: il box
sigillato NON e' spiegabile come "media delle sue carte" - ha una dinamica di
prezzo propria (premio da sigillato/scarsita' fisica, veicolo preferito da
collezionisti/speculatori, piu' liquido e senza rischio di grading/contraffazione
rispetto alle singole) che va oltre quello che le sue stesse carte producono.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.engine.strategies.time_series_momentum import TimeSeriesMomentumStrategy
from poke_quant.validation.statistical_validation import deflated_sharpe_ratio, pbo_cscv
from poke_quant.validation.bootstrap import block_bootstrap_metrics, summarize_bootstrap
from scripts.optimize_and_falsify import run_bt, MODERN_ERA_CUTOFF


def main():
    metadata = load_metadata()
    prices_full = load_price_matrix()
    single_to_box = json.load(open("data_cache/single_to_box_map.json"))
    box_to_singles = {}
    for single_id, box_id in single_to_box.items():
        box_to_singles.setdefault(box_id, []).append(single_id)

    sealed_ids = [
        k for k, v in metadata.items()
        if v.get("type") == "sealed" and v.get("data_quality") != "thin_unreliable"
        and k in prices_full.columns and v.get("release_date") and v["release_date"] >= MODERN_ERA_CUTOFF
    ]
    meta_sub = {k: v for k, v in metadata.items() if k in sealed_ids}
    prices_sub = prices_full[sealed_ids]

    # Costruisce l'indice sintetico "paniere di singole" per ogni box con >=3 singole note
    basket_series = {}
    for box_id in sealed_ids:
        singles = [s for s in box_to_singles.get(box_id, []) if s in prices_full.columns]
        if len(singles) < 3:
            continue
        singles_df = prices_full[singles]
        basket_ret = singles_df.pct_change().mean(axis=1, skipna=True)
        idx = (1.0 + basket_ret.fillna(0.0)).cumprod() * 100.0
        idx.iloc[0] = 100.0
        basket_series[box_id] = idx

    if len(basket_series) < 3:
        print("Troppi pochi box con >=3 singole mappate - impossibile costruire un paniere sufficiente.")
        return

    basket_df = pd.DataFrame(basket_series).reindex(prices_sub.index)
    basket_meta = {k: meta_sub[k] for k in basket_series}
    print(f"Paniere costruito per {len(basket_series)} box (su {len(sealed_ids)} nell'universo box).")

    # BOX REALE (produzione) sul sottoinsieme comparabile
    real_prices_sub = prices_sub[list(basket_series.keys())]
    strat_box = TimeSeriesMomentumStrategy(real_prices_sub, lookback_months=12)
    res_box = run_bt(strat_box, real_prices_sub, basket_meta)
    print(f"\nBOX REALE (stesso sottoinsieme, {len(basket_series)} box) | "
          f"CAGR {res_box.cagr*100:+6.2f}% | Sharpe {res_box.sharpe:5.2f} | "
          f"MaxDD {res_box.max_drawdown*100:6.2f}% | Trade {res_box.total_trades:3d}")

    # PANIERE SINGOLE (stessa strategia, sull'indice sintetico invece del box)
    strat_basket = TimeSeriesMomentumStrategy(basket_df, lookback_months=12)
    res_basket = run_bt(strat_basket, basket_df, basket_meta)
    print(f"PANIERE SINGOLE (stessi {len(basket_series)} set)      | "
          f"CAGR {res_basket.cagr*100:+6.2f}% | Sharpe {res_basket.sharpe:5.2f} | "
          f"MaxDD {res_basket.max_drawdown*100:6.2f}% | Trade {res_basket.total_trades:3d}")

    common_idx = res_box.monthly_returns.index.intersection(res_basket.monthly_returns.index)
    corr = res_box.monthly_returns.loc[common_idx].corr(res_basket.monthly_returns.loc[common_idx])
    print(f"\nCorrelazione dei rendimenti mensili di STRATEGIA (non dei prezzi grezzi): {corr:.3f}")

    perf_matrix = np.column_stack([
        res_box.monthly_returns.loc[common_idx].values,
        res_basket.monthly_returns.loc[common_idx].values,
    ])
    t_len = len(perf_matrix)
    splits = 8 if t_len >= 32 else 4
    rem = t_len % splits
    pbo = pbo_cscv(perf_matrix[rem:, :] if rem else perf_matrix, n_splits=splits)
    print(f"PBO (2 candidati, {splits} split): {pbo:.3f}")

    dsr_basket = deflated_sharpe_ratio(observed_sr=res_basket.sharpe / np.sqrt(12), n_trials=2, n_obs=len(res_basket.monthly_returns))
    print(f"DSR paniere singole (n_trials=2): {dsr_basket:.3f}")

    sims = block_bootstrap_metrics(res_basket.monthly_returns, n_sims=500, block_size=6)
    print(f"\nBlock bootstrap (500 sim, blocchi 6m) sul paniere singole:")
    print(summarize_bootstrap(sims))

    mid = len(basket_df) // 2
    h1_dates, h2_dates = basket_df.index[:mid], basket_df.index[mid:]
    print(f"\nWalk-forward H1/H2 sul paniere singole:")
    for label, dates in [("H1", h1_dates), ("H2", h2_dates)]:
        sub = basket_df.loc[dates]
        strat = TimeSeriesMomentumStrategy(sub, lookback_months=12)
        res = run_bt(strat, sub, basket_meta)
        print(f"  {label} ({dates[0].strftime('%Y-%m')}->{dates[-1].strftime('%Y-%m')}) | Sharpe {res.sharpe:5.2f} | CAGR {res.cagr*100:+6.2f}%")


if __name__ == "__main__":
    main()
