#!/usr/bin/env python3
"""
scripts/box_max_price_ratio_test.py — Analogo, per i box sigillati, del
"prezzo massimo che preserva l'edge" gia' costruito per le singole (confine
del quantile del fattore scarsita', vedi generate_singles_signal.py). Il TS
Momentum non ha un modello di valore che produca un residuo/confine dello
stesso tipo (compra su trend, non su sconto vs. pari) - quindi non esiste un
equivalente esatto. Il candidato piu' onesto e' il rapporto prezzo/MSRP gia'
usato per decidere QUALI box vintage entrano nell'universo investibile
(poke_quant.data.liquidity_filter.MAX_PRICE_TO_MSRP_RATIO = 21.6x, calibrato
UNA VOLTA sull'universo moderno prima di guardare l'effetto sul backtest -
vedi scripts/sealed_universe_expansion_test.py): qui si testa se applicare lo
STESSO numero (non una nuova griglia - risarebbe overfitting il confine
stesso) anche come tetto per ogni NUOVO ingresso live, non solo come filtro
di ammissione all'universo, protegge davvero o e' irrilevante/dannoso.

ESITO: vedi output. Nessuna modifica silenziosa alla produzione - il
parametro max_price_msrp_ratio di TimeSeriesMomentumStrategy resta None
(nessun tetto) di default finche' non si decide esplicitamente qui.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.data.liquidity_filter import liquid_sealed_ids, MAX_PRICE_TO_MSRP_RATIO
from poke_quant.engine.backtester import Backtester
from poke_quant.engine.strategies.time_series_momentum import TimeSeriesMomentumStrategy
from poke_quant.validation.statistical_validation import deflated_sharpe_ratio, pbo_cscv


def run_bt(strat, prices_df, metadata):
    bt = Backtester(strat, prices_df, metadata, initial_cash=10000.0, platform="cardmarket",
                     apply_liquidity_slippage=True, apply_holding_cost=True, apply_buy_side_shipping=True)
    return bt.run()


def main():
    metadata = load_metadata()
    prices_full = load_price_matrix()
    sealed_ids = liquid_sealed_ids(metadata, prices_full)
    meta_sub = {k: metadata[k] for k in sealed_ids}
    prices_sub = prices_full[sealed_ids]

    n_with_msrp = sum(1 for k in sealed_ids if metadata[k].get("msrp"))
    print(f"Universo: {len(sealed_ids)} box, di cui {n_with_msrp} con MSRP noto (solo questi sono soggetti al tetto).\n")

    variants = {
        f"SENZA tetto prezzo/MSRP (prod. attuale)": None,
        f"CON tetto {MAX_PRICE_TO_MSRP_RATIO:.1f}x all'ingresso (stesso numero dell'universo)": MAX_PRICE_TO_MSRP_RATIO,
    }
    results = {}
    for label, ratio in variants.items():
        strat = TimeSeriesMomentumStrategy(prices_sub, lookback_months=12, max_price_msrp_ratio=ratio)
        res = run_bt(strat, prices_sub, meta_sub)
        results[label] = res
        print(f"  {label:60s} | Sharpe {res.sharpe:5.2f} | CAGR {res.cagr*100:+7.2f}% | "
              f"MaxDD {res.max_drawdown*100:7.2f}% | Trade {res.total_trades:3d}")

    labels = list(results.keys())
    common_idx = results[labels[0]].monthly_returns.index.intersection(results[labels[1]].monthly_returns.index)
    perf_matrix = np.column_stack([results[l].monthly_returns.loc[common_idx].values for l in labels])
    t_len = len(perf_matrix)
    splits = 8 if t_len >= 32 else 4
    rem = t_len % splits
    pbo = pbo_cscv(perf_matrix[rem:, :] if rem else perf_matrix, n_splits=splits)
    print(f"\nPBO (2 candidati, {splits} split): {pbo:.3f}")

    res_capped = results[labels[1]]
    dsr_capped = deflated_sharpe_ratio(observed_sr=res_capped.sharpe / np.sqrt(12), n_trials=2, n_obs=len(res_capped.monthly_returns))
    print(f"DSR variante con tetto (n_trials=2, solo i due candidati di questo test): {dsr_capped:.3f}")

    mid = len(prices_sub) // 2
    h1_dates, h2_dates = prices_sub.index[:mid], prices_sub.index[mid:]
    print(f"\nWalk-forward H1/H2 della variante CON tetto:")
    for wf_label, dates in [("H1", h1_dates), ("H2", h2_dates)]:
        sub = prices_sub.loc[dates]
        strat = TimeSeriesMomentumStrategy(sub, lookback_months=12, max_price_msrp_ratio=MAX_PRICE_TO_MSRP_RATIO)
        res = run_bt(strat, sub, meta_sub)
        print(f"  {wf_label} ({dates[0].strftime('%Y-%m')}->{dates[-1].strftime('%Y-%m')}) | "
              f"Sharpe {res.sharpe:5.2f} | CAGR {res.cagr*100:+6.2f}% | Trade {res.total_trades:3d}")

    n_blocked = 0
    strat_check = TimeSeriesMomentumStrategy(prices_sub, lookback_months=12)
    for item_id in sealed_ids:
        msrp = metadata[item_id].get("msrp")
        if not msrp:
            continue
        s = prices_sub[item_id].dropna()
        s = s[s > 0]
        if s.empty:
            continue
        if float(s.iloc[-1]) / float(msrp) > MAX_PRICE_TO_MSRP_RATIO:
            n_blocked += 1
    print(f"\nBox con MSRP noto oggi SOPRA il tetto {MAX_PRICE_TO_MSRP_RATIO:.1f}x (bloccati da un nuovo ingresso oggi): {n_blocked}/{n_with_msrp}")


if __name__ == "__main__":
    main()
