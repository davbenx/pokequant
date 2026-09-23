#!/usr/bin/env python3
"""
scripts/grade_lead_lag_test.py — Ipotesi diversa dal semplice spread di
livello (vedi grade_spread_test.py): la stessa carta gradata diversamente
potrebbe "sottovalutata" nel senso che un grado si muove e l'altro segue con
ritardo (trasmissione di momentum tra gradazioni), non solo che il rapporto
tra i due si allontana dalla propria media storica.

Due passi, sullo stesso campione di 40 carte chase (seed=42):
  1. Diagnostica pura: correlazione contemporanea e a un mese di ritardo (in
     entrambe le direzioni) tra rendimenti mensili di gradi adiacenti della
     STESSA carta.
  2. Se la diagnostica non mostra una direzione di ritardo chiara, testa
     comunque momentum_factor sul rapporto tra gradi (continuazione E
     inversione, non solo la direzione che "sembra giusta") con lo stesso
     rigore (griglia, PBO, DSR, walk-forward).

ESITO: NON VALIDATO, e la diagnostica spiega perche' prima ancora di arrivare
al backtest. La correlazione a 1 mese di ritardo e' praticamente identica in
ENTRAMBE le direzioni per ogni coppia di gradi (es. psa10/grade9: "alto guida
basso" 0,136 contro "basso guida alto" 0,141) - nessun grado anticipa
sistematicamente l'altro. Il co-movimento tra gradi della stessa carta e'
quasi tutto CONTEMPORANEO (correlazione 0,21-0,57 nello stesso mese), non
sfasato - a risoluzione mensile non c'e' ritardo di trasmissione da inseguire.

Il backtest sul momentum del rapporto confirma: PBO 61,4% (8 candidati),
DSR del vincitore 0,749 (peggio del test di mean-reversion sullo stesso
campione, DSR 0,850). Il segno vincente cambia da coppia a coppia
(momentum-continua vince in 3/4 coppie, momentum-inverti in 1/4) - se ci
fosse un vero meccanismo di trasmissione ci si aspetterebbe la stessa
direzione ovunque, non un risultato misto. Walk-forward sul vincitore:
H1 Sharpe -0,59 (solo 4 trade), H2 +1,26 (28 trade) - stesso schema di
sempre. Il campione (40 carte) non è il collo di bottiglia: e' l'assenza di
struttura lead-lag nella diagnostica stessa a chiudere la questione.
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata
from poke_quant.engine.strategies.cross_sectional_factor import CrossSectionalFactorStrategy, momentum_factor
from poke_quant.validation.statistical_validation import deflated_sharpe_ratio, pbo_cscv
from scripts.optimize_and_falsify import run_bt
from scripts.grade_spread_test import build_tier_panel

PAIRS = [("grade8", "grade7"), ("grade9", "grade8"), ("grade9_5", "grade9"),
         ("psa10", "grade9_5"), ("psa10", "grade9")]


def lead_lag_diagnostic(panels: dict):
    print(f"{'Coppia (alto/basso)':22s} {'n carte':>8s} {'corr contemp':>13s} {'alto guida basso':>17s} {'basso guida alto':>17s}")
    for high, low in PAIRS:
        common = [c for c in panels[high].columns if c in panels[low].columns]
        if len(common) < 8:
            print(f"{high + '/' + low:22s}  troppo pochi carte ({len(common)})")
            continue
        contemp, high_leads, low_leads = [], [], []
        for c in common:
            h, l = panels[high][c].dropna(), panels[low][c].dropna()
            idx = h.index.intersection(l.index)
            if len(idx) < 12:
                continue
            h_ret, l_ret = h.reindex(idx).pct_change(), l.reindex(idx).pct_change()
            ci = h_ret.dropna().index.intersection(l_ret.dropna().index)
            if len(ci) < 10:
                continue
            h_ret, l_ret = h_ret.loc[ci], l_ret.loc[ci]
            contemp.append(h_ret.corr(l_ret))
            hl = h_ret.shift(1).corr(l_ret)
            ll = l_ret.shift(1).corr(h_ret)
            if not np.isnan(hl):
                high_leads.append(hl)
            if not np.isnan(ll):
                low_leads.append(ll)
        m = lambda lst: np.nanmean(lst) if lst else float("nan")
        print(f"{high + '/' + low:22s} {len(common):>8d} {m(contemp):>13.3f} {m(high_leads):>17.3f} {m(low_leads):>17.3f}")


def momentum_on_ratio_grid(panels: dict, metadata: dict):
    results = {}
    for high, low in [("psa10", "grade9"), ("grade9_5", "grade9"), ("grade9", "grade8"), ("grade8", "grade7")]:
        common_ids = [c for c in panels[low].columns if c in panels[high].columns]
        if len(common_ids) < 10:
            continue
        low_df = panels[low][common_ids].reindex(panels[low].index.union(panels[high].index)).sort_index()
        high_df = panels[high][common_ids].reindex(low_df.index)
        ratio_df = (high_df / low_df).replace([np.inf, -np.inf], np.nan)
        meta_sub = {c: {**metadata.get(c, {}), "type": "single"} for c in common_ids}
        for direction, ascending in [("momentum-continua", False), ("momentum-inverti", True)]:
            strat = CrossSectionalFactorStrategy(ratio_df, momentum_factor, lookback_months=6, top_quantile=0.30,
                                                  ascending=ascending, rebalance_every_months=6, item_type_filter="single")
            res = run_bt(strat, low_df, meta_sub)
            name = f"{high}/{low} {direction}"
            results[name] = (res, low_df, ratio_df, meta_sub)
            print(f"  {name:32s} | CAGR {res.cagr*100:+6.2f}% | Sharpe {res.sharpe:5.2f} | "
                  f"MaxDD {res.max_drawdown*100:6.2f}% | Trade {res.total_trades:4d}")

    common_idx = None
    for res, _, _, _ in results.values():
        common_idx = res.monthly_returns.index if common_idx is None else common_idx.intersection(res.monthly_returns.index)
    perf_matrix = np.column_stack([results[k][0].monthly_returns.loc[common_idx].values for k in results])
    t_len = len(perf_matrix)
    splits = 8 if t_len >= 32 else 4
    rem = t_len % splits
    pbo = pbo_cscv(perf_matrix[rem:, :] if rem else perf_matrix, n_splits=splits)
    print(f"\nPBO ({len(results)} candidati, {splits} split): {pbo:.3f}")

    best_name = max(results, key=lambda k: results[k][0].sharpe)
    best, best_low_df, best_ratio_df, best_meta = results[best_name]
    dsr = deflated_sharpe_ratio(observed_sr=best.sharpe / np.sqrt(12), n_trials=len(results), n_obs=len(best.monthly_returns))
    print(f"Migliore: '{best_name}' Sharpe {best.sharpe:.2f} | Trade {best.total_trades} | DSR: {dsr:.3f}")

    mid = len(best_low_df) // 2
    h1_dates, h2_dates = best_low_df.index[:mid], best_low_df.index[mid:]
    ascending = "inverti" in best_name
    print(f"\nWalk-forward H1/H2 sul vincitore '{best_name}':")
    for label, dates in [("H1", h1_dates), ("H2", h2_dates)]:
        sub_low = best_low_df.loc[dates]
        sub_ratio = best_ratio_df.loc[best_ratio_df.index.intersection(dates)]
        strat = CrossSectionalFactorStrategy(sub_ratio, momentum_factor, lookback_months=6, top_quantile=0.30,
                                              ascending=ascending, rebalance_every_months=6, item_type_filter="single")
        r = run_bt(strat, sub_low, best_meta)
        print(f"  {label} ({dates[0].strftime('%Y-%m')}->{dates[-1].strftime('%Y-%m')}) | "
              f"Sharpe {r.sharpe:5.2f} | CAGR {r.cagr*100:+6.2f}% | Trade {r.total_trades}")


def main():
    ladder = json.loads((Path(__file__).resolve().parent.parent / "data_cache" / "grade_ladder_prices.json").read_text())
    metadata = load_metadata()
    panels = {t: build_tier_panel(ladder, t) for t in ["grade7", "grade8", "grade9", "grade9_5", "psa10"]}

    print("=== Diagnostica: correlazione contemporanea e a 1 mese di ritardo tra gradi adiacenti ===\n")
    lead_lag_diagnostic(panels)

    print("\n=== Momentum sul rapporto tra gradi (continuazione vs inversione) ===\n")
    momentum_on_ratio_grid(panels, metadata)


if __name__ == "__main__":
    main()
