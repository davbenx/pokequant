#!/usr/bin/env python3
"""
scripts/scarcity_value_singles_test.py — Testa ScarcityValueFactorStrategy
(regressione cross-sezionale con scarsita' continua invece di rarita'
categoriale, vedi poke_quant/engine/strategies/scarcity_value_factor.py) su
864 singole (chase + controllo). Stesso standard di rigore di tutta questa
ricerca, con la correzione per multiple comparisons applicata FIN DALL'INIZIO
(lezione imparata dal fattore di valore relativo e dalla teoria EV del box,
entrambi promettenti sulla propria griglia e falliti sotto correzione piena).

ESITO: PRIMO FATTORE SULLE SINGOLE A SUPERARE IL DSR CORRETTO PER L'INTERA
RICERCA (0,943 su 51 trial totali). Vedi il docstring completo in
poke_quant/engine/strategies/scarcity_value_factor.py per tutti i numeri e le
verifiche di robustezza (walk-forward, campione di controllo, perturbazione
della tabella di probabilita').
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.engine.strategies.scarcity_value_factor import ScarcityValueFactorStrategy
from poke_quant.validation.statistical_validation import deflated_sharpe_ratio, pbo_cscv
from poke_quant.validation.bootstrap import block_bootstrap_metrics, summarize_bootstrap
from scripts.optimize_and_falsify import run_bt

# Trial gia' tentati sulle singole in questa sessione (vedi
# scripts/singles_session_correction.py) + questa griglia.
PRIOR_SINGLES_TRIALS = 46


def main():
    metadata = load_metadata()
    prices_full = load_price_matrix("historical_prices_graded_singles_grade9.csv")
    singles_ids = [
        k for k, v in metadata.items()
        if v.get("type") == "single" and v.get("data_quality") != "thin_unreliable" and k in prices_full.columns
    ]
    meta_sub = {k: metadata[k] for k in singles_ids}
    prices_sub = prices_full[singles_ids]
    print(f"Universo: {len(singles_ids)} singole (chase + controllo)")

    configs = {
        "SCARCITY rebal=6 q=0.20": dict(rebalance_every_months=6, top_quantile=0.20, min_age_months=6),
        "SCARCITY rebal=3 q=0.20": dict(rebalance_every_months=3, top_quantile=0.20, min_age_months=6),
        "SCARCITY rebal=6 q=0.10": dict(rebalance_every_months=6, top_quantile=0.10, min_age_months=6),
        "SCARCITY rebal=6 q=0.30": dict(rebalance_every_months=6, top_quantile=0.30, min_age_months=6),
        "SCARCITY rebal=12 q=0.20": dict(rebalance_every_months=12, top_quantile=0.20, min_age_months=6),
    }
    results = {}
    for name, kw in configs.items():
        strat = ScarcityValueFactorStrategy(**kw)
        res = run_bt(strat, prices_sub, meta_sub)
        results[name] = res
        print(f"  {name:26s} | CAGR {res.cagr*100:+6.2f}% | Sharpe {res.sharpe:5.2f} | "
              f"MaxDD {res.max_drawdown*100:6.2f}% | Trade {res.total_trades:4d}")

    common_idx = None
    for res in results.values():
        common_idx = res.monthly_returns.index if common_idx is None else common_idx.intersection(res.monthly_returns.index)
    perf_matrix = np.column_stack([results[k].monthly_returns.loc[common_idx].values for k in configs])
    t_len = len(perf_matrix)
    splits = 8 if t_len >= 32 else 4
    rem = t_len % splits
    pbo = pbo_cscv(perf_matrix[rem:, :] if rem else perf_matrix, n_splits=splits)
    print(f"\nPBO (5 candidati, {splits} split): {pbo:.3f}")

    best_name = max(results, key=lambda k: results[k].sharpe)
    best = results[best_name]
    total_trials = PRIOR_SINGLES_TRIALS + len(configs)
    dsr_own = deflated_sharpe_ratio(observed_sr=best.sharpe / np.sqrt(12), n_trials=len(configs), n_obs=len(best.monthly_returns))
    dsr_full = deflated_sharpe_ratio(observed_sr=best.sharpe / np.sqrt(12), n_trials=total_trials, n_obs=len(best.monthly_returns))
    print(f"Migliore: '{best_name}' Sharpe {best.sharpe:.2f}")
    print(f"DSR (solo griglia propria, n_trials=5): {dsr_own:.3f}")
    print(f"DSR (intera ricerca sulle singole, n_trials={total_trials} = {PRIOR_SINGLES_TRIALS} precedenti + 5 di qui): {dsr_full:.3f}")

    sims = block_bootstrap_metrics(best.monthly_returns, n_sims=500, block_size=6)
    print(summarize_bootstrap(sims))

    mid = len(prices_sub) // 2
    h1_dates, h2_dates = prices_sub.index[:mid], prices_sub.index[mid:]
    print(f"\nWalk-forward H1/H2 sul vincitore '{best_name}':")
    for label, dates in [("H1", h1_dates), ("H2", h2_dates)]:
        sub = prices_sub.loc[dates]
        strat = ScarcityValueFactorStrategy(**configs[best_name])
        res = run_bt(strat, sub, meta_sub)
        print(f"  {label} ({dates[0].strftime('%Y-%m')}->{dates[-1].strftime('%Y-%m')}) | Sharpe {res.sharpe:5.2f} | CAGR {res.cagr*100:+6.2f}%")

    # Verifica decisiva: tiene anche nel solo campione di controllo casuale
    # (mai selezionato sul prezzo, libero da survivorship bias)?
    print(f"\nVerifica su solo controllo casuale (no survivorship bias):")
    control_ids = [k for k in singles_ids if metadata[k].get("selection_method") == "random_control"]
    control_meta = {k: metadata[k] for k in control_ids}
    control_prices = prices_full[control_ids]
    strat = ScarcityValueFactorStrategy(**configs[best_name], min_cross_section=10)
    res_control = run_bt(strat, control_prices, control_meta)
    print(f"  Solo controllo (n={len(control_ids)}) | CAGR {res_control.cagr*100:+6.2f}% | Sharpe {res_control.sharpe:5.2f} | "
          f"MaxDD {res_control.max_drawdown*100:6.2f}% | Trade {res_control.total_trades}")
    mid_c = len(control_prices) // 2
    for label, dates in [("H1", control_prices.index[:mid_c]), ("H2", control_prices.index[mid_c:])]:
        sub = control_prices.loc[dates]
        s = ScarcityValueFactorStrategy(**configs[best_name], min_cross_section=10)
        r = run_bt(s, sub, control_meta)
        print(f"    {label} | Sharpe {r.sharpe:5.2f} | CAGR {r.cagr*100:+6.2f}%")


if __name__ == "__main__":
    main()
