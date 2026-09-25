#!/usr/bin/env python3
"""
scripts/production_singles_revalidation_expanded_universe.py — Riverifica la
strategia PRODUZIONE delle singole (ScarcityValueFactorStrategy, PRODUCTION_PARAMS
di scripts/generate_singles_signal.py) sull'universo ESPANSO dopo:
  1. discover_random_control_singles.py --per-set 30 (campione di controllo casuale
     6x più grande: 329->2.356 carte di controllo, per correggere la contaminazione
     94-96% chase trovata testando la rarità premium - vedi
     scripts/rarity_premium_clean_control_retest.py).
  2. rebuild_prices_with_real_fx.py (pannelli prezzo ricostruiti per tutti i 3.207
     item di metadata, incluse le nuove carte di controllo).

NON è un nuovo candidato/trial - è la STESSA strategia già in produzione
(PRODUCTION_PARAMS, mai cambiati), riverificata con più dati. Non consuma un nuovo
slot nel conteggio DSR full-session cumulativo (resta n_trials_full_session=69) -
stesso trattamento già usato per i bug-fix di universo in questa ricerca (es.
grading_cost_floor_test.py, il fix della finestra Sandslash).

Domanda: più dati (quasi 2x l'universo di singole) cambiano il verdetto o lo
rafforzano? Se il fattore "scarsità" è un vero effetto strutturale (sconto vs.
pari corretto per rarità/età/set), aggiungere altre ~2.000 carte casuali (nessun
survivorship bias - scelte senza guardare il prezzo) non dovrebbe peggiorarlo.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.data.liquidity_filter import liquid_singles_ids
from poke_quant.engine.backtester import Backtester
from poke_quant.engine.strategies.scarcity_value_factor import ScarcityValueFactorStrategy
from poke_quant.validation.statistical_validation import deflated_sharpe_ratio, pbo_cscv
from poke_quant.validation.bootstrap import block_bootstrap_metrics, summarize_bootstrap
from scripts.generate_singles_signal import PRODUCTION_PARAMS

N_TRIALS_FULL_SESSION = 69  # invariato: stessa strategia, non un nuovo trial


def run_bt(strat, prices_df, metadata):
    bt = Backtester(strat, prices_df, metadata, initial_cash=10000.0, platform="cardmarket",
                     apply_liquidity_slippage=True, apply_holding_cost=True, apply_buy_side_shipping=True)
    return bt.run()


def walk_forward(res, prices_sub, meta_sub, config):
    mid = len(prices_sub) // 2
    out = []
    for label, dates in [("H1", prices_sub.index[:mid]), ("H2", prices_sub.index[mid:])]:
        sub = prices_sub.loc[dates]
        strat = ScarcityValueFactorStrategy(**config)
        r = run_bt(strat, sub, meta_sub)
        out.append((label, r.sharpe, r.cagr))
    return out


def main():
    metadata = load_metadata()
    prices_full = load_price_matrix("historical_prices_graded_singles_grade9.csv")
    singles_ids = liquid_singles_ids(metadata, prices_full)
    n_chase = sum(1 for k in singles_ids if metadata[k].get("selection_method") == "chase_price_filter_survivorship_biased")
    n_control = sum(1 for k in singles_ids if metadata[k].get("selection_method") == "random_control")
    print(f"Universo (liquid_singles_ids, espanso): {len(singles_ids)} carte "
          f"(chase: {n_chase}, controllo: {n_control} - era 653/675, ~85%/15% prima dell'espansione)")

    meta_sub = {k: metadata[k] for k in singles_ids}
    prices_sub = prices_full[singles_ids]

    print(f"\n=== Produzione (PRODUCTION_PARAMS={PRODUCTION_PARAMS}) ===")
    strat = ScarcityValueFactorStrategy(**PRODUCTION_PARAMS)
    res = run_bt(strat, prices_sub, meta_sub)
    dsr_full = deflated_sharpe_ratio(observed_sr=res.sharpe / np.sqrt(12), n_trials=N_TRIALS_FULL_SESSION, n_obs=len(res.monthly_returns))
    print(f"  Sharpe {res.sharpe:.2f} | CAGR {res.cagr*100:+.2f}% | MaxDD {res.max_drawdown*100:.2f}% | "
          f"Trade {res.total_trades} | DSR(full-session, {N_TRIALS_FULL_SESSION}) {dsr_full:.3f}")

    print("\n=== Own-grid (vicinato dei parametri di produzione, per DSR/PBO own-grid) ===")
    grid = {
        "rebal=3 q=0.20 (produzione)": PRODUCTION_PARAMS,
        "rebal=6 q=0.20": dict(PRODUCTION_PARAMS, rebalance_every_months=6),
        "rebal=3 q=0.10": dict(PRODUCTION_PARAMS, top_quantile=0.10),
        "rebal=3 q=0.30": dict(PRODUCTION_PARAMS, top_quantile=0.30),
        "rebal=12 q=0.20": dict(PRODUCTION_PARAMS, rebalance_every_months=12),
    }
    grid_results = {}
    for name, cfg in grid.items():
        s = ScarcityValueFactorStrategy(**cfg)
        r = run_bt(s, prices_sub, meta_sub)
        grid_results[name] = r
        print(f"  {name:28s} | Sharpe {r.sharpe:5.2f} | CAGR {r.cagr*100:+6.2f}% | MaxDD {r.max_drawdown*100:6.2f}%")

    common_idx = None
    for r in grid_results.values():
        common_idx = r.monthly_returns.index if common_idx is None else common_idx.intersection(r.monthly_returns.index)
    perf_matrix = np.column_stack([grid_results[k].monthly_returns.loc[common_idx].values for k in grid])
    t_len = len(perf_matrix)
    splits = 8 if t_len >= 32 else 4
    rem = t_len % splits
    pbo = pbo_cscv(perf_matrix[rem:, :] if rem else perf_matrix, n_splits=splits)
    best_name = max(grid_results, key=lambda k: grid_results[k].sharpe)
    dsr_own = deflated_sharpe_ratio(observed_sr=grid_results[best_name].sharpe / np.sqrt(12),
                                     n_trials=len(grid), n_obs=len(grid_results[best_name].monthly_returns))
    print(f"\nPBO own-grid ({len(grid)} candidati, {splits} split): {pbo:.3f}")
    print(f"DSR own-grid (migliore='{best_name}'): {dsr_own:.3f}")

    print("\n=== Bootstrap a blocchi (produzione) ===")
    sims = block_bootstrap_metrics(res.monthly_returns, n_sims=500, block_size=6)
    print(summarize_bootstrap(sims))

    print("\n=== Walk-forward H1/H2 (produzione) ===")
    for label, sharpe, cagr in walk_forward(res, prices_sub, meta_sub, PRODUCTION_PARAMS):
        print(f"  {label} | Sharpe {sharpe:5.2f} | CAGR {cagr*100:+6.2f}%")

    print(f"\n=== SOLO controllo casuale (isola dal survivorship bias, n={n_control}) ===")
    control_ids = [k for k in singles_ids if metadata[k].get("selection_method") == "random_control"]
    control_meta = {k: metadata[k] for k in control_ids}
    control_prices = prices_sub[control_ids]
    strat_c = ScarcityValueFactorStrategy(**dict(PRODUCTION_PARAMS, min_cross_section=10))
    res_c = run_bt(strat_c, control_prices, control_meta)
    print(f"  Sharpe {res_c.sharpe:.2f} | CAGR {res_c.cagr*100:+.2f}% | MaxDD {res_c.max_drawdown*100:.2f}% | Trade {res_c.total_trades}")
    for label, sharpe, cagr in walk_forward(res_c, control_prices, control_meta, dict(PRODUCTION_PARAMS, min_cross_section=10)):
        print(f"    {label} | Sharpe {sharpe:5.2f} | CAGR {cagr*100:+6.2f}%")


if __name__ == "__main__":
    main()
