#!/usr/bin/env python3
"""
scripts/optimize_and_falsify.py — Round di ottimizzazione/falsificazione rigorosa
per i due candidati validati:
  1. TS Momentum su box sigillati era moderna (2019+)
  2. Carry/Scarsità su singole gradate Grade 9

REGOLA METODOLOGICA (per non ricadere nell'errore che abbiamo già trovato e
corretto più volte in questa sessione): non si cerca "il parametro migliore" su
questi 69 mesi — si verifica la STABILITÀ su una griglia di parametri vicini
(un picco netto su un solo valore è un segnale di overfitting, un plateau è
rassicurante), si calcola il PBO su TUTTA la griglia (non solo tra strategie
diverse, l'uso classico di Bailey&LopezDePrado è esattamente "ho provato N
varianti, quanto è probabile che la migliore in-sample non lo sia out-of-sample"),
e si stima un intervallo di confidenza via block bootstrap invece di un singolo
numero puntuale.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.engine.backtester import Backtester
from poke_quant.engine.strategies.time_series_momentum import TimeSeriesMomentumStrategy
from poke_quant.engine.strategies.carry_scarcity_factor import CarryScarcityFactorStrategy
from poke_quant.validation.statistical_validation import deflated_sharpe_ratio, pbo_cscv
from poke_quant.validation.bootstrap import block_bootstrap_metrics, summarize_bootstrap

MODERN_ERA_CUTOFF = "2019-01-01"


def run_bt(strategy, prices_df, metadata):
    bt = Backtester(strategy, prices_df, metadata, initial_cash=10000.0, platform="cardmarket",
                     apply_liquidity_slippage=True, apply_holding_cost=True)
    return bt.run()


def section_tsmom_sealed():
    print("=" * 100)
    print("  1) TS MOMENTUM — BOX SIGILLATI ERA MODERNA (2019+)")
    print("=" * 100)
    metadata = load_metadata()
    prices_full = load_price_matrix()
    sealed_ids = [
        k for k, v in metadata.items()
        if v.get("type") == "sealed" and v.get("data_quality") != "thin_unreliable"
        and k in prices_full.columns and v.get("release_date") and v["release_date"] >= MODERN_ERA_CUTOFF
    ]
    meta_sub = {k: v for k, v in metadata.items() if k in sealed_ids}
    prices_sub = prices_full[sealed_ids]

    lookbacks = [6, 9, 12, 15, 18]
    results = {}
    print(f"\nGriglia lookback (mesi) — stabilità, non 'il migliore':")
    for lb in lookbacks:
        strat = TimeSeriesMomentumStrategy(prices_sub, lookback_months=lb)
        res = run_bt(strat, prices_sub, meta_sub)
        results[lb] = res
        print(f"  lookback={lb:2d}m | CAGR {res.cagr*100:+6.2f}% | Sharpe {res.sharpe:5.2f} | "
              f"MaxDD {res.max_drawdown*100:6.2f}% | Trade {res.total_trades:3d}")

    common_idx = None
    for res in results.values():
        common_idx = res.monthly_returns.index if common_idx is None else common_idx.intersection(res.monthly_returns.index)
    perf_matrix = np.column_stack([results[lb].monthly_returns.loc[common_idx].values for lb in lookbacks])
    t_len = len(perf_matrix)
    splits = 8 if t_len >= 32 else 4
    rem = t_len % splits
    pbo = pbo_cscv(perf_matrix[rem:, :] if rem else perf_matrix, n_splits=splits)
    print(f"\nPBO sulla griglia lookback ({splits} split): {pbo:.2f} ({pbo*100:.1f}%)")

    base = results[12]
    dsr = deflated_sharpe_ratio(observed_sr=base.sharpe / np.sqrt(12), n_trials=len(lookbacks), n_obs=len(base.monthly_returns))
    print(f"DSR lookback=12m (n_trials={len(lookbacks)}): {dsr:.3f}")

    sims = block_bootstrap_metrics(base.monthly_returns, n_sims=500, block_size=6)
    print(f"\nBlock bootstrap (500 sim, blocchi 6m) su lookback=12m:")
    print(summarize_bootstrap(sims))


def section_carry_singles():
    print("\n\n" + "=" * 100)
    print("  2) CARRY/SCARSITÀ — SINGOLE GRADATE GRADE 9")
    print("=" * 100)
    metadata = load_metadata()
    prices_full = load_price_matrix("historical_prices_graded_singles_grade9.csv")
    singles_ids = [
        k for k, v in metadata.items()
        if v.get("type") == "single" and v.get("data_quality") != "thin_unreliable" and k in prices_full.columns
    ]
    meta_sub = {k: v for k, v in metadata.items() if k in singles_ids}
    prices_sub = prices_full[singles_ids]

    quantiles = [0.20, 0.25, 0.30, 0.40, 0.50]
    results = {}
    print(f"\nGriglia top_quantile — stabilità:")
    for q in quantiles:
        strat = CarryScarcityFactorStrategy(top_quantile=q, item_type_filter="single")
        res = run_bt(strat, prices_sub, meta_sub)
        results[q] = res
        print(f"  quantile={q:.2f} | CAGR {res.cagr*100:+6.2f}% | Sharpe {res.sharpe:5.2f} | "
              f"MaxDD {res.max_drawdown*100:6.2f}% | Trade {res.total_trades:3d}")

    common_idx = None
    for res in results.values():
        common_idx = res.monthly_returns.index if common_idx is None else common_idx.intersection(res.monthly_returns.index)
    perf_matrix = np.column_stack([results[q].monthly_returns.loc[common_idx].values for q in quantiles])
    t_len = len(perf_matrix)
    splits = 8 if t_len >= 32 else 4
    rem = t_len % splits
    pbo = pbo_cscv(perf_matrix[rem:, :] if rem else perf_matrix, n_splits=splits)
    print(f"\nPBO sulla griglia top_quantile ({splits} split): {pbo:.2f} ({pbo*100:.1f}%)")

    base = results[0.30]
    dsr = deflated_sharpe_ratio(observed_sr=base.sharpe / np.sqrt(12), n_trials=len(quantiles), n_obs=len(base.monthly_returns))
    print(f"DSR quantile=0.30 (n_trials={len(quantiles)}): {dsr:.3f}")

    sims = block_bootstrap_metrics(base.monthly_returns, n_sims=500, block_size=6)
    print(f"\nBlock bootstrap (500 sim, blocchi 6m) su quantile=0.30:")
    print(summarize_bootstrap(sims))


if __name__ == "__main__":
    section_tsmom_sealed()
    section_carry_singles()
