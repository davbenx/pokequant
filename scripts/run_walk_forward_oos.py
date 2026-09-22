#!/usr/bin/env python3
"""
scripts/run_walk_forward_oos.py — Fase 2: walk-forward / out-of-sample sul perimetro
sealed box (le 9 carte "single" nel metadata sono escluse su richiesta esplicita —
focus su sealed box + slab già certificate, basso rischio operativo umano).

NOTA PERIMETRO: le strategie su slab (poke_quant/slabs/) non sono testabili qui.
Non esiste in data_cache/ alcuna serie storica di prezzi per slab gradate — solo
scanner "live" (slab_scanner.py) senza storico salvato. Prerequisito non ancora
soddisfatto per qualunque validazione statistica lì: serve prima costruire un
pannello storico, non è uno scope di questo script.

Due test distinti, entrambi assenti dalla suite di falsificazione originale:

  A) SPLIT TEMPORALE (stabilità di regime): stesso universo, ogni strategia
     rieseguita da zero su prima metà e seconda metà della finestra storica.
     Un edge strutturale dovrebbe reggere su entrambe; un edge curve-fit su un
     singolo regime tende a sgonfiarsi nella metà che non l'ha "generato".

  B) SPLIT PER ANNATA DI ASSET (rottura dell'overfitting sull'universo): stessa
     finestra storica intera, ma universo diviso in "vecchi" (release < 2023-01-01,
     31 set) e "nuovi" (release >= 2023-01-01, 16 set) — questi ultimi con molto
     meno storico realizzato al momento in cui le regole delle strategie sono
     state scritte/discusse.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_price_matrix, load_metadata
from poke_quant.engine.backtester import Backtester
from poke_quant.engine.strategies.optimal_sealed_strategy import OptimalSealedStrategy
from poke_quant.engine.strategies.time_series_momentum import TimeSeriesMomentumStrategy
from poke_quant.engine.strategies.cross_sectional_momentum import CrossSectionalMomentumStrategy
from poke_quant.engine.strategies.carry_scarcity_factor import CarryScarcityFactorStrategy
from poke_quant.engine.strategies.equal_weight_benchmark import EqualWeightBenchmarkStrategy

VINTAGE_CUTOFF = "2023-01-01"


def make_strategies(prices_df: pd.DataFrame):
    """Factory: alcune strategie tengono uno stato interno (self._call_count, ecc.)
    e vanno ricostruite ex-novo per ogni sotto-backtest, non riusate."""
    return {
        "TS Momentum (12m)": TimeSeriesMomentumStrategy(prices_df, lookback_months=12),
        "Cross-Sectional Momentum": CrossSectionalMomentumStrategy(prices_df, lookback_months=6, top_quantile=0.30),
        "Carry/Scarsità": CarryScarcityFactorStrategy(top_quantile=0.30),
        "Equal-Weight Buy&Hold": EqualWeightBenchmarkStrategy(),
        "Optimal Sealed (baseline)": OptimalSealedStrategy(allowed_tiers=["S", "A"], min_buy_age_months=4, max_buy_age_months=14),
    }


def run_one(strategy, prices_df, metadata):
    bt = Backtester(
        strategy=strategy, historical_prices_df=prices_df, items_metadata=metadata,
        initial_cash=10000.0, platform="cardmarket",
        apply_liquidity_slippage=True, apply_holding_cost=True,
    )
    return bt.run()


def fmt(res, n_assets, n_months):
    if res is None:
        return f"  {'n/d (dati insufficienti)':60s}"
    return (f"  CAGR {res.cagr*100:+7.2f}% | Sharpe {res.sharpe:5.2f} | MaxDD {res.max_drawdown*100:6.2f}% | "
            f"Trade {res.total_trades:3d} | asset={n_assets:2d} mesi={n_months:2d}")


def main():
    prices_full = load_price_matrix()
    metadata_full = load_metadata()

    sealed_ids = [
        k for k, v in metadata_full.items()
        if v.get("type") == "sealed" and v.get("data_quality") != "thin_unreliable"
    ]
    metadata_sealed = {k: v for k, v in metadata_full.items() if k in sealed_ids}
    prices_sealed = prices_full[[c for c in prices_full.columns if c in sealed_ids]]

    print("=" * 100)
    print("  FASE 2 — WALK-FORWARD / OUT-OF-SAMPLE (perimetro: sealed box, single escluse)")
    print(f"  Universo sealed: {len(metadata_sealed)} asset | Escluse (single, fuori perimetro): "
          f"{len(metadata_full) - len(metadata_sealed)}")
    print("=" * 100)

    # -----------------------------------------------------------------
    # TEST A: SPLIT TEMPORALE
    # -----------------------------------------------------------------
    mid_idx = len(prices_sealed) // 2
    first_half = prices_sealed.iloc[:mid_idx]
    second_half = prices_sealed.iloc[mid_idx:]
    print(f"\nTEST A — SPLIT TEMPORALE: "
          f"H1 {first_half.index[0].strftime('%Y-%m')}->{first_half.index[-1].strftime('%Y-%m')} "
          f"({len(first_half)}m) | H2 {second_half.index[0].strftime('%Y-%m')}->{second_half.index[-1].strftime('%Y-%m')} "
          f"({len(second_half)}m)")
    print("-" * 100)

    for name in make_strategies(prices_sealed).keys():
        strat_h1 = make_strategies(first_half)[name]
        strat_h2 = make_strategies(second_half)[name]
        try:
            res_h1 = run_one(strat_h1, first_half, metadata_sealed)
        except Exception as e:
            res_h1 = None
        try:
            res_h2 = run_one(strat_h2, second_half, metadata_sealed)
        except Exception as e:
            res_h2 = None
        n_assets = len(prices_sealed.columns)
        print(f"[{name}]")
        print(f"  H1: {fmt(res_h1, n_assets, len(first_half))}")
        print(f"  H2: {fmt(res_h2, n_assets, len(second_half))}")

    # -----------------------------------------------------------------
    # TEST B: SPLIT PER ANNATA DI ASSET
    # -----------------------------------------------------------------
    old_ids = [k for k in sealed_ids if metadata_sealed[k]["release_date"] < VINTAGE_CUTOFF]
    new_ids = [k for k in sealed_ids if metadata_sealed[k]["release_date"] >= VINTAGE_CUTOFF]
    prices_old = prices_sealed[[c for c in prices_sealed.columns if c in old_ids]]
    prices_new = prices_sealed[[c for c in prices_sealed.columns if c in new_ids]]
    meta_old = {k: v for k, v in metadata_sealed.items() if k in old_ids}
    meta_new = {k: v for k, v in metadata_sealed.items() if k in new_ids}

    print(f"\nTEST B — SPLIT PER ANNATA: VECCHI (release < {VINTAGE_CUTOFF}, n={len(old_ids)}) "
          f"vs NUOVI (release >= {VINTAGE_CUTOFF}, n={len(new_ids)})")
    print("-" * 100)

    for name in make_strategies(prices_sealed).keys():
        strat_old = make_strategies(prices_old)[name]
        strat_new = make_strategies(prices_new)[name]
        try:
            res_old = run_one(strat_old, prices_old, meta_old)
        except Exception:
            res_old = None
        try:
            res_new = run_one(strat_new, prices_new, meta_new)
        except Exception:
            res_new = None
        print(f"[{name}]")
        print(f"  VECCHI: {fmt(res_old, len(old_ids), len(prices_old))}")
        print(f"  NUOVI:  {fmt(res_new, len(new_ids), len(prices_new))}")

    print("\n" + "=" * 100)
    print("Lettura: un edge strutturale regge su ENTRAMBE le metà temporali ed ENTRAMBE le")
    print("annate di asset. Se un numero forte in Fase 1 sparisce qui in una delle 4 celle,")
    print("quell'edge è probabilmente concentrato in un regime/coorte specifico, non generale.")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
