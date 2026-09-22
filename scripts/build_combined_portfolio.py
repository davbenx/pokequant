#!/usr/bin/env python3
"""
scripts/build_combined_portfolio.py — Combina i due segnali validati (TS Momentum
su box sigillati, Carry/Scarsità su singole gradate) in un'allocazione unica di
capitale, con pesatura inverse-volatilità tra i due sleeve (stessa filosofia di
vol-targeting già usata in ApexEngine — vedi apex_v2_engine.py in apex-engine) e
dimensionamento per età sulle singole posizioni (poke_quant/engine/position_sizing.py).

NON verifica liquidità reale — vedi OPERATIONS_ITALIA.md.

Uso: python scripts/build_combined_portfolio.py [--capital 10000]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.engine.backtester import Backtester
from poke_quant.engine.strategies.time_series_momentum import TimeSeriesMomentumStrategy
from poke_quant.engine.strategies.carry_scarcity_factor import CarryScarcityFactorStrategy
from poke_quant.engine.position_sizing import age_weight
from scripts.generate_monthly_signal import compute_signal_rows, MODERN_ERA_CUTOFF
from scripts.generate_carry_signal_singles import compute_top_ranked


def sleeve_volatility(strategy_factory, prices_df, metadata) -> float:
    """Volatilità annualizzata dei rendimenti mensili storici della strategia."""
    strat = strategy_factory(prices_df)
    bt = Backtester(strat, prices_df, metadata, initial_cash=10000.0, platform="cardmarket",
                     apply_liquidity_slippage=True, apply_holding_cost=True)
    res = bt.run()
    return float(res.monthly_returns.std(ddof=1) * (12 ** 0.5))


def compute_sleeve_weights():
    metadata = load_metadata()

    prices_sealed_full = load_price_matrix()
    sealed_ids = [
        k for k, v in metadata.items()
        if v.get("type") == "sealed" and v.get("data_quality") != "thin_unreliable"
        and k in prices_sealed_full.columns and v.get("release_date") and v["release_date"] >= MODERN_ERA_CUTOFF
    ]
    prices_sealed = prices_sealed_full[sealed_ids]
    meta_sealed = {k: v for k, v in metadata.items() if k in sealed_ids}
    vol_sealed = sleeve_volatility(lambda p: TimeSeriesMomentumStrategy(p, lookback_months=12), prices_sealed, meta_sealed)

    prices_singles_full = load_price_matrix("historical_prices_graded_singles_grade9.csv")
    singles_ids = [
        k for k, v in metadata.items()
        if v.get("type") == "single" and v.get("data_quality") != "thin_unreliable" and k in prices_singles_full.columns
    ]
    prices_singles = prices_singles_full[singles_ids]
    meta_singles = {k: v for k, v in metadata.items() if k in singles_ids}
    vol_singles = sleeve_volatility(lambda p: CarryScarcityFactorStrategy(top_quantile=0.30, item_type_filter="single"), prices_singles, meta_singles)

    inv_sealed, inv_singles = 1.0 / vol_sealed, 1.0 / vol_singles
    w_sealed = inv_sealed / (inv_sealed + inv_singles)
    w_singles = 1.0 - w_sealed
    return w_sealed, w_singles, vol_sealed, vol_singles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capital", type=float, default=10000.0)
    args = parser.parse_args()

    w_sealed, w_singles, vol_sealed, vol_singles = compute_sleeve_weights()
    budget_sealed = args.capital * w_sealed
    budget_singles = args.capital * w_singles

    print("=" * 100)
    print(f"  ALLOCAZIONE COMBINATA — capitale totale: {args.capital:,.2f} €")
    print(f"  Volatilità annualizzata: sealed={vol_sealed*100:.1f}% | singole={vol_singles*100:.1f}%")
    print(f"  Pesi inverse-vol: sealed={w_sealed*100:.1f}% ({budget_sealed:,.2f}€) | "
          f"singole={w_singles*100:.1f}% ({budget_singles:,.2f}€)")
    print("  NON verifica liquidità reale — vedi OPERATIONS_ITALIA.md")
    print("=" * 100)

    metadata = load_metadata()
    latest_date = load_price_matrix().index[-1]

    # --- Sleeve sealed: solo i segnali BUY/HOLD, pesati per età (nessuna release_date -> peso floor) ---
    sealed_rows, _ = compute_signal_rows()
    buy_rows = [r for r in sealed_rows if r["signal"] == "BUY/HOLD"]
    weighted = []
    for r in buy_rows:
        rel_dt = metadata[r["item_id"]].get("release_date")
        age_m = None
        if rel_dt:
            rd = pd.to_datetime(rel_dt)
            age_m = (latest_date.year - rd.year) * 12 + (latest_date.month - rd.month)
        weighted.append((r, age_weight(age_m)))
    total_w = sum(w for _, w in weighted) or 1.0

    print(f"\n— SLEEVE BOX SIGILLATI ({len(buy_rows)} posizioni BUY/HOLD) —")
    for r, w in sorted(weighted, key=lambda x: -x[1]):
        alloc = budget_sealed * (w / total_w)
        print(f"  {alloc:>8.2f}€  (peso età {w:.2f})  {r['name']}")

    # --- Sleeve singole: quantile top, pesate per età (già filtrato per min_age nel ranking) ---
    top_singles, ranked, _, singles_meta = compute_top_ranked()
    weighted_s = []
    for item_id, age_m, price in top_singles:
        weighted_s.append(((item_id, age_m, price), age_weight(age_m)))
    total_w_s = sum(w for _, w in weighted_s) or 1.0

    print(f"\n— SLEEVE SINGOLE GRADATE ({len(top_singles)} carte nel quantile) —")
    for (item_id, age_m, price), w in sorted(weighted_s, key=lambda x: -x[1])[:20]:
        alloc = budget_singles * (w / total_w_s)
        print(f"  {alloc:>8.2f}€  (peso età {w:.2f}, {age_m}m)  {singles_meta[item_id].get('name', item_id)}")
    if len(weighted_s) > 20:
        print(f"  ... e altre {len(weighted_s) - 20} carte")


if __name__ == "__main__":
    main()
