#!/usr/bin/env python3
"""
scripts/generate_singles_signal.py — Segnale mensile per il fattore scarsita'
sulle singole (poke_quant/engine/strategies/scarcity_value_factor.py, DSR
0,943 corretto per l'intera ricerca sulle singole - vedi
scripts/scarcity_value_singles_test.py). Stesso schema di
scripts/generate_monthly_signal.py per i box, riusato dalla dashboard.
"""

from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.engine.strategies.scarcity_value_factor import ScarcityValueFactorStrategy

PRODUCTION_PARAMS = dict(top_quantile=0.20, min_age_months=6, max_positions=60, min_cross_section=20)


def compute_singles_signal_rows():
    """Ritorna (rows, latest_date). rows contiene solo il quantile BUY (residuo
    piu' negativo) - il fattore non genera un segnale AVOID esplicito come il
    momentum (e' un ranking relativo continuo, non una soglia assoluta)."""
    metadata = load_metadata()
    prices_full = load_price_matrix("historical_prices_graded_singles_grade9.csv")
    latest_date = prices_full.index[-1]
    price_row = prices_full.loc[latest_date]

    market_snapshot = {}
    for item_id, info in metadata.items():
        if item_id in price_row.index and price_row[item_id] > 0 and not pd.isna(price_row[item_id]):
            snap = dict(info)
            snap["current_price"] = float(price_row[item_id])
            market_snapshot[item_id] = snap

    strat = ScarcityValueFactorStrategy(**PRODUCTION_PARAMS)
    residuals = strat._fit_residuals(pd.to_datetime(latest_date), market_snapshot)

    n_buy = max(1, int(len(residuals) * strat.top_quantile))
    ranked = sorted(residuals.items(), key=lambda x: x[1])[:n_buy][: strat.max_positions]

    rows = []
    for item_id, residual in ranked:
        info = metadata[item_id]
        rows.append({
            "item_id": item_id,
            "name": info.get("name", item_id),
            "current_price_eur": market_snapshot[item_id]["current_price"],
            "residual": residual,
            "rarity": info.get("rarity"),
            "franchise": info.get("franchise", "pokemon"),
            "language": info.get("language", "en"),
        })
    return rows, latest_date


def main():
    rows, latest_date = compute_singles_signal_rows()
    print(f"Data segnale: {latest_date} | {len(rows)} carte nel quantile BUY (fattore scarsita')\n")
    for r in rows[:20]:
        print(f"  {r['name']:38s} {r['current_price_eur']:8.2f}€ | residuo {r['residual']:+.2f} | {r['rarity']}")


if __name__ == "__main__":
    main()
