#!/usr/bin/env python3
"""
scripts/flag_unreliable_assets.py — Applica il filtro di liquidità/attendibilità
(poke_quant/data/liquidity_filter.py) a historical_prices.csv e ai pannelli
graded-singles, scrivendo un campo "data_quality" in items_metadata.json per
ogni asset che non lo supera. Non cancella nessun dato: li rende visibili e
facilmente esclusi dagli script di validazione.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, save_metadata, load_price_matrix
from poke_quant.data.liquidity_filter import compute_reliability_flags


def main():
    metadata = load_metadata()
    prices_df = load_price_matrix("historical_prices.csv")

    flags = compute_reliability_flags(prices_df)
    n_flagged = 0
    for item_id, (ok, reason) in flags.items():
        if item_id not in metadata:
            continue
        if ok:
            metadata[item_id].pop("data_quality", None)
            metadata[item_id].pop("data_quality_reason", None)
        else:
            metadata[item_id]["data_quality"] = "thin_unreliable"
            metadata[item_id]["data_quality_reason"] = reason
            n_flagged += 1

    save_metadata(metadata)
    print(f"Asset valutati: {len(flags)} | flaggati come thin_unreliable: {n_flagged}")
    n_sealed = sum(1 for k, (ok, _) in flags.items() if not ok and metadata.get(k, {}).get("type") == "sealed")
    n_single = sum(1 for k, (ok, _) in flags.items() if not ok and metadata.get(k, {}).get("type") == "single")
    print(f"  di cui sealed: {n_sealed} | single: {n_single}")


if __name__ == "__main__":
    main()
