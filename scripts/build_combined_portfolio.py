#!/usr/bin/env python3
"""
scripts/build_combined_portfolio.py — Allocazione di capitale per la produzione.

STATO: SOLO SEALED. La sleeve singole (Carry/Scarsita' e i 4 fattori alternativi
testati in scripts/optimize_and_falsify.py::section_singles_factor_search - TS
momentum, cross-sectional momentum, dip mean-reversion, rarita' ex-ante) e' stata
chiusa: nessuno supera la soglia istituzionale sull'universo reale e bias-auditato
(928 carte). Il migliore (dip mean-reversion) ha PBO=0.514 e inverte segno tra prima
e seconda meta' del campione - non un fattore stabile. Vedi i docstring di
poke_quant/engine/strategies/{carry_scarcity_factor,dip_mean_reversion,
rarity_tier_factor,cross_sectional_momentum}.py per lo stato di ciascuno.

Finche' non emerge un fattore validato sulle singole, il 100% del capitale va sulla
sleeve sealed (TS Momentum, DSR 0.913, l'unica strategia validata a livello
istituzionale in questo progetto). Se in futuro un fattore singole viene validato,
reintrodurre la pesatura inverse-volatilita' tra le due sleeve (vedi git history di
questo file prima di questo commit per la formula gia' pronta) invece di riscriverla
da zero, e dimensionare per eta' con poke_quant/engine/position_sizing.py come gia'
fatto qui per la sleeve sealed.

NON verifica liquidità reale — vedi OPERATIONS_ITALIA.md.

Uso: python scripts/build_combined_portfolio.py [--capital 10000]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.engine.position_sizing import age_weight
from scripts.generate_monthly_signal import compute_signal_rows


def build_sealed_allocation(capital: float):
    """Ritorna la lista (row, alloc_eur) pesata per eta' per le posizioni BUY/HOLD correnti."""
    metadata = load_metadata()
    latest_date = load_price_matrix().index[-1]

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

    return [(r, capital * (w / total_w)) for r, w in sorted(weighted, key=lambda x: -x[1])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capital", type=float, default=10000.0)
    args = parser.parse_args()

    print("=" * 100)
    print(f"  ALLOCAZIONE DI PRODUZIONE — capitale totale: {args.capital:,.2f} €")
    print("  100% box sigillati (TS Momentum) — sleeve singole sospesa, nessun fattore validato")
    print("  NON verifica liquidità reale — vedi OPERATIONS_ITALIA.md")
    print("=" * 100)

    allocation = build_sealed_allocation(args.capital)
    print(f"\n— BOX SIGILLATI ({len(allocation)} posizioni BUY/HOLD) —")
    for r, alloc in allocation:
        print(f"  {alloc:>8.2f}€  {r['name']}")


if __name__ == "__main__":
    main()
