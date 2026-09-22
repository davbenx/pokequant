#!/usr/bin/env python3
"""
scripts/generate_carry_signal_singles.py — Segnale operativo Carry/Scarsità per
singole già gradate (tier Grade 9). Candidato migliore per questa categoria,
non Time-Series Momentum: TSMOM sulle singole è NEGATIVO nella prima metà dello
storico (CAGR -17,5%, Sharpe -1,23, 70 trade — troppo whipsaw), mentre
Carry/Scarsità resta positivo o quasi-flat in ogni split temporale/per-annata
testato (vedi commit git per i numeri completi).

Regola meccanica: ogni 3 mesi, tieni il quantile del 30% di carte PIÙ VECCHIE
(per data di release) tra quelle correntemente prezzate, pesate equamente.
Zero trade attivi tra un ribilanciamento e l'altro — automatizzabile al massimo.

Stesso cap di plausibilità e stessa nota sulla liquidità reale non verificata
di scripts/generate_monthly_signal.py.

Uso: python scripts/generate_carry_signal_singles.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix

TOP_QUANTILE = 0.30
MIN_AGE_MONTHS = 6
PLAUSIBILITY_CAP_EUR = 3000.0  # sopra, quasi certamente rumore da mercato sottile (verificato: solo 2/226 carte lo superano)


def compute_top_ranked():
    """Ritorna (top, ranked, latest_date). Riutilizzabile da altri script/orchestratori."""
    metadata = load_metadata()
    prices_df = load_price_matrix("historical_prices_graded_singles_grade9.csv")

    singles_ids = [
        k for k, v in metadata.items()
        if v.get("type") == "single" and v.get("data_quality") != "thin_unreliable" and k in prices_df.columns
    ]
    latest_date = prices_df.index[-1]

    ranked = []
    for item_id in singles_ids:
        series = prices_df[item_id].dropna()
        series = series[series > 0]
        if series.empty:
            continue
        cur_price = float(series.iloc[-1])
        rel_dt = metadata[item_id].get("release_date")
        age_m = None
        if rel_dt:
            rd = pd.to_datetime(rel_dt)
            age_m = (latest_date.year - rd.year) * 12 + (latest_date.month - rd.month)
        if age_m is None or age_m < MIN_AGE_MONTHS:
            continue
        ranked.append((item_id, age_m, cur_price))

    ranked.sort(key=lambda x: -x[1])
    n_top = max(1, int(round(len(ranked) * TOP_QUANTILE)))
    top = ranked[:n_top]
    return top, ranked, latest_date, metadata


def main():
    top, ranked, latest_date, metadata = compute_top_ranked()

    print("=" * 100)
    print(f"  SEGNALE CARRY/SCARSITÀ — singole gradate Grade 9 — {latest_date.strftime('%Y-%m')}")
    print(f"  Universo eleggibile: {len(ranked)} carte | Quantile top {int(TOP_QUANTILE*100)}%: {len(top)} carte")
    print("  NON verifica disponibilità/prezzo eseguibile reale.")
    print("=" * 100 + "\n")

    print(f"{'Età (mesi)':>10s}  {'Prezzo':>10s}  Nome")
    print("-" * 100)
    for item_id, age_m, price in top:
        flag = "  <- VERIFICARE (prezzo sopra soglia plausibilità)" if price > PLAUSIBILITY_CAP_EUR else ""
        print(f"{age_m:>10d}  {price:>9.2f}€  {metadata[item_id].get('name', item_id)}{flag}")


if __name__ == "__main__":
    main()
