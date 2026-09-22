#!/usr/bin/env python3
"""
scripts/build_ratio_matrix.py — Costruisce la matrice storica del rapporto
prezzo_singola / prezzo_box_dello_stesso_set, per testare un fattore ibrido
(cross_sectional_factor.py::momentum_factor o zscore_factor applicati al RAPPORTO,
non al prezzo puro della singola) mai provato prima in questa sessione.

Abbinamento singola->box: stesso release_date esatto (proxy per "stesso set" - 109
date box su 111 sono univoche, le 2 collisioni sono risolte deterministicamente
scegliendo l'item_id alfabeticamente inferiore). 825 delle 983 singole (84%) trovano
un abbinamento.

ESITO (vedi poke_quant/engine/strategies/cross_sectional_factor.py per i numeri):
NON VALIDATO - il momentum sul rapporto ha PBO 65.7% e inverte segno tra H1/H2
(Sharpe -1.35 -> +2.63), lo stesso schema trovato su ogni fattore tecnico testato.

Uso: python scripts/build_ratio_matrix.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix, ensure_cache_dir, save_price_matrix

RATIO_MATRIX_FILENAME = "historical_ratio_single_to_box.csv"
MAP_FILENAME = "single_to_box_map.json"


def build_single_to_box_map(metadata: dict, prices_sealed: pd.DataFrame) -> dict:
    box_by_date = {}
    for k, v in sorted(metadata.items()):  # sorted per determinismo sulle collisioni di data
        if v.get("type") == "sealed" and v.get("release_date") and k in prices_sealed.columns:
            box_by_date.setdefault(v["release_date"], k)

    pairs = {}
    for k, v in metadata.items():
        if v.get("type") == "single" and v.get("release_date") in box_by_date:
            pairs[k] = box_by_date[v["release_date"]]
    return pairs


def build_ratio_matrix(min_observations: int = 15) -> pd.DataFrame:
    metadata = load_metadata()
    prices_singles = load_price_matrix("historical_prices_graded_singles_grade9.csv")
    prices_sealed = load_price_matrix()

    pairs = build_single_to_box_map(metadata, prices_sealed)
    print(f"Coppie singola->box trovate: {len(pairs)}")

    ratio_data = {}
    for single_id, box_id in pairs.items():
        if single_id not in prices_singles.columns:
            continue
        s, b = prices_singles[single_id], prices_sealed[box_id]
        common = s.index.intersection(b.index)
        ratio = (s.loc[common] / b.loc[common]).replace([float("inf"), -float("inf")], None)
        if ratio.notna().sum() >= min_observations:
            ratio_data[single_id] = ratio

    ratio_df = pd.DataFrame(ratio_data).sort_index()
    save_price_matrix(ratio_df, filename=RATIO_MATRIX_FILENAME)
    with open(ensure_cache_dir() / MAP_FILENAME, "w") as f:
        json.dump(pairs, f, indent=2)
    return ratio_df


if __name__ == "__main__":
    df = build_ratio_matrix()
    print(f"Matrice rapporto: {df.shape}")
    print(f"Salvato {RATIO_MATRIX_FILENAME} e {MAP_FILENAME}")
