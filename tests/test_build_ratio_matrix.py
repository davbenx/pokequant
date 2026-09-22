"""
tests/test_build_ratio_matrix.py — Copertura per l'abbinamento singola->box
(stesso release_date) usato dal fattore ibrido rapporto singola/box.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_ratio_matrix import build_single_to_box_map


def test_maps_single_to_the_box_with_same_release_date():
    metadata = {
        "box_a": {"type": "sealed", "release_date": "2021-08-27"},
        "single_a": {"type": "single", "release_date": "2021-08-27"},
        "single_orphan": {"type": "single", "release_date": "1999-01-09"},  # nessun box con questa data
    }
    prices_sealed = pd.DataFrame({"box_a": [100.0, 110.0]})
    pairs = build_single_to_box_map(metadata, prices_sealed)
    assert pairs == {"single_a": "box_a"}


def test_date_collision_resolved_deterministically_by_item_id():
    metadata = {
        "box_b": {"type": "sealed", "release_date": "2023-09-22"},
        "box_a": {"type": "sealed", "release_date": "2023-09-22"},  # stessa data, item_id alfabeticamente prima
        "single_x": {"type": "single", "release_date": "2023-09-22"},
    }
    prices_sealed = pd.DataFrame({"box_a": [100.0], "box_b": [200.0]})
    pairs = build_single_to_box_map(metadata, prices_sealed)
    assert pairs["single_x"] == "box_a", "a parita' di data va scelto l'item_id alfabeticamente inferiore, non l'ordine di iterazione del dict"


def test_box_missing_from_price_matrix_is_not_used_for_matching():
    metadata = {
        "box_no_prices": {"type": "sealed", "release_date": "2021-08-27"},
        "single_a": {"type": "single", "release_date": "2021-08-27"},
    }
    prices_sealed = pd.DataFrame({"other_box": [100.0]})
    pairs = build_single_to_box_map(metadata, prices_sealed)
    assert pairs == {}, "un box senza storico prezzi non deve essere usato come abbinamento"
