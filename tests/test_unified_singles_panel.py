"""
tests/test_unified_singles_panel.py — Unit test per il pannello prezzo
unificato multi-franchise (poke_quant/data/unified_singles_panel.py),
progetto pilota Magic: The Gathering.
"""

import pandas as pd

from poke_quant.data.unified_singles_panel import build_unified_singles_price_panel


def test_prefers_grade9_when_present():
    idx = pd.date_range("2024-01-01", periods=3, freq="MS")
    grade9 = pd.DataFrame({"pokemon_card": [10.0, 11.0, 12.0]}, index=idx)
    raw = pd.DataFrame({"pokemon_card": [5.0, 5.5, 6.0]}, index=idx)
    result = build_unified_singles_price_panel(grade9, raw)
    assert list(result["pokemon_card"]) == [10.0, 11.0, 12.0]


def test_falls_back_to_raw_when_grade9_missing():
    idx = pd.date_range("2024-01-01", periods=3, freq="MS")
    grade9 = pd.DataFrame({"pokemon_card": [10.0, 11.0, 12.0]}, index=idx)
    raw = pd.DataFrame({"mtg_card": [2.0, 2.5, 3.0]}, index=idx)
    result = build_unified_singles_price_panel(grade9, raw)
    assert "mtg_card" in result.columns
    assert list(result["mtg_card"]) == [2.0, 2.5, 3.0]
    assert list(result["pokemon_card"]) == [10.0, 11.0, 12.0]


def test_falls_back_to_raw_when_grade9_all_nan():
    """Una carta puo' avere una colonna in grade9_df ma tutta NaN (nessun
    dato gradato reale, solo un placeholder) - deve comunque cadere sul raw."""
    idx = pd.date_range("2024-01-01", periods=3, freq="MS")
    grade9 = pd.DataFrame({"mtg_card": [None, None, None]}, index=idx)
    raw = pd.DataFrame({"mtg_card": [2.0, 2.5, 3.0]}, index=idx)
    result = build_unified_singles_price_panel(grade9, raw)
    assert list(result["mtg_card"]) == [2.0, 2.5, 3.0]


def test_handles_none_grade9_df():
    idx = pd.date_range("2024-01-01", periods=3, freq="MS")
    raw = pd.DataFrame({"mtg_card": [2.0, 2.5, 3.0]}, index=idx)
    result = build_unified_singles_price_panel(None, raw)
    assert list(result["mtg_card"]) == [2.0, 2.5, 3.0]


def test_union_of_date_indices():
    idx1 = pd.date_range("2024-01-01", periods=2, freq="MS")
    idx2 = pd.date_range("2024-02-01", periods=2, freq="MS")
    grade9 = pd.DataFrame({"a": [1.0, 2.0]}, index=idx1)
    raw = pd.DataFrame({"b": [3.0, 4.0]}, index=idx2)
    result = build_unified_singles_price_panel(grade9, raw)
    assert len(result.index) == 3  # 2024-01, 2024-02, 2024-03
