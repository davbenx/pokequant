"""
tests/test_liquidity_filter.py — Unit test per il filtro di attendibilità/liquidità.
"""

import pandas as pd

from poke_quant.data.liquidity_filter import compute_reliability_flags, filter_reliable


def _make_df():
    idx = pd.date_range("2021-01-01", periods=12, freq="MS")
    stable = pd.Series([100 + i for i in range(12)], index=idx)  # crescita lineare, ok
    thin_market = pd.Series([100] * 5 + [13000] + [100] * 6, index=idx)  # salto assurdo
    too_short = pd.Series([100, 110, 120], index=idx[:3])
    return pd.DataFrame({"stable_item": stable, "thin_item": thin_market, "short_item": too_short})


def test_stable_series_marked_reliable():
    flags = compute_reliability_flags(_make_df())
    ok, reason = flags["stable_item"]
    assert ok is True
    assert reason == ""


def test_extreme_monthly_jump_flagged_unreliable():
    flags = compute_reliability_flags(_make_df())
    ok, reason = flags["thin_item"]
    assert ok is False
    assert "salto mensile" in reason


def test_too_short_series_flagged_unreliable():
    flags = compute_reliability_flags(_make_df())
    ok, reason = flags["short_item"]
    assert ok is False
    assert "troppo corta" in reason


def test_filter_reliable_keeps_only_stable_column():
    filtered = filter_reliable(_make_df())
    assert list(filtered.columns) == ["stable_item"]
