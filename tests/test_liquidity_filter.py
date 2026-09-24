"""
tests/test_liquidity_filter.py — Unit test per il filtro di attendibilità/liquidità.
"""

import pandas as pd

from poke_quant.data.liquidity_filter import compute_reliability_flags, filter_reliable, is_liquid_sealed, liquid_sealed_ids


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


def _sealed_prices():
    idx = pd.date_range("2021-01-01", periods=6, freq="MS")
    return pd.DataFrame({
        "modern_bb": [100.0] * 6,
        "vintage_liquid_bb": [500.0] * 6,   # 5x MSRP 100 - dentro range
        "vintage_grail_bb": [5000.0] * 6,   # 50x MSRP 100 - fuori range (grail)
        "vintage_no_msrp_bb": [500.0] * 6,  # nessun MSRP - non fabbricato, escluso
    })


def _sealed_meta():
    return {
        "modern_bb": {"type": "sealed", "release_date": "2021-01-01", "msrp": 100.0},
        "vintage_liquid_bb": {"type": "sealed", "release_date": "2010-01-01", "msrp": 100.0},
        "vintage_grail_bb": {"type": "sealed", "release_date": "2005-01-01", "msrp": 100.0},
        "vintage_no_msrp_bb": {"type": "sealed", "release_date": "2008-01-01", "msrp": None},
    }


def test_modern_box_always_included_regardless_of_ratio():
    meta, prices = _sealed_meta(), _sealed_prices()
    assert is_liquid_sealed("modern_bb", meta["modern_bb"], prices, max_price_msrp_ratio=10.0)


def test_vintage_box_included_if_price_msrp_ratio_within_range():
    meta, prices = _sealed_meta(), _sealed_prices()
    assert is_liquid_sealed("vintage_liquid_bb", meta["vintage_liquid_bb"], prices, max_price_msrp_ratio=10.0)


def test_vintage_grail_excluded_when_ratio_exceeds_range():
    meta, prices = _sealed_meta(), _sealed_prices()
    assert not is_liquid_sealed("vintage_grail_bb", meta["vintage_grail_bb"], prices, max_price_msrp_ratio=10.0)


def test_vintage_box_without_real_msrp_stays_excluded_not_fabricated():
    meta, prices = _sealed_meta(), _sealed_prices()
    assert not is_liquid_sealed("vintage_no_msrp_bb", meta["vintage_no_msrp_bb"], prices, max_price_msrp_ratio=10.0)


def test_liquid_sealed_ids_returns_expected_subset():
    metadata, prices = _sealed_meta(), _sealed_prices()
    ids = liquid_sealed_ids(metadata, prices, max_price_msrp_ratio=10.0)
    assert set(ids) == {"modern_bb", "vintage_liquid_bb"}
