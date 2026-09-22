"""
tests/test_execution_gap_calibrator.py — Unit test per la calibrazione empirica
dello scarto dashboard-price vs verified-lowest-ask.
"""

import pandas as pd
import pytest

from poke_quant.data.execution_gap_calibrator import (
    append_execution_observation, load_execution_log, get_calibrated_slippage,
    compute_premium_stats_by_tier, MIN_OBS_FOR_CALIBRATION,
)


@pytest.fixture(autouse=True)
def isolate_cache(tmp_path, monkeypatch):
    import poke_quant.data.storage as storage
    monkeypatch.setattr(storage, "CACHE_DIR", tmp_path)
    yield tmp_path


def test_append_and_load_roundtrip():
    row = append_execution_observation(
        item_id="evolving_skies_bb", dashboard_price_eur=2200.0,
        verified_lowest_ask_eur=2400.0, source="cardmarket", active_listing_count=2,
    )
    assert pytest.approx(row["premium_pct"], abs=1e-2) == 9.09

    log_df = load_execution_log()
    assert log_df is not None
    assert len(log_df) == 1
    assert pytest.approx(log_df.iloc[0]["premium_pct"], abs=1e-2) == 9.09


def test_no_log_returns_uncalibrated_default():
    value, is_calibrated, n_obs = get_calibrated_slippage(
        "some_item", metadata={"some_item": {"slippage_pct": 0.02}}, log_df=None
    )
    assert value == 0.02
    assert is_calibrated is False
    assert n_obs == 0


def test_below_threshold_stays_uncalibrated():
    for i in range(MIN_OBS_FOR_CALIBRATION - 1):
        append_execution_observation("item_a", dashboard_price_eur=100.0, verified_lowest_ask_eur=110.0)
    log_df = load_execution_log()
    value, is_calibrated, n_obs = get_calibrated_slippage(
        "item_a", metadata={"item_a": {"slippage_pct": 0.02}}, log_df=log_df
    )
    assert is_calibrated is False
    assert n_obs == MIN_OBS_FOR_CALIBRATION - 1
    assert value == 0.02


def test_above_threshold_uses_empirical_mean():
    for _ in range(MIN_OBS_FOR_CALIBRATION):
        append_execution_observation("item_b", dashboard_price_eur=100.0, verified_lowest_ask_eur=120.0)
    log_df = load_execution_log()
    value, is_calibrated, n_obs = get_calibrated_slippage(
        "item_b", metadata={"item_b": {"slippage_pct": 0.02}}, log_df=log_df
    )
    assert is_calibrated is True
    assert n_obs == MIN_OBS_FOR_CALIBRATION
    assert pytest.approx(value, abs=1e-3) == 0.20


def test_tier_fallback_when_item_below_threshold_but_tier_above():
    metadata = {
        "item_c": {"set_tier": "S", "slippage_pct": 0.02},
        "item_d": {"set_tier": "S", "slippage_pct": 0.02},
    }
    for _ in range(MIN_OBS_FOR_CALIBRATION):
        append_execution_observation("item_d", dashboard_price_eur=100.0, verified_lowest_ask_eur=130.0)
    # item_c ha zero osservazioni proprie ma condivide il tier "S" con item_d
    log_df = load_execution_log()
    value, is_calibrated, n_obs = get_calibrated_slippage("item_c", metadata=metadata, log_df=log_df)
    assert is_calibrated is True
    assert pytest.approx(value, abs=1e-3) == 0.30


def test_premium_stats_by_tier_groups_correctly():
    metadata = {"item_e": {"set_tier": "A"}, "item_f": {"set_tier": "A"}}
    append_execution_observation("item_e", dashboard_price_eur=100.0, verified_lowest_ask_eur=110.0)
    append_execution_observation("item_f", dashboard_price_eur=100.0, verified_lowest_ask_eur=130.0)
    log_df = load_execution_log()
    stats = compute_premium_stats_by_tier(log_df, metadata)
    row = stats[stats["set_tier"] == "A"].iloc[0]
    assert row["n_obs"] == 2
    assert pytest.approx(row["mean"], abs=1e-2) == 20.0
