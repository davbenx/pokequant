"""
tests/test_position_sizing.py — Unit test per la regola di dimensionamento per età.
"""

import pytest

from poke_quant.engine.position_sizing import age_weight, FLOOR_MULTIPLIER, MATURITY_MONTHS


def test_zero_age_gives_floor():
    assert age_weight(0) == pytest.approx(FLOOR_MULTIPLIER)


def test_mature_age_gives_full_weight():
    assert age_weight(MATURITY_MONTHS) == 1.0
    assert age_weight(MATURITY_MONTHS * 2) == 1.0


def test_ramp_is_monotonic_increasing():
    weights = [age_weight(m) for m in range(0, MATURITY_MONTHS + 1, 2)]
    assert weights == sorted(weights)


def test_none_or_negative_age_gives_floor():
    assert age_weight(None) == FLOOR_MULTIPLIER
    assert age_weight(-5) == FLOOR_MULTIPLIER


def test_midpoint_is_between_floor_and_one():
    mid = age_weight(MATURITY_MONTHS / 2)
    assert FLOOR_MULTIPLIER < mid < 1.0
