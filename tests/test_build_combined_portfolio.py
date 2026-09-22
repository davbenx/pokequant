"""
tests/test_build_combined_portfolio.py — Smoke test per l'allocazione combinata.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "build_combined_portfolio.py"


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location("build_combined_portfolio", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_combined_portfolio"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_sleeve_weights_sum_to_one_and_are_positive(module):
    w_sealed, w_singles, vol_sealed, vol_singles = module.compute_sleeve_weights()
    assert w_sealed > 0 and w_singles > 0
    assert w_sealed + w_singles == pytest.approx(1.0)
    assert vol_sealed > 0 and vol_singles > 0


def test_higher_volatility_sleeve_gets_lower_weight(module):
    w_sealed, w_singles, vol_sealed, vol_singles = module.compute_sleeve_weights()
    if vol_sealed > vol_singles:
        assert w_sealed < w_singles
    else:
        assert w_singles < w_sealed
