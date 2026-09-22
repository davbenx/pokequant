"""
tests/test_generate_carry_signal_singles.py — Smoke test per lo script del
segnale Carry/Scarsità su singole gradate.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "generate_carry_signal_singles.py"


@pytest.fixture(scope="module")
def signal_module():
    spec = importlib.util.spec_from_file_location("generate_carry_signal_singles", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_carry_signal_singles"] = module
    spec.loader.exec_module(module)
    return module


def test_constants_are_sane(signal_module):
    assert 0 < signal_module.TOP_QUANTILE <= 1.0
    assert signal_module.MIN_AGE_MONTHS >= 0
    assert signal_module.PLAUSIBILITY_CAP_EUR > 0
