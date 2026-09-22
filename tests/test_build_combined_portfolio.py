"""
tests/test_build_combined_portfolio.py — Smoke test per l'allocazione di produzione
(solo sealed: la sleeve singole e' sospesa, vedi docstring dello script).
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


def test_allocation_uses_full_capital_across_buy_hold_positions(module):
    allocation = module.build_sealed_allocation(10000.0)
    if not allocation:
        pytest.skip("nessuna posizione BUY/HOLD nel mese corrente")
    total_alloc = sum(alloc for _, alloc in allocation)
    assert total_alloc == pytest.approx(10000.0)
    assert all(alloc > 0 for _, alloc in allocation)


def test_allocation_has_no_singles_sleeve(module):
    # La sleeve singole e' sospesa: il modulo non deve esporre piu' compute_sleeve_weights
    # (pesatura inverse-vol tra due sleeve) - solo l'allocazione sealed singola.
    assert not hasattr(module, "compute_sleeve_weights")
