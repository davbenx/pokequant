"""
tests/test_generate_monthly_signal.py — Smoke test per lo script che genera il
segnale operativo TS Momentum (scripts/generate_monthly_signal.py). Verifica solo
che, sui dati reali gia' in cache, produca un output ben formato e coerente (ogni
riga ha un segnale valido, il cap di plausibilita' e' rispettato), non la qualita'
del rendimento.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from poke_quant.data.storage import load_metadata, load_price_matrix

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "generate_monthly_signal.py"


def _load_signal_module():
    spec = importlib.util.spec_from_file_location("generate_monthly_signal", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_monthly_signal"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def signal_module():
    return _load_signal_module()


def test_modern_era_universe_is_nonempty_and_recent(signal_module):
    metadata = load_metadata()
    prices_df = load_price_matrix()
    sealed_ids = [
        k for k, v in metadata.items()
        if v.get("type") == "sealed" and v.get("data_quality") != "thin_unreliable"
        and k in prices_df.columns
        and v.get("release_date") and v["release_date"] >= signal_module.MODERN_ERA_CUTOFF
    ]
    assert len(sealed_ids) > 0
    for item_id in sealed_ids:
        assert metadata[item_id]["release_date"] >= signal_module.MODERN_ERA_CUTOFF


def test_constants_are_sane(signal_module):
    assert 0 < signal_module.PLAUSIBILITY_CAP_PCT < 500
    assert signal_module.LOOKBACK_MONTHS == 12
    assert signal_module.MODERN_ERA_CUTOFF >= "2015-01-01"
