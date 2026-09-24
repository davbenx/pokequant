"""
tests/test_log_execution_price.py — Regressione per il bug trovato loggando
Raichu #14: _latest_dashboard_price() controllava solo i pannelli legacy
(historical_prices.csv/historical_prices_europe.csv), che per le singole
hanno un valore DIVERSO dal pannello grade9 che il dashboard mostra davvero -
lo stesso bug gia' trovato e corretto in scripts/flag_unreliable_assets.py.
"""

import pandas as pd
import pytest

from scripts.log_execution_price import _latest_dashboard_price


@pytest.fixture(autouse=True)
def isolate_price_cache(tmp_path, monkeypatch):
    import poke_quant.data.storage as storage
    monkeypatch.setattr(storage, "CACHE_DIR", tmp_path)

    idx = pd.date_range("2026-08-01", periods=2, freq="MS")
    legacy_raw = pd.DataFrame({"raichu_14": [21.0, 22.86]}, index=idx)
    grade9 = pd.DataFrame({"raichu_14": [108.84, 107.60]}, index=idx)
    legacy_raw.to_csv(tmp_path / "historical_prices.csv", index_label="date")
    legacy_raw.to_csv(tmp_path / "historical_prices_europe.csv", index_label="date")
    grade9.to_csv(tmp_path / "historical_prices_graded_singles_grade9.csv", index_label="date")
    yield tmp_path


def test_single_reads_grade9_panel_not_legacy_raw():
    metadata = {"raichu_14": {"type": "single"}}
    price = _latest_dashboard_price("raichu_14", metadata)
    assert price == pytest.approx(107.60)


def test_sealed_item_falls_back_to_legacy_panels():
    metadata = {"raichu_14": {"type": "sealed"}}
    price = _latest_dashboard_price("raichu_14", metadata)
    assert price == pytest.approx(22.86)


def test_unknown_item_returns_zero():
    assert _latest_dashboard_price("does_not_exist", {}) == 0.0
