"""
tests/test_scarcity_value_factor_sealed.py — Copertura per
SealedScarcityValueFactorStrategy (regressione cross-sezionale log-prezzo ~
log-MSRP + eta' + franchise, equivalente box del fattore scarsita' delle
singole - vedi poke_quant/engine/strategies/scarcity_value_factor_sealed.py
per il perche' MSRP sostituisce la rarita' come proxy di scarsita').
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.engine.portfolio import Portfolio, Position
from poke_quant.engine.strategies.scarcity_value_factor_sealed import SealedScarcityValueFactorStrategy


def _snapshot(prices: dict, meta: dict) -> dict:
    snap = {}
    for item_id, px in prices.items():
        info = dict(meta[item_id])
        info["current_price"] = px
        snap[item_id] = info
    return snap


def _make_meta(n: int, msrp: float = 100.0) -> dict:
    return {
        f"box_{i}": {"type": "sealed", "release_date": "2019-01-01", "msrp": msrp,
                     "franchise": "pokemon", "language": "en"}
        for i in range(n)
    }


def test_buys_the_underpriced_box_given_its_msrp_tier():
    portfolio = Portfolio(initial_cash=10000.0)
    meta = _make_meta(25)
    prices = {f"box_{i}": 300.0 for i in range(25)}
    prices["box_5"] = 120.0  # stesso MSRP delle altre, molto piu' a buon mercato

    strat = SealedScarcityValueFactorStrategy(rebalance_every_months=1, top_quantile=0.10,
                                               min_age_months=0, min_cross_section=10)
    signals = strat.generate_signals("2024-01-01", portfolio, _snapshot(prices, meta))
    buys = [s for s in signals if s.action == "BUY"]
    assert any(s.item_id == "box_5" for s in buys)


def test_higher_msrp_tier_gets_a_higher_implied_fair_price():
    """Due fasce di MSRP con il loro prezzo 'normale' (MSRP 50 -> ~150EUR,
    MSRP 140 -> ~450EUR) danno al modello varianza reale da spiegare - un box
    MSRP 140 anomalo, prezzato come un box MSRP 50, deve risultare il piu'
    sottovalutato di tutti."""
    portfolio = Portfolio(initial_cash=10000.0)
    meta, prices = {}, {}
    for i in range(15):
        meta[f"small_{i}"] = {"type": "sealed", "release_date": "2019-01-01", "msrp": 50.0,
                               "franchise": "pokemon", "language": "en"}
        prices[f"small_{i}"] = 150.0 + i % 5
    for i in range(15):
        meta[f"big_{i}"] = {"type": "sealed", "release_date": "2019-01-01", "msrp": 140.0,
                             "franchise": "pokemon", "language": "en"}
        prices[f"big_{i}"] = 450.0 + i % 5
    meta["big_anomaly"] = {"type": "sealed", "release_date": "2019-01-01", "msrp": 140.0,
                            "franchise": "pokemon", "language": "en"}
    prices["big_anomaly"] = 150.0  # MSRP alto ma prezzato come un box MSRP basso

    strat = SealedScarcityValueFactorStrategy(rebalance_every_months=1, top_quantile=0.10,
                                               min_age_months=0, min_cross_section=10)
    cur_dt = pd.to_datetime("2024-01-01")
    residuals = strat._fit_residuals(cur_dt, _snapshot(prices, meta))
    assert residuals["big_anomaly"] == min(residuals.values())


def test_returns_no_signals_below_min_cross_section():
    portfolio = Portfolio(initial_cash=10000.0)
    meta = _make_meta(5)
    prices = {f"box_{i}": 100.0 for i in range(5)}
    strat = SealedScarcityValueFactorStrategy(rebalance_every_months=1, top_quantile=0.20,
                                               min_age_months=0, min_cross_section=10)
    signals = strat.generate_signals("2024-01-01", portfolio, _snapshot(prices, meta))
    assert signals == []


def test_box_without_msrp_excluded_from_regression():
    portfolio = Portfolio(initial_cash=10000.0)
    meta = _make_meta(20)
    for i in range(20, 25):
        meta[f"box_{i}"] = {"type": "sealed", "release_date": "2019-01-01", "msrp": None,
                             "franchise": "pokemon", "language": "en"}
    prices = {f"box_{i}": 100.0 for i in range(25)}
    strat = SealedScarcityValueFactorStrategy(rebalance_every_months=1, top_quantile=0.20,
                                               min_age_months=0, min_cross_section=10)
    cur_dt = pd.to_datetime("2024-01-01")
    residuals = strat._fit_residuals(cur_dt, _snapshot(prices, meta))
    assert all(f"box_{i}" not in residuals for i in range(20, 25))


def test_min_age_months_excludes_recently_released_box():
    portfolio = Portfolio(initial_cash=10000.0)
    meta = _make_meta(20)
    meta["box_recent"] = {"type": "sealed", "release_date": "2023-12-15", "msrp": 100.0,
                           "franchise": "pokemon", "language": "en"}
    prices = {f"box_{i}": 100.0 for i in range(20)}
    prices["box_recent"] = 50.0
    strat = SealedScarcityValueFactorStrategy(rebalance_every_months=1, top_quantile=0.20,
                                               min_age_months=4, min_cross_section=10)
    cur_dt = pd.to_datetime("2024-01-01")  # box_recent ha 0 mesi di eta'
    residuals = strat._fit_residuals(cur_dt, _snapshot(prices, meta))
    assert "box_recent" not in residuals


def test_sell_signal_when_box_leaves_the_cheap_quantile():
    portfolio = Portfolio(initial_cash=10000.0)
    meta = _make_meta(20)
    prices = {f"box_{i}": 100.0 for i in range(20)}
    snap = _snapshot(prices, meta)
    portfolio.positions = {
        "box_15": Position(item_id="box_15", item_name="box_15", item_type="sealed",
                            quantity=1, buy_date="2023-01-01", buy_price_unit=100.0, total_cost=100.0),
    }
    strat = SealedScarcityValueFactorStrategy(rebalance_every_months=1, top_quantile=0.10,
                                               min_age_months=0, min_cross_section=10)
    signals = strat.generate_signals("2024-01-01", portfolio, snap)
    assert any(s.action == "SELL" and s.item_id == "box_15" for s in signals)


def test_only_rebalances_on_scheduled_months():
    portfolio = Portfolio(initial_cash=10000.0)
    meta = _make_meta(20)
    prices = {f"box_{i}": 100.0 for i in range(20)}
    prices["box_5"] = 40.0
    strat = SealedScarcityValueFactorStrategy(rebalance_every_months=3, top_quantile=0.10,
                                               min_age_months=0, min_cross_section=10)
    snap = _snapshot(prices, meta)
    first = strat.generate_signals("2024-01-01", portfolio, snap)
    second = strat.generate_signals("2024-02-01", portfolio, snap)
    assert any(s.action == "BUY" for s in first)
    assert second == []
