"""
tests/test_relative_value_factor.py — Copertura per RelativeValueFactorStrategy
(regressione cross-sezionale log-prezzo ~ caratteristiche, compra il residuo
piu' negativo), aggiunta per testare l'ipotesi "carta sottovalutata rispetto
ai suoi pari" richiesta dall'utente.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.engine.portfolio import Portfolio
from poke_quant.engine.strategies.relative_value_factor import RelativeValueFactorStrategy


def _snapshot(prices: dict, meta: dict) -> dict:
    snap = {}
    for item_id, px in prices.items():
        info = dict(meta[item_id])
        info["current_price"] = px
        snap[item_id] = info
    return snap


def _make_meta(n: int, release_date: str = "2019-01-01") -> dict:
    return {
        f"card_{i}": {"type": "single", "release_date": release_date, "rarity": "Common",
                      "franchise": "pokemon", "language": "en", "selection_method": "random_control"}
        for i in range(n)
    }


def test_buys_the_card_priced_below_its_peers():
    portfolio = Portfolio(initial_cash=10000.0)
    meta = _make_meta(25)
    # Tutte le carte a 100, tranne una a 40 (nessuna differenza di eta'/rarita'/franchise/lingua)
    # -> il residuo della regressione deve essere piu' negativo per la carta sottoprezzata.
    prices = {f"card_{i}": 100.0 for i in range(25)}
    prices["card_5"] = 40.0

    strat = RelativeValueFactorStrategy(rebalance_every_months=1, top_quantile=0.10,
                                         min_age_months=0, min_cross_section=10)
    snap = _snapshot(prices, meta)
    signals = strat.generate_signals("2024-01-01", portfolio, snap)

    buys = [s for s in signals if s.action == "BUY"]
    assert len(buys) >= 1
    assert any(s.item_id == "card_5" for s in buys)


def test_skips_rebalance_months_and_respects_min_cross_section():
    portfolio = Portfolio(initial_cash=10000.0)
    meta = _make_meta(25)
    prices = {f"card_{i}": 100.0 for i in range(25)}
    prices["card_5"] = 40.0

    strat = RelativeValueFactorStrategy(rebalance_every_months=3, top_quantile=0.10,
                                         min_age_months=0, min_cross_section=10)
    snap = _snapshot(prices, meta)
    # Primo mese: ribilancia (call_count 0 % 3 == 0)
    first = strat.generate_signals("2024-01-01", portfolio, snap)
    assert any(s.action == "BUY" for s in first)
    # Secondo mese: non ribilancia
    second = strat.generate_signals("2024-02-01", portfolio, snap)
    assert second == []


def test_returns_no_signals_below_min_cross_section():
    portfolio = Portfolio(initial_cash=10000.0)
    meta = _make_meta(5)
    prices = {f"card_{i}": 100.0 for i in range(5)}

    strat = RelativeValueFactorStrategy(rebalance_every_months=1, top_quantile=0.20,
                                         min_age_months=0, min_cross_section=10)
    snap = _snapshot(prices, meta)
    signals = strat.generate_signals("2024-01-01", portfolio, snap)
    assert signals == []


def test_is_chase_control_variable_present_in_regression():
    portfolio = Portfolio(initial_cash=10000.0)
    meta = _make_meta(20)
    for i in range(10):
        meta[f"card_{i}"]["selection_method"] = "chase_price_filter_survivorship_biased"
    # Tutte le chase a 150 (prezzo alto "di base" per essere chase), tutte le control a 100,
    # tranne una control sottoprezzata a 40 - deve emergere comunque come piu' sottovalutata.
    prices = {}
    for i in range(20):
        is_chase = i < 10
        prices[f"card_{i}"] = 150.0 if is_chase else 100.0
    prices["card_15"] = 40.0

    strat = RelativeValueFactorStrategy(rebalance_every_months=1, top_quantile=0.10,
                                         min_age_months=0, min_cross_section=10)
    snap = _snapshot(prices, meta)
    signals = strat.generate_signals("2024-01-01", portfolio, snap)
    buys = [s for s in signals if s.action == "BUY"]
    assert any(s.item_id == "card_15" for s in buys)
