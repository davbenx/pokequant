"""
tests/test_scarcity_value_factor.py — Copertura per ScarcityValueFactorStrategy
(regressione cross-sezionale log-prezzo ~ log-scarsita' continua + controlli),
il primo fattore sulle singole a superare il DSR corretto per l'intera
ricerca (0,943 su 51 trial totali - vedi scripts/scarcity_value_singles_test.py).
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.engine.portfolio import Portfolio
from poke_quant.engine.strategies.scarcity_value_factor import ScarcityValueFactorStrategy


def _snapshot(prices: dict, meta: dict) -> dict:
    snap = {}
    for item_id, px in prices.items():
        info = dict(meta[item_id])
        info["current_price"] = px
        snap[item_id] = info
    return snap


def _make_meta(n: int, rarity: str = "Rare Holo") -> dict:
    return {
        f"card_{i}": {"type": "single", "release_date": "2019-01-01", "rarity": rarity,
                      "franchise": "pokemon", "language": "en", "selection_method": "random_control"}
        for i in range(n)
    }


def test_buys_the_underpriced_card_given_its_rarity_tier():
    portfolio = Portfolio(initial_cash=10000.0)
    meta = _make_meta(25)
    prices = {f"card_{i}": 100.0 for i in range(25)}
    prices["card_5"] = 40.0  # stessa fascia di rarita' delle altre, ma molto piu' a buon mercato

    strat = ScarcityValueFactorStrategy(rebalance_every_months=1, top_quantile=0.10,
                                         min_age_months=0, min_cross_section=10)
    signals = strat.generate_signals("2024-01-01", portfolio, _snapshot(prices, meta))
    buys = [s for s in signals if s.action == "BUY"]
    assert any(s.item_id == "card_5" for s in buys)


def test_rarer_tier_gets_a_higher_implied_fair_price():
    """Tre fasce di rarita' con il loro prezzo 'normale' (Common ~50, Rare Ultra
    ~200, Rare Secret ~500) danno al modello varianza reale da spiegare - poi
    una carta Rare Secret anomala, prezzata come una Common, deve risultare la
    piu' sottovalutata di tutte (residuo molto piu' negativo), perche' la sua
    fascia implicherebbe un prezzo atteso molto piu' alto di quello che ha."""
    portfolio = Portfolio(initial_cash=10000.0)
    meta, prices = {}, {}
    for i in range(15):
        meta[f"common_{i}"] = {"type": "single", "release_date": "2019-01-01", "rarity": "Common",
                                "franchise": "pokemon", "language": "en", "selection_method": "random_control"}
        prices[f"common_{i}"] = 48.0 + i % 5
    for i in range(10):
        meta[f"ultra_{i}"] = {"type": "single", "release_date": "2019-01-01", "rarity": "Rare Ultra",
                               "franchise": "pokemon", "language": "en", "selection_method": "random_control"}
        prices[f"ultra_{i}"] = 195.0 + i % 5
    for i in range(10):
        meta[f"secret_{i}"] = {"type": "single", "release_date": "2019-01-01", "rarity": "Rare Secret",
                                "franchise": "pokemon", "language": "en", "selection_method": "random_control"}
        prices[f"secret_{i}"] = 495.0 + i % 5
    meta["secret_anomaly"] = {"type": "single", "release_date": "2019-01-01", "rarity": "Rare Secret",
                               "franchise": "pokemon", "language": "en", "selection_method": "random_control"}
    prices["secret_anomaly"] = 50.0  # Rare Secret ma prezzata come una Common

    strat = ScarcityValueFactorStrategy(rebalance_every_months=1, top_quantile=0.10,
                                         min_age_months=0, min_cross_section=10)
    cur_dt = pd.to_datetime("2024-01-01")
    residuals = strat._fit_residuals(cur_dt, _snapshot(prices, meta))
    assert residuals["secret_anomaly"] == min(residuals.values())


def test_returns_no_signals_below_min_cross_section():
    portfolio = Portfolio(initial_cash=10000.0)
    meta = _make_meta(5)
    prices = {f"card_{i}": 100.0 for i in range(5)}
    strat = ScarcityValueFactorStrategy(rebalance_every_months=1, top_quantile=0.20,
                                         min_age_months=0, min_cross_section=10)
    signals = strat.generate_signals("2024-01-01", portfolio, _snapshot(prices, meta))
    assert signals == []


def test_legacy_continuous_scarcity_mode_still_works():
    """use_rank=False, use_log_age=False, extra_controls=False riproduce la
    specifica originale (scarsita' continua, eta' lineare, nessun controllo
    extra) - deve restare disponibile e funzionante, non solo i nuovi default."""
    portfolio = Portfolio(initial_cash=10000.0)
    meta = _make_meta(25)
    prices = {f"card_{i}": 100.0 for i in range(25)}
    prices["card_5"] = 40.0

    strat = ScarcityValueFactorStrategy(rebalance_every_months=1, top_quantile=0.10, min_age_months=0,
                                         min_cross_section=10, use_rank=False, use_log_age=False, extra_controls=False)
    signals = strat.generate_signals("2024-01-01", portfolio, _snapshot(prices, meta))
    buys = [s for s in signals if s.action == "BUY"]
    assert any(s.item_id == "card_5" for s in buys)


def test_promo_and_unknown_rarity_excluded_from_regression():
    portfolio = Portfolio(initial_cash=10000.0)
    meta = _make_meta(20)
    for i in range(20, 25):
        meta[f"card_{i}"] = {"type": "single", "release_date": "2019-01-01", "rarity": "Promo",
                              "franchise": "pokemon", "language": "en", "selection_method": "random_control"}
    prices = {f"card_{i}": 100.0 for i in range(25)}
    strat = ScarcityValueFactorStrategy(rebalance_every_months=1, top_quantile=0.20,
                                         min_age_months=0, min_cross_section=10)
    cur_dt = pd.to_datetime("2024-01-01")
    residuals = strat._fit_residuals(cur_dt, _snapshot(prices, meta))
    assert all(f"card_{i}" not in residuals for i in range(20, 25))
