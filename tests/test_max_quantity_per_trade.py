"""
tests/test_max_quantity_per_trade.py — Copertura per max_quantity_per_trade
su TimeSeriesMomentumStrategy e ScarcityValueFactorStrategy. Trovato indagando
"e se compro più copie di box o carte?": senza tetto, qty = budget_posizione
// prezzo compra decine di copie identiche su un item economico (nel backtest
validato: fino a 95 copie di una singola carta in un mese) - nessun mercato
reale ha mai cosi' tante copie identiche disponibili insieme allo stesso
prezzo. Default None deve riprodurre esattamente il comportamento storico
(nessun tetto); l'impatto misurato di un tetto realistico è in
scripts/max_quantity_per_trade_test.py (box quasi insensibile, singole
molto sensibile - Sharpe 1,57->0,30 al tetto di 1 copia).
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.engine.portfolio import Portfolio
from poke_quant.engine.strategies.scarcity_value_factor import ScarcityValueFactorStrategy
from poke_quant.engine.strategies.time_series_momentum import TimeSeriesMomentumStrategy


def _snapshot(prices: dict, meta: dict) -> dict:
    snap = {}
    for item_id, px in prices.items():
        info = dict(meta[item_id])
        info["current_price"] = px
        snap[item_id] = info
    return snap


def test_scarcity_factor_default_has_no_cap_and_buys_many_units():
    portfolio = Portfolio(initial_cash=10000.0)
    meta = {
        f"card_{i}": {"type": "single", "release_date": "2019-01-01", "rarity": "Rare Holo",
                      "franchise": "pokemon", "language": "en", "selection_method": "random_control"}
        for i in range(25)
    }
    prices = {f"card_{i}": 100.0 for i in range(25)}
    prices["card_5"] = 2.0  # molto a buon mercato: senza tetto compra decine di copie

    strat = ScarcityValueFactorStrategy(rebalance_every_months=1, top_quantile=0.10,
                                         min_age_months=0, min_cross_section=10)
    signals = strat.generate_signals("2024-01-01", portfolio, _snapshot(prices, meta))
    buy = next(s for s in signals if s.item_id == "card_5")
    assert buy.quantity > 10, "senza tetto, una carta economica deve comprare piu' di 10 copie identiche"


def test_scarcity_factor_cap_limits_quantity():
    portfolio = Portfolio(initial_cash=10000.0)
    meta = {
        f"card_{i}": {"type": "single", "release_date": "2019-01-01", "rarity": "Rare Holo",
                      "franchise": "pokemon", "language": "en", "selection_method": "random_control"}
        for i in range(25)
    }
    prices = {f"card_{i}": 100.0 for i in range(25)}
    prices["card_5"] = 2.0

    strat = ScarcityValueFactorStrategy(rebalance_every_months=1, top_quantile=0.10,
                                         min_age_months=0, min_cross_section=10, max_quantity_per_trade=3)
    signals = strat.generate_signals("2024-01-01", portfolio, _snapshot(prices, meta))
    buy = next(s for s in signals if s.item_id == "card_5")
    assert buy.quantity == 3


def test_tsmom_default_has_no_cap_and_buys_many_units():
    dates = pd.date_range("2021-01-01", periods=14, freq="MS")
    prices_df = pd.DataFrame({"box_a": [1.0] * 13 + [1.5]}, index=dates)  # +50% a 12m, prezzo bassissimo
    meta = {"box_a": {"type": "sealed"}}
    strat = TimeSeriesMomentumStrategy(prices_df, lookback_months=12)
    portfolio = Portfolio(initial_cash=10000.0)
    signals = strat.generate_signals("2022-02-01", portfolio, _snapshot({"box_a": 1.5}, meta))
    buy = next(s for s in signals if s.item_id == "box_a")
    assert buy.quantity > 10, "senza tetto, un item a prezzo bassissimo deve comprare piu' di 10 unita'"


def test_tsmom_cap_limits_quantity():
    dates = pd.date_range("2021-01-01", periods=14, freq="MS")
    prices_df = pd.DataFrame({"box_a": [1.0] * 13 + [1.5]}, index=dates)
    meta = {"box_a": {"type": "sealed"}}
    strat = TimeSeriesMomentumStrategy(prices_df, lookback_months=12, max_quantity_per_trade=2)
    portfolio = Portfolio(initial_cash=10000.0)
    signals = strat.generate_signals("2022-02-01", portfolio, _snapshot({"box_a": 1.5}, meta))
    buy = next(s for s in signals if s.item_id == "box_a")
    assert buy.quantity == 2
