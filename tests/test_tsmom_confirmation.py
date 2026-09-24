"""
tests/test_tsmom_confirmation.py — Copertura per min_confirm_months su
TimeSeriesMomentumStrategy (poke_quant/engine/strategies/time_series_momentum.py):
richiede N mesi CONSECUTIVI di momentum positivo prima di aprire una posizione,
non solo il mese corrente. Parametro NON adottato in produzione (default=1,
comportamento originale) - vedi scripts/sealed_momentum_confirmation_search.py
per l'esito (NON validato: PBO 87%, DSR 0,634 sotto anche il baseline). Resta
nel codice, testato, per chi volesse esplorarlo ulteriormente.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.engine.portfolio import Portfolio
from poke_quant.engine.strategies.time_series_momentum import TimeSeriesMomentumStrategy


def _snapshot(prices: dict, meta: dict) -> dict:
    snap = {}
    for item_id, px in prices.items():
        info = dict(meta[item_id])
        info["current_price"] = px
        snap[item_id] = info
    return snap


# d0->d1 negativo, d1->d2/d2->d3/d3->d4 positivi consecutivi (streak 1,2,3 a d2,d3,d4).
_DATES = pd.date_range("2021-01-01", periods=5, freq="MS")
_VALUES = [100.0, 90.0, 95.0, 99.0, 104.0]
_PRICES_DF = pd.DataFrame({"box_a": _VALUES}, index=_DATES)
_META = {"box_a": {"type": "sealed"}}


def test_default_min_confirm_one_buys_on_first_positive_month():
    strat = TimeSeriesMomentumStrategy(_PRICES_DF, lookback_months=1)  # default min_confirm_months=1
    portfolio = Portfolio(initial_cash=10000.0)
    snap = _snapshot({"box_a": 95.0}, _META)  # d2, streak=1 (primo mese positivo)
    signals = strat.generate_signals("2021-03-01", portfolio, snap)
    assert any(s.action == "BUY" and s.item_id == "box_a" for s in signals)


def test_confirm_months_blocks_entry_before_streak_is_long_enough():
    strat = TimeSeriesMomentumStrategy(_PRICES_DF, lookback_months=1, min_confirm_months=3)
    portfolio = Portfolio(initial_cash=10000.0)
    snap = _snapshot({"box_a": 99.0}, _META)  # d3, streak=2 (< 3 richiesti)
    signals = strat.generate_signals("2021-04-01", portfolio, snap)
    assert not any(s.action == "BUY" and s.item_id == "box_a" for s in signals), (
        "solo 2 mesi consecutivi positivi, min_confirm_months=3 non deve ancora comprare"
    )


def test_confirm_months_allows_entry_once_streak_reached():
    strat = TimeSeriesMomentumStrategy(_PRICES_DF, lookback_months=1, min_confirm_months=3)
    portfolio = Portfolio(initial_cash=10000.0)
    snap = _snapshot({"box_a": 104.0}, _META)  # d4, streak=3 (== 3 richiesti)
    signals = strat.generate_signals("2021-05-01", portfolio, snap)
    assert any(s.action == "BUY" and s.item_id == "box_a" for s in signals), (
        "3 mesi consecutivi positivi devono soddisfare min_confirm_months=3"
    )
