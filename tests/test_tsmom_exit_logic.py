"""
tests/test_tsmom_exit_logic.py — Copertura per i nuovi parametri di uscita di
TimeSeriesMomentumStrategy (exit_lookback_months, exit_threshold, trailing_stop_pct).
Il default di ciascuno deve riprodurre esattamente la regola originale (mom<=0 sulla
stessa finestra di ingresso) - verificato a parte con un match bit-a-bit sul backtest
reale prima di scrivere questi test.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.engine.portfolio import Portfolio, Position
from poke_quant.engine.strategies.time_series_momentum import TimeSeriesMomentumStrategy


def _snapshot(prices: dict, meta: dict) -> dict:
    snap = {}
    for item_id, px in prices.items():
        info = dict(meta[item_id])
        info["current_price"] = px
        snap[item_id] = info
    return snap


def _make_prices(values, start="2021-01-01"):
    dates = pd.date_range(start, periods=len(values), freq="MS")
    return pd.Series(values, index=dates)


def test_default_exit_matches_original_rule_mom_leq_zero():
    dates = pd.date_range("2021-01-01", periods=14, freq="MS")
    prices_df = pd.DataFrame({"box_a": [100.0] * 13 + [95.0]}, index=dates)  # -5% a 12m
    meta = {"box_a": {"type": "sealed"}}
    strat = TimeSeriesMomentumStrategy(prices_df, lookback_months=12)
    portfolio = Portfolio(initial_cash=10000.0)
    portfolio.positions = {"box_a": Position(item_id="box_a", item_name="Box A", item_type="sealed",
                                              quantity=1, buy_date="2021-01-01", buy_price_unit=100.0, total_cost=100.0)}
    snap = _snapshot({"box_a": 95.0}, meta)
    signals = strat.generate_signals("2022-02-01", portfolio, snap)
    assert any(s.action == "SELL" and s.item_id == "box_a" for s in signals)


def test_exit_threshold_gives_buffer_against_small_negative_momentum():
    dates = pd.date_range("2021-01-01", periods=14, freq="MS")
    prices_df = pd.DataFrame({"box_a": [100.0] * 13 + [97.0]}, index=dates)  # -3% a 12m
    meta = {"box_a": {"type": "sealed"}}
    # soglia -10%: un -3% non deve far vendere
    strat = TimeSeriesMomentumStrategy(prices_df, lookback_months=12, exit_threshold=-0.10)
    portfolio = Portfolio(initial_cash=10000.0)
    portfolio.positions = {"box_a": Position(item_id="box_a", item_name="Box A", item_type="sealed",
                                              quantity=1, buy_date="2021-01-01", buy_price_unit=100.0, total_cost=100.0)}
    snap = _snapshot({"box_a": 97.0}, meta)
    signals = strat.generate_signals("2022-02-01", portfolio, snap)
    assert not any(s.action == "SELL" for s in signals), "con buffer -10% un -3% non deve attivare la vendita"


def test_exit_lookback_shorter_than_entry_reacts_faster():
    dates = pd.date_range("2021-01-01", periods=14, freq="MS")
    # Rendimento 12m ancora positivo (100->110) ma ultimi 3 mesi in forte discesa (130->110)
    values = [100.0, 105.0, 108.0, 112.0, 118.0, 122.0, 125.0, 128.0, 130.0, 128.0, 122.0, 118.0, 114.0, 110.0]
    prices_df = pd.DataFrame({"box_a": values}, index=dates)
    meta = {"box_a": {"type": "sealed"}}
    strat = TimeSeriesMomentumStrategy(prices_df, lookback_months=12, exit_lookback_months=3)
    portfolio = Portfolio(initial_cash=10000.0)
    portfolio.positions = {"box_a": Position(item_id="box_a", item_name="Box A", item_type="sealed",
                                              quantity=1, buy_date="2021-01-01", buy_price_unit=100.0, total_cost=100.0)}
    snap = _snapshot({"box_a": 110.0}, meta)
    signals = strat.generate_signals("2022-02-01", portfolio, snap)
    assert any(s.action == "SELL" and s.item_id == "box_a" for s in signals), (
        "il momentum a 3m e' negativo (130->110) anche se il 12m e' ancora positivo (100->110)"
    )


def test_trailing_stop_triggers_independent_of_momentum_sign():
    dates = pd.date_range("2021-01-01", periods=14, freq="MS")
    # 12m momentum ancora ampiamente positivo (100->150) ma crollo del 30% dal massimo (200->140)
    values = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 200.0, 180.0, 160.0, 150.0, 140.0]
    prices_df = pd.DataFrame({"box_a": values}, index=dates)
    meta = {"box_a": {"type": "sealed"}}
    strat = TimeSeriesMomentumStrategy(prices_df, lookback_months=12, trailing_stop_pct=0.20)
    portfolio = Portfolio(initial_cash=10000.0)
    portfolio.positions = {"box_a": Position(item_id="box_a", item_name="Box A", item_type="sealed",
                                              quantity=1, buy_date="2021-01-01", buy_price_unit=100.0, total_cost=100.0)}
    snap = _snapshot({"box_a": 140.0}, meta)
    signals = strat.generate_signals("2022-02-01", portfolio, snap)
    sells = [s for s in signals if s.action == "SELL" and s.item_id == "box_a"]
    assert sells, "un calo del 30% dal massimo (200) deve attivare il trailing stop al 20%, anche con 12m momentum positivo"
    assert "trailing stop" in sells[0].reason.lower()


def test_trailing_stop_not_triggered_when_drawdown_from_peak_is_small():
    dates = pd.date_range("2021-01-01", periods=14, freq="MS")
    values = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 200.0, 195.0, 192.0, 190.0, 188.0]
    prices_df = pd.DataFrame({"box_a": values}, index=dates)
    meta = {"box_a": {"type": "sealed"}}
    strat = TimeSeriesMomentumStrategy(prices_df, lookback_months=12, trailing_stop_pct=0.20)
    portfolio = Portfolio(initial_cash=10000.0)
    portfolio.positions = {"box_a": Position(item_id="box_a", item_name="Box A", item_type="sealed",
                                              quantity=1, buy_date="2021-01-01", buy_price_unit=100.0, total_cost=100.0)}
    snap = _snapshot({"box_a": 188.0}, meta)
    signals = strat.generate_signals("2022-02-01", portfolio, snap)
    assert not any(s.action == "SELL" for s in signals), "un calo del 6% dal massimo non deve attivare uno stop al 20%"
