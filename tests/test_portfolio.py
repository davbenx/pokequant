"""
tests/test_portfolio.py — Unit test per la gestione del portafoglio e inventario.
"""

import pytest
from poke_quant.engine.portfolio import Portfolio


def test_portfolio_buy_and_cash():
    p = Portfolio(initial_cash=1000.0)
    # Compra 2 box a 140€
    ok = p.buy("es_bb", "Evolving Skies", "sealed", 2, 140.0, "2021-09-01")
    assert ok is True
    assert p.cash == 720.0
    assert len(p.positions) == 1
    assert p.positions["es_bb"].quantity == 2
    assert p.positions["es_bb"].total_cost == 280.0


def test_portfolio_insufficient_cash():
    p = Portfolio(initial_cash=100.0)
    ok = p.buy("es_bb", "Evolving Skies", "sealed", 1, 140.0, "2021-09-01")
    assert ok is False
    assert p.cash == 100.0
    assert len(p.positions) == 0


def test_portfolio_sell_and_pnl():
    p = Portfolio(initial_cash=1000.0)
    p.buy("es_bb", "Evolving Skies", "sealed", 2, 140.0, "2021-09-01")
    
    # Vende 1 box a 300€ dopo 1 anno
    trade = p.sell("es_bb", 1, 300.0, "2022-09-01", platform="cardmarket")
    assert trade is not None
    assert trade.quantity == 1
    assert trade.gross_proceeds == 300.0
    assert trade.net_proceeds > 280.0
    assert trade.net_pnl > 140.0  # Guadagno netto > 140€ (300 - 15 fee - 140 costo)
    assert p.positions["es_bb"].quantity == 1
    assert len(p.closed_trades) == 1
