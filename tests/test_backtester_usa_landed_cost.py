"""
tests/test_backtester_usa_landed_cost.py — Unit test per
Backtester.buy_at_usa_landed_cost, aggiunto per verificare (richiesto
esplicitamente dall'utente) se comprare sistematicamente al costo sdoganato
da un venditore USA (config.py::estimate_usa_import_landed_cost) preservi
l'edge testato - vedi scripts/usa_landed_cost_edge_test.py per l'esito
(no: distrugge l'edge, specialmente sulle singole).
"""

import pandas as pd
import pytest

from poke_quant.engine.backtester import Backtester
from poke_quant.engine.strategies import Signal
from poke_quant.config import estimate_usa_import_landed_cost


class _BuyOnceStrategy:
    """Compra 1 unita' al primo mese e non fa nulla dopo - il minimo per
    isolare il calcolo di unit_price all'acquisto dal resto della logica."""

    def __init__(self, item_id: str, item_type: str):
        self.item_id = item_id
        self.item_type = item_type
        self._bought = False

    def reset(self):
        self._bought = False

    def generate_signals(self, current_date, portfolio, market_snapshot):
        if self._bought or self.item_id not in market_snapshot:
            return []
        self._bought = True
        price = market_snapshot[self.item_id]["current_price"]
        return [Signal(action="BUY", item_id=self.item_id, item_name=self.item_id,
                        item_type=self.item_type, quantity=1, target_price=price, reason="test")]


def _prices_and_meta(item_id: str, item_type: str, price: float = 100.0):
    idx = pd.date_range("2024-01-01", periods=3, freq="MS")
    prices = pd.DataFrame({item_id: [price] * 3}, index=idx)
    metadata = {item_id: {"type": item_type, "name": item_id}}
    return prices, metadata


def test_buy_at_usa_landed_cost_inflates_unit_price_for_single():
    prices, metadata = _prices_and_meta("card_a", "single", price=100.0)
    strat = _BuyOnceStrategy("card_a", "single")
    bt = Backtester(strat, prices, metadata, initial_cash=10_000.0, buy_at_usa_landed_cost=True)
    res = bt.run()

    expected_unit_price = estimate_usa_import_landed_cost(100.0, item_type="single")
    assert expected_unit_price > 100.0  # sanity: il costo sdoganato e' sempre sopra il prezzo nudo

    cash_after_buy = float(res.nav_history["cash"].iloc[0])
    assert cash_after_buy == pytest.approx(10_000.0 - expected_unit_price, abs=1e-2)


def test_buy_without_usa_landed_cost_pays_bare_price():
    """Controllo: senza il flag, l'acquisto resta al prezzo nudo (comportamento
    storico invariato) - la differenza qui sotto e' interamente dovuta al
    costo sdoganato, non ad altro."""
    prices, metadata = _prices_and_meta("card_a", "single", price=100.0)
    strat = _BuyOnceStrategy("card_a", "single")
    bt = Backtester(strat, prices, metadata, initial_cash=10_000.0)
    res = bt.run()
    cash_after_buy = float(res.nav_history["cash"].iloc[0])
    assert cash_after_buy == pytest.approx(10_000.0 - 100.0, abs=1e-2)


def test_buy_at_usa_landed_cost_and_apply_buy_side_shipping_are_mutually_exclusive():
    """buy_at_usa_landed_cost ha priorita' su apply_buy_side_shipping se
    entrambi True - non si somma la spedizione EU flat AL costo sdoganato
    pieno, sarebbe doppio conteggio."""
    prices, metadata = _prices_and_meta("card_b", "single", price=50.0)
    strat = _BuyOnceStrategy("card_b", "single")
    bt = Backtester(strat, prices, metadata, initial_cash=10_000.0,
                     apply_buy_side_shipping=True, buy_at_usa_landed_cost=True)
    res = bt.run()
    assert res.final_nav > 0  # esegue senza errori con entrambi i flag settati
