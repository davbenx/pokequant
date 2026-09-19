"""
tests/test_friction.py — Unit test per le frizioni e costi di mercato.
"""

import pytest
from poke_quant.engine.friction import calculate_sale_friction, evaluate_grading_arbitrage


def test_cardmarket_friction():
    # Vendita 100€ su Cardmarket, no shipping a carico venditore, 0.60€ imballaggio
    res = calculate_sale_friction(
        gross_price=100.0,
        item_type="single",
        platform="cardmarket",
        seller_absorbs_shipping=False,
        packaging_cost=0.60
    )
    # Fee = 5% di 100 = 5.0€ + 0.60€ imballaggio = 5.60€
    assert res.platform_fee == 5.0
    assert res.shipping_and_packaging == 0.60
    assert pytest.approx(res.net_proceeds, abs=1e-2) == 94.40
    assert pytest.approx(res.effective_friction_pct, abs=1e-3) == 0.056


def test_ebay_friction():
    # Vendita 200€ su eBay (12.5% + 0.35€ fisso) + imballo 0.60€
    res = calculate_sale_friction(
        gross_price=200.0,
        item_type="single",
        platform="ebay",
        seller_absorbs_shipping=False,
        packaging_cost=0.60
    )
    expected_fee = 200.0 * 0.125 + 0.35  # 25.35€
    assert pytest.approx(res.platform_fee, abs=1e-2) == expected_fee
    assert pytest.approx(res.net_proceeds, abs=1e-2) == 200.0 - expected_fee - 0.60


def test_grading_arbitrage_positive_ev():
    # Carta raw comprata a 50€, PSA 10 vale 300€, PSA 9 vale 45€, gem rate 70%
    # Costo grading 25€
    opp = evaluate_grading_arbitrage(
        raw_price=50.0,
        psa10_price=300.0,
        psa9_price=45.0,
        gem_rate=0.70,
        platform="cardmarket"
    )
    assert opp.raw_price == 50.0
    assert opp.grading_cost == 25.0
    assert opp.expected_graded_gross > 200.0
    assert opp.expected_net_profit > 50.0
    assert opp.expected_net_roi > 0.50
    assert opp.is_favorable is True


def test_grading_arbitrage_unfavorable():
    # Carta raw a 100€, PSA 10 solo a 120€: chiaramente in perdita dopo 25€ fee e rischio
    opp = evaluate_grading_arbitrage(
        raw_price=100.0,
        psa10_price=120.0,
        psa9_price=80.0,
        gem_rate=0.50,
        platform="cardmarket"
    )
    assert opp.is_favorable is False
    assert opp.expected_net_profit < 0.0
