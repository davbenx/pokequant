"""
tests/test_config_import_usa.py — Unit test per la stima del costo sdoganato
(config.py::estimate_usa_import_landed_cost), richiesta esplicitamente dopo
che l'utente ha segnalato di avere poche opportunità comprando solo da
Cardmarket EU: comprare da un venditore USA (TCGplayer/eBay.com) significa
aggiungere dogana, non solo spedizione internazionale - dal 2026-07-01 (Reg.
UE 382/2026) ogni spedizione extra-UE paga dazio, senza soglia di franchigia.
"""

import pytest

from poke_quant.config import estimate_usa_import_landed_cost, IMPORT_FROM_USA


def test_landed_cost_includes_vat_on_item_and_shipping():
    price, shipping = 100.0, IMPORT_FROM_USA.intl_shipping_single_eur
    result = estimate_usa_import_landed_cost(price, item_type="single")
    expected_vat = (price + shipping) * IMPORT_FROM_USA.vat_rate
    expected = round(price + shipping + expected_vat + IMPORT_FROM_USA.eu_customs_duty_flat_eur
                      + IMPORT_FROM_USA.courier_handling_fee_eur, 2)
    assert result == expected


def test_sealed_uses_higher_shipping_estimate_than_single():
    single_cost = estimate_usa_import_landed_cost(100.0, item_type="single")
    sealed_cost = estimate_usa_import_landed_cost(100.0, item_type="sealed")
    assert sealed_cost > single_cost


def test_national_contribution_excluded_by_default():
    without = estimate_usa_import_landed_cost(100.0, item_type="single")
    with_contribution = estimate_usa_import_landed_cost(100.0, item_type="single", include_national_contribution=True)
    assert with_contribution == pytest.approx(without + IMPORT_FROM_USA.italy_national_contribution_eur, abs=1e-2)


def test_landed_cost_always_exceeds_bare_item_price():
    """Nessuna spedizione extra-UE e' mai gratis o senza dazio - vedi la nota
    normativa in config.py."""
    assert estimate_usa_import_landed_cost(50.0, item_type="single") > 50.0
