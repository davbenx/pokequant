import pytest
import datetime
from poke_quant.data.cardmarket_bridge import (
    load_cardmarket_quotes,
    save_cardmarket_quotes,
    update_cardmarket_quote,
    get_cardmarket_live_prices,
    get_cardmarket_deep_link,
    evaluate_cardmarket_item
)


def test_seed_and_load_quotes():
    quotes = load_cardmarket_quotes()
    assert "stellar_crown_bb" in quotes
    assert "temporal_forces_bb" in quotes
    assert "jp_vstar_universe_bb" in quotes
    assert "op06_wings_captain_bb" in quotes
    assert quotes["stellar_crown_bb"]["last_verified_price"] == 249.65
    assert quotes["temporal_forces_bb"]["last_verified_price"] == 225.00


def test_update_quote():
    ok = update_cardmarket_quote("temporal_forces_bb", 230.00, language="en", source="Test Source")
    assert ok is True
    quotes = load_cardmarket_quotes()
    assert quotes["temporal_forces_bb"]["last_verified_price"] == 230.00
    # Ripristina
    update_cardmarket_quote("temporal_forces_bb", 225.00, language="en")


def test_get_live_prices_filtered():
    en_prices = get_cardmarket_live_prices(language_filter="en")
    assert "stellar_crown_bb" in en_prices
    assert "jp_vstar_universe_bb" not in en_prices

    jp_prices = get_cardmarket_live_prices(language_filter="jp")
    assert "jp_vstar_universe_bb" in jp_prices
    assert "stellar_crown_bb" not in jp_prices

    op_prices = get_cardmarket_live_prices(franchise_filter="one_piece")
    assert "op06_wings_captain_bb" in op_prices
    assert "temporal_forces_bb" not in op_prices


def test_deep_link_generation():
    url_en = get_cardmarket_deep_link("Temporal Forces", franchise="pokemon", language="en")
    assert "idLanguage=1" in url_en
    assert "Pokemon" in url_en

    url_jp = get_cardmarket_deep_link("VSTAR Universe", franchise="pokemon", language="jp")
    assert "idLanguage=2" in url_jp

    url_op = get_cardmarket_deep_link("Wings of the Captain", franchise="one_piece", language="en")
    assert "OnePiece" in url_op
    assert "idLanguage=1" in url_op


def test_evaluate_item_blocked_vs_buy():
    meta_tf = {
        "name": "Temporal Forces Booster Box",
        "msrp": 160.0,
        "release_date": "2024-03-22",
        "set_tier": "B"
    }
    # Simuliamo mese 6 con prezzo 225€ (Cap 184€) -> BLOCKED
    res_high = evaluate_cardmarket_item("temporal_forces_bb", meta_tf, live_price=225.00, current_date=datetime.date(2024, 9, 1))
    assert res_high["status_code"] == "BLOCKED_OVERPRICED"
    assert res_high["delta_vs_cap"] > 0

    # Simuliamo mese 6 con prezzo 125€ -> BUY_SIGNAL
    res_low = evaluate_cardmarket_item("temporal_forces_bb", meta_tf, live_price=125.00, current_date=datetime.date(2024, 9, 1))
    assert res_low["status_code"] == "BUY_SIGNAL"
    assert res_low["delta_vs_cap"] < 0
