import pytest
import datetime
import poke_quant.data.cardmarket_bridge as cardmarket_bridge
from poke_quant.data.cardmarket_bridge import (
    load_cardmarket_quotes,
    save_cardmarket_quotes,
    update_cardmarket_quote,
    get_cardmarket_live_prices,
    get_cardmarket_deep_link,
    evaluate_cardmarket_item
)


@pytest.fixture(autouse=True)
def isolate_cardmarket_cache(tmp_path, monkeypatch):
    """Evita che questi test scrivano sulla cache reale data_cache/cardmarket_live_quotes.json.
    load_cardmarket_quotes() seeda automaticamente DEFAULT_CARDMARKET_SEED su un file
    mancante, quindi puntare CACHE_FILE a un percorso vuoto in tmp_path basta."""
    monkeypatch.setattr(cardmarket_bridge, "CACHE_FILE", tmp_path / "cardmarket_live_quotes.json")
    yield


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

    # id Cardmarket per il giapponese e' 7, non 2 (2 e' il francese) - bug reale
    # trovato verificando i link, correggeva silenziosamente ogni box JP a
    # filtrare per francese se il parametro viene onorato dalla pagina.
    url_jp = get_cardmarket_deep_link("VSTAR Universe", franchise="pokemon", language="jp")
    assert "idLanguage=7" in url_jp

    url_op = get_cardmarket_deep_link("Wings of the Captain", franchise="one_piece", language="en")
    assert "OnePiece" in url_op
    assert "idLanguage=1" in url_op


def test_deep_link_single_uses_set_name_not_card_number_or_grade_text():
    """Trovato verificando i link: senza il set, una ricerca per un nome carta
    comune e' ambigua fra decine di espansioni - il game_slug (da metadata,
    stesso identificatore PriceCharting) e' il vero disambiguante. "#203"
    (indice interno PriceCharting) e "PSA 9" (testo letterale che Cardmarket
    non indicizza nel titolo prodotto raw) vanno rimossi dalla ricerca, non
    aggiunti."""
    url = get_cardmarket_deep_link(
        "Magikarp #203 Illustration Rare", franchise="pokemon", language="en",
        item_type="single", game_slug="pokemon-paldea-evolved",
    )
    assert "Paldea" in url and "Evolved" in url
    assert "%23203" not in url and "#203" not in url
    assert "PSA" not in url


def test_deep_link_sealed_still_appends_booster_box():
    url = get_cardmarket_deep_link("Shining Fates Elite Trainer Box", franchise="pokemon", language="en")
    assert "Elite%20Trainer%20Box" in url or "Elite+Trainer+Box" in url
    url2 = get_cardmarket_deep_link("Crown Zenith", franchise="pokemon", language="en")
    assert "Booster%20Box" in url2


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
