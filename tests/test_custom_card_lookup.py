"""
tests/test_custom_card_lookup.py — Test unitari per il parsing di URL/slug PriceCharting
e il matching intelligente di carte nel database per il Calcolatore Slab.
"""

import pytest
from poke_quant.data.price_fetcher import (
    parse_pricecharting_url_or_slug,
    search_metadata_card_by_query,
)


def test_parse_pricecharting_url_standard():
    url = "https://www.pricecharting.com/game/pokemon-base-set/charizard-4"
    res = parse_pricecharting_url_or_slug(url)
    assert res == ("pokemon-base-set", "charizard-4")


def test_parse_pricecharting_url_with_fragment_and_query():
    url = "https://www.pricecharting.com/game/pokemon-surging-sparks/pikachu-ex-238#graded"
    res = parse_pricecharting_url_or_slug(url)
    assert res == ("pokemon-surging-sparks", "pikachu-ex-238")

    url_query = "https://www.pricecharting.com/game/pokemon-team-up/gengar-&-mimikyu-gx-165?sort=highest"
    res2 = parse_pricecharting_url_or_slug(url_query)
    assert res2 == ("pokemon-team-up", "gengar-&-mimikyu-gx-165")


def test_parse_pricecharting_slug_direct():
    slug = "pokemon-base-set/charizard-4"
    res = parse_pricecharting_url_or_slug(slug)
    assert res == ("pokemon-base-set", "charizard-4")


def test_parse_pricecharting_invalid_or_empty():
    assert parse_pricecharting_url_or_slug("") is None
    assert parse_pricecharting_url_or_slug("not a valid url") is None
    assert parse_pricecharting_url_or_slug("https://google.com") is None


def test_search_metadata_card_by_query():
    mock_meta = {
        "charizard_4": {
            "name": "Charizard #4",
            "type": "single",
            "game_slug": "pokemon-base-set",
            "item_slug": "charizard-4",
            "era": "vintage",
        },
        "umbreon_vmax_215": {
            "name": "Umbreon VMAX Alternate Art (Moonbreon)",
            "type": "single",
            "game_slug": "pokemon-evolving-skies",
            "item_slug": "umbreon-vmax-215",
            "era": "modern",
        },
        "pikachu_ex_238": {
            "name": "Pikachu ex #238",
            "type": "single",
            "game_slug": "pokemon-surging-sparks",
            "item_slug": "pikachu-ex-238",
            "era": "modern",
        },
        "sealed_box": {
            "name": "Base Set Booster Box",
            "type": "sealed",
            "game_slug": "pokemon-base-set",
            "item_slug": "booster-box",
        },
    }

    # Match esatto slug
    res1 = search_metadata_card_by_query("charizard-4", mock_meta)
    assert res1 is not None
    assert res1[0] == "charizard_4"

    # Match parole chiave
    res2 = search_metadata_card_by_query("moonbreon", mock_meta)
    assert res2 is not None
    assert res2[0] == "umbreon_vmax_215"

    res3 = search_metadata_card_by_query("pikachu surging", mock_meta)
    assert res3 is not None
    assert res3[0] == "pikachu_ex_238"

    # Ignora non-singles
    res4 = search_metadata_card_by_query("booster box", mock_meta)
    assert res4 is None
