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


def test_search_ignores_grade_number_substring_false_match():
    """BUG TROVATO (l'utente: "Jolteon PSA 7 no symbol mi dice Fair value PSA
    151.01 EUR, ma su pricecharting e' molto piu' basso", 2026-09-30): un voto
    scritto come numero puro nel campo nome carta (es. "7", parte del form ma
    l'utente lo scrive comunque nel nome) combaciava per SOTTOSTRINGA con
    qualunque numero di catalogo che lo contenesse (es. "72", "109") invece
    che per parola intera - una query per una carta reale ("Charizard #4")
    con un voto scritto in coda ("charizard 7") poteva quindi far vincere
    una carta completamente estranea il cui numero contenesse "7" come
    sottostringa, invece di restare sulla carta corretta o non matchare
    affatto."""
    mock_meta = {
        "charizard_4": {
            "name": "Charizard #4",
            "type": "single",
            "game_slug": "pokemon-base-set",
            "item_slug": "charizard-4",
            "era": "vintage",
        },
        "unrelated_72": {
            "name": "Some Unrelated Card #72",
            "type": "single",
            "game_slug": "pokemon-unrelated-set",
            "item_slug": "some-unrelated-card-72",
            "era": "modern",
        },
    }
    res = search_metadata_card_by_query("charizard 7", mock_meta)
    assert res is None or res[0] == "charizard_4"
    assert res is None or res[0] != "unrelated_72"


def test_search_strips_grading_company_stopword():
    """Stessa richiesta: la casa di gradazione (gia' un campo separato nel
    form) non deve mai contribuire al match o diluire il rapporto di parole
    combacianti."""
    mock_meta = {
        "charizard_4": {
            "name": "Charizard #4",
            "type": "single",
            "game_slug": "pokemon-base-set",
            "item_slug": "charizard-4",
            "era": "vintage",
        },
    }
    res = search_metadata_card_by_query("charizard psa", mock_meta)
    assert res is not None
    assert res[0] == "charizard_4"
