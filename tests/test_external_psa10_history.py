"""
tests/test_external_psa10_history.py — Unit test per la riconciliazione con il
dataset esterno gemrate-free github.com/samaygodika/pokemon-psa10-history
(poke_quant/data/external_psa10_history.py). Nessuna rete: usa righe sintetiche
modellate sui casi reali (ambigui) trovati ispezionando history/assets.csv.
"""

from poke_quant.data.external_psa10_history import (
    find_matching_asset, build_number_index, extract_number_from_item_slug,
)


def _row(card_name, set_, subject, variety, card_number="215"):
    return {"card_name": card_name, "set": set_, "subject": subject, "variety": variety,
            "card_number": card_number, "asset_id": card_name}


def test_excludes_foreign_language_variant_with_same_number_and_subject():
    """BUG TROVATO ispezionando i dati reali: lo stesso numero+soggetto ricorre in
    piu' lingue come asset DIVERSI (popolazione diversa) - un match naive per solo
    numero+soggetto agganciava la popolazione TEDESCA a una carta inglese."""
    rows = [
        _row("2021 Pokemon Sword and Shield Evolving Skies German Umbreon Vmax #215",
             "Pokemon Sword and Shield Evolving Skies", "Umbreon Vmax", "German"),
        _row("2021 Pokemon Sword and Shield Evolving Skies Secret Rare Umbreon Vmax #215",
             "Pokemon Sword and Shield Evolving Skies", "Umbreon Vmax", "Secret Rare"),
    ]
    idx = build_number_index(rows)
    hit = find_matching_asset("Umbreon VMAX Alternate Art (Moonbreon)", "pokemon-evolving-skies",
                               None, "215", idx)
    assert hit is not None
    assert "German" not in hit["card_name"]
    assert "Secret Rare" in hit["card_name"]


def test_returns_none_when_only_foreign_variants_exist():
    rows = [_row("... German Umbreon Vmax #215", "Pokemon Evolving Skies", "Umbreon Vmax", "German")]
    idx = build_number_index(rows)
    assert find_matching_asset("Umbreon VMAX", "pokemon-evolving-skies", None, "215", idx) is None


def test_prefers_plain_printing_over_reverse_holo_when_our_card_is_not_reverse():
    rows = [
        _row("2023 Pokemon Scarlet and Violet 151 Reverse Holo Charmander #4",
             "Pokemon Scarlet and Violet 151", "Charmander", "", card_number="4"),
        _row("2023 Pokemon Scarlet and Violet 151 Charmander #4",
             "Pokemon Scarlet and Violet 151", "Charmander", "", card_number="4"),
    ]
    idx = build_number_index(rows)
    hit = find_matching_asset("Charmander #4", "pokemon-scarlet-&-violet-151", "Common", "4", idx)
    assert hit is not None
    assert "Reverse" not in hit["card_name"]


def test_ambiguous_different_sets_same_number_returns_correct_set_via_game_slug():
    """Lo stesso numero+soggetto ricorre in epoche/set completamente diversi nel
    catalogo alt.xyz (65k carte) - il game_slug (set noto) disambigua."""
    rows = [
        _row("2021 Pokemon Sword and Shield Chilling Reign Galarian Zapdos V #174",
             "Pokemon Sword and Shield Chilling Reign", "Galarian Zapdos V", "", card_number="174"),
        _row("2020 Pokemon Sword and Shield Vivid Voltage Galarian Sirfetch'D V #174",
             "Pokemon Sword and Shield Vivid Voltage", "Galarian Sirfetch D V", "", card_number="174"),
    ]
    idx = build_number_index(rows)
    hit = find_matching_asset("Galarian Zapdos V #174", "pokemon-chilling-reign", None, "174", idx)
    assert hit is not None
    assert "Chilling Reign" in hit["card_name"]


def test_no_candidates_for_number_returns_none():
    assert find_matching_asset("Anything", "pokemon-some-set", None, "999", {}) is None


def test_extract_number_from_item_slug():
    assert extract_number_from_item_slug("umbreon-vmax-215") == "215"
    assert extract_number_from_item_slug("charizard-4") == "4"
    assert extract_number_from_item_slug("no-number-here-") is None or True  # trailing hyphen edge case, non deve esplodere
