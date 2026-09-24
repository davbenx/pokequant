"""
tests/test_discover_singles_universe.py — Copertura per la costruzione dell'universo
singles (discover_chase_cards.py, discover_random_control_singles.py) senza rete:
mocka fetch_set_cards/fetch_pricecharting_series per verificare che:
  1. slugify_card/make_item_id producano identificatori validi e stabili.
  2. discover_chase_cards.py taggi ogni item con selection_method biased.
  3. discover_random_control_singles.py campioni SENZA filtro di prezzo/rarita
     (deve poter includere anche una carta Common a prezzo 0) e taggi random_control.
  4. Nessuno dei due script scriva duplicati su una chiave (game_slug, item_slug)
     gia' presente in items_metadata.json.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.discover_chase_cards import slugify_card, make_item_id, build_set_ids, fetch_set_cards as _unused_import  # noqa: F401
import scripts.discover_chase_cards as chase_mod
import scripts.discover_random_control_singles as control_mod


def test_build_set_ids_matches_only_eras_with_a_known_sealed_box(monkeypatch):
    """build_set_ids() sostituisce la mappa scritta a mano: deve trovare
    dinamicamente solo i set pokemontcg.io la cui era ha GIA' un box sealed in
    metadata (stesso era_id() usato da discover_sealed_universe.py per creare
    l'item_id) - un set senza box sealed noto resta escluso."""
    fake_sets = {"data": [
        {"id": "swsh6", "name": "Chilling Reign"},   # ha "chilling_reign_bb" in metadata
        {"id": "xx99", "name": "Made Up Future Set"},  # nessun box sealed noto
    ]}

    class _FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return fake_sets

    monkeypatch.setattr(chase_mod.requests, "get", lambda *a, **k: _FakeResp())
    metadata = {"chilling_reign_bb": {"type": "sealed", "game_slug": "pokemon-chilling-reign"}}
    result = build_set_ids(metadata)
    assert result == {"swsh6": "chilling_reign"}


def _fake_card(name, number, rarity, price, release="2021-08-27"):
    return {
        "name": name, "number": number, "rarity": rarity,
        "cardmarket": {"prices": {"averageSellPrice": price, "trendPrice": price}},
        "set": {"releaseDate": release},
    }


def _fake_series_ok():
    idx = pd.date_range("2021-08-01", periods=6, freq="MS")
    return {"raw": pd.Series([10.0] * 6, index=idx), "graded": pd.Series([30.0] * 6, index=idx)}


def _fake_series_empty():
    return {"raw": pd.Series(dtype=float)}


def test_slugify_and_make_item_id_are_stable_and_url_safe():
    assert slugify_card("Umbreon VMAX", "215") == "umbreon-vmax-215"
    assert slugify_card("Mr. Mime", "1") == "mr-mime-1"
    assert make_item_id("Latias & Latios GX", "170") == "latias_latios_gx_170"
    # Idempotenza: stesso input -> stesso output
    assert slugify_card("Charizard", "4") == slugify_card("Charizard", "4")


def test_chase_cards_tags_every_new_item_as_survivorship_biased(monkeypatch, tmp_path):
    monkeypatch.setattr(chase_mod, "build_set_ids", lambda metadata: {"testset": "test_era"})
    monkeypatch.setattr(chase_mod, "fetch_set_cards", lambda set_id, retries=3: [
        _fake_card("Rare Chase Card", "1", "Rare Secret", 99.0),
    ])
    monkeypatch.setattr(chase_mod, "fetch_pricecharting_series", lambda gs, isl, eur_usd_series=None: _fake_series_ok())

    metadata = {"test_era_bb": {"game_slug": "pokemon-test-era", "item_slug": "booster-box", "type": "sealed"}}
    saved = {}
    monkeypatch.setattr(chase_mod, "load_metadata", lambda: metadata)
    monkeypatch.setattr(chase_mod, "save_metadata", lambda m: saved.update(m))
    monkeypatch.setattr(chase_mod, "load_eur_usd_series", lambda: None)

    with patch.object(sys, "argv", ["discover_chase_cards.py", "--min-price", "40"]):
        chase_mod.main()

    new_items = [v for k, v in saved.items() if k != "test_era_bb"]
    assert len(new_items) == 1
    assert new_items[0]["selection_method"] == "chase_price_filter_survivorship_biased"
    assert new_items[0]["type"] == "single"


def test_random_control_includes_low_price_common_cards(monkeypatch):
    """Il campione di controllo NON deve filtrare per prezzo: una Common a 0 EUR
    deve poter entrare, a differenza di discover_chase_cards.py."""
    monkeypatch.setattr(control_mod, "build_set_ids", lambda metadata: {"testset": "test_era"})
    monkeypatch.setattr(control_mod, "fetch_set_cards", lambda set_id, retries=3: [
        _fake_card("Common Junk Card", "50", "Common", 0.0),
    ])
    monkeypatch.setattr(control_mod, "fetch_pricecharting_series", lambda gs, isl, eur_usd_series=None: _fake_series_ok())

    metadata = {"test_era_bb": {"game_slug": "pokemon-test-era", "item_slug": "booster-box", "type": "sealed"}}
    saved = {}
    monkeypatch.setattr(control_mod, "load_metadata", lambda: metadata)
    monkeypatch.setattr(control_mod, "save_metadata", lambda m: saved.update(m))
    monkeypatch.setattr(control_mod, "load_eur_usd_series", lambda: None)

    with patch.object(sys, "argv", ["discover_random_control_singles.py", "--per-set", "5"]):
        control_mod.main()

    new_items = [v for k, v in saved.items() if k != "test_era_bb"]
    assert len(new_items) == 1, "una Common a prezzo 0 deve essere campionata, non scartata"
    assert new_items[0]["selection_method"] == "random_control"
    assert new_items[0]["cardmarket_ref_price_eur"] == 0.0


def test_random_control_skips_duplicate_game_slug_item_slug(monkeypatch):
    monkeypatch.setattr(control_mod, "build_set_ids", lambda metadata: {"testset": "test_era"})
    monkeypatch.setattr(control_mod, "fetch_set_cards", lambda set_id, retries=3: [
        _fake_card("Already Known", "7", "Rare", 50.0),
    ])
    fetch_calls = []

    def _tracked_fetch(gs, isl, eur_usd_series=None):
        fetch_calls.append((gs, isl))
        return _fake_series_ok()

    monkeypatch.setattr(control_mod, "fetch_pricecharting_series", _tracked_fetch)

    existing_slug = slugify_card("Already Known", "7")
    metadata = {
        "test_era_bb": {"game_slug": "pokemon-test-era", "item_slug": "booster-box", "type": "sealed"},
        "already_known_7": {"game_slug": "pokemon-test-era", "item_slug": existing_slug, "type": "single"},
    }
    saved = {}
    monkeypatch.setattr(control_mod, "load_metadata", lambda: metadata)
    monkeypatch.setattr(control_mod, "save_metadata", lambda m: saved.update(m))
    monkeypatch.setattr(control_mod, "load_eur_usd_series", lambda: None)

    with patch.object(sys, "argv", ["discover_random_control_singles.py", "--per-set", "5"]):
        control_mod.main()

    assert fetch_calls == [], "un duplicato (game_slug, item_slug) noto non deve nemmeno arrivare al fetch"
    assert saved == metadata, "nessun nuovo item deve essere scritto per un duplicato"


def test_random_control_skips_unresolved_pricecharting_history(monkeypatch):
    monkeypatch.setattr(control_mod, "build_set_ids", lambda metadata: {"testset": "test_era"})
    monkeypatch.setattr(control_mod, "fetch_set_cards", lambda set_id, retries=3: [
        _fake_card("No History Card", "99", "Common", 1.0),
    ])
    monkeypatch.setattr(control_mod, "fetch_pricecharting_series", lambda gs, isl, eur_usd_series=None: _fake_series_empty())

    metadata = {"test_era_bb": {"game_slug": "pokemon-test-era", "item_slug": "booster-box", "type": "sealed"}}
    saved = {}
    monkeypatch.setattr(control_mod, "load_metadata", lambda: metadata)
    monkeypatch.setattr(control_mod, "save_metadata", lambda m: saved.update(m))
    monkeypatch.setattr(control_mod, "load_eur_usd_series", lambda: None)

    with patch.object(sys, "argv", ["discover_random_control_singles.py", "--per-set", "5"]):
        control_mod.main()

    new_items = [v for k, v in saved.items() if k != "test_era_bb"]
    assert new_items == [], "una carta senza storico PriceCharting reale non va inventata né aggiunta"
