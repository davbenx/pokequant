"""
tests/test_population_storage.py — Unit test per il log di popolazione
(append_population_snapshot/load_population_history in poke_quant/data/storage.py).
E' un LOG che cresce nel tempo (uno snapshot per fetch), non una matrice
sovrascrivibile - PriceCharting non pubblica la popolazione passata.
"""

import pytest

import poke_quant.data.storage as storage


@pytest.fixture(autouse=True)
def isolate_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "CACHE_DIR", tmp_path)
    yield tmp_path


def test_append_creates_file_with_header_on_first_call():
    rows = [{"date": "2026-09-25", "item_id": "charizard_4", "grade": "9",
             "psa_pop": 8455, "cgc_pop": 2, "total_pop": 8457, "price_usd": 2976.69}]
    storage.append_population_snapshot(rows)
    df = storage.load_population_history()
    assert len(df) == 1
    assert df.iloc[0]["item_id"] == "charizard_4"
    assert df.iloc[0]["psa_pop"] == 8455


def test_append_accumulates_across_multiple_calls_not_overwrite():
    storage.append_population_snapshot([{"date": "2026-09-25", "item_id": "a", "grade": "9",
                                          "psa_pop": 100, "cgc_pop": 1, "total_pop": 101, "price_usd": 10.0}])
    storage.append_population_snapshot([{"date": "2026-10-25", "item_id": "a", "grade": "9",
                                          "psa_pop": 110, "cgc_pop": 1, "total_pop": 111, "price_usd": 11.0}])
    df = storage.load_population_history()
    assert len(df) == 2
    assert list(df["psa_pop"]) == [100, 110]


def test_load_returns_none_when_no_snapshot_taken_yet():
    assert storage.load_population_history() is None


def test_append_with_empty_rows_is_a_noop():
    storage.append_population_snapshot([])
    assert storage.load_population_history() is None
