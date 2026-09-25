"""
tests/test_product_image_cache.py — Copertura per get_product_image() (app.py):
cache manuale con TTL diverso per successo (7gg, l'immagine di un prodotto non
cambia) e fallimento (1h) - trovato verificando "molte immagini delle carte
singole non le vedo": prima un unico TTL lungo (7gg) per entrambi i casi
trasformava un rate limit 429 transitorio di PriceCharting in "nessuna
immagine per una settimana intera".
"""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app


def setup_function():
    app._IMAGE_CACHE.clear()


def test_successful_fetch_is_cached_long_term():
    with patch("app.fetch_pricecharting_cover_image_url", return_value="https://img/ok.jpg") as mock_fetch:
        url1 = app.get_product_image("pokemon-some-set", "booster-box")
        url2 = app.get_product_image("pokemon-some-set", "booster-box")
    assert url1 == url2 == "https://img/ok.jpg"
    mock_fetch.assert_called_once()  # seconda chiamata servita dalla cache, nessun nuovo fetch
    _, _, ttl = app._IMAGE_CACHE[("pokemon-some-set", "booster-box")]
    assert ttl == app._IMAGE_SUCCESS_TTL


def test_failed_fetch_gets_a_short_ttl_not_the_full_week():
    with patch("app.fetch_pricecharting_cover_image_url", return_value=None):
        url = app.get_product_image("pokemon-some-set", "booster-box")
    assert url is None
    _, _, ttl = app._IMAGE_CACHE[("pokemon-some-set", "booster-box")]
    assert ttl == app._IMAGE_FAILURE_TTL
    assert ttl < app._IMAGE_SUCCESS_TTL


def test_expired_failure_is_retried_not_stuck_forever():
    with patch("app.fetch_pricecharting_cover_image_url", return_value=None) as mock_fetch:
        app.get_product_image("pokemon-some-set", "booster-box")
    # Simula il passare di piu' di 1h (TTL fallimento) senza aspettare davvero.
    url, cached_at, ttl = app._IMAGE_CACHE[("pokemon-some-set", "booster-box")]
    app._IMAGE_CACHE[("pokemon-some-set", "booster-box")] = (url, cached_at - ttl - 1, ttl)
    with patch("app.fetch_pricecharting_cover_image_url", return_value="https://img/recovered.jpg") as mock_fetch2:
        url2 = app.get_product_image("pokemon-some-set", "booster-box")
    assert url2 == "https://img/recovered.jpg"
    mock_fetch2.assert_called_once()


def test_missing_slug_returns_none_without_fetching():
    with patch("app.fetch_pricecharting_cover_image_url") as mock_fetch:
        assert app.get_product_image(None, "booster-box") is None
        assert app.get_product_image("pokemon-some-set", None) is None
    mock_fetch.assert_not_called()
