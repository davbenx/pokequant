"""
tests/test_product_image_cache.py — Copertura per get_product_image() (app.py):
cache manuale con TTL diverso per successo (7gg, l'immagine di un prodotto non
cambia) e fallimento (1h) - trovato verificando "molte immagini delle carte
singole non le vedo": prima un unico TTL lungo (7gg) per entrambi i casi
trasformava un rate limit 429 transitorio di PriceCharting in "nessuna
immagine per una settimana intera".
"""

import sys
import time
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


def test_prefetch_fetches_all_missing_pairs_in_parallel_and_populates_cache():
    """Trovato verificando "la dashboard continua a ricaricare": un
    caricamento a cache fredda richiedeva ~66s (misurato) perché fino a
    50-60 immagini venivano scaricate in SERIE, una alla volta - abbastanza
    lento da causare timeout/refresh e un loop di reload apparente. Con un
    thread pool, il tempo totale deve avvicinarsi al fetch PIU' LENTO, non
    alla somma di tutti."""
    def slow_fetch(game_slug, item_slug):
        time.sleep(0.2)
        return f"https://img/{game_slug}-{item_slug}.jpg"

    pairs = [("pokemon-set", f"card-{i}") for i in range(8)]
    with patch("app.fetch_pricecharting_cover_image_url", side_effect=slow_fetch):
        t0 = time.time()
        app.prefetch_product_images(pairs)
        elapsed = time.time() - t0
    assert elapsed < 0.2 * len(pairs)  # molto piu' rapido della somma seriale (8 * 0.2s = 1.6s)
    for game_slug, item_slug in pairs:
        url, _, ttl = app._IMAGE_CACHE[(game_slug, item_slug)]
        assert url == f"https://img/{game_slug}-{item_slug}.jpg"
        assert ttl == app._IMAGE_SUCCESS_TTL


def test_prefetch_skips_pairs_already_cached():
    app._IMAGE_CACHE[("pokemon-set", "card-0")] = ("https://img/cached.jpg", time.time(), app._IMAGE_SUCCESS_TTL)
    with patch("app.fetch_pricecharting_cover_image_url", return_value="https://img/new.jpg") as mock_fetch:
        app.prefetch_product_images([("pokemon-set", "card-0"), ("pokemon-set", "card-1")])
    mock_fetch.assert_called_once_with("pokemon-set", "card-1")


def test_prefetch_ignores_pairs_with_missing_slug():
    with patch("app.fetch_pricecharting_cover_image_url") as mock_fetch:
        app.prefetch_product_images([(None, "card-0"), ("pokemon-set", None), (None, None)])
    mock_fetch.assert_not_called()
