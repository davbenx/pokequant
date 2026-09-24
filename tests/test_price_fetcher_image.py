"""
tests/test_price_fetcher_image.py — Copertura per fetch_pricecharting_cover_image_url
(poke_quant/data/price_fetcher.py): estrazione dell'URL immagine reale del
prodotto, senza rete (risposta HTTP mockata).
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.price_fetcher import fetch_pricecharting_cover_image_url

# HTML minimale ma realistico: una tabella di "prodotti simili" PRIMA di
# product_details (che il parser deve ignorare) e la copertina vera dopo.
_FAKE_HTML = """
<html><body>
<table>
<tr><td class="image"><img class="photo" src="https://storage.googleapis.com/images.pricecharting.com/WRONG_ONE/60.jpg" /></td></tr>
</table>
<div id="product_details">
    <div class="cover">
        <a href="#">
            <img src='https://storage.googleapis.com/images.pricecharting.com/correct-hash-here/240.jpg'
                 alt="Booster Box Prices" />
        </a>
    </div>
</div>
</body></html>
"""


def test_extracts_image_after_product_details_marker_not_the_similar_products_table():
    mock_resp = MagicMock()
    mock_resp.text = _FAKE_HTML
    mock_resp.raise_for_status = MagicMock()
    with patch("poke_quant.data.price_fetcher.requests.get", return_value=mock_resp):
        url = fetch_pricecharting_cover_image_url("pokemon-some-set", "booster-box")
    assert url == "https://storage.googleapis.com/images.pricecharting.com/correct-hash-here/240.jpg"


def test_returns_none_when_product_details_marker_missing():
    mock_resp = MagicMock()
    mock_resp.text = "<html><body>pagina di ricerca generica, nessun prodotto</body></html>"
    mock_resp.raise_for_status = MagicMock()
    with patch("poke_quant.data.price_fetcher.requests.get", return_value=mock_resp):
        url = fetch_pricecharting_cover_image_url("pokemon-broken-slug", "booster-box")
    assert url is None


def test_returns_none_on_network_error():
    with patch("poke_quant.data.price_fetcher.requests.get", side_effect=Exception("timeout")):
        url = fetch_pricecharting_cover_image_url("pokemon-some-set", "booster-box")
    assert url is None
