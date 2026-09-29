"""
tests/test_eval_slab_listing.py — Test per scripts/eval_slab_listing.py.

BUG TROVATO (richiesta esplicita dell'utente: valutazione di acquisti reali,
"e' importante mantenere edge", 2026-09-29): quando una variante speciale
(1st Edition/No Symbol/Shadowless) aveva gia' un prezzo REALE trovato via
fetch_pricecharting_variant_grade9 (la pagina dedicata della variante),
evaluate_listing() interrogava SEMPRE ANCHE fetch_pricecharting_grade_tier_price
(la pagina della stampa STANDARD, senza variante) e - se trovava un prezzo -
lo usava per sovrascrivere silenziosamente il benchmark della variante gia'
risolto. Caso reale: una CGC 9.0 1st Edition Gym Challenge Sabrina #20
(prezzo reale 1st Ed. $229.99) veniva valutata contro il benchmark Unlimited
($106.12), producendo un falso "SCARTARE / OVERPRICED +72.6%" su una carta
in realta' vicina al fair value ("BUY CONSIGLIATO" con il benchmark corretto).
"""

from unittest.mock import patch
import io
import contextlib

from scripts.eval_slab_listing import evaluate_listing


def _run_and_capture(**kwargs) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        evaluate_listing(**kwargs)
    return buf.getvalue()


@patch("scripts.eval_slab_listing.fetch_pricecharting_grade_tier_price")
@patch("scripts.eval_slab_listing.fetch_pricecharting_variant_grade9")
def test_variant_price_not_overwritten_by_standard_print_price(mock_variant, mock_tier):
    # Prezzo reale 1st Edition (quello corretto da usare).
    mock_variant.return_value = (197.97, 229.99, "https://pricecharting.com/fake-1st-ed")
    # Prezzo reale della stampa Unlimited (NON deve vincere sul benchmark).
    mock_tier.return_value = (91.35, 106.12, "https://pricecharting.com/fake-unlimited", "PriceCharting")

    out = _run_and_capture(
        card_query="sabrina_20",
        company="CGC",
        grade="9.0",
        price_eur=152.34,
        variant="1st_edition",
    )

    assert "197.97" in out or "229.99" in out
    assert "106.12" not in out
    assert "BUY CONSIGLIATO" in out or "DEEP VALUE" in out or "FAIR VALUE" in out
    assert "SCARTARE" not in out
