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
import re
import contextlib

from scripts.eval_slab_listing import evaluate_listing


def _run_and_capture(**kwargs) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        evaluate_listing(**kwargs)
    return buf.getvalue()


def _fair_value(out: str) -> float:
    m = re.search(r"Fair Value Slab:\s*([\d.]+)\s*€", out)
    assert m, f"Fair Value non trovato nell'output:\n{out}"
    return float(m.group(1))


@patch("scripts.eval_slab_listing.fetch_pricecharting_grade_tier_price")
@patch("scripts.eval_slab_listing.fetch_pricecharting_variant_grade_tier")
@patch("scripts.eval_slab_listing.fetch_pricecharting_variant_grade9")
def test_variant_price_not_overwritten_by_standard_print_price(mock_variant, mock_variant_tier, mock_tier):
    # Nessun dato reale per il grado ESATTO richiesto sulla pagina variante -
    # cade sul Grado 9 reale sotto (comportamento testato qui).
    mock_variant_tier.return_value = None
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


@patch("scripts.eval_slab_listing.fetch_pricecharting_grade_tier_price")
@patch("scripts.eval_slab_listing.fetch_pricecharting_variant_grade_tier")
@patch("scripts.eval_slab_listing.fetch_pricecharting_variant_grade9")
def test_variant_price_scales_with_selected_grade(mock_variant, mock_variant_tier, mock_tier):
    """BUG TROVATO (l'utente: "modificare il voto slab non modifica i prezzi
    consigliati", 2026-09-29, su Jolteon Holo No Symbol Error): quando la
    variante aveva un prezzo reale (sempre al Grado 9, unico dato che
    fetch_pricecharting_variant_grade9 restituisce), is_grade_benchmark_resolved
    veniva fissato a True per QUALUNQUE voto scelto, bloccando il fair value
    al valore Grado 9 indipendentemente dal voto - verificato: un 10.0 dava un
    fair value piu' BASSO di un 9.0 sulla stessa carta (invertito). Nessuna
    pagina standard deve mai essere interrogata per una variante gia' risolta
    (mock_tier.return_value = None -> se venisse chiamata, il benchmark
    sparirebbe e il test fallirebbe). mock_variant_tier=None simula "nessun
    dato reale per il grado esatto sulla pagina variante" per isolare e
    testare la scalatura di FALLBACK (vedi
    test_variant_uses_real_tier_data_when_available per il percorso nuovo che
    trova un dato reale per ogni grado)."""
    mock_variant_tier.return_value = None
    mock_variant.return_value = (314.61, 365.50, "https://pricecharting.com/fake-no-symbol")
    mock_tier.return_value = None

    fair_values = {}
    for grade in ["7.0", "8.0", "8.5", "9.0", "9.5", "10.0"]:
        out = _run_and_capture(
            card_query="jolteon_4", company="CGC", grade=grade, price_eur=100.0, variant="no_symbol",
        )
        fair_values[grade] = _fair_value(out)
        mock_tier.assert_not_called()

    ordered = [fair_values[g] for g in ["7.0", "8.0", "8.5", "9.0", "9.5", "10.0"]]
    assert ordered == sorted(ordered), f"il fair value deve crescere col voto: {fair_values}"
    assert len(set(ordered)) == 6, f"ogni voto deve dare un fair value diverso: {fair_values}"


@patch("scripts.eval_slab_listing.fetch_pricecharting_grade_tier_price")
@patch("scripts.eval_slab_listing.fetch_pricecharting_variant_grade9")
@patch("scripts.eval_slab_listing.fetch_pricecharting_variant_grade_tier")
def test_variant_uses_real_tier_data_when_available(mock_variant_tier, mock_variant_grade9, mock_tier):
    """Richiesto dall'utente dopo il fix del voto slab: "deve prendere i dati
    reali quanto possibile". Se PriceCharting ha un dato reale per il grado
    ESATTO richiesto sulla pagina della variante (non solo Grado 9), quello va
    usato al posto della stima algoritmica - e il fetch del solo Grado 9 non
    va nemmeno chiamato, perche' il dato migliore e' gia' disponibile."""
    mock_variant_tier.return_value = (450.0, 522.5, "https://pricecharting.com/fake-no-symbol-psa10")

    out = _run_and_capture(
        card_query="jolteon_4", company="CGC", grade="10.0", price_eur=100.0, variant="no_symbol",
    )

    mock_variant_tier.assert_called_once()
    _, kwargs = mock_variant_tier.call_args
    assert kwargs.get("tier") == "psa10"
    mock_variant_grade9.assert_not_called()
    mock_tier.assert_not_called()
    assert "450.00" in out or "522.5" in out


def test_grade_95_algorithmic_path_not_double_counted():
    """BUG TROVATO (trovato indagando il bug del voto slab sopra): il ramo
    algoritmico (nessun dato reale PriceCharting) pre-scalava il benchmark
    grado 9.5 con ERA_BGS95_TO_PSA9_RATIO e POI adjust_price_for_grading
    applicava DI NUOVO un moltiplicatore gia' "vs PSA9" da
    EMPIRICAL_RATIOS_GRADE9 - doppio conteggio (+61% su CGC Moderno). Usa una
    carta senza dato reale (nessun mock di fetch = tutte le fetch live
    tornano un valore plausibile o None; qui basta confrontare 9.5 con 9.0 e
    verificare che il rapporto sia vicino al moltiplicatore CGC/9.5/MODERN
    reale (1.18x), non al suo quadrato (1.18*1.61=1.90x)."""
    from poke_quant.slabs.grading_multipliers import EMPIRICAL_RATIOS_GRADE9, GradingCompany, Era

    with patch("scripts.eval_slab_listing.fetch_pricecharting_grade_tier_price", return_value=None), \
         patch("scripts.eval_slab_listing.fetch_pricecharting_variant_grade9", return_value=None):
        out_90 = _run_and_capture(card_query="jolteon_4", company="CGC", grade="9.0", price_eur=100.0)
        out_95 = _run_and_capture(card_query="jolteon_4", company="CGC", grade="9.5", price_eur=100.0)

    fv_90 = _fair_value(out_90)
    fv_95 = _fair_value(out_95)
    mult_90 = EMPIRICAL_RATIOS_GRADE9[(GradingCompany.CGC, "9.0", Era.VINTAGE)][0]
    mult_95 = EMPIRICAL_RATIOS_GRADE9[(GradingCompany.CGC, "9.5", Era.VINTAGE)][0]
    expected_ratio = mult_95 / mult_90  # entrambi i multipliers sono "vs PSA9", il base cancella
    observed_ratio = fv_95 / fv_90
    assert abs(observed_ratio - expected_ratio) < 0.05, (
        f"rapporto 9.5/9.0 osservato {observed_ratio:.3f} vs atteso {expected_ratio:.3f} "
        f"(un rapporto molto piu' alto indicherebbe doppio conteggio della pre-scalatura ERA_BGS95_TO_PSA9_RATIO)"
    )


def _max_edge(out: str) -> float:
    m = re.search(r"Tetto Max Edge:\s*([\d.]+)\s*€", out)
    assert m, f"Tetto Max Edge non trovato nell'output:\n{out}"
    return float(m.group(1))


@patch("scripts.eval_slab_listing.fetch_pricecharting_grade_tier_price")
@patch("scripts.eval_slab_listing.fetch_pricecharting_variant_grade9")
def test_psa_blend_premium_applied_to_fair_value_and_ceiling_consistently(mock_variant, mock_tier):
    """BUG TROVATO verificando il fix PSA_BLEND_PREMIUM_FACTOR (l'utente:
    "trovo molte slab ben sopra il prezzo max Edge sul mercato europeo...
    valuta se questo è calcolato correttamente", 2026-10-02): il Tetto Max
    Edge veniva SEMPRE ricalcolato con una formula parallela
    (base_max_edge_price * adj.sniper_ceiling_factor) che ignorava
    sniper_ceiling_raw gia' calcolato da adjust_price_for_grading() - quindi
    quando quella funzione e' stata corretta per applicare un premio PSA sul
    blend cross-company PriceCharting (PSA_BLEND_PREMIUM_FACTOR), il Fair
    Value si aggiornava ma il Tetto Max Edge (e il verdetto finale) no.
    Verifica qui che siano consistenti: il Tetto deve riflettere lo stesso
    premio del Fair Value, non il blend grezzo pre-fix."""
    from poke_quant.slabs.grading_multipliers import PSA_BLEND_PREMIUM_FACTOR

    mock_variant.return_value = None
    blend_price_usd, blend_price_eur = 99.99, 86.07  # Dragonite-EX #106, caso reale (PriceCharting Grado 9)
    mock_tier.return_value = (blend_price_eur, blend_price_usd, "https://pricecharting.com/fake", "PriceCharting")

    out = _run_and_capture(card_query="dragonite_ex_106", company="PSA", grade="9.0", price_eur=94.30, shipping_eur=7.00)

    fv = _fair_value(out)
    ceiling = _max_edge(out)
    assert fv == round(blend_price_eur * PSA_BLEND_PREMIUM_FACTOR, 2), (
        f"Fair Value {fv} non riflette il premio PSA sul blend ({blend_price_eur} x {PSA_BLEND_PREMIUM_FACTOR})"
    )
    assert ceiling >= fv, f"Tetto Max Edge ({ceiling}) sotto il Fair Value ({fv}) - formula del tetto non corretta"
    assert ceiling > round(blend_price_eur * 1.05, 2), (
        "Tetto Max Edge identico al valore pre-fix - la formula parallela sta ancora ignorando "
        "il premio PSA calcolato da adjust_price_for_grading()"
    )
