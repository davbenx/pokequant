"""
tests/test_app_slab_calculator.py — Test end-to-end (via streamlit.testing.v1.AppTest)
del Valutatore Slab in app.py.

BUG TROVATO (richiesta esplicita dell'utente: "le valutazioni sulle slab
singole sono corrette? e' importante per mantenere edge", 2026-09-29):
app.py ha una propria implementazione DUPLICATA (non importata da
scripts/eval_slab_listing.py) del calcolo fair_value/sniper_ceiling per il
Valutatore Slab live - la stessa classe di bug trovata e corretta in
poke_quant/slabs/grading_multipliers.py (sniper_ceiling_factor sotto
multiplier per grado 9.5/premium 10) poteva quindi ripresentarsi qui anche
dopo quel fix, se qualcuno modifica questo percorso senza toccare la
funzione condivisa. Questo test verifica l'invariante end-to-end sul
percorso REALE che l'utente usa dal browser (submit del form), non solo
sulla funzione di libreria sottostante.
"""

from pathlib import Path
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def _submit_calculator(company: str, grade: str, offer_price: float, shipping: float = 0.0):
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    assert not at.exception

    at.selectbox[1].set_value(company)  # Casa di Gradazione
    at.selectbox[2].set_value(grade)    # Voto Slab
    at.number_input[1].set_value(offer_price)  # Prezzo Annuncio / Offerta
    at.number_input[2].set_value(shipping)     # Spese Sped.
    at.run(timeout=60)
    assert not at.exception

    at.button[0].click()
    at.run(timeout=60)
    assert not at.exception
    return at.session_state.get("slab_eval_res")


def test_sniper_ceiling_never_below_fair_value_grade_95():
    res = _submit_calculator(company="BGS", grade="9.5 Gem Mint", offer_price=75.0)
    assert res is not None
    assert res["sniper_ceiling_calib"] >= res["fair_value_calib"]


def test_verdict_at_fair_value_is_not_overpriced():
    """Un'offerta appena sopra il fair value (dentro il tetto max edge) deve
    finire in 'FAIR VALUE / AL LIMITE', non 'OVERPRICED' - il sintomo diretto
    del bug: con sniper_ceiling < fair_value, QUALUNQUE prezzo vicino al fair
    value veniva marcato overpriced."""
    probe = _submit_calculator(company="BGS", grade="9.5 Gem Mint", offer_price=1.0)
    fair_value = probe["fair_value_calib"]
    res = _submit_calculator(company="BGS", grade="9.5 Gem Mint", offer_price=round(fair_value + 1, 2))
    assert "OVERPRICED" not in res["v_badge"]
    assert "FAIR VALUE" in res["v_badge"] or "BUY" in res["v_badge"] or "DEEP VALUE" in res["v_badge"]


def test_verdict_well_above_ceiling_is_overpriced():
    probe = _submit_calculator(company="BGS", grade="9.5 Gem Mint", offer_price=1.0)
    ceiling = probe["sniper_ceiling_calib"]
    res = _submit_calculator(company="BGS", grade="9.5 Gem Mint", offer_price=round(ceiling * 1.2, 2))
    assert "OVERPRICED" in res["v_badge"]
