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
from unittest.mock import patch
import pytest
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


def test_thin_market_flagged_card_shows_warning():
    """Trovato indagando 'e' possibile aggiungere un controllo per mercato
    sottile?' (2026-09-29): una carta con data_quality=='thin_unreliable'
    (scripts/flag_unreliable_assets.py) deve mostrare un avviso live nel
    calcolatore, non solo finire silenziosamente esclusa dai segnali BUY."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    assert not at.exception

    at.radio[0].set_value("✏️ Carta Personalizzata / Inserimento Libero (o Link PriceCharting)")
    at.run(timeout=60)
    assert not at.exception

    at.text_input[0].set_value("Clefable Call of Legends")
    at.selectbox[0].set_value("CGC")
    at.selectbox[1].set_value("9.0 Mint")
    at.number_input[1].set_value(50.0)
    at.run(timeout=60)
    assert not at.exception

    at.button[0].click()
    at.run(timeout=60)
    assert not at.exception

    res = at.session_state.get("slab_eval_res")
    assert res is not None
    assert res["matched_db_info"] is not None
    assert res["matched_db_info"].get("data_quality") == "thin_unreliable"
    warnings_text = " ".join(w.value for w in at.warning)
    assert "mercato sottile" in warnings_text.lower()


def _submit_variant_calculator(card_name: str, variant: str, company: str, grade: str, offer_price: float):
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    assert not at.exception

    at.radio[0].set_value("✏️ Carta Personalizzata / Inserimento Libero (o Link PriceCharting)")
    at.run(timeout=60)
    assert not at.exception

    at.text_input[0].set_value(card_name)
    at.selectbox[0].set_value(company)
    at.selectbox[1].set_value(grade)
    at.selectbox[2].set_value(variant)
    at.number_input[1].set_value(offer_price)
    at.run(timeout=60)
    assert not at.exception

    at.button[0].click()
    at.run(timeout=60)
    assert not at.exception
    return at.session_state.get("slab_eval_res")


def test_variant_with_real_grade9_price_scales_with_selected_grade():
    """BUG TROVATO (l'utente stava valutando "Jolteon Holo No Symbol Error"
    nel Valutatore Slab: "modificare il voto slab non modifica i prezzi
    consigliati", 2026-09-29): quando una variante speciale aveva un prezzo
    reale trovato su PriceCharting (sempre al Grado 9, l'unico dato che
    fetch_pricecharting_variant_grade9 restituisce), is_pc_grade_resolved
    restava True per QUALUNQUE voto scelto nel menu - il fair value restava
    bloccato al valore Grado 9 indipendentemente dal voto. Sintomo piu'
    grave riscontrato dal vivo: un CGC 10.0 dava un fair value piu' BASSO di
    un CGC 9.0 sulla stessa identica carta (invertito). Richiede rete (dato
    reale PriceCharting per Jolteon #4 No Symbol Error) - se la variante non
    ha piu' un prezzo reale disponibile il test si salta invece di fallire
    a causa di una carta terza non piu' raggiungibile, non del bug stesso."""
    variant_opt = "No Symbol Error (Rileva reale da PriceCharting o ~1.4x)"
    fair_values = {}
    for grade in ["7.0 Near Mint", "8.5 NM-Mint+", "9.0 Mint", "9.5 Gem Mint", "10.0 Gem Mint"]:
        res = _submit_variant_calculator("Jolteon Jungle", variant_opt, "CGC", grade, 100.0)
        if res is None or "dato reale" not in res.get("v_desc", "").lower():
            pytest.skip("Nessun prezzo reale PriceCharting disponibile per questa variante al momento del test")
        fair_values[grade] = res["fair_value_calib"]

    ordered = [fair_values[g] for g in ["7.0 Near Mint", "8.5 NM-Mint+", "9.0 Mint", "9.5 Gem Mint", "10.0 Gem Mint"]]
    assert ordered == sorted(ordered), f"il fair value deve crescere col voto: {fair_values}"
    assert len(set(ordered)) == len(ordered), f"ogni voto deve dare un fair value diverso: {fair_values}"


def test_grade_ladder_thin_market_flag_shows_warning():
    """BUCO STRUTTURALE TROVATO (l'utente: "tappa il buco del filtro mercato
    sottile", 2026-09-30): un dato reale per un grado specifico (9.5/10/8/7,
    da data_cache/grade_ladder_prices.json) non passava da nessun controllo
    di attendibilita' - solo il pannello legacy grade9(PSA9)/raw era
    coperto. Questo test verifica che una carta gia' flaggata da
    check_grade_ladder_tier_reliable (Colress #135, psa10 - salto 5.1x
    seguito da 2 mesi fermo, vedi scripts/grade_ladder_thin_market_test.py)
    mostri un avviso live nel calcolatore."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    assert not at.exception

    at.radio[0].set_value("✏️ Carta Personalizzata / Inserimento Libero (o Link PriceCharting)")
    at.run(timeout=60)
    assert not at.exception

    at.text_input[0].set_value("Colress Plasma Storm")
    at.selectbox[0].set_value("CGC")
    at.selectbox[1].set_value("10.0 Gem Mint")
    at.number_input[1].set_value(50.0)
    at.run(timeout=60)
    assert not at.exception

    at.button[0].click()
    at.run(timeout=60)
    assert not at.exception

    res = at.session_state.get("slab_eval_res")
    assert res is not None
    if not res.get("ladder_flag_reason"):
        pytest.skip("Colress #135 non e' piu' flaggato a psa10 nella cache attuale (ricalibrazione mensile)")
    warnings_text = " ".join(w.value for w in at.warning)
    assert "mercato sottile su questo grado specifico" in warnings_text.lower()


def test_psa_benchmark_includes_blend_premium_on_real_data():
    """BUG TROVATO (l'utente: "trovo molte slab ben sopra il prezzo max Edge
    sul mercato europeo... valuta se questo è calcolato correttamente",
    2026-10-02, verificato con 5 inserzioni reali indipendenti per
    Dragonite-EX #106 PSA 9 su eBay/Vinted, gap minimo +9,6% sul comp piu'
    affidabile): PriceCharting non separa il prezzo per casa di gradazione
    sotto il Grado 10 - il benchmark "PSA" letto da PriceCharting e' un
    blend cross-company, non un prezzo PSA puro. Il Fair Value per un
    acquisto PSA con dato reale deve ora riflettere PSA_BLEND_PREMIUM_FACTOR
    (1.08x), non il blend grezzo - verificato qui sul percorso REALE che
    l'utente usa dal browser, con dati live (richiede rete)."""
    from poke_quant.slabs.grading_multipliers import PSA_BLEND_PREMIUM_FACTOR

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    at.radio[0].set_value("🔍 Cerca tra tutte le 3.100+ carte del Database PokeQuant")
    at.run(timeout=60)
    sb = at.selectbox[0]
    match = [o for o in sb.options if "Dragonite-EX #106" in o]
    if not match:
        pytest.skip("Dragonite-EX #106 non trovato nel database attuale")
    sb.set_value(match[0])
    at.run(timeout=60)
    at.selectbox[1].set_value("PSA")
    at.selectbox[2].set_value("9.0 Mint")
    at.number_input[1].set_value(94.30)
    at.run(timeout=60)
    at.button[0].click()
    at.run(timeout=60)

    res = at.session_state.get("slab_eval_res")
    assert res is not None
    if "Dato Reale" not in res.get("benchmark_source", ""):
        pytest.skip("Nessun dato reale PriceCharting disponibile per questa carta al momento del test")
    blend_price = res["base_psa_raw"]
    assert res["fair_value_calib"] == round(blend_price * PSA_BLEND_PREMIUM_FACTOR, 2)
    assert res["sniper_ceiling_calib"] >= res["fair_value_calib"]


def test_psa_blend_premium_also_applied_on_db_fallback_without_real_tier_data():
    """GAP NOTO CHIUSO (dichiarato esplicitamente come non risolto nel fix
    PSA_BLEND_PREMIUM_FACTOR, 2026-10-02: "il ramo algoritmico di fallback
    [...] non riceve ancora il premio, perche' condivide il percorso con
    input manual-override/Cardmarket che NON vanno corretti" - l'utente ha
    poi chiesto esplicitamente di risolverlo, 2026-10-03: "Tackle into known,
    deliberately unfixed gap"). Il test precedente copre SOLO il caso in cui
    PriceCharting ha un dato reale per il grado ESATTO richiesto (Priorità 2,
    ramo is_grade_benchmark_price=True). Qui si forza invece il grado 7.0 -
    quasi mai disponibile come dato reale per-tier su PriceCharting - per
    cadere sul fallback Priorità 3 (pannello storico "Database PokeQuant",
    lo stesso identico blend cross-company, letto da cache invece che
    fetchato dal vivo in questo momento) e verificare che riceva la STESSA
    correzione, non solo il percorso con dato reale. fetch_pricecharting_
    grade_tier_price e' mockato a None (nessun dato reale per QUALUNQUE
    grado/carta) per rendere deterministico il percorso di fallback, invece
    di dipendere dal fatto che PriceCharting non abbia per caso un dato
    reale per il grado scelto su quella carta in quel momento - verificato
    dal vivo durante lo sviluppo di questo test: una prima versione senza
    mock era flaky (risolveva un dato reale Grado 7 reale su una carta
    scelta a caso)."""
    from poke_quant.slabs.grading_multipliers import PSA_BLEND_PREMIUM_FACTOR
    from poke_quant.data.storage import load_metadata, load_price_matrix

    with patch("poke_quant.data.price_fetcher.fetch_pricecharting_grade_tier_price", return_value=None):
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=60)
        at.radio[0].set_value("🔍 Cerca tra tutte le 3.100+ carte del Database PokeQuant")
        at.run(timeout=60)
        sb = at.selectbox[0]
        if not sb.options:
            pytest.skip("Nessuna carta disponibile nel database attuale")
        chosen = sb.options[len(sb.options) // 2]  # carta generica, non una mega-chase nota
        sb.set_value(chosen)
        at.run(timeout=60)
        at.selectbox[1].set_value("PSA")
        at.selectbox[2].set_value("7.0 Near Mint")
        at.number_input[1].set_value(1.0)
        at.run(timeout=60)
        at.button[0].click()
        at.run(timeout=60)

    res = at.session_state.get("slab_eval_res")
    assert res is not None
    if "Database PokeQuant" not in res.get("benchmark_source", "") and "Riconosciuta da DB" not in res.get("benchmark_source", ""):
        pytest.skip(f"Questa carta ha risolto un dato reale per-tier invece del fallback DB: {res.get('benchmark_source')}")

    metadata = load_metadata()
    item_id = res["target_item_id"]
    prices_full = load_price_matrix("historical_prices_graded_singles_grade9.csv")
    assert item_id in prices_full.columns, "item_id scelto non nel pannello prezzi"
    raw_blend_price = float(prices_full[item_id].iloc[-1])

    assert res["base_psa_final"] == round(raw_blend_price * PSA_BLEND_PREMIUM_FACTOR, 2), (
        f"base_psa_final ({res['base_psa_final']}) non riflette il premio PSA sul blend grezzo "
        f"({raw_blend_price} × {PSA_BLEND_PREMIUM_FACTOR}) nel ramo di fallback DB"
    )
    assert "×" in res["benchmark_source"] or f"{PSA_BLEND_PREMIUM_FACTOR:.2f}" in res["benchmark_source"]
    assert res["sniper_ceiling_calib"] >= res["fair_value_calib"]


def test_cardmarket_fallback_not_affected_by_psa_blend_premium():
    """Guardia simmetrica al test sopra: cardmarket_ref_price_eur e' una
    fonte diversa (Cardmarket, non PriceCharting) senza il bias di blending
    cross-company qui diagnosticato - NON deve mai ricevere
    PSA_BLEND_PREMIUM_FACTOR, a differenza di last_psa_price. Mocka
    search_metadata_card_by_query per restituire un metadato sintetico con
    SOLO cardmarket_ref_price_eur (nessun last_psa_price) - senza il mock il
    test sarebbe flaky: dipenderebbe dal fatto che una carta reale scelta a
    caso non abbia per caso un last_psa_price valido in cache (il ramo
    Cardmarket e' il fallback di un fallback, raro nei dati reali)."""
    fake_match = (
        "fake_cardmarket_only_card",
        {
            "name": "Carta Sintetica Solo Cardmarket",
            "game_slug": "pokemon-test-set",
            "item_slug": "carta-sintetica-solo-cardmarket",
            "cardmarket_ref_price_eur": 42.0,
            "release_date": "2015-01-01",
        },
    )
    with patch("poke_quant.data.price_fetcher.search_metadata_card_by_query", return_value=fake_match), \
         patch("poke_quant.data.price_fetcher.fetch_pricecharting_grade_tier_price", return_value=None):
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=60)
        at.radio[0].set_value("✏️ Carta Personalizzata / Inserimento Libero (o Link PriceCharting)")
        at.run(timeout=60)
        at.text_input[0].set_value("Carta Sintetica Solo Cardmarket")
        at.selectbox[0].set_value("CGC")
        at.selectbox[1].set_value("9.0 Mint")
        at.number_input[1].set_value(50.0)
        at.run(timeout=60)
        at.button[0].click()
        at.run(timeout=60)

    res = at.session_state.get("slab_eval_res")
    assert res is not None
    assert "Cardmarket" in res.get("benchmark_source", ""), res.get("benchmark_source")
    assert "×" not in res["benchmark_source"]
    assert res["base_psa_final"] == 42.0, "il valore Cardmarket non deve subire nessuna correzione"
