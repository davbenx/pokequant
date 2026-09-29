"""
tests/test_grading_multipliers.py — Test unitari per il modulo di calibrazione
dei moltiplicatori di prezzo tra case di gradazione (poke_quant/slabs/grading_multipliers.py).
"""

import pytest
from poke_quant.slabs.grading_multipliers import (
    GradingCompany,
    Era,
    normalize_company,
    normalize_era,
    get_grading_adjustment,
    adjust_price_for_grading,
    EMPIRICAL_RATIOS_GRADE9,
    EMPIRICAL_RATIOS_GRADE10,
)


def test_normalize_company_aliases():
    assert normalize_company("BGS") == GradingCompany.BGS
    assert normalize_company("Beckett") == GradingCompany.BGS
    assert normalize_company("CGC Cards") == GradingCompany.CGC
    assert normalize_company("TAG") == GradingCompany.TAG
    assert normalize_company("CCC") == GradingCompany.CCC
    assert normalize_company("Classic Card Collector") == GradingCompany.CCC
    assert normalize_company("aigrading") == GradingCompany.AIGRADING
    assert normalize_company("AI Grading") == GradingCompany.AIGRADING
    assert normalize_company("graad") == GradingCompany.GRAAD
    assert normalize_company("PCA France") == GradingCompany.PCA
    assert normalize_company("sgc tuxedo") == GradingCompany.SGC
    assert normalize_company("ACE Grading") == GradingCompany.ACE
    assert normalize_company("Unknown") == GradingCompany.PSA


def test_normalize_era():
    assert normalize_era("vintage") == Era.VINTAGE
    assert normalize_era("WotC 1999") == Era.VINTAGE
    assert normalize_era("mid_era") == Era.MID_ERA
    assert normalize_era("EX Series 2005") == Era.MID_ERA
    assert normalize_era("modern") == Era.MODERN
    assert normalize_era("Scarlet & Violet") == Era.MODERN


def test_hierarchy_grade9_order():
    """Verifica che la gerarchia di valore rispetti la logica di mercato."""
    adj_bgs95 = get_grading_adjustment(GradingCompany.BGS, 9.5, Era.VINTAGE)
    adj_psa9 = get_grading_adjustment(GradingCompany.PSA, 9.0, Era.VINTAGE)
    adj_cgc9 = get_grading_adjustment(GradingCompany.CGC, 9.0, Era.VINTAGE)
    adj_sgc9 = get_grading_adjustment(GradingCompany.SGC, 9.0, Era.VINTAGE)
    adj_pca9 = get_grading_adjustment(GradingCompany.PCA, 9.0, Era.VINTAGE)
    adj_graad9 = get_grading_adjustment(GradingCompany.GRAAD, 9.0, Era.VINTAGE)

    # BGS 9.5 Gem Mint > PSA 9 Mint
    assert adj_bgs95.multiplier > adj_psa9.multiplier
    # PSA 9 Mint > CGC 9 Mint
    assert adj_psa9.multiplier > adj_cgc9.multiplier
    # CGC 9 Mint > SGC 9 Mint
    assert adj_cgc9.multiplier > adj_sgc9.multiplier
    # SGC 9 Mint > PCA 9 Mint
    assert adj_sgc9.multiplier > adj_pca9.multiplier
    # PCA 9 Mint > GRAAD 9 Mint (o comparabile con lieve vantaggio PCA per storico)
    assert adj_pca9.multiplier >= adj_graad9.multiplier


def test_hierarchy_grade10_order():
    """Verifica la gerarchia al Grado 10."""
    adj_black = get_grading_adjustment(GradingCompany.BGS, 10.0, Era.MODERN, subgrades_black_label=True)
    adj_pristine = get_grading_adjustment(GradingCompany.BGS, 10.0, Era.MODERN, is_pristine=True)
    adj_psa10 = get_grading_adjustment(GradingCompany.PSA, 10.0, Era.MODERN)
    adj_cgc_gem = get_grading_adjustment(GradingCompany.CGC, 10.0, Era.MODERN)
    adj_graad10 = get_grading_adjustment(GradingCompany.GRAAD, 10.0, Era.MODERN)

    assert adj_black.multiplier > adj_pristine.multiplier
    assert adj_pristine.multiplier > adj_psa10.multiplier
    assert adj_psa10.multiplier > adj_cgc_gem.multiplier
    assert adj_cgc_gem.multiplier > adj_graad10.multiplier


def test_adjust_price_for_grading_calculations():
    base_psa9 = 100.0
    fair_cgc9, sniper_cgc9, adj_cgc = adjust_price_for_grading(base_psa9, "CGC", 9.0, Era.VINTAGE)
    # CGC 9 vintage multiplier ~0.901
    assert 85.0 <= fair_cgc9 <= 95.0
    assert sniper_cgc9 <= base_psa9

    fair_bgs95, sniper_bgs95, adj_bgs = adjust_price_for_grading(base_psa9, "BGS", 9.5, Era.VINTAGE)
    # BGS 9.5 vintage multiplier ~1.809
    assert fair_bgs95 > base_psa9
    assert sniper_bgs95 > base_psa9

    fair_graad9, sniper_graad9, adj_graad = adjust_price_for_grading(base_psa9, "GRAAD", 9.0, Era.VINTAGE)
    # GRAAD 9 vintage multiplier ~0.681
    assert 65.0 <= fair_graad9 <= 72.0
    assert sniper_graad9 < 80.0


def test_special_variants():
    from poke_quant.slabs.grading_multipliers import get_variant_multiplier

    v_std, _ = get_variant_multiplier("standard")
    v_1st_wotc, _ = get_variant_multiplier("1st Edition WotC")
    v_1st_base, _ = get_variant_multiplier("1st Edition Base Set")
    v_nosym, _ = get_variant_multiplier("No Symbol")
    v_shadow, _ = get_variant_multiplier("Shadowless")

    assert v_std == 1.0
    assert v_1st_wotc == 2.5
    assert v_1st_base == 6.0
    assert v_nosym == 1.4
    assert v_shadow == 3.0

    # Test set-awareness
    v_fossil_1st, _ = get_variant_multiplier("1st Edition Base Set", game_slug="pokemon-fossil")
    assert v_fossil_1st == 2.5  # Corretto a WotC standard anche se l'utente ha cliccato per sbaglio Base Set
    v_base_1st, _ = get_variant_multiplier("1st Edition", game_slug="pokemon-base-set")
    assert v_base_1st == 6.0

    # Test regressione stringhe complesse con numeri (es. etichette dropdown)
    from poke_quant.slabs.grading_multipliers import variant_to_pricecharting_key, normalize_variant
    ui_label = "No Symbol Error (Rileva reale da PriceCharting o ~1.4x)"
    assert normalize_variant(ui_label) == "no_symbol"
    assert variant_to_pricecharting_key(ui_label) == "no-symbol"
    v_nosym_ui, _ = get_variant_multiplier(ui_label, game_slug="pokemon-jungle")
    assert v_nosym_ui == 1.4  # Mai piu confuso con 1st edition solo per via di '1.4x'!

    # Test adjust_price_for_grading con variante
    base_unlimited = 100.0
    fair_1st, _, _ = adjust_price_for_grading(base_unlimited, "PSA", 9.0, Era.VINTAGE, variant="1st Edition WotC")
    assert fair_1st == 250.0


def test_black_label_multiplier():
    base_psa10 = 500.0
    fair_black, sniper_black, adj_black = adjust_price_for_grading(
        base_psa_price_eur=base_psa10,
        company="BGS",
        grade=10.0,
        era=Era.MODERN,
        subgrades_black_label=True,
    )
    assert adj_black.multiplier >= 4.0
    assert fair_black >= 2000.0
    assert sniper_black >= 1500.0


def test_current_fx_rate_integration():
    from poke_quant.data.fx_rates import get_current_eur_usd_rate
    rate = get_current_eur_usd_rate()
    assert 1.0 <= rate <= 1.4  # Range realistico per EUR/USD


def test_era_ratios_and_estimation():
    from poke_quant.slabs.grading_multipliers import (
        ERA_PSA10_TO_PSA9_RATIO,
        ERA_BGS95_TO_PSA9_RATIO,
        estimate_psa10_from_psa9,
        estimate_grade95_from_psa9,
    )

    # Verifica moltiplicatori empirici
    assert ERA_PSA10_TO_PSA9_RATIO[Era.VINTAGE] == 3.80
    assert ERA_PSA10_TO_PSA9_RATIO[Era.MID_ERA] == 3.40
    assert ERA_PSA10_TO_PSA9_RATIO[Era.MODERN] == 2.80

    assert ERA_BGS95_TO_PSA9_RATIO[Era.VINTAGE] == 1.81
    assert ERA_BGS95_TO_PSA9_RATIO[Era.MID_ERA] == 1.55
    assert ERA_BGS95_TO_PSA9_RATIO[Era.MODERN] == 1.70

    # Test stima da PSA 9
    psa9_price = 100.0
    assert estimate_psa10_from_psa9(psa9_price, Era.VINTAGE) == 380.0
    assert estimate_psa10_from_psa9(psa9_price, Era.MID_ERA) == 340.0
    assert estimate_psa10_from_psa9(psa9_price, Era.MODERN) == 280.0

    assert estimate_grade95_from_psa9(psa9_price, Era.VINTAGE) == 181.0
    assert estimate_grade95_from_psa9(psa9_price, Era.MID_ERA) == 155.0
    assert estimate_grade95_from_psa9(psa9_price, Era.MODERN) == 170.0


def test_get_recommended_grade_for_card():
    from poke_quant.slabs.grading_multipliers import get_recommended_grade_for_card

    # Vintage (anno <= 2003)
    rec_v = get_recommended_grade_for_card(rel_year=2000)
    assert rec_v["era"] == "vintage"
    assert "PSA 9" in rec_v["target_grade"]
    assert rec_v["is_grade9_viable"] is True
    assert rec_v["warning_modern_g9"] is False
    assert "Sweet Spot" in rec_v["rationale"]

    # Mid-Era (2004 <= anno <= 2016)
    rec_m = get_recommended_grade_for_card(rel_year=2010)
    assert rec_m["era"] == "mid_era"
    assert "PSA 9" in rec_m["target_grade"]
    assert "9.5" in rec_m["target_grade"]
    assert rec_m["is_grade9_viable"] is True
    assert rec_m["warning_modern_g9"] is False

    # Moderno (anno >= 2017)
    rec_mod = get_recommended_grade_for_card(rel_year=2021)
    assert rec_mod["era"] == "modern"
    assert "PSA 10" in rec_mod["target_grade"]
    assert rec_mod["is_grade9_viable"] is False
    assert rec_mod["warning_modern_g9"] is True
    assert "Evita Grado 9" in rec_mod["target_badge"] or "Evita G9" in rec_mod["target_badge"]
    assert "trappola" in rec_mod["rationale"].lower()


def test_fetch_pricecharting_grade_tier_price():
    from poke_quant.data.price_fetcher import fetch_pricecharting_grade_tier_price

    # Verifica lookup su carta con dati reali (Dragonite-EX)
    res = fetch_pricecharting_grade_tier_price("pokemon-evolutions", "dragonite-ex-106", tier="psa10")
    assert res is not None
    eur, usd, url, src = res
    assert usd == 480.0
    assert eur > 350.0
    assert "pricecharting.com" in url


def test_lower_grades_monotonicity():
    """Verifica che i moltiplicatori scendano monotonicamente da 9.0 a 7.0 per tutte le compagnie."""
    from poke_quant.slabs.grading_multipliers import get_grading_adjustment, GradingCompany, Era

    grades = ["9.0", "8.5", "8.0", "7.5", "7.0"]
    for comp in GradingCompany:
        for era in [Era.VINTAGE, Era.MID_ERA, Era.MODERN]:
            multipliers = [get_grading_adjustment(comp, g, era).multiplier for g in grades]
            for i in range(len(multipliers) - 1):
                assert multipliers[i] > multipliers[i + 1], (
                    f"Violazione monotonicità per {comp.value} in {era.value}: "
                    f"{grades[i]} ({multipliers[i]}) <= {grades[i+1]} ({multipliers[i+1]})"
                )


def test_company_relative_factor_vs_psa():
    from poke_quant.slabs.grading_multipliers import get_company_relative_factor_vs_psa, GradingCompany, Era

    # PSA vs PSA è sempre 1.0
    assert get_company_relative_factor_vs_psa(GradingCompany.PSA, "8.5", Era.VINTAGE) == 1.0
    assert get_company_relative_factor_vs_psa("PSA", "7.0", Era.MID_ERA) == 1.0

    # BGS e CGC tengono il valore meglio degli enti regionali su gradi inferiori
    bgs_factor = get_company_relative_factor_vs_psa(GradingCompany.BGS, "8.0", Era.VINTAGE)
    cgc_factor = get_company_relative_factor_vs_psa(GradingCompany.CGC, "8.0", Era.VINTAGE)
    graad_factor = get_company_relative_factor_vs_psa(GradingCompany.GRAAD, "8.0", Era.VINTAGE)

    assert bgs_factor >= 0.88
    assert cgc_factor >= 0.85
    assert graad_factor <= 0.70
    assert bgs_factor > graad_factor
    assert cgc_factor > graad_factor


def test_get_grade_benchmarks_ladder():
    from poke_quant.slabs.grading_multipliers import get_grade_benchmarks_ladder, Era

    # 1. Test con carta reale presente in cache ladder (m_rayquaza_ex_105)
    ladder_real = get_grade_benchmarks_ladder(base_psa9_eur=100.0, era=Era.MID_ERA, item_id="m_rayquaza_ex_105")
    assert "10.0" in ladder_real
    assert "9.5" in ladder_real
    assert "9.0" in ladder_real
    assert "8.5" in ladder_real
    assert "8.0" in ladder_real
    assert "7.5" in ladder_real
    assert "7.0" in ladder_real

    # Verifica monotonicità prezzi reali
    assert ladder_real["10.0"]["price_eur"] > ladder_real["9.5"]["price_eur"]
    assert ladder_real["9.5"]["price_eur"] > ladder_real["9.0"]["price_eur"]
    assert ladder_real["9.0"]["price_eur"] > ladder_real["8.5"]["price_eur"]
    assert ladder_real["8.5"]["price_eur"] > ladder_real["8.0"]["price_eur"]
    assert ladder_real["8.0"]["price_eur"] > ladder_real["7.5"]["price_eur"]
    assert ladder_real["7.5"]["price_eur"] > ladder_real["7.0"]["price_eur"]
    assert ladder_real["8.0"]["is_real"] is True

    # 2. Test fallback puramente algoritmico
    ladder_algo = get_grade_benchmarks_ladder(base_psa9_eur=100.0, era=Era.VINTAGE)
    assert ladder_algo["9.0"]["price_eur"] == 100.0
    assert ladder_algo["8.5"]["price_eur"] == 78.0
    assert ladder_algo["8.0"]["price_eur"] == 65.0
    assert ladder_algo["7.5"]["price_eur"] == 55.0
    assert ladder_algo["7.0"]["price_eur"] == 48.0
    assert ladder_algo["8.0"]["is_real"] is False


def test_fetch_pricecharting_grade_tier_price_lower_grades():
    from poke_quant.data.price_fetcher import fetch_pricecharting_grade_tier_price

    # m_rayquaza_ex_105 ha dati reali per grade8, grade7
    res_g8 = fetch_pricecharting_grade_tier_price("pokemon-roaring-skies", "m-rayquaza-ex-105", tier="grade8", item_id="m_rayquaza_ex_105")
    assert res_g8 is not None
    eur8, usd8, url8, src8 = res_g8
    assert eur8 == 680.01

    res_g7 = fetch_pricecharting_grade_tier_price("pokemon-roaring-skies", "m-rayquaza-ex-105", tier="grade7", item_id="m_rayquaza_ex_105")
    assert res_g7 is not None
    eur7, usd7, url7, src7 = res_g7
    assert eur7 == 448.9

    # Mezzi voti interpolati
    res_g85 = fetch_pricecharting_grade_tier_price("pokemon-roaring-skies", "m-rayquaza-ex-105", tier="grade8_5", item_id="m_rayquaza_ex_105")
    assert res_g85 is not None
    eur85, _, _, src85 = res_g85
    assert eur8 < eur85 < 1955.26
    assert "Interpolato" in src85


def test_get_recommended_grade_targets():
    from poke_quant.slabs.grading_multipliers import get_recommended_grade_targets, Era

    # 1. Moderno: Target PSA 10, minor alt BGS 9.5
    mod = get_recommended_grade_targets(base_psa9_eur=50.0, era=Era.MODERN)
    assert mod["target_grade"] == "PSA 10"
    assert mod["target_price_eur"] == 140.0  # 50.0 * 2.80
    assert "PSA 10" in mod["target_badge"]
    assert mod["is_grade9_viable"] is False
    assert len(mod["minor_alternatives"]) == 1
    assert mod["minor_alternatives"][0]["grade"] == "9.5"
    assert mod["minor_alternatives"][0]["company"] == "BGS"
    assert "BGS 9.5" in mod["minor_alternatives_str"]
    assert "Sconsigliati gradi ≤ 9.0" in mod["minor_alternatives_str"]

    # 2. Vintage: Target PSA 9, minor alts PSA 8.5, 8.0, 7.0
    vint = get_recommended_grade_targets(base_psa9_eur=200.0, era=Era.VINTAGE)
    assert vint["target_grade"] == "PSA 9"
    assert vint["target_price_eur"] == 200.0
    assert vint["is_grade9_viable"] is True
    assert len(vint["minor_alternatives"]) == 3
    grades = [a["grade"] for a in vint["minor_alternatives"]]
    assert grades == ["8.5", "8.0", "7.0"]
    assert "PSA 8.5" in vint["minor_alternatives_str"]
    assert "PSA 8.0" in vint["minor_alternatives_str"]
    assert "PSA 7.0" in vint["minor_alternatives_str"]
    assert vint["minor_alternatives"][0]["price_eur"] == 156.0  # 200 * 0.78
    assert vint["minor_alternatives"][1]["price_eur"] == 130.0  # 200 * 0.65
    assert vint["minor_alternatives"][2]["price_eur"] == 96.0   # 200 * 0.48

    # 3. Mid-Era: Target PSA 9, minor alts PSA 8.5, 8.0
    mid = get_recommended_grade_targets(base_psa9_eur=100.0, era=Era.MID_ERA)
    assert mid["target_grade"] == "PSA 9"
    assert mid["target_price_eur"] == 100.0
    assert mid["is_grade9_viable"] is True
    assert len(mid["minor_alternatives"]) == 2
    assert [a["grade"] for a in mid["minor_alternatives"]] == ["8.5", "8.0"]

    # 4. Con dati reali PC (m_rayquaza_ex_105)
    ray = get_recommended_grade_targets(base_psa9_eur=1000.0, era=Era.MID_ERA, item_id="m_rayquaza_ex_105")
    assert ray["minor_alternatives"][1]["is_real"] is True
    assert ray["minor_alternatives"][1]["price_eur"] == 680.01
    assert "pop_pressure" in ray


def test_get_card_pop_pressure():
    from poke_quant.slabs.grading_multipliers import get_card_pop_pressure, Era

    # Test con carta con dati di popolazione noti
    pop_info = get_card_pop_pressure("alakazam_1", Era.VINTAGE)
    assert pop_info is not None
    assert "tier" in pop_info
    assert "badge_html" in pop_info
    assert "ratio_8_9" in pop_info
    if pop_info["ratio_8_9"] is not None:
        assert pop_info["ratio_8_9"] >= 0.0
        assert 0.0 <= pop_info["percentile"] <= 100.0

    # Test con carta sconosciuta (fallback graceful)
    unknown = get_card_pop_pressure("carta_inesistente_xyz_999", Era.MODERN)
    assert unknown["tier"] == "unknown"
    assert "Pop N/D" in unknown["badge_html"]
    assert unknown["is_overcrowded"] is False


def test_adjust_price_for_grading_grade_benchmark():
    from poke_quant.slabs.grading_multipliers import adjust_price_for_grading, Era

    # Caso Dark Vileplume: PriceCharting restituisce il prezzo reale del grado 8.5 (78.54 €)
    fv, sc, adj = adjust_price_for_grading(
        base_psa_price_eur=78.54,
        company="BGS",
        grade="8.5",
        era=Era.VINTAGE,
        is_grade_benchmark_price=True,
    )
    # Fair Value BGS 8.5 deve essere calibrato con comp_rel vs PSA 8.5 (0.921x -> 72.34 €)
    assert fv == 72.34
    # Lo sniper ceiling deve essere 72.34 * 1.05 = 75.96 € e NON collassare a 60.20 € per doppio sconto
    assert sc == 75.96
    assert sc >= fv


def test_get_card_strategy_and_pop_details():
    from poke_quant.slabs.grading_multipliers import get_card_strategy_and_pop_details

    # 1. Test carta Core BUY (Dark Vileplume #13)
    dv = get_card_strategy_and_pop_details(item_id="dark_vileplume_13")
    assert dv["item_id"] == "dark_vileplume_13"
    assert dv["strategy_tier"] == "CORE_BUY"
    assert "Tier 1: Core Conviction" in dv["strategy_badge"]
    assert dv["discount_pct"] < -70.0
    assert dv["psa_census"]["total"] > 1000
    assert dv["psa_census"]["10"] == 22
    assert dv["psa_census"]["9"] == 434
    assert dv["cgc_census"]["10"] >= 20
    assert dv["pricecharting_pop_url"] is not None
    assert "pricecharting.com/pop/item/pokemon-team-rocket/dark-vileplume-13" in dv["pricecharting_pop_url"]
    assert "psacard.com/pop/search" in dv["psa_search_url"]
    assert "copie a box" in dv["pull_rate_desc"]
    assert "Fuori Stampa" in dv["supply_status"]

    # 2. Test carta Alternativa BUY (Alakazam #1)
    al = get_card_strategy_and_pop_details(item_id="alakazam_1")
    assert al["strategy_tier"] == "ALT_BUY"
    assert "Alternativa" in al["strategy_badge"]
    assert al["discount_pct"] < -50.0

    # 3. Test carta fuori catalogo / custom (graceful fallback)
    custom = get_card_strategy_and_pop_details(
        item_id=None,
        game_slug="pokemon-custom-set",
        item_slug="custom-card-99",
        card_name="Pikachu Custom",
    )
    assert custom["strategy_tier"] in ["CUSTOM", "NEUTRAL"]
    assert "psacard.com/pop/search" in custom["psa_search_url"]







def test_sniper_ceiling_never_below_fair_value():
    """BUG TROVATO (verifica richiesta dall'utente: "verifica che le
    valutazioni dei prezzi massimi sulle slab per avere edge siano
    corretti", 2026-09-29): sniper_ceiling_factor e' il fattore MASSIMO
    consentito per preservare l'edge - per costruzione non puo' mai essere
    sotto multiplier (il fair value che delimita), altrimenti
    eval_slab_listing.py marca come "OVERPRICED" uno slab venduto esattamente
    al fair value del modello. Vero per 90/90 celle a grado 9.0-7.0, ma era
    rotto per le 9 celle a grado 9.5 (PSA/BGS/CGC) e per 8 sotto-varianti
    premium di Grado 10 (BGS Pristine/Black Label, CGC/TAG Pristine Modern) -
    es. BGS 9.5 vintage: mult=1.809 ma sniper_factor=1.250 (-31%). Corretto
    con un floor in get_grading_adjustment(). Questo test verifica
    l'invariante su OGNI cella delle due tabelle, non solo i casi trovati."""
    for table in (EMPIRICAL_RATIOS_GRADE9, EMPIRICAL_RATIOS_GRADE10):
        for (company, grade, era) in table.keys():
            adj = get_grading_adjustment(
                company=company, grade=grade, era=era,
                subgrades_black_label=("black_label" in grade),
                is_pristine=("pristine" in grade),
            )
            assert adj.sniper_ceiling_factor >= adj.multiplier - 1e-9, (
                f"{company.value} {grade} {era.value}: sniper_ceiling_factor "
                f"{adj.sniper_ceiling_factor} < multiplier {adj.multiplier}"
            )


def test_adjust_price_for_grading_sniper_ceiling_covers_fair_value():
    """Caso concreto del bug: prima della correzione BGS 9.5 vintage produceva
    fair_value=180.90 ma sniper_ceiling=125.00 (sotto il fair value stesso)."""
    fair_value, sniper_ceiling, _ = adjust_price_for_grading(
        base_psa_price_eur=100.0, company=GradingCompany.BGS, grade="9.5", era=Era.VINTAGE,
    )
    assert sniper_ceiling >= fair_value
