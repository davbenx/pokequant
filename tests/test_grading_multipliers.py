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

    # Test adjust_price_for_grading con variante
    base_unlimited = 100.0
    fair_1st, _, _ = adjust_price_for_grading(base_unlimited, "PSA", 9.0, Era.VINTAGE, variant="1st Edition WotC")
    assert fair_1st == 250.0
