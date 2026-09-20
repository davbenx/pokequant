"""
tests/test_slab_edges.py — Test unitari per i 10 Edge Matematici delle Carte Gradate (PSA, BGS, CGC).
Verifica:
  - Spread cross-grader e Z-Score (Edge 1)
  - Filtri Pop Saturation e blocco diluizione (Edge 2)
  - Moltiplicatore di scarsità condizione / Gem-Rate (Edge 3)
  - Cost-to-Grade Manufacturing Floor (Edge 4)
  - Geo-dislocation (Edge 5)
  - Presa di profitto a 2 tranche e allerta diluizione (Edge 6-8)
  - Rotazione ottimizzata del capitale (Edge 9-10)
"""

import pytest
from poke_quant.slabs.models import GradingCompany, SlabGrade, EdgeType, Subgrades, AvailabilityStatus
from poke_quant.slabs.edge_calculator import (
    calc_cross_grading_spread,
    calc_pop_saturation_edge,
    calc_gem_scarcity_edge,
    calc_manufacturing_cost_floor,
    calc_geo_dislocation,
    calc_pop_dilution_wave_sell,
    calc_parabolic_exhaustion_sell,
    calc_spread_convergence_sell,
    calc_opportunity_cost_rotation,
    calc_era_cycle_rotation
)
from poke_quant.slabs.slab_scanner import scan_slabs_market
from poke_quant.slabs.slab_universe import get_cardmarket_direct_link, SLAB_UNIVERSE


def test_cross_grading_spread_edge_triggers_on_deep_discount():
    # PSA 10 a 1000€, BGS 9.5 normalmente scambia a 780€ (ratio 0.78).
    # Se BGS 9.5 è listata a 500€ (ratio 0.50), Z-score = (0.50 - 0.78) / 0.07 = -4.0
    res = calc_cross_grading_spread(
        psa_10_price=1000.0,
        target_price=500.0,
        company=GradingCompany.BGS,
        grade=SlabGrade.BGS_9_5_GEM
    )
    assert res["edge_active"] is True
    assert res["z_score"] <= -2.0
    assert res["margin_of_safety_pct"] > 30.0
    assert res["crossover_probability"] >= 0.80


def test_cross_grading_spread_inactive_on_fair_pricing():
    # BGS 9.5 a 780€ vs PSA 10 a 1000€ (prezzo equo perfetto)
    res = calc_cross_grading_spread(
        psa_10_price=1000.0,
        target_price=780.0,
        company=GradingCompany.BGS,
        grade=SlabGrade.BGS_9_5_GEM
    )
    assert res["edge_active"] is False
    assert abs(res["z_score"]) < 1.0


def test_cross_grading_quad_subgrades_bonus():
    # Subgrades quad 9.5 aumentano la probabilità di crossover a 0.90
    subs = Subgrades(centering=9.5, corners=9.5, edges=9.5, surface=9.5)
    res = calc_cross_grading_spread(
        psa_10_price=1000.0,
        target_price=600.0,
        company=GradingCompany.BGS,
        grade=SlabGrade.BGS_9_5_GEM,
        subgrades=subs
    )
    assert res["crossover_probability"] == pytest.approx(0.85)
    assert res["edge_active"] is True


def test_pop_saturation_blocks_dilution():
    # Pop velocity alta (5.0%/mese) -> acquisto bloccato!
    res = calc_pop_saturation_edge(
        pop_growth_30d_pct=5.0,
        pop_acceleration_pct=20.0,
        gem_rate=0.70
    )
    assert res["edge_active"] is False
    assert res["is_dilution_blocked"] is True


def test_pop_saturation_plateau_confirmed():
    # Pop velocity piatta (0.5%/mese) con accelerazione negativa
    res = calc_pop_saturation_edge(
        pop_growth_30d_pct=0.5,
        pop_acceleration_pct=-5.0,
        gem_rate=0.60
    )
    assert res["edge_active"] is True
    assert res["is_dilution_blocked"] is False
    assert res["confidence"] >= 0.70


def test_gem_scarcity_edge():
    # Carta vintage/difficile: Gem-rate 10%, PSA 9 a 100€
    # Multiplier equo = (1 / 0.10)^0.65 = 4.47x -> Fair PSA 10 = 447€
    # Se PSA 10 è quotata a 250€ (solo 2.5x vs 4.47x equo), scatta l'edge
    res = calc_gem_scarcity_edge(
        gem_rate=0.10,
        psa_10_price=250.0,
        psa_9_price=100.0
    )
    assert res["edge_active"] is True
    assert res["margin_of_safety_pct"] > 35.0


def test_cost_to_grade_floor_arbitrage():
    # Raw a 100€, Gem-rate 50%, costi perizia 40€
    # Floor industriale = (100 + 40 - 0.5*80) / 0.5 = 200€
    # Se venduta a 150€ (<= 0.90 * 200€ = 180€), scatta arbitraggio
    res = calc_manufacturing_cost_floor(
        raw_price=100.0,
        gem_rate=0.50,
        grading_fee=25.0,
        shipping_customs=15.0,
        psa_9_price=80.0,
        listing_price=150.0
    )
    assert res["edge_active"] is True
    assert res["cost_floor_eur"] == 200.0
    assert res["margin_of_safety_pct"] == 25.0


def test_geo_dislocation_edge():
    # US comp: 1200$ (pari a 1111€ con EUR/USD 1.08)
    # Cardmarket ask: 800€
    # Net spread: ((1111 - 800 - 12) / 1111) * 100 = 26.9% (> 18%)
    res = calc_geo_dislocation(
        us_comp_sold_usd=1200.0,
        cardmarket_ask_eur=800.0,
        eur_usd=1.08
    )
    assert res["edge_active"] is True
    assert res["net_spread_pct"] > 20.0


def test_parabolic_exhaustion_tranche_1_and_2():
    # Test Tranche 1 (+70% ROI)
    res_t1 = calc_parabolic_exhaustion_sell(
        current_price=175.0,
        buy_price=100.0,
        tranche_1_already_sold=False
    )
    assert res_t1["edge_active"] is True
    assert res_t1["tranche"] == "TRANCHE_1_CAPITAL_RECOVERY"
    assert res_t1["sell_percentage"] == 50.0

    # Test Tranche 2 (Z >= 2.5)
    res_t2 = calc_parabolic_exhaustion_sell(
        current_price=300.0,
        buy_price=100.0,
        price_1y_mean=150.0,
        price_1y_std=50.0,  # Z = (300 - 150) / 50 = 3.0 >= 2.5
        tranche_1_already_sold=True
    )
    assert res_t2["edge_active"] is True
    assert res_t2["tranche"] == "TRANCHE_2_FULL_EXIT"
    assert res_t2["sell_percentage"] == 100.0


def test_opportunity_cost_rotation():
    # Posizione attuale stagnante da 8 mesi (CAGR atteso 3%)
    # Target con upside stimato +35% -> Alpha differenziale netto = 35 - 3 - 8 = +24% >= 20%
    res = calc_opportunity_cost_rotation(
        holding_expected_cagr_pct=3.0,
        target_expected_cagr_pct=35.0,
        holding_months_stagnant=8
    )
    assert res["should_rotate"] is True
    assert res["net_alpha_differential_pct"] >= 20.0


def test_era_cycle_rotation():
    # Modern Grail a 2000€ vs Vintage Grail a 1200€ -> EraRatio = 1.67 >= 1.30
    res = calc_era_cycle_rotation(
        modern_grail_psa_10_price=2000.0,
        vintage_grail_psa_9_or_10_price=1200.0,
        era_ratio_95th_percentile=1.30
    )
    assert res["should_rotate"] is True
    assert res["era_ratio"] == 1.67


def test_verified_availability_filter_isolates_oos():
    """
    Verifica che il filtro di disponibilità reale scarti i comp teorici OOS
    (come Charizard Base Set BGS 9.5 a 2200€ quando l'ask reale è 4800€)
    e mantenga SOLO offerte eseguibili con pezzi reali in vendita.
    """
    # 1. Con filtro attivo: esclude OOS e popola unverified_or_out_of_stock
    verified_res = scan_slabs_market(require_verified_available=True)
    assert verified_res["require_verified_available"] is True
    assert len(verified_res["unverified_or_out_of_stock"]) >= 1

    oos_cards = [u["card_id"] for u in verified_res["unverified_or_out_of_stock"]]
    assert "base_set_charizard_unlimited" in oos_cards

    # Tutte le opportunità BUY rimanenti devono avere status VERIFIED_AVAILABLE e pezzi > 0
    for b in verified_res["buy_signals"]:
        assert b.availability_status == AvailabilityStatus.VERIFIED_AVAILABLE
        assert b.active_listing_count >= 1
        assert b.cardmarket_direct_url is not None
        assert "isGraded=Y" in b.cardmarket_direct_url
        assert b.seller_country != ""

    # 2. Con filtro disattivato: accetta comp storici teorici
    unfiltered_res = scan_slabs_market(require_verified_available=False)
    assert unfiltered_res["require_verified_available"] is False
    assert len(unfiltered_res["unverified_or_out_of_stock"]) == 0
    unfiltered_buy_ids = [b.card_id for b in unfiltered_res["buy_signals"]]
    assert "base_set_charizard_unlimited" in unfiltered_buy_ids


def test_cardmarket_direct_link_generator():
    """Verifica che il generatore di deep-link Cardmarket produca URL validi con filtri per lastre."""
    card = SLAB_UNIVERSE["pikachu_van_gogh_085"]
    url = get_cardmarket_direct_link(card, "BGS_9_5_GEM")
    assert url.startswith("https://www.cardmarket.com/en/")
    assert "Pikachu-with-Grey-Felt-Hat" in url
    assert "isGraded=Y" in url

