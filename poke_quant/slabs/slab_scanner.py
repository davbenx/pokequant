"""
poke_quant/slabs/slab_scanner.py — Scanner operativo e generatore di segnali su Carte Gradate (PSA, BGS, CGC).
Esegue la scansione in tempo reale calcolando i 10 Edge matematici:
  - Genera segnali BUY su dislocazioni statistiche
  - Genera segnali SELL (Tranche 1, Tranche 2, Dilution Wave)
  - Genera raccomandazioni di ROTAZIONE del capitale
  - Dimostra il rigetto degli asset spazzatura/iper-inflazionati (Anti-Survivorship Bias)
"""

from __future__ import annotations
import uuid
import datetime
from typing import Dict, List, Any, Optional

from poke_quant.slabs.models import (
    GradingCompany, SlabGrade, EdgeType, SignalAction, AvailabilityStatus,
    SlabSignal, RotationRecommendation, SlabHolding
)
from poke_quant.slabs.slab_universe import (
    get_slab_universe, get_curated_grails, get_failed_controls, get_cardmarket_direct_link
)
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


def scan_slabs_market(
    user_holdings: Optional[List[SlabHolding]] = None,
    us_auction_comps_usd: Optional[Dict[str, float]] = None,
    custom_universe: Optional[Dict[str, Dict[str, Any]]] = None,
    require_verified_available: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Esegue la scansione completa del mercato Slabs con verifica di disponibilità reale in vendita.
    Se require_verified_available=True, filtra ed esclude i prezzi teorici o storici
    dove non esiste un'inserzione attiva al prezzo dichiarato.
    """
    universe = custom_universe or get_slab_universe()
    if user_holdings is None:
        user_holdings = []
    if us_auction_comps_usd is None:
        us_auction_comps_usd = {}

    buy_signals: List[SlabSignal] = []
    sell_signals: List[SlabSignal] = []
    rotations: List[RotationRecommendation] = []
    rejected_controls: List[Dict[str, Any]] = []
    unverified_or_out_of_stock: List[Dict[str, Any]] = []
    all_evaluated: List[Dict[str, Any]] = []

    # =========================================================================
    # 1. SCANSIONE ACQUISTO (BUY) ED EVALUATION GENERALE
    # =========================================================================
    for card_id, item in universe.items():
        name = item["name"]
        psa_10_px = float(item.get("psa_10_price_eur", 0.0))
        psa_9_px = float(item.get("psa_9_price_eur", 0.0))
        raw_px = float(item.get("raw_price_eur", 0.0))
        bgs_9_5_px = float(item.get("bgs_9_5_price_eur", 0.0))
        cgc_10_px = float(item.get("cgc_10_gem_price_eur", 0.0))
        gem_rate = float(item.get("gem_rate", 0.70))
        pop_growth = float(item.get("pop_growth_30d_pct", 0.0))
        pop_acc = float(item.get("pop_acceleration_pct", 0.0))
        is_failed = item.get("is_failed_control", False)

        # 1.1 FILTRO RIGETTO ASSET CONTROLLO (Anti-Survivorship Test)
        if is_failed:
            rejected_controls.append({
                "card_id": card_id,
                "name": name,
                "gem_rate": gem_rate,
                "pop_growth_30d_pct": pop_growth,
                "rejection_reason": "Asset privo di moat collezionistico / Gem-rate eccessivamente alto / Iper-offerta."
            })
            continue

        # 1.2 TEST EDGE 2: POP SATURATION PLATEAU (Supply Check)
        pop_edge = calc_pop_saturation_edge(pop_growth, pop_acc, gem_rate)
        if pop_edge["is_dilution_blocked"]:
            # Se la carta è in fase di diluizione attiva, nessun BUY è permesso!
            all_evaluated.append({
                "card_id": card_id,
                "name": name,
                "status": "BLOCKED_BY_DILUTION",
                "reason": pop_edge["reason"]
            })
            continue

        def _evaluate_listing(grade_key: str, target_px: float):
            live_dict = item.get("live_listings", {}).get(grade_key, {})
            status_str = live_dict.get("status", "HISTORICAL_COMP")
            lowest_ask = float(live_dict.get("lowest_ask", target_px))
            units = int(live_dict.get("units", 0))
            country = live_dict.get("country", "EU")
            direct_url = get_cardmarket_direct_link(item, grade_key)
            is_available = (status_str == "VERIFIED_AVAILABLE") and (units > 0) and (lowest_ask <= target_px * 1.05)
            return is_available, status_str, lowest_ask, units, country, direct_url

        # 1.3 TEST EDGE 1: CROSS-COMPANY SPREAD (BGS 9.5 vs PSA 10)
        if bgs_9_5_px > 0 and psa_10_px > 0:
            bgs_res = calc_cross_grading_spread(
                psa_10_price=psa_10_px,
                target_price=bgs_9_5_px,
                company=GradingCompany.BGS,
                grade=SlabGrade.BGS_9_5_GEM
            )
            if bgs_res["edge_active"]:
                is_avail, st_str, low_ask, units, country, d_url = _evaluate_listing("BGS_9_5_GEM", bgs_9_5_px)
                if require_verified_available and not is_avail:
                    unverified_or_out_of_stock.append({
                        "card_id": card_id,
                        "name": name,
                        "grade": SlabGrade.BGS_9_5_GEM.value,
                        "target_price": bgs_9_5_px,
                        "lowest_active_ask": low_ask,
                        "status": st_str,
                        "reason": f"Edge teorico attivo (Z={bgs_res['z_score']:.2f}), ma attualmente esaurito/ask troppo alto su Cardmarket ({low_ask:,.0f} € vs target {bgs_9_5_px:,.0f} €).",
                        "direct_url": d_url
                    })
                else:
                    buy_signals.append(SlabSignal(
                        signal_id=f"SIG-BUY-BGS-{card_id}",
                        action=SignalAction.BUY,
                        card_id=card_id,
                        card_name=name,
                        target_company=GradingCompany.BGS,
                        target_grade=SlabGrade.BGS_9_5_GEM,
                        current_price_eur=bgs_9_5_px,
                        fair_value_eur=bgs_res["fair_value_eur"],
                        margin_of_safety_pct=bgs_res["margin_of_safety_pct"],
                        primary_edge=EdgeType.CROSS_COMPANY_DISLOCATION,
                        confidence_score=0.85,
                        reason=bgs_res["reason"],
                        availability_status=AvailabilityStatus.VERIFIED_AVAILABLE if is_avail else AvailabilityStatus.HISTORICAL_COMP,
                        active_listing_count=units,
                        lowest_active_ask_eur=low_ask,
                        cardmarket_direct_url=d_url,
                        seller_country=country,
                        metrics=bgs_res
                    ))

        # 1.4 TEST EDGE 1: CROSS-COMPANY SPREAD (CGC 10 vs PSA 10)
        if cgc_10_px > 0 and psa_10_px > 0:
            cgc_res = calc_cross_grading_spread(
                psa_10_price=psa_10_px,
                target_price=cgc_10_px,
                company=GradingCompany.CGC,
                grade=SlabGrade.CGC_10_GEM
            )
            if cgc_res["edge_active"]:
                is_avail, st_str, low_ask, units, country, d_url = _evaluate_listing("CGC_10_GEM", cgc_10_px)
                if require_verified_available and not is_avail:
                    unverified_or_out_of_stock.append({
                        "card_id": card_id,
                        "name": name,
                        "grade": SlabGrade.CGC_10_GEM.value,
                        "target_price": cgc_10_px,
                        "lowest_active_ask": low_ask,
                        "status": st_str,
                        "reason": f"Edge teorico attivo (Z={cgc_res['z_score']:.2f}), ma attualmente esaurito/ask troppo alto su Cardmarket ({low_ask:,.0f} € vs target {cgc_10_px:,.0f} €).",
                        "direct_url": d_url
                    })
                else:
                    buy_signals.append(SlabSignal(
                        signal_id=f"SIG-BUY-CGC-{card_id}",
                        action=SignalAction.BUY,
                        card_id=card_id,
                        card_name=name,
                        target_company=GradingCompany.CGC,
                        target_grade=SlabGrade.CGC_10_GEM,
                        current_price_eur=cgc_10_px,
                        fair_value_eur=cgc_res["fair_value_eur"],
                        margin_of_safety_pct=cgc_res["margin_of_safety_pct"],
                        primary_edge=EdgeType.CROSS_COMPANY_DISLOCATION,
                        confidence_score=0.80,
                        reason=cgc_res["reason"],
                        availability_status=AvailabilityStatus.VERIFIED_AVAILABLE if is_avail else AvailabilityStatus.HISTORICAL_COMP,
                        active_listing_count=units,
                        lowest_active_ask_eur=low_ask,
                        cardmarket_direct_url=d_url,
                        seller_country=country,
                        metrics=cgc_res
                    ))

        # 1.5 TEST EDGE 3: CONDITION RARITY MULTIPLIER (Gem-Rate basso)
        gem_res = calc_gem_scarcity_edge(gem_rate, psa_10_px, psa_9_px, raw_px)
        if gem_res["edge_active"]:
            is_avail, st_str, low_ask, units, country, d_url = _evaluate_listing("PSA_10", psa_10_px)
            if require_verified_available and not is_avail:
                unverified_or_out_of_stock.append({
                    "card_id": card_id,
                    "name": name,
                    "grade": SlabGrade.PSA_10.value,
                    "target_price": psa_10_px,
                    "lowest_active_ask": low_ask,
                    "status": st_str,
                    "reason": f"Condition rarity attiva (Gem-rate {gem_rate*100:.1f}%), ma attualmente esaurito sul book a {psa_10_px:,.0f} €.",
                    "direct_url": d_url
                })
            else:
                buy_signals.append(SlabSignal(
                    signal_id=f"SIG-BUY-GEM-{card_id}",
                    action=SignalAction.BUY,
                    card_id=card_id,
                    card_name=name,
                    target_company=GradingCompany.PSA,
                    target_grade=SlabGrade.PSA_10,
                    current_price_eur=psa_10_px,
                    fair_value_eur=gem_res["fair_psa_10_price"],
                    margin_of_safety_pct=gem_res["margin_of_safety_pct"],
                    primary_edge=EdgeType.GEM_RATE_SCARCITY,
                    confidence_score=0.88,
                    reason=gem_res["reason"],
                    availability_status=AvailabilityStatus.VERIFIED_AVAILABLE if is_avail else AvailabilityStatus.HISTORICAL_COMP,
                    active_listing_count=units,
                    lowest_active_ask_eur=low_ask,
                    cardmarket_direct_url=d_url,
                    seller_country=country,
                    metrics=gem_res
                ))

        # 1.6 TEST EDGE 5: GEO-DISLOCATION (se presente comp USA)
        us_comp = us_auction_comps_usd.get(card_id)
        if us_comp and psa_10_px > 0:
            geo_res = calc_geo_dislocation(us_comp, psa_10_px)
            if geo_res["edge_active"]:
                is_avail, st_str, low_ask, units, country, d_url = _evaluate_listing("PSA_10", psa_10_px)
                if not (require_verified_available and not is_avail):
                    buy_signals.append(SlabSignal(
                        signal_id=f"SIG-BUY-GEO-{card_id}",
                        action=SignalAction.BUY,
                        card_id=card_id,
                        card_name=name,
                        target_company=GradingCompany.PSA,
                        target_grade=SlabGrade.PSA_10,
                        current_price_eur=psa_10_px,
                        fair_value_eur=geo_res["us_comp_eur"],
                        margin_of_safety_pct=geo_res["net_spread_pct"],
                        primary_edge=EdgeType.GEO_DISLOCATION,
                        confidence_score=0.78,
                        reason=geo_res["reason"],
                        availability_status=AvailabilityStatus.VERIFIED_AVAILABLE if is_avail else AvailabilityStatus.HISTORICAL_COMP,
                        active_listing_count=units,
                        lowest_active_ask_eur=low_ask,
                        cardmarket_direct_url=d_url,
                        seller_country=country,
                        metrics=geo_res
                    ))

    # =========================================================================
    # 2. SCANSIONE POSIZIONI POSSEDUTE (SELL & TAKE-PROFIT)
    # =========================================================================
    for holding in user_holdings:
        card_data = universe.get(holding.card_id)
        if not card_data:
            continue

        pop_growth = float(card_data.get("pop_growth_30d_pct", 0.0))
        pop_acc = float(card_data.get("pop_acceleration_pct", 0.0))

        # 2.1 TEST EDGE 6: ALLERTA DILUIZIONE POP (Uscita Preventiva)
        dilution_res = calc_pop_dilution_wave_sell(pop_growth, pop_acc)
        if dilution_res["edge_active"]:
            sell_signals.append(SlabSignal(
                signal_id=f"SIG-SELL-DIL-{holding.holding_id}",
                action=SignalAction.SELL,
                card_id=holding.card_id,
                card_name=holding.card_name,
                target_company=holding.company,
                target_grade=holding.grade,
                current_price_eur=holding.current_price_eur,
                fair_value_eur=holding.current_price_eur,
                margin_of_safety_pct=0.0,
                primary_edge=EdgeType.POP_DILUTION_WAVE,
                confidence_score=0.90,
                reason=dilution_res["reason"],
                metrics=dilution_res
            ))

        # 2.2 TEST EDGE 7: TAKE-PROFIT A 2 TRANCHE
        take_px_res = calc_parabolic_exhaustion_sell(
            current_price=holding.current_price_eur,
            buy_price=holding.buy_price_eur,
            tranche_1_already_sold=holding.tranche_1_sold
        )
        if take_px_res["edge_active"]:
            sell_signals.append(SlabSignal(
                signal_id=f"SIG-SELL-TP-{holding.holding_id}",
                action=SignalAction.SELL,
                card_id=holding.card_id,
                card_name=holding.card_name,
                target_company=holding.company,
                target_grade=holding.grade,
                current_price_eur=holding.current_price_eur,
                fair_value_eur=holding.current_price_eur,
                margin_of_safety_pct=0.0,
                primary_edge=EdgeType.PARABOLIC_EXHAUSTION,
                confidence_score=0.92,
                reason=take_px_res["reason"],
                metrics=take_px_res
            ))

    # =========================================================================
    # 3. SCANSIONE ROTAZIONI DEL CAPITALE (ROTATE)
    # =========================================================================
    if buy_signals and user_holdings:
        # Trova il miglior segnale BUY disponibile (più alto margine di sicurezza)
        best_target = max(buy_signals, key=lambda s: s.margin_of_safety_pct)
        target_expected_roi = best_target.margin_of_safety_pct + 15.0  # Stima di riallineamento a 12 mesi

        for holding in user_holdings:
            # Stima della stagnazione della posizione attuale
            holding_stagnant_months = 7  # Default per posizioni storiche
            holding_expected_roi = 4.0   # Rendimento laterale atteso

            rot_res = calc_opportunity_cost_rotation(
                holding_expected_cagr_pct=holding_expected_roi,
                target_expected_cagr_pct=target_expected_roi,
                holding_months_stagnant=holding_stagnant_months
            )

            if rot_res["should_rotate"]:
                rotations.append(RotationRecommendation(
                    holding_card_id=holding.card_id,
                    holding_card_name=holding.card_name,
                    holding_grade=holding.grade,
                    holding_current_price=holding.current_price_eur,
                    holding_unrealized_roi_pct=holding.unrealized_roi_pct,
                    target_card_id=best_target.card_id,
                    target_card_name=best_target.card_name,
                    target_grade=best_target.target_grade,
                    target_current_price=best_target.current_price_eur,
                    target_fair_value=best_target.fair_value_eur,
                    target_margin_of_safety_pct=best_target.margin_of_safety_pct,
                    net_alpha_differential_pct=rot_res["net_alpha_differential_pct"],
                    rationale=rot_res["reason"]
                ))

    return {
        "timestamp": datetime.date.today().isoformat(),
        "total_universe_scanned": len(universe),
        "require_verified_available": require_verified_available,
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "rotation_signals": rotations,
        "rejected_controls": rejected_controls,
        "unverified_or_out_of_stock": unverified_or_out_of_stock,
        "all_evaluated": all_evaluated
    }
