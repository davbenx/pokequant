"""
poke_quant/slabs/edge_calculator.py — I 10 Edge Matematici per Carte Gradate (PSA, BGS, CGC).
Implementa le formule quantitative per:
  - 5 Edge di Acquisto (BUY)
  - 3 Edge di Vendita / Presa di Beneficio (SELL)
  - 2 Regole di Rotazione del Capitale (ROTATE)
"""

from __future__ import annotations
import math
from typing import Dict, Any, Optional
from poke_quant.slabs.models import GradingCompany, SlabGrade, EdgeType, Subgrades


# ATTENZIONE: NON CALIBRATI su dati reali, nonostante il commento originale dicesse
# il contrario. Verificato durante l'audit Fase 0: non esiste in questo repo (né in
# data_cache/) alcun dataset o script che calcoli questi numeri da aste/Cardmarket
# reali. Sono priors plausibili scritti a mano, non misurati. Tutti gli Edge 1, 3 e 8
# di questo modulo (che li usano) vanno trattati come SOLO SEGNALE ESPLORATIVO finché
# non vengono ricalibrati su comp di vendita reali (population report PSA + prezzi
# slab reali — bloccato oggi da Cloudflare su psacard.com, vedi discussione sessione).
HISTORICAL_GRADE_RATIOS: Dict[str, Dict[str, float]] = {
    "BGS_9_5_GEM": {"mean": 0.78, "std": 0.07, "crossover_prob": 0.80},
    "BGS_10_PRISTINE": {"mean": 2.10, "std": 0.25, "crossover_prob": 1.00},
    "BGS_10_BLACK_LABEL": {"mean": 4.50, "std": 0.80, "crossover_prob": 1.00},
    "CGC_10_GEM": {"mean": 0.82, "std": 0.06, "crossover_prob": 0.82},
    "CGC_10_PRISTINE": {"mean": 1.35, "std": 0.12, "crossover_prob": 0.95},
    "PSA_9": {"mean": 0.35, "std": 0.08, "crossover_prob": 0.00},
    "BGS_9": {"mean": 0.30, "std": 0.06, "crossover_prob": 0.00},
    "CGC_9": {"mean": 0.28, "std": 0.06, "crossover_prob": 0.00},
}


# =====================================================================
# 1. BUY EDGES (1 - 5)
# =====================================================================

def calc_cross_grading_spread(
    psa_10_price: float,
    target_price: float,
    company: GradingCompany,
    grade: SlabGrade,
    subgrades: Optional[Subgrades] = None
) -> Dict[str, Any]:
    """
    Edge 1: Cross-Company Grade Dislocation (Spread Z-Score).
    Valuta se una lastra BGS o CGC è scambiata a un forte sconto statistico
    rispetto al benchmark PSA 10 della stessa carta.
    """
    if psa_10_price <= 0 or target_price <= 0:
        return {"edge_active": False, "z_score": 0.0, "reason": "Prezzi non validi"}

    # Mappatura della chiave
    key = None
    if company == GradingCompany.BGS:
        if grade == SlabGrade.BGS_9_5_GEM:
            key = "BGS_9_5_GEM"
        elif grade == SlabGrade.BGS_10_PRISTINE:
            key = "BGS_10_PRISTINE"
        elif grade == SlabGrade.BGS_10_BLACK_LABEL:
            key = "BGS_10_BLACK_LABEL"
        elif grade == SlabGrade.BGS_9:
            key = "BGS_9"
    elif company == GradingCompany.CGC:
        if grade == SlabGrade.CGC_10_GEM:
            key = "CGC_10_GEM"
        elif grade == SlabGrade.CGC_10_PRISTINE:
            key = "CGC_10_PRISTINE"
        elif grade == SlabGrade.CGC_9:
            key = "CGC_9"

    if not key or key not in HISTORICAL_GRADE_RATIOS:
        return {"edge_active": False, "z_score": 0.0, "reason": f"Nessun benchmark storico per {company} {grade}"}

    bench = HISTORICAL_GRADE_RATIOS[key]
    mean_ratio = bench["mean"]
    std_ratio = bench["std"]
    crossover_prob = bench["crossover_prob"]

    # Bonus probabilità se BGS ha subgrades eccezionali (True Gem+ o Quad 9.5)
    if subgrades:
        if subgrades.is_true_gem_plus:
            crossover_prob = min(0.95, crossover_prob + 0.10)
            mean_ratio = min(0.90, mean_ratio + 0.05)
        elif subgrades.is_quad_9_5:
            crossover_prob = min(0.90, crossover_prob + 0.05)

    observed_ratio = target_price / psa_10_price
    z_score = (observed_ratio - mean_ratio) / std_ratio if std_ratio > 0 else 0.0
    fair_value = round(psa_10_price * mean_ratio, 2)
    margin_of_safety_pct = round(((fair_value - target_price) / fair_value) * 100.0, 1) if fair_value > 0 else 0.0

    # Condizione di BUY: Z-Score < -2.0 (dislocazione a 2 deviazioni standard)
    # oppure margine di sicurezza >= 22% con probabilità di crossover >= 75%
    edge_active = (z_score <= -2.0) or (margin_of_safety_pct >= 22.0 and crossover_prob >= 0.75)

    return {
        "edge_active": edge_active,
        "edge_type": EdgeType.CROSS_COMPANY_DISLOCATION,
        "observed_ratio": round(observed_ratio, 3),
        "benchmark_mean_ratio": mean_ratio,
        "z_score": round(z_score, 2),
        "fair_value_eur": fair_value,
        "current_price_eur": target_price,
        "margin_of_safety_pct": margin_of_safety_pct,
        "crossover_probability": crossover_prob,
        "reason": (
            f"{company.value} {grade.value} scambiato a ratio {observed_ratio:.2f} vs PSA 10 "
            f"(storico: {mean_ratio:.2f}, Z={z_score:.2f}). Sconto del {margin_of_safety_pct}% vs Fair Value."
        )
    }


def calc_pop_saturation_edge(
    pop_growth_30d_pct: float,
    pop_acceleration_pct: float,
    gem_rate: float
) -> Dict[str, Any]:
    """
    Edge 2: Pop Report Saturation Plateau (Supply Exhaustion).
    Blocca gli acquisti se la diluizione è in corso (V_pop > 3.0%/mese).
    Segnala un edge BUY quando la curva di sottomissione si appiattisce
    (V_pop <= 0.8%/mese e accelerazione <= 0).
    """
    is_blocked = pop_growth_30d_pct > 3.0
    is_plateau = (pop_growth_30d_pct <= 0.8) and (pop_acceleration_pct <= 0.0)

    confidence = 0.0
    if is_plateau:
        # Più basso è il Gem-rate, più forte è l'esaurimento della supply mint
        confidence = min(0.95, 0.70 + (0.30 * max(0.0, 1.0 - gem_rate)))

    return {
        "edge_active": is_plateau and not is_blocked,
        "is_dilution_blocked": is_blocked,
        "edge_type": EdgeType.POP_SATURATION_PLATEAU,
        "pop_growth_30d_pct": pop_growth_30d_pct,
        "pop_acceleration_pct": pop_acceleration_pct,
        "confidence": round(confidence, 2),
        "reason": (
            "DILUIZIONE ATTIVA: sottomissioni a PSA/BGS in rapida crescita (>3%/mese). Acquisto bloccato."
            if is_blocked else
            (
                f"SATURAZIONE SUPPLY RAGGIUNTA: crescita pop al {pop_growth_30d_pct:.1f}%/mese (curva piatta). "
                "Nuova offerta primaria esaurita."
                if is_plateau else "Dinamica pop neutrale."
            )
        )
    }


def calc_gem_scarcity_edge(
    gem_rate: float,
    psa_10_price: float,
    psa_9_price: float,
    raw_price: float = 0.0
) -> Dict[str, Any]:
    """
    Edge 3: Condition Rarity Multiplier (Gem-MT Rate Scarcity).
    Se una carta ha un Gem-rate basso (<25%), il moltiplicatore PSA 10 / PSA 9
    teorico M_fair = (1 / G_rate)^0.65 deve remunerare la rarità.
    Se M_mkt < 0.70 * M_fair, la PSA 10 è gravemente sottoprezzata.
    """
    if gem_rate <= 0.0 or psa_9_price <= 0.0 or psa_10_price <= 0.0:
        return {"edge_active": False, "reason": "Dati insufficienti"}

    gem_rate = min(1.0, max(0.01, gem_rate))
    fair_multiplier = round((1.0 / gem_rate) ** 0.65, 2)
    observed_multiplier = round(psa_10_price / psa_9_price, 2)

    fair_psa_10_price = round(psa_9_price * fair_multiplier, 2)
    margin_of_safety_pct = round(((fair_psa_10_price - psa_10_price) / fair_psa_10_price) * 100.0, 1)

    # Condizione: Gem-rate basso (< 30%) e moltiplicatore osservato fortemente a sconto
    edge_active = (gem_rate <= 0.30) and (observed_multiplier < 0.70 * fair_multiplier) and (margin_of_safety_pct >= 20.0)

    return {
        "edge_active": edge_active,
        "edge_type": EdgeType.GEM_RATE_SCARCITY,
        "gem_rate": round(gem_rate, 3),
        "observed_multiplier": observed_multiplier,
        "fair_multiplier": fair_multiplier,
        "fair_psa_10_price": fair_psa_10_price,
        "current_psa_10_price": psa_10_price,
        "margin_of_safety_pct": margin_of_safety_pct,
        "reason": (
            f"CONDITION RARITY EDGE: Gem-Rate solo {gem_rate*100:.1f}%. "
            f"Moltiplicatore attuale 10/9 pari a {observed_multiplier:.1f}x vs {fair_multiplier:.1f}x equo. "
            f"Sconto rarità: {margin_of_safety_pct}%."
        )
    }


def calc_manufacturing_cost_floor(
    raw_price: float,
    gem_rate: float,
    grade_company: GradingCompany = GradingCompany.PSA,
    grading_fee: float = 25.0,
    shipping_customs: float = 15.0,
    psa_9_price: Optional[float] = None,
    listing_price: float = 0.0
) -> Dict[str, Any]:
    """
    Edge 4: Cost-to-Grade Manufacturing Floor (Distress Arbitrage).
    Calcola il costo atteso industriale E[C_10] per produrre una PSA 10:
      E[C_10] = (P_raw + C_grade + C_ship - (1 - G)*P_9) / G
    Se il listing price di mercato è <= 0.90 * E[C_10], c'è una svendita
    sotto il costo di produzione.
    """
    if raw_price <= 0 or gem_rate <= 0:
        return {"edge_active": False, "reason": "Dati raw non validi"}

    gem_rate = min(1.0, max(0.05, gem_rate))
    if psa_9_price is None or psa_9_price <= 0:
        psa_9_price = max(raw_price * 0.85, 20.0)

    unit_production_cost = raw_price + grading_fee + shipping_customs
    # Valore atteso del recupero della copia 9 in caso di fallimento del 10
    recovery_if_fail = (1.0 - gem_rate) * psa_9_price
    cost_floor_10 = (unit_production_cost - recovery_if_fail) / gem_rate
    cost_floor_10 = max(cost_floor_10, unit_production_cost)  # Floor non può essere inferiore al raw+grading
    cost_floor_10 = round(cost_floor_10, 2)

    edge_active = False
    margin_pct = 0.0
    if listing_price > 0:
        margin_pct = round(((cost_floor_10 - listing_price) / cost_floor_10) * 100.0, 1)
        edge_active = (listing_price <= 0.90 * cost_floor_10) and (margin_pct >= 10.0)

    return {
        "edge_active": edge_active,
        "edge_type": EdgeType.MANUFACTURING_COST_FLOOR,
        "raw_price": raw_price,
        "gem_rate": gem_rate,
        "cost_floor_eur": cost_floor_10,
        "listing_price_eur": listing_price,
        "margin_of_safety_pct": margin_pct,
        "reason": (
            f"MANUFACTURING FLOOR ARBITRAGE: Prezzo di vendita {listing_price:.1f} € inferiore al costo "
            f"industriale di perizia E[C_10] = {cost_floor_10:.1f} € ({margin_pct}% sotto costo)."
            if edge_active else f"Prezzo sopra il floor di perizia ({cost_floor_10:.1f} €)."
        )
    }


def calc_geo_dislocation(
    us_comp_sold_usd: float,
    cardmarket_ask_eur: float,
    eur_usd: float = 1.08,
    cardmarket_fee_pct: float = 0.05,
    shipping_cost_eur: float = 12.0
) -> Dict[str, Any]:
    """
    Edge 5: Geo-Dislocation (Cardmarket vs US Comps).
    Verifica se il prezzo ask su Cardmarket presenta uno sconto > 18%
    rispetto all'equivalente in EUR dell'ultima asta registrata in USA (Goldin/Heritage/eBay US).
    """
    if us_comp_sold_usd <= 0 or cardmarket_ask_eur <= 0 or eur_usd <= 0:
        return {"edge_active": False, "reason": "Prezzi transatlantici non validi"}

    us_price_eur = round(us_comp_sold_usd / eur_usd, 2)
    # Calcolo al netto di costi di spedizione e fee Cardmarket
    net_spread_pct = round(((us_price_eur - cardmarket_ask_eur - shipping_cost_eur) / us_price_eur) * 100.0, 1)

    edge_active = net_spread_pct >= 18.0

    return {
        "edge_active": edge_active,
        "edge_type": EdgeType.GEO_DISLOCATION,
        "us_comp_usd": us_comp_sold_usd,
        "us_comp_eur": us_price_eur,
        "cardmarket_ask_eur": cardmarket_ask_eur,
        "net_spread_pct": net_spread_pct,
        "reason": (
            f"GEO-DISLOCATION: Cardmarket ask a {cardmarket_ask_eur:.1f} € vs {us_price_eur:.1f} € "
            f"comp USA ({us_comp_sold_usd:.0f} $). Spread netto del {net_spread_pct}%."
        )
    }


# =====================================================================
# 2. SELL & TAKE-PROFIT EDGES (6 - 8)
# =====================================================================

def calc_pop_dilution_wave_sell(
    pop_growth_30d_pct: float,
    pop_acceleration_pct: float
) -> Dict[str, Any]:
    """
    Edge 6: Pop Dilution Wave (Pre-emptive De-risk).
    Se la crescita della pop supera il 4.5%/mese e l'accelerazione è > +50%,
    le lastre stanno inondando il mercato. Uscita preventiva per proteggere i profitti.
    """
    is_dilution_trigger = (pop_growth_30d_pct >= 4.5) and (pop_acceleration_pct >= 50.0)

    return {
        "edge_active": is_dilution_trigger,
        "edge_type": EdgeType.POP_DILUTION_WAVE,
        "pop_growth_30d_pct": pop_growth_30d_pct,
        "pop_acceleration_pct": pop_acceleration_pct,
        "recommended_action": "SELL / DE-RISK IMMEDIATO",
        "reason": (
            f"DILUTION WAVE WARNING: Aumento Pop del {pop_growth_30d_pct:.1f}%/30gg con accelerazione +{pop_acceleration_pct:.1f}%. "
            "Rischio crollo prezzi su Cardmarket/eBay entro 30 giorni."
        )
    }


def calc_parabolic_exhaustion_sell(
    current_price: float,
    buy_price: float,
    ema_180: Optional[float] = None,
    price_1y_mean: Optional[float] = None,
    price_1y_std: Optional[float] = None,
    tranche_1_already_sold: bool = False
) -> Dict[str, Any]:
    """
    Edge 7: Parabolic Euphoria Exhaustion (Take-Profit Istituzionale a 2 Tranche).
      - Tranche 1: Net ROI >= +70% o estensione > +35% sopra l'EMA 180gg -> Vendi 50%.
      - Tranche 2: Z-Score prezzo 1y >= +2.5 -> Liquidazione completa (esaurimento euforico).
    """
    if buy_price <= 0 or current_price <= 0:
        return {"edge_active": False, "tranche": None}

    net_roi_pct = ((current_price - buy_price) / buy_price) * 100.0

    # Calcolo estensione sopra EMA
    ema_distance_pct = 0.0
    if ema_180 and ema_180 > 0:
        ema_distance_pct = ((current_price - ema_180) / ema_180) * 100.0

    # Calcolo Z-score annuale
    z_score_1y = 0.0
    if price_1y_mean and price_1y_std and price_1y_std > 0:
        z_score_1y = (current_price - price_1y_mean) / price_1y_std

    # Tranche 2: Euphoria Peak (Z >= 2.5) -> Uscita totale
    if z_score_1y >= 2.5 and net_roi_pct >= 50.0:
        return {
            "edge_active": True,
            "edge_type": EdgeType.PARABOLIC_EXHAUSTION,
            "tranche": "TRANCHE_2_FULL_EXIT",
            "sell_percentage": 100.0,
            "net_roi_pct": round(net_roi_pct, 1),
            "z_score_1y": round(z_score_1y, 2),
            "reason": (
                f"TAKE-PROFIT TRANCHE 2 (Top Parabolico): Z-Score storico {z_score_1y:.2f} >= 2.5. "
                f"ROI realizzato: +{net_roi_pct:.1f}%. Liquidare il 100% della posizione."
            )
        }

    # Tranche 1: Recupero Capitale (+70% ROI o +35% sopra EMA) se non ancora venduta
    if not tranche_1_already_sold and (net_roi_pct >= 70.0 or ema_distance_pct >= 35.0):
        return {
            "edge_active": True,
            "edge_type": EdgeType.PARABOLIC_EXHAUSTION,
            "tranche": "TRANCHE_1_CAPITAL_RECOVERY",
            "sell_percentage": 50.0,
            "net_roi_pct": round(net_roi_pct, 1),
            "ema_distance_pct": round(ema_distance_pct, 1),
            "reason": (
                f"TAKE-PROFIT TRANCHE 1: Raggiunto target +{net_roi_pct:.1f}% (EMA distance: +{ema_distance_pct:.1f}%). "
                "Vendere il 50% delle copie per recuperare il capitale di carico. Rimanente a Rischio Zero."
            )
        }

    return {"edge_active": False, "tranche": None, "net_roi_pct": round(net_roi_pct, 1)}


def calc_spread_convergence_sell(
    current_ratio: float,
    target_key: str = "BGS_9_5_GEM"
) -> Dict[str, Any]:
    """
    Edge 8: Spread Convergence (Chiusura dell'Arbitraggio Cross-Grader).
    Quando il rapporto BGS/CGC vs PSA 10 ritorna nella media storica (+/- 0.5 sigma),
    l'alfa è stato estratto e la posizione va monetizzata o ruotata.
    """
    if target_key not in HISTORICAL_GRADE_RATIOS:
        return {"edge_active": False}

    bench = HISTORICAL_GRADE_RATIOS[target_key]
    mean_r = bench["mean"]
    std_r = bench["std"]

    z = (current_ratio - mean_r) / std_r if std_r > 0 else 0.0
    converged = abs(z) <= 0.50

    return {
        "edge_active": converged,
        "edge_type": EdgeType.SPREAD_CONVERGENCE,
        "current_ratio": round(current_ratio, 3),
        "mean_ratio": mean_r,
        "z_score": round(z, 2),
        "reason": (
            f"ARBITRAGGIO CHIUSO: Ratio {current_ratio:.2f} rientrato nella norma storica "
            f"({mean_r:.2f}, Z={z:.2f}). Alfa cross-grade completamente monetizzato."
            if converged else f"Ratio in espansione/dislocazione (Z={z:.2f})."
        )
    }


# =====================================================================
# 3. ROTATION EDGES (9 - 10)
# =====================================================================

def calc_opportunity_cost_rotation(
    holding_expected_cagr_pct: float,
    target_expected_cagr_pct: float,
    holding_months_stagnant: int,
    exit_friction_pct: float = 0.05,
    entry_friction_pct: float = 0.03
) -> Dict[str, Any]:
    """
    Edge 9: Opportunity Cost Rotation (Alpha Differential).
    Se la lastra in portafoglio è stagnante da >= 6 mesi (CAGR atteso <= 5%)
    e c'è un'opportunità target con Alpha differenziale netto >= 20%, scatta la rotazione.
    """
    total_friction_pct = (exit_friction_pct + entry_friction_pct) * 100.0
    net_alpha_differential_pct = round(target_expected_cagr_pct - holding_expected_cagr_pct - total_friction_pct, 1)

    is_stagnant = holding_months_stagnant >= 6 and holding_expected_cagr_pct <= 6.0
    should_rotate = is_stagnant and (net_alpha_differential_pct >= 20.0)

    return {
        "should_rotate": should_rotate,
        "edge_type": EdgeType.OPPORTUNITY_COST_ROTATION,
        "holding_expected_cagr_pct": holding_expected_cagr_pct,
        "target_expected_cagr_pct": target_expected_cagr_pct,
        "net_alpha_differential_pct": net_alpha_differential_pct,
        "holding_months_stagnant": holding_months_stagnant,
        "reason": (
            f"ROTAZIONE CONSIGLIATA: La posizione attuale è ferma da {holding_months_stagnant} mesi "
            f"(rendimento atteso {holding_expected_cagr_pct}%). Ruotando sul target, "
            f"l'Alpha netto differenziale è +{net_alpha_differential_pct}% (al netto di fee)."
            if should_rotate else "Nessuna rotazione conveniente."
        )
    }


def calc_era_cycle_rotation(
    modern_grail_psa_10_price: float,
    vintage_grail_psa_9_or_10_price: float,
    era_ratio_95th_percentile: float = 1.30
) -> Dict[str, Any]:
    """
    Edge 10: Era Cycle Rotation (Modern to Vintage).
    Se il prezzo di un modern grail PSA 10 supera il prezzo di un pezzo vintage
    a fornitura fissa (EraRatio > soglia 95° percentile), scatta la raccomandazione
    di prendere profitto sul moderno e ruotare su vintage solido.
    """
    if modern_grail_psa_10_price <= 0 or vintage_grail_psa_9_or_10_price <= 0:
        return {"should_rotate": False}

    era_ratio = modern_grail_psa_10_price / vintage_grail_psa_9_or_10_price
    should_rotate = era_ratio >= era_ratio_95th_percentile

    return {
        "should_rotate": should_rotate,
        "edge_type": EdgeType.ERA_CYCLE_ROTATION,
        "era_ratio": round(era_ratio, 2),
        "threshold": era_ratio_95th_percentile,
        "reason": (
            f"ROTAZIONE EPOCALE (Modern -> Vintage): Il Modern Grail ({modern_grail_psa_10_price:.0f} €) "
            f"vale {era_ratio:.2f}x il Vintage Grail ({vintage_grail_psa_9_or_10_price:.0f} €). "
            "Prendere profitto sul Moderno ed allocare su asset con Pop Report fisso."
            if should_rotate else f"Rapporto Modern/Vintage nella norma ({era_ratio:.2f}x)."
        )
    }
