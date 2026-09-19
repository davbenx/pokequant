"""
poke_quant/engine/friction.py — Modello analitico delle frizioni del mercato collezionabili.
Calcola commissioni di piattaforma, costi di spedizione/imballaggio, e valore atteso di grading.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from poke_quant.config import PLATFORM_FEES, SHIPPING_COSTS, GRADING_DEFAULT, GradingConfig


@dataclass
class SaleFrictionResult:
    gross_price: float
    platform: str
    platform_fee: float
    shipping_and_packaging: float
    net_proceeds: float
    effective_friction_pct: float


def calculate_sale_friction(
    gross_price: float,
    item_type: str = "single",  # "single" oppure "sealed"
    platform: str = "cardmarket",
    seller_absorbs_shipping: bool = False,
    packaging_cost: float = 0.60
) -> SaleFrictionResult:
    """
    Calcola l'importo netto incassato dalla vendita di un articolo tenendo conto
    di commissioni di piattaforma e spese a carico del venditore.
    """
    if gross_price <= 0:
        return SaleFrictionResult(0.0, platform, 0.0, 0.0, 0.0, 0.0)

    fee_info = PLATFORM_FEES.get(platform, PLATFORM_FEES["cardmarket"])
    platform_fee = gross_price * fee_info["percentage"] + fee_info["fixed_fee"]

    # Spese vive di spedizione/imballaggio se assorbite dal venditore
    shipping_cost = 0.0
    if seller_absorbs_shipping:
        if item_type == "sealed":
            shipping_cost = SHIPPING_COSTS["sealed_box"]
        else:
            shipping_cost = SHIPPING_COSTS["single_tracked"]

    total_shipping_pack = shipping_cost + packaging_cost
    net_proceeds = gross_price - platform_fee - total_shipping_pack
    effective_friction_pct = (gross_price - net_proceeds) / gross_price

    return SaleFrictionResult(
        gross_price=gross_price,
        platform=platform,
        platform_fee=platform_fee,
        shipping_and_packaging=total_shipping_pack,
        net_proceeds=net_proceeds,
        effective_friction_pct=effective_friction_pct
    )


@dataclass
class GradingArbitrageOpportunity:
    raw_price: float
    psa10_price: float
    psa9_price: float
    psa8_or_less_price: float
    gem_rate: float
    grading_cost: float
    expected_graded_gross: float
    expected_graded_net: float
    expected_net_profit: float
    expected_net_roi: float
    is_favorable: bool


def evaluate_grading_arbitrage(
    raw_price: float,
    psa10_price: float,
    psa9_price: Optional[float] = None,
    gem_rate: Optional[float] = None,
    item_era: str = "modern",
    platform: str = "cardmarket",
    config: GradingConfig = GRADING_DEFAULT
) -> GradingArbitrageOpportunity:
    """
    Valuta la convenienza matematica (valore atteso) di mandare a gradare una carta raw
    acquistata a raw_price, considerando i costi fissi e le probabilità di voto (PSA 10 vs 9 vs <=8).
    """
    if gem_rate is None:
        gem_rate = config.default_modern_gem_rate if item_era == "modern" else config.default_vintage_gem_rate

    # Se non specificato, stimiamo PSA 9 come ~75% del prezzo raw per modern (o ~raw_price per vintage)
    if psa9_price is None:
        psa9_price = max(raw_price * 0.85, 10.0)

    # Grado 8 o inferiore copre il residuo (tipicamente venduto a sconto pesante sul raw)
    psa8_price = max(raw_price * 0.50, 5.0)

    prob_10 = gem_rate
    prob_9 = (1.0 - gem_rate) * 0.80
    prob_rest = max(0.0, 1.0 - prob_10 - prob_9)

    expected_gross = (prob_10 * psa10_price) + (prob_9 * psa9_price) + (prob_rest * psa8_price)

    # Calcolo ricavo netto dopo vendita su piattaforma
    friction = calculate_sale_friction(expected_gross, item_type="single", platform=platform)
    expected_net = friction.net_proceeds

    total_cost_basis = raw_price + config.fee_per_card_eur
    expected_net_profit = expected_net - total_cost_basis
    expected_net_roi = expected_net_profit / total_cost_basis if total_cost_basis > 0 else 0.0

    # Consideriamo l'opportunità favorevole solo se l'atteso supera un hurdle rate minimo (es. +25% netto per compensare il rischio di grade e il fermo di 2 mesi)
    is_favorable = expected_net_roi >= 0.25

    return GradingArbitrageOpportunity(
        raw_price=raw_price,
        psa10_price=psa10_price,
        psa9_price=psa9_price,
        psa8_or_less_price=psa8_price,
        gem_rate=gem_rate,
        grading_cost=config.fee_per_card_eur,
        expected_graded_gross=expected_gross,
        expected_graded_net=expected_net,
        expected_net_profit=expected_net_profit,
        expected_net_roi=expected_net_roi,
        is_favorable=is_favorable
    )
