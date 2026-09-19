"""
poke_quant/config.py — Parametri globali di mercato e frizioni reali.
"""

from dataclasses import dataclass
from typing import Dict

# Tassi di cambio medi EUR/USD per conversione coerente
DEFAULT_EUR_USD = 1.08

# Commissioni percentuali e fisse delle piattaforme di vendita
PLATFORM_FEES: Dict[str, Dict[str, float]] = {
    "cardmarket": {
        "percentage": 0.05,        # 5% commissione base venditore Cardmarket
        "fixed_fee": 0.0,
        "payment_fee_pct": 0.0,    # a carico acquirente
    },
    "ebay": {
        "percentage": 0.125,       # 12.5% medio categoria collezionabili/carte
        "fixed_fee": 0.35,         # 0.35 EUR fisso per ordine
        "payment_fee_pct": 0.0,
    },
    "direct_private": {
        "percentage": 0.0,         # Scambio a mano / fiere / gruppi senza intermediari
        "fixed_fee": 0.0,
        "payment_fee_pct": 0.0,
    }
}

# Costi di spedizione e imballaggio tracciato/assicurato (Italia / EU)
SHIPPING_COSTS: Dict[str, float] = {
    "single_tracked": 7.00,       # Raccomandata / Corriere con tracking (toploader + cardboard)
    "sealed_box": 10.00,          # Pacco standard assicurato (bubble wrap + scatola rigida)
    "packaging_material": 0.60,   # Busta imbottita, toploader, sleeve, scatola
}

# Parametri di grading PSA (costo unitario comprensivo di spedizione round-trip e dogana)
@dataclass(frozen=True)
class GradingConfig:
    fee_per_card_eur: float = 25.00     # Costo totale all-in servizio bulk (fee + spedizione USA + IVA)
    turnaround_months: int = 2          # Mesi di fermo del capitale prima che la carta sia vendibile
    default_modern_gem_rate: float = 0.70 # Probabilità stima PSA 10 su carte modern pack-fresh
    default_vintage_gem_rate: float = 0.20 # Probabilità stima PSA 10 su carte vintage (WotC)

GRADING_DEFAULT = GradingConfig()
