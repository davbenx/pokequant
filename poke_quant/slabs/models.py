"""
poke_quant/slabs/models.py — Modelli di dati per carte gradate (PSA, BGS, CGC).
Definisce le aziende di gradazione, gradi standardizzati, quote di mercato,
segnali quantitativi (BUY, SELL, ROTATE) e posizioni di portafoglio.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
import datetime


class GradingCompany(str, Enum):
    PSA = "PSA"
    BGS = "BGS"       # Beckett Grading Services
    CGC = "CGC"       # Certified Guaranty Company


class SlabGrade(str, Enum):
    # Tier Massimi / Pristine
    BGS_10_BLACK_LABEL = "BGS 10 Black Label"
    BGS_10_PRISTINE = "BGS 10 Pristine"
    CGC_10_PRISTINE = "CGC Pristine 10"
    
    # Tier Gem Mint (Standard d'investimento)
    PSA_10 = "PSA 10 Gem Mint"
    BGS_9_5_GEM = "BGS 9.5 Gem Mint"
    CGC_10_GEM = "CGC 10 Gem Mint"
    
    # Tier Mint (Spesso trappola su moderne, ma benchmark per Vintage)
    PSA_9 = "PSA 9 Mint"
    BGS_9 = "BGS 9 Mint"
    CGC_9 = "CGC 9 Mint"
    
    # Raw (Pack Fresh non gradata)
    RAW = "Raw Mint"


class EdgeType(str, Enum):
    # BUY EDGES (1-5)
    CROSS_COMPANY_DISLOCATION = "Edge 1: Cross-Company Spread (Z-Score)"
    POP_SATURATION_PLATEAU = "Edge 2: Pop Saturation Plateau (Supply Exhaustion)"
    GEM_RATE_SCARCITY = "Edge 3: Gem-Rate Scarcity Multiplier (Condition Rarity)"
    MANUFACTURING_COST_FLOOR = "Edge 4: Cost-to-Grade Floor (Distress Arbitrage)"
    GEO_DISLOCATION = "Edge 5: Geo-Dislocation (Cardmarket vs US Comps)"
    
    # SELL EDGES (6-8)
    POP_DILUTION_WAVE = "Edge 6: Pop Dilution Wave (Pre-emptive De-risk)"
    PARABOLIC_EXHAUSTION = "Edge 7: Parabolic Euphoria Exhaustion (Take-Profit)"
    SPREAD_CONVERGENCE = "Edge 8: Spread Convergence (Arbitrage Closed)"
    
    # ROTATION EDGES (9-10)
    OPPORTUNITY_COST_ROTATION = "Edge 9: Opportunity Cost Rotation (Alpha Diff)"
    ERA_CYCLE_ROTATION = "Edge 10: Era Valuation Divergence (Modern to Vintage)"


class SignalAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    ROTATE = "ROTATE"
    HOLD = "HOLD"


class AvailabilityStatus(str, Enum):
    VERIFIED_AVAILABLE = "VERIFIED_AVAILABLE"  # Inserzione attiva verificata al prezzo dichiarato
    HISTORICAL_COMP = "HISTORICAL_COMP"        # Basato su comp d'asta (nessun ask attivo al target)
    OUT_OF_STOCK = "OUT_OF_STOCK"              # Nessun pezzo disponibile sotto il prezzo target


@dataclass
class Subgrades:
    centering: float
    corners: float
    edges: float
    surface: float

    @property
    def is_quad_9_5(self) -> bool:
        """True se tutte e 4 le subgrades sono almeno 9.5."""
        return all(g >= 9.5 for g in [self.centering, self.corners, self.edges, self.surface])

    @property
    def is_true_gem_plus(self) -> bool:
        """True se almeno una subgrade è 10 e le altre sono almeno 9.5."""
        vals = [self.centering, self.corners, self.edges, self.surface]
        return any(g >= 10.0 for g in vals) and all(g >= 9.5 for g in vals)


@dataclass
class SlabQuote:
    """Rappresenta una quotazione di mercato per una specifica lastra gradata."""
    card_id: str
    card_name: str
    company: GradingCompany
    grade: SlabGrade
    price_eur: float
    currency: str = "EUR"
    venue: str = "Cardmarket"
    pop_count: int = 0
    pop_growth_30d_pct: float = 0.0
    subgrades: Optional[Subgrades] = None
    listing_url: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.date.today().isoformat())


@dataclass
class PopReportSnapshot:
    """Snapshot storico del population report per una specifica carta."""
    card_id: str
    total_graded: int
    pop_psa_10: int
    pop_psa_9: int
    pop_bgs_9_5_plus: int
    pop_cgc_10: int
    gem_rate: float            # pop_10 / total_graded
    growth_30d_pct: float      # V_pop
    acceleration_30d_pct: float = 0.0 # Acc_pop


@dataclass
class SlabSignal:
    """Rappresenta un segnale operativo quantitativo (BUY, SELL o ROTATE)."""
    signal_id: str
    action: SignalAction
    card_id: str
    card_name: str
    target_company: GradingCompany
    target_grade: SlabGrade
    current_price_eur: float
    fair_value_eur: float
    margin_of_safety_pct: float  # (Fair - Current) / Fair * 100
    primary_edge: EdgeType
    confidence_score: float = 0.80
    reason: str = ""
    availability_status: AvailabilityStatus = AvailabilityStatus.VERIFIED_AVAILABLE
    active_listing_count: int = 1
    lowest_active_ask_eur: float = 0.0
    cardmarket_direct_url: Optional[str] = None
    seller_country: str = "EU"
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.date.today().isoformat())


@dataclass
class RotationRecommendation:
    """Raccomandazione esplicita di rotazione del capitale da un asset a un altro."""
    holding_card_id: str
    holding_card_name: str
    holding_grade: SlabGrade
    holding_current_price: float
    holding_unrealized_roi_pct: float
    target_card_id: str
    target_card_name: str
    target_grade: SlabGrade
    target_current_price: float
    target_fair_value: float
    target_margin_of_safety_pct: float
    net_alpha_differential_pct: float  # Expected gain target - Expected gain holding - fees
    rationale: str
    timestamp: str = field(default_factory=lambda: datetime.date.today().isoformat())


@dataclass
class SlabHolding:
    """Posizione posseduta in portafoglio di una specifica lastra."""
    holding_id: str
    card_id: str
    card_name: str
    company: GradingCompany
    grade: SlabGrade
    quantity: int
    buy_price_eur: float
    buy_date: str
    current_price_eur: float
    tranche_1_sold: bool = False
    notes: str = ""

    @property
    def unrealized_pnl_eur(self) -> float:
        return (self.current_price_eur - self.buy_price_eur) * self.quantity

    @property
    def unrealized_roi_pct(self) -> float:
        if self.buy_price_eur <= 0:
            return 0.0
        return ((self.current_price_eur - self.buy_price_eur) / self.buy_price_eur) * 100.0
