"""
poke_quant/slabs/grading_multipliers.py — Modulo di calibrazione empirica
dei moltiplicatori di prezzo e sconti di liquidità per case di gradazione.

Calibrato sui risultati della ricerca empirica cross-sezionale (scripts/grading_company_multiplier_research.py).
Fornisce:
  - Rapporti empirici di prezzo vs benchmark PSA 9 (per gradi 8.5 - 9.5) e vs PSA 10 (per gradi 10).
  - Penalità prudenziale di liquidità per enti regionali (GRAAD, PCA, ACE) vs major globali (PSA, BGS, CGC).
  - Fattore di tetto massimo per lo sniper di eBay/Cardmarket.
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


class GradingCompany(str, Enum):
    PSA = "PSA"
    BGS = "BGS"       # Beckett Grading Services
    CGC = "CGC"       # Certified Guaranty Company
    SGC = "SGC"       # Sportscard Guaranty Corporation (The Tuxedo)
    TAG = "TAG"       # Technical Authentication & Grading (USA)
    GRAAD = "GRAAD"   # Gem Rate Authentication And Diagnostic (Italia)
    PCA = "PCA"       # Professional Cards Authenticator (Francia)
    CCC = "CCC"       # Classic Card Collector (Austria/Germania)
    AIGRADING = "AiGrading"  # AiGrading (Italia)
    ACE = "ACE"       # ACE Grading (UK)


class Era(str, Enum):
    VINTAGE = "vintage"    # 1999–2003 (WotC: Base Set, Jungle, Fossil, Rocket, Gym, Neo, e-Series)
    MID_ERA = "mid_era"    # 2004–2016 (EX, Diamond & Pearl, Platinum, HGSS, Black & White, XY)
    MODERN = "modern"      # 2017–2026 (Sun & Moon, Sword & Shield, Scarlet & Violet)


@dataclass(frozen=True)
class GradingAdjustment:
    company: GradingCompany
    grade_label: str
    benchmark_ref: str            # "PSA_9" o "PSA_10"
    multiplier: float             # Moltiplicatore centrale sul prezzo PSA benchmark
    liquidity_penalty_pct: float  # Sconto prudenziale di liquidità internazionale (%)
    sniper_ceiling_factor: float  # Fattore massimo consentito per preservare l'Edge nello sniper
    era: Era
    notes: str


# =============================================================================
# MOLTIPLICATORI EMPIRICI PER VARIANTI SPECIALI (1st Edition, No Symbol, Shadowless)
# =============================================================================

SPECIAL_VARIANTS: Dict[str, Tuple[float, str]] = {
    "standard": (1.00, "Versione Standard / Unlimited"),
    "first_edition_wotc": (2.50, "1ª Edizione WotC (Jungle, Fossil, Rocket, Gym, Neo: premio ~2.5x vs Unlimited)"),
    "first_edition_base": (6.00, "1ª Edizione Base Set (rarità estrema: premio ~6.0x vs Unlimited)"),
    "no_symbol": (1.40, "No Symbol Error (Jungle Holo senza simbolo fiore: premio ~1.4x vs Unlimited)"),
    "shadowless": (3.00, "Shadowless (Base Set senza ombra: premio ~3.0x vs Unlimited)"),
    "reverse_holo_lc": (3.00, "Reverse Holo Legendary Collection (Fireworks: premio ~3.0x vs Unlimited)"),
}


def normalize_variant(variant: str) -> str:
    """Normalizza la variante in una chiave canonica univoca."""
    v = str(variant).strip().lower()
    # Verifica errori specifici prima di 1st edition per evitare che '1.4x' attivi 1st edition
    if any(k in v for k in ["no_symbol", "no-symbol", "no symbol", "senza simbolo", "senza logo"]):
        return "no_symbol"
    if "shadowless" in v:
        return "shadowless"
    if any(k in v for k in ["reverse", "firework"]):
        return "reverse_holo_lc"
    # Per 1st edition, cerca specificamente '1st', '1ª', '1a' o 'first' (evitando '1' isolato o in decimali)
    if any(k in v for k in ["1st", "1ª", "1a", "first", "prima edizione"]):
        if "base" in v:
            return "first_edition_base"
        return "first_edition_wotc"
    return "standard"


def variant_to_pricecharting_key(variant: str) -> Optional[str]:
    """Converte una variante nella chiave slug usata da PriceCharting."""
    canon = normalize_variant(variant)
    if canon == "no_symbol":
        return "no-symbol"
    elif canon == "shadowless":
        return "shadowless"
    elif canon in ("first_edition_wotc", "first_edition_base"):
        return "1st-edition"
    return None


def get_variant_multiplier(variant: str, game_slug: Optional[str] = None) -> Tuple[float, str]:
    """Ritorna (moltiplicatore, descrizione) per la variante richiesta."""
    canon = normalize_variant(variant)
    if canon in ("first_edition_wotc", "first_edition_base"):
        if game_slug:
            gs = game_slug.lower()
            if "base-set" in gs or "base_set" in gs:
                return SPECIAL_VARIANTS["first_edition_base"]
            else:
                return SPECIAL_VARIANTS["first_edition_wotc"]
        return SPECIAL_VARIANTS[canon]
    return SPECIAL_VARIANTS.get(canon, SPECIAL_VARIANTS["standard"])


# =============================================================================
# MATRICE CALIBRATA DEI MOLTIPLICATORI EMPIRICI (scripts/grading_company_multiplier_research.py)
# =============================================================================

# Formato: (Company, Grade_Key, Era) -> (Multiplier vs Ref, Liquidity Penalty %, Sniper Factor)
# Dove:
#   - Se Grade_Key appartiene alla famiglia 9 (9.0, 9.5), il Benchmark di Riferimento è PSA 9 = 1.00x
#   - Se Grade_Key appartiene alla famiglia 10, il Benchmark di Riferimento è PSA 10 = 1.00x

EMPIRICAL_RATIOS_GRADE9: Dict[Tuple[GradingCompany, str, Era], Tuple[float, float, float, str]] = {
    # --- PSA (Benchmark) ---
    (GradingCompany.PSA, "9.5", Era.VINTAGE): (1.500, 0.0, 1.150, "PSA 9.5 Mint+ vintage"),
    (GradingCompany.PSA, "9.5", Era.MID_ERA): (1.400, 0.0, 1.120, "PSA 9.5 Mint+ mid-era"),
    (GradingCompany.PSA, "9.5", Era.MODERN): (1.300, 0.0, 1.100, "PSA 9.5 Mint+ moderno"),
    (GradingCompany.PSA, "9.0", Era.VINTAGE): (1.000, 0.0, 1.000, "Benchmark base del modello"),
    (GradingCompany.PSA, "9.0", Era.MID_ERA): (1.000, 0.0, 1.000, "Benchmark base del modello"),
    (GradingCompany.PSA, "9.0", Era.MODERN): (1.000, 0.0, 1.000, "Benchmark base del modello"),
    (GradingCompany.PSA, "8.5", Era.VINTAGE): (0.780, 0.0, 0.800, "PSA 8.5 NM-Mint+ vintage: ottimo compromesso collezionistico"),
    (GradingCompany.PSA, "8.5", Era.MID_ERA): (0.750, 0.0, 0.770, "PSA 8.5 NM-Mint+ mid-era"),
    (GradingCompany.PSA, "8.5", Era.MODERN): (0.550, 5.0, 0.580, "PSA 8.5 moderno: forte compressione vs Raw"),
    (GradingCompany.PSA, "8.0", Era.VINTAGE): (0.650, 0.0, 0.670, "PSA 8.0 NM-Mint vintage: soglia d'ingresso accessibile e solida"),
    (GradingCompany.PSA, "8.0", Era.MID_ERA): (0.600, 0.0, 0.620, "PSA 8.0 NM-Mint mid-era"),
    (GradingCompany.PSA, "8.0", Era.MODERN): (0.420, 10.0, 0.440, "PSA 8.0 moderno: liquidità compressa"),
    (GradingCompany.PSA, "7.5", Era.VINTAGE): (0.550, 0.0, 0.570, "PSA 7.5 Near Mint+ vintage"),
    (GradingCompany.PSA, "7.5", Era.MID_ERA): (0.480, 0.0, 0.500, "PSA 7.5 Near Mint+ mid-era"),
    (GradingCompany.PSA, "7.5", Era.MODERN): (0.330, 15.0, 0.350, "PSA 7.5 moderno: sconsigliato investimento"),
    (GradingCompany.PSA, "7.0", Era.VINTAGE): (0.480, 0.0, 0.500, "PSA 7.0 Near Mint vintage: minimo collezionistico consigliato WotC"),
    (GradingCompany.PSA, "7.0", Era.MID_ERA): (0.400, 0.0, 0.420, "PSA 7.0 Near Mint mid-era"),
    (GradingCompany.PSA, "7.0", Era.MODERN): (0.250, 20.0, 0.270, "PSA 7.0 moderno: valore depresso rispetto a Raw"),

    # --- BGS (Beckett) ---
    (GradingCompany.BGS, "9.5", Era.VINTAGE): (1.809, 0.0, 1.250, "BGS 9.5 Gem Mint vintage: premio forte vs PSA 9"),
    (GradingCompany.BGS, "9.5", Era.MID_ERA): (1.552, 0.0, 1.200, "BGS 9.5 Gem Mint mid-era: premio strutturale"),
    (GradingCompany.BGS, "9.5", Era.MODERN): (1.611, 0.0, 1.150, "BGS 9.5 Gem Mint moderno: posizionato a 0.65x di PSA 10"),
    (GradingCompany.BGS, "9.0", Era.VINTAGE): (0.884, 5.0, 0.900, "BGS 9.0 Mint vintage: lieve sconto di liquidità vs PSA 9"),
    (GradingCompany.BGS, "9.0", Era.MID_ERA): (0.887, 5.0, 0.900, "BGS 9.0 Mint mid-era"),
    (GradingCompany.BGS, "9.0", Era.MODERN): (0.903, 3.0, 0.920, "BGS 9.0 Mint moderno"),
    (GradingCompany.BGS, "8.5", Era.VINTAGE): (0.718, 6.0, 0.730, "BGS 8.5 NM-Mint+ vintage: subgrades rispettati"),
    (GradingCompany.BGS, "8.5", Era.MID_ERA): (0.690, 6.0, 0.700, "BGS 8.5 NM-Mint+ mid-era"),
    (GradingCompany.BGS, "8.5", Era.MODERN): (0.506, 5.0, 0.520, "BGS 8.5 moderno"),
    (GradingCompany.BGS, "8.0", Era.VINTAGE): (0.585, 7.0, 0.600, "BGS 8.0 NM-Mint vintage"),
    (GradingCompany.BGS, "8.0", Era.MID_ERA): (0.540, 7.0, 0.550, "BGS 8.0 NM-Mint mid-era"),
    (GradingCompany.BGS, "8.0", Era.MODERN): (0.378, 6.0, 0.390, "BGS 8.0 moderno"),
    (GradingCompany.BGS, "7.5", Era.VINTAGE): (0.484, 8.0, 0.500, "BGS 7.5 Near Mint+ vintage"),
    (GradingCompany.BGS, "7.5", Era.MID_ERA): (0.422, 8.0, 0.430, "BGS 7.5 Near Mint+ mid-era"),
    (GradingCompany.BGS, "7.5", Era.MODERN): (0.290, 8.0, 0.300, "BGS 7.5 moderno"),
    (GradingCompany.BGS, "7.0", Era.VINTAGE): (0.422, 9.0, 0.430, "BGS 7.0 Near Mint vintage"),
    (GradingCompany.BGS, "7.0", Era.MID_ERA): (0.352, 9.0, 0.360, "BGS 7.0 Near Mint mid-era"),
    (GradingCompany.BGS, "7.0", Era.MODERN): (0.220, 10.0, 0.230, "BGS 7.0 moderno"),

    # --- CGC Cards ---
    (GradingCompany.CGC, "9.5", Era.VINTAGE): (1.350, 0.0, 1.150, "CGC 9.5 Gem Mint vintage"),
    (GradingCompany.CGC, "9.5", Era.MID_ERA): (1.250, 0.0, 1.120, "CGC 9.5 Gem Mint mid-era"),
    (GradingCompany.CGC, "9.5", Era.MODERN): (1.180, 0.0, 1.100, "CGC 9.5 Gem Mint moderno"),
    (GradingCompany.CGC, "9.0", Era.VINTAGE): (0.901, 7.0, 0.920, "CGC 9.0 Mint vintage: sconto liquidità medio 8-10%"),
    (GradingCompany.CGC, "9.0", Era.MID_ERA): (0.913, 6.0, 0.930, "CGC 9.0 Mint mid-era"),
    (GradingCompany.CGC, "9.0", Era.MODERN): (0.931, 5.0, 0.940, "CGC 9.0 Mint moderno: molto vicino a PSA 9"),
    (GradingCompany.CGC, "8.5", Era.VINTAGE): (0.702, 8.0, 0.710, "CGC 8.5 NM-Mint+ vintage"),
    (GradingCompany.CGC, "8.5", Era.MID_ERA): (0.675, 7.0, 0.680, "CGC 8.5 NM-Mint+ mid-era"),
    (GradingCompany.CGC, "8.5", Era.MODERN): (0.506, 7.0, 0.510, "CGC 8.5 moderno"),
    (GradingCompany.CGC, "8.0", Era.VINTAGE): (0.572, 9.0, 0.580, "CGC 8.0 NM-Mint vintage"),
    (GradingCompany.CGC, "8.0", Era.MID_ERA): (0.528, 8.0, 0.530, "CGC 8.0 NM-Mint mid-era"),
    (GradingCompany.CGC, "8.0", Era.MODERN): (0.370, 8.0, 0.380, "CGC 8.0 moderno"),
    (GradingCompany.CGC, "7.5", Era.VINTAGE): (0.468, 10.0, 0.470, "CGC 7.5 Near Mint+ vintage"),
    (GradingCompany.CGC, "7.5", Era.MID_ERA): (0.408, 10.0, 0.410, "CGC 7.5 Near Mint+ mid-era"),
    (GradingCompany.CGC, "7.5", Era.MODERN): (0.281, 10.0, 0.290, "CGC 7.5 moderno"),
    (GradingCompany.CGC, "7.0", Era.VINTAGE): (0.408, 11.0, 0.410, "CGC 7.0 Near Mint vintage"),
    (GradingCompany.CGC, "7.0", Era.MID_ERA): (0.340, 11.0, 0.350, "CGC 7.0 Near Mint mid-era"),
    (GradingCompany.CGC, "7.0", Era.MODERN): (0.213, 12.0, 0.220, "CGC 7.0 moderno"),

    # --- TAG Grading (USA, Digital 1000 DPI) ---
    (GradingCompany.TAG, "9.0", Era.VINTAGE): (0.800, 15.0, 0.830, "TAG 9.0 vintage: diffidenza collezionisti storici"),
    (GradingCompany.TAG, "9.0", Era.MID_ERA): (0.840, 12.0, 0.860, "TAG 9.0 mid-era: community emergente"),
    (GradingCompany.TAG, "9.0", Era.MODERN): (0.880, 8.0, 0.900, "TAG 9.0 moderno: forte interesse tech-oriented USA"),
    (GradingCompany.TAG, "8.5", Era.VINTAGE): (0.624, 18.0, 0.640, "TAG 8.5 vintage"),
    (GradingCompany.TAG, "8.5", Era.MID_ERA): (0.615, 15.0, 0.630, "TAG 8.5 mid-era"),
    (GradingCompany.TAG, "8.5", Era.MODERN): (0.473, 10.0, 0.480, "TAG 8.5 moderno"),
    (GradingCompany.TAG, "8.0", Era.VINTAGE): (0.507, 20.0, 0.520, "TAG 8.0 vintage"),
    (GradingCompany.TAG, "8.0", Era.MID_ERA): (0.480, 17.0, 0.490, "TAG 8.0 mid-era"),
    (GradingCompany.TAG, "8.0", Era.MODERN): (0.353, 12.0, 0.360, "TAG 8.0 moderno"),
    (GradingCompany.TAG, "7.5", Era.VINTAGE): (0.413, 22.0, 0.420, "TAG 7.5 vintage"),
    (GradingCompany.TAG, "7.5", Era.MID_ERA): (0.374, 19.0, 0.380, "TAG 7.5 mid-era"),
    (GradingCompany.TAG, "7.5", Era.MODERN): (0.271, 14.0, 0.280, "TAG 7.5 moderno"),
    (GradingCompany.TAG, "7.0", Era.VINTAGE): (0.346, 24.0, 0.350, "TAG 7.0 vintage"),
    (GradingCompany.TAG, "7.0", Era.MID_ERA): (0.300, 20.0, 0.310, "TAG 7.0 mid-era"),
    (GradingCompany.TAG, "7.0", Era.MODERN): (0.200, 16.0, 0.210, "TAG 7.0 moderno"),

    # --- SGC (Tuxedo) ---
    (GradingCompany.SGC, "9.0", Era.VINTAGE): (0.785, 15.0, 0.820, "SGC 9.0 Mint vintage: sconto estero significativo"),
    (GradingCompany.SGC, "9.0", Era.MID_ERA): (0.800, 14.0, 0.830, "SGC 9.0 Mint mid-era"),
    (GradingCompany.SGC, "9.0", Era.MODERN): (0.836, 12.0, 0.850, "SGC 9.0 Mint moderno"),
    (GradingCompany.SGC, "8.5", Era.VINTAGE): (0.640, 16.0, 0.660, "SGC 8.5 vintage"),
    (GradingCompany.SGC, "8.5", Era.MID_ERA): (0.615, 15.0, 0.630, "SGC 8.5 mid-era"),
    (GradingCompany.SGC, "8.5", Era.MODERN): (0.451, 14.0, 0.460, "SGC 8.5 moderno"),
    (GradingCompany.SGC, "8.0", Era.VINTAGE): (0.520, 18.0, 0.540, "SGC 8.0 vintage"),
    (GradingCompany.SGC, "8.0", Era.MID_ERA): (0.480, 17.0, 0.490, "SGC 8.0 mid-era"),
    (GradingCompany.SGC, "8.0", Era.MODERN): (0.336, 16.0, 0.340, "SGC 8.0 moderno"),
    (GradingCompany.SGC, "7.5", Era.VINTAGE): (0.429, 20.0, 0.440, "SGC 7.5 vintage"),
    (GradingCompany.SGC, "7.5", Era.MID_ERA): (0.374, 19.0, 0.380, "SGC 7.5 mid-era"),
    (GradingCompany.SGC, "7.5", Era.MODERN): (0.257, 18.0, 0.260, "SGC 7.5 moderno"),
    (GradingCompany.SGC, "7.0", Era.VINTAGE): (0.360, 22.0, 0.370, "SGC 7.0 vintage"),
    (GradingCompany.SGC, "7.0", Era.MID_ERA): (0.300, 20.0, 0.310, "SGC 7.0 mid-era"),
    (GradingCompany.SGC, "7.0", Era.MODERN): (0.188, 20.0, 0.190, "SGC 7.0 moderno"),

    # --- GRAAD (Italia) ---
    (GradingCompany.GRAAD, "9.0", Era.VINTAGE): (0.681, 28.0, 0.720, "GRAAD 9.0 vintage: sconto pesante fuori dall'Italia"),
    (GradingCompany.GRAAD, "9.0", Era.MID_ERA): (0.711, 25.0, 0.740, "GRAAD 9.0 mid-era"),
    (GradingCompany.GRAAD, "9.0", Era.MODERN): (0.750, 20.0, 0.770, "GRAAD 9.0 moderno"),
    (GradingCompany.GRAAD, "8.5", Era.VINTAGE): (0.510, 30.0, 0.540, "GRAAD 8.5 vintage"),
    (GradingCompany.GRAAD, "8.5", Era.MID_ERA): (0.510, 28.0, 0.530, "GRAAD 8.5 mid-era"),
    (GradingCompany.GRAAD, "8.5", Era.MODERN): (0.380, 25.0, 0.390, "GRAAD 8.5 moderno"),
    (GradingCompany.GRAAD, "8.0", Era.VINTAGE): (0.410, 32.0, 0.430, "GRAAD 8.0 vintage"),
    (GradingCompany.GRAAD, "8.0", Era.MID_ERA): (0.400, 30.0, 0.420, "GRAAD 8.0 mid-era"),
    (GradingCompany.GRAAD, "8.0", Era.MODERN): (0.280, 28.0, 0.290, "GRAAD 8.0 moderno"),
    (GradingCompany.GRAAD, "7.5", Era.VINTAGE): (0.330, 35.0, 0.350, "GRAAD 7.5 vintage"),
    (GradingCompany.GRAAD, "7.5", Era.MID_ERA): (0.300, 33.0, 0.320, "GRAAD 7.5 mid-era"),
    (GradingCompany.GRAAD, "7.5", Era.MODERN): (0.210, 30.0, 0.220, "GRAAD 7.5 moderno"),
    (GradingCompany.GRAAD, "7.0", Era.VINTAGE): (0.280, 38.0, 0.300, "GRAAD 7.0 vintage"),
    (GradingCompany.GRAAD, "7.0", Era.MID_ERA): (0.240, 35.0, 0.260, "GRAAD 7.0 mid-era"),
    (GradingCompany.GRAAD, "7.0", Era.MODERN): (0.150, 35.0, 0.160, "GRAAD 7.0 moderno"),

    # --- AiGrading (Italia) ---
    (GradingCompany.AIGRADING, "9.0", Era.VINTAGE): (0.680, 28.0, 0.710, "AiGrading 9.0 vintage"),
    (GradingCompany.AIGRADING, "9.0", Era.MID_ERA): (0.700, 25.0, 0.730, "AiGrading 9.0 mid-era"),
    (GradingCompany.AIGRADING, "9.0", Era.MODERN): (0.740, 20.0, 0.760, "AiGrading 9.0 moderno"),
    (GradingCompany.AIGRADING, "8.5", Era.VINTAGE): (0.510, 30.0, 0.540, "AiGrading 8.5 vintage"),
    (GradingCompany.AIGRADING, "8.5", Era.MID_ERA): (0.500, 28.0, 0.520, "AiGrading 8.5 mid-era"),
    (GradingCompany.AIGRADING, "8.5", Era.MODERN): (0.380, 25.0, 0.390, "AiGrading 8.5 moderno"),
    (GradingCompany.AIGRADING, "8.0", Era.VINTAGE): (0.410, 32.0, 0.430, "AiGrading 8.0 vintage"),
    (GradingCompany.AIGRADING, "8.0", Era.MID_ERA): (0.400, 30.0, 0.420, "AiGrading 8.0 mid-era"),
    (GradingCompany.AIGRADING, "8.0", Era.MODERN): (0.280, 28.0, 0.290, "AiGrading 8.0 moderno"),
    (GradingCompany.AIGRADING, "7.5", Era.VINTAGE): (0.330, 35.0, 0.350, "AiGrading 7.5 vintage"),
    (GradingCompany.AIGRADING, "7.5", Era.MID_ERA): (0.300, 33.0, 0.320, "AiGrading 7.5 mid-era"),
    (GradingCompany.AIGRADING, "7.5", Era.MODERN): (0.210, 30.0, 0.220, "AiGrading 7.5 moderno"),
    (GradingCompany.AIGRADING, "7.0", Era.VINTAGE): (0.280, 38.0, 0.300, "AiGrading 7.0 vintage"),
    (GradingCompany.AIGRADING, "7.0", Era.MID_ERA): (0.240, 35.0, 0.260, "AiGrading 7.0 mid-era"),
    (GradingCompany.AIGRADING, "7.0", Era.MODERN): (0.150, 35.0, 0.160, "AiGrading 7.0 moderno"),

    # --- PCA (Francia) ---
    (GradingCompany.PCA, "9.0", Era.VINTAGE): (0.726, 25.0, 0.740, "PCA 9.0 vintage: liquido principalmente in Francia"),
    (GradingCompany.PCA, "9.0", Era.MID_ERA): (0.737, 23.0, 0.750, "PCA 9.0 mid-era"),
    (GradingCompany.PCA, "9.0", Era.MODERN): (0.776, 18.0, 0.780, "PCA 9.0 moderno"),
    (GradingCompany.PCA, "8.5", Era.VINTAGE): (0.550, 26.0, 0.570, "PCA 8.5 vintage"),
    (GradingCompany.PCA, "8.5", Era.MID_ERA): (0.540, 24.0, 0.550, "PCA 8.5 mid-era"),
    (GradingCompany.PCA, "8.5", Era.MODERN): (0.410, 20.0, 0.420, "PCA 8.5 moderno"),
    (GradingCompany.PCA, "8.0", Era.VINTAGE): (0.440, 28.0, 0.460, "PCA 8.0 vintage"),
    (GradingCompany.PCA, "8.0", Era.MID_ERA): (0.420, 26.0, 0.440, "PCA 8.0 mid-era"),
    (GradingCompany.PCA, "8.0", Era.MODERN): (0.300, 22.0, 0.310, "PCA 8.0 moderno"),
    (GradingCompany.PCA, "7.5", Era.VINTAGE): (0.360, 30.0, 0.370, "PCA 7.5 vintage"),
    (GradingCompany.PCA, "7.5", Era.MID_ERA): (0.320, 28.0, 0.340, "PCA 7.5 mid-era"),
    (GradingCompany.PCA, "7.5", Era.MODERN): (0.230, 25.0, 0.240, "PCA 7.5 moderno"),
    (GradingCompany.PCA, "7.0", Era.VINTAGE): (0.300, 32.0, 0.320, "PCA 7.0 vintage"),
    (GradingCompany.PCA, "7.0", Era.MID_ERA): (0.260, 30.0, 0.270, "PCA 7.0 mid-era"),
    (GradingCompany.PCA, "7.0", Era.MODERN): (0.170, 28.0, 0.180, "PCA 7.0 moderno"),

    # --- CCC (Classic Card Collector, DACH) ---
    (GradingCompany.CCC, "9.0", Era.VINTAGE): (0.690, 28.0, 0.720, "CCC 9.0 vintage: ente regionale DACH/Germania"),
    (GradingCompany.CCC, "9.0", Era.MID_ERA): (0.710, 25.0, 0.740, "CCC 9.0 mid-era"),
    (GradingCompany.CCC, "9.0", Era.MODERN): (0.750, 20.0, 0.770, "CCC 9.0 moderno"),
    (GradingCompany.CCC, "8.5", Era.VINTAGE): (0.510, 30.0, 0.540, "CCC 8.5 vintage"),
    (GradingCompany.CCC, "8.5", Era.MID_ERA): (0.510, 28.0, 0.530, "CCC 8.5 mid-era"),
    (GradingCompany.CCC, "8.5", Era.MODERN): (0.380, 25.0, 0.390, "CCC 8.5 moderno"),
    (GradingCompany.CCC, "8.0", Era.VINTAGE): (0.410, 32.0, 0.430, "CCC 8.0 vintage"),
    (GradingCompany.CCC, "8.0", Era.MID_ERA): (0.400, 30.0, 0.420, "CCC 8.0 mid-era"),
    (GradingCompany.CCC, "8.0", Era.MODERN): (0.280, 28.0, 0.290, "CCC 8.0 moderno"),
    (GradingCompany.CCC, "7.5", Era.VINTAGE): (0.330, 35.0, 0.350, "CCC 7.5 vintage"),
    (GradingCompany.CCC, "7.5", Era.MID_ERA): (0.300, 33.0, 0.320, "CCC 7.5 mid-era"),
    (GradingCompany.CCC, "7.5", Era.MODERN): (0.210, 30.0, 0.220, "CCC 7.5 moderno"),
    (GradingCompany.CCC, "7.0", Era.VINTAGE): (0.280, 38.0, 0.300, "CCC 7.0 vintage"),
    (GradingCompany.CCC, "7.0", Era.MID_ERA): (0.240, 35.0, 0.260, "CCC 7.0 mid-era"),
    (GradingCompany.CCC, "7.0", Era.MODERN): (0.150, 35.0, 0.160, "CCC 7.0 moderno"),

    # --- ACE Grading (UK) ---
    (GradingCompany.ACE, "9.0", Era.VINTAGE): (0.740, 22.0, 0.760, "ACE 9.0 vintage: mercato prevalentemente UK"),
    (GradingCompany.ACE, "9.0", Era.MID_ERA): (0.760, 20.0, 0.780, "ACE 9.0 mid-era"),
    (GradingCompany.ACE, "9.0", Era.MODERN): (0.800, 16.0, 0.810, "ACE 9.0 moderno"),
    (GradingCompany.ACE, "8.5", Era.VINTAGE): (0.550, 24.0, 0.570, "ACE 8.5 vintage"),
    (GradingCompany.ACE, "8.5", Era.MID_ERA): (0.540, 22.0, 0.550, "ACE 8.5 mid-era"),
    (GradingCompany.ACE, "8.5", Era.MODERN): (0.410, 18.0, 0.420, "ACE 8.5 moderno"),
    (GradingCompany.ACE, "8.0", Era.VINTAGE): (0.440, 26.0, 0.460, "ACE 8.0 vintage"),
    (GradingCompany.ACE, "8.0", Era.MID_ERA): (0.420, 24.0, 0.440, "ACE 8.0 mid-era"),
    (GradingCompany.ACE, "8.0", Era.MODERN): (0.300, 20.0, 0.310, "ACE 8.0 moderno"),
    (GradingCompany.ACE, "7.5", Era.VINTAGE): (0.360, 28.0, 0.370, "ACE 7.5 vintage"),
    (GradingCompany.ACE, "7.5", Era.MID_ERA): (0.320, 26.0, 0.340, "ACE 7.5 mid-era"),
    (GradingCompany.ACE, "7.5", Era.MODERN): (0.230, 22.0, 0.240, "ACE 7.5 moderno"),
    (GradingCompany.ACE, "7.0", Era.VINTAGE): (0.300, 30.0, 0.320, "ACE 7.0 vintage"),
    (GradingCompany.ACE, "7.0", Era.MID_ERA): (0.260, 28.0, 0.270, "ACE 7.0 mid-era"),
    (GradingCompany.ACE, "7.0", Era.MODERN): (0.170, 25.0, 0.180, "ACE 7.0 moderno"),
}

EMPIRICAL_RATIOS_GRADE10: Dict[Tuple[GradingCompany, str, Era], Tuple[float, float, float, str]] = {
    # --- PSA 10 (Benchmark) ---
    (GradingCompany.PSA, "10.0", Era.VINTAGE): (1.000, 0.0, 1.000, "Benchmark base Grado 10"),
    (GradingCompany.PSA, "10.0", Era.MID_ERA): (1.000, 0.0, 1.000, "Benchmark base Grado 10"),
    (GradingCompany.PSA, "10.0", Era.MODERN): (1.000, 0.0, 1.000, "Benchmark base Grado 10"),

    # --- BGS 10 Pristine & Black Label ---
    (GradingCompany.BGS, "10.0_pristine", Era.VINTAGE): (1.889, 0.0, 1.600, "BGS 10 Pristine vintage: rarità estrema"),
    (GradingCompany.BGS, "10.0_pristine", Era.MID_ERA): (1.860, 0.0, 1.550, "BGS 10 Pristine mid-era"),
    (GradingCompany.BGS, "10.0_pristine", Era.MODERN): (2.100, 0.0, 1.700, "BGS 10 Pristine moderno"),
    (GradingCompany.BGS, "10.0_black_label", Era.VINTAGE): (4.500, 0.0, 3.500, "BGS Black Label Quad 10"),
    (GradingCompany.BGS, "10.0_black_label", Era.MID_ERA): (4.200, 0.0, 3.200, "BGS Black Label Quad 10"),
    (GradingCompany.BGS, "10.0_black_label", Era.MODERN): (4.800, 0.0, 3.600, "BGS Black Label Quad 10"),

    # --- CGC 10 Pristine & Gem Mint ---
    (GradingCompany.CGC, "10.0_pristine", Era.VINTAGE): (0.619, 10.0, 0.700, "CGC 10 Pristine Gold Label"),
    (GradingCompany.CGC, "10.0_pristine", Era.MID_ERA): (0.750, 8.0, 0.800, "CGC 10 Pristine Gold Label"),
    (GradingCompany.CGC, "10.0_pristine", Era.MODERN): (1.150, 0.0, 1.050, "CGC 10 Pristine moderno"),
    (GradingCompany.CGC, "10.0_gem", Era.VINTAGE): (0.428, 20.0, 0.480, "CGC 10 Gem Mint vintage sconta forte gap su PSA 10"),
    (GradingCompany.CGC, "10.0_gem", Era.MID_ERA): (0.550, 15.0, 0.600, "CGC 10 Gem Mint mid-era"),
    (GradingCompany.CGC, "10.0_gem", Era.MODERN): (0.780, 10.0, 0.820, "CGC 10 Gem Mint moderno"),

    # --- TAG 10 Gem Mint & Pristine ---
    (GradingCompany.TAG, "10.0", Era.VINTAGE): (0.500, 25.0, 0.550, "TAG 10 vintage"),
    (GradingCompany.TAG, "10.0", Era.MID_ERA): (0.600, 20.0, 0.650, "TAG 10 mid-era"),
    (GradingCompany.TAG, "10.0", Era.MODERN): (0.820, 12.0, 0.850, "TAG 10 moderno"),
    (GradingCompany.TAG, "10.0_pristine", Era.VINTAGE): (0.800, 15.0, 0.850, "TAG 10 Pristine vintage"),
    (GradingCompany.TAG, "10.0_pristine", Era.MID_ERA): (0.900, 12.0, 0.900, "TAG 10 Pristine mid-era"),
    (GradingCompany.TAG, "10.0_pristine", Era.MODERN): (1.100, 5.0, 1.050, "TAG 10 Pristine moderno"),

    # --- GRAAD, PCA, CCC, AiGrading 10 ---
    (GradingCompany.GRAAD, "10.0", Era.VINTAGE): (0.350, 35.0, 0.400, "GRAAD 10 vintage scambia molto sotto PSA 10"),
    (GradingCompany.GRAAD, "10.0", Era.MID_ERA): (0.450, 30.0, 0.500, "GRAAD 10 mid-era"),
    (GradingCompany.GRAAD, "10.0", Era.MODERN): (0.550, 25.0, 0.600, "GRAAD 10 moderno"),
    (GradingCompany.PCA, "10.0", Era.VINTAGE): (0.400, 32.0, 0.450, "PCA 10 vintage"),
    (GradingCompany.PCA, "10.0", Era.MID_ERA): (0.500, 28.0, 0.550, "PCA 10 mid-era"),
    (GradingCompany.PCA, "10.0", Era.MODERN): (0.600, 22.0, 0.650, "PCA 10 moderno"),
    (GradingCompany.CCC, "10.0", Era.VINTAGE): (0.350, 35.0, 0.400, "CCC 10 vintage scambia molto sotto PSA 10"),
    (GradingCompany.CCC, "10.0", Era.MID_ERA): (0.450, 30.0, 0.500, "CCC 10 mid-era"),
    (GradingCompany.CCC, "10.0", Era.MODERN): (0.550, 25.0, 0.600, "CCC 10 moderno"),
    (GradingCompany.AIGRADING, "10.0", Era.VINTAGE): (0.350, 35.0, 0.400, "AiGrading 10 vintage"),
    (GradingCompany.AIGRADING, "10.0", Era.MID_ERA): (0.450, 30.0, 0.500, "AiGrading 10 mid-era"),
    (GradingCompany.AIGRADING, "10.0", Era.MODERN): (0.550, 25.0, 0.600, "AiGrading 10 moderno"),
}


def normalize_company(comp: str | GradingCompany) -> GradingCompany:
    """Normalizza la stringa della compagnia all'Enum GradingCompany."""
    if isinstance(comp, GradingCompany):
        return comp
    c = str(comp).strip().upper()
    if "BECKETT" in c or "BGS" in c:
        return GradingCompany.BGS
    if "CGC" in c:
        return GradingCompany.CGC
    if "TAG" in c:
        return GradingCompany.TAG
    if "AIGRAD" in c or "AI GRAD" in c:
        return GradingCompany.AIGRADING
    if "CCC" in c or "CLASSIC" in c:
        return GradingCompany.CCC
    if "GRAAD" in c:
        return GradingCompany.GRAAD
    if "PCA" in c:
        return GradingCompany.PCA
    if "SGC" in c:
        return GradingCompany.SGC
    if "ACE" in c:
        return GradingCompany.ACE
    return GradingCompany.PSA


def normalize_era(era: str | Era) -> Era:
    if isinstance(era, Era):
        return era
    e = str(era).strip().lower()
    if "vint" in e or "wotc" in e:
        return Era.VINTAGE
    if "mid" in e or "ex" in e or "dp" in e or "xy" in e or "bw" in e:
        return Era.MID_ERA
    return Era.MODERN


# Rapporti medi empirici PSA 10 / PSA 9 calibrati per Era collezionistica
# Fonti: scripts/grading_company_multiplier_research.py e dataset PriceCharting
ERA_PSA10_TO_PSA9_RATIO: Dict[Era, float] = {
    Era.VINTAGE: 3.80,   # Vintage (1999–2003): rarità estrema delle gemme (~3.8x vs PSA 9)
    Era.MID_ERA: 3.40,   # Mid-Era (2004–2016): popolazioni contenute, premio solido (~3.4x vs PSA 9)
    Era.MODERN: 2.80,    # Moderno (2017+): gem rate elevato, premio più compresso (~2.8x vs PSA 9)
}

ERA_BGS95_TO_PSA9_RATIO: Dict[Era, float] = {
    Era.VINTAGE: 1.81,   # BGS 9.5 nel Vintage
    Era.MID_ERA: 1.55,   # BGS 9.5 nel Mid-Era
    Era.MODERN: 1.70,    # BGS 9.5 nel Moderno
}


def estimate_psa10_from_psa9(psa9_price_eur: float, era: Era | str) -> float:
    """Calcola la stima algoritmica del benchmark PSA 10 partendo dal benchmark PSA 9 in assenza di dati reali PriceCharting."""
    era_enum = normalize_era(era)
    ratio = ERA_PSA10_TO_PSA9_RATIO.get(era_enum, 3.00)
    return round(psa9_price_eur * ratio, 2)


def estimate_grade95_from_psa9(psa9_price_eur: float, era: Era | str) -> float:
    """Calcola la stima algoritmica del benchmark Grado 9.5 partendo dal benchmark PSA 9 in assenza di dati reali PriceCharting."""
    era_enum = normalize_era(era)
    ratio = ERA_BGS95_TO_PSA9_RATIO.get(era_enum, 1.65)
    return round(psa9_price_eur * ratio, 2)


def get_recommended_grade_for_card(
    rel_year: Optional[int] = None,
    era: Optional[str | Era] = None,
) -> Dict[str, Any]:
    """
    Ritorna la raccomandazione quantitativa istituzionale sul grado su cui puntare
    in base all'era collezionistica della carta (Vintage vs Mid-Era vs Modern).
    
    Regole di Liquidità ed Efficienza di Mercato:
      - Vintage (1999–2003): Sweet Spot su PSA 9. Ottima liquidità, fair value accessibile.
        PSA 10 ha premi estremi (3.8x+) e scambi rarefatti.
      - Mid-Era (2004–2016): Bilanciato su PSA 9 o BGS 9.5 / CGC 9.5. Sano premium su Raw.
      - Moderno (2017+): Puntare tassativamente su PSA 10 (o BGS 9.5).
        ATTENZIONE: Grado 9 nel moderno è una trappola di liquidità (pop report saturo di 10,
        prezzi compressi a ridosso del Raw).
    """
    if era is not None:
        era_enum = normalize_era(era)
    elif rel_year is not None:
        if rel_year <= 2003:
            era_enum = Era.VINTAGE
        elif rel_year <= 2016:
            era_enum = Era.MID_ERA
        else:
            era_enum = Era.MODERN
    else:
        era_enum = Era.MODERN

    if era_enum == Era.VINTAGE:
        return {
            "era": "vintage",
            "era_label": "Vintage (1999–2003)",
            "target_grade": "PSA 9",
            "target_badge": "🎯 Target: PSA 9 (Mint)",
            "badge_color": "#10b981",
            "is_grade9_viable": True,
            "warning_modern_g9": False,
            "accessible_grades": ["8.5", "8.0", "7.5", "7.0"],
            "rationale": (
                "Vintage (1999–2003): PSA 9 è lo Sweet Spot Istituzionale (massima liquidità e fair value). "
                "Per carte rare ad alto costo (WotC Holo, 1st Edition, Shinings), anche i gradi intermedi 8.5, 8.0, 7.5 e 7.0 "
                "rappresentano ottime soglie d'ingresso collezionistiche a forte sconto (-22% / -52% vs G9) con solida conservazione del valore."
            ),
            "short_advice": "Vintage: Punta a PSA 9 (sweet spot), ma su carte ad alto valore anche PSA 7–8.5 offrono ottime entrate accessibili (-22%/-52%). PSA 10 ha premi estremi (3.8x+).",
        }
    elif era_enum == Era.MID_ERA:
        return {
            "era": "mid_era",
            "era_label": "Mid-Era (2004–2016)",
            "target_grade": "PSA 9 / BGS 9.5",
            "target_badge": "🎯 Target: PSA 9 o BGS 9.5",
            "badge_color": "#38bdf8",
            "is_grade9_viable": True,
            "warning_modern_g9": False,
            "accessible_grades": ["8.5", "8.0"],
            "rationale": (
                "Mid-Era (2004–2016, EX/DP/HGSS/BW/XY): Equilibrio solido tra liquidità e premium. "
                "I gradi 9 e 9.5 scambiano con frequenza e mantengono un sano margine sul Raw. "
                "Per le carte più costose, PSA 8 e 8.5 sono alternative liquide con sconti del -25%/-40% sul Grado 9."
            ),
            "short_advice": "Mid-Era: Bilanciato su PSA 9 o BGS 9.5. Per carte rare, PSA 8 e 8.5 offrono entrate liquide con sconto (-25%/-40%).",
        }
    else:
        return {
            "era": "modern",
            "era_label": "Moderno (2017+)",
            "target_grade": "PSA 10",
            "target_badge": "🎯 Target: PSA 10 (Evita G9/G8/G7)",
            "badge_color": "#f59e0b",
            "is_grade9_viable": False,
            "warning_modern_g9": True,
            "accessible_grades": [],
            "rationale": (
                "Moderno (2017+): Il Pop Report è saturo di Gem Mint (>70-80%). I gradi ≤ 9.0 (inclusi 8.5, 8.0, 7.5, 7.0) rappresentano una trappola "
                "di liquidità che distruggono valore rispetto al costo di gradazione e scambiano a sconto persino sul Raw. "
                "Nel moderno comprare SOLO Grado 10 (PSA 10, BGS 9.5/10 o CGC Pristine 10)."
            ),
            "short_advice": "⚠️ Moderno: Punta SOLO a Grado 10! Evita gradi ≤ 9.0 (inclusi 8 e 7: pop report saturo di 10, forte distruzione di valore vs Raw).",
        }


def get_grading_adjustment(
    company: str | GradingCompany,
    grade: str | float,
    era: str | Era = Era.MODERN,
    subgrades_black_label: bool = False,
    is_pristine: bool = False,
) -> GradingAdjustment:
    """
    Ritorna la rettifica quantitativa calibrata per una combinazione (Compagnia, Grado, Era).
    Supporta gradi da 10.0 fino a 7.0 con mezzi voti (9.5, 8.5, 7.5).
    """
    comp_enum = normalize_company(company)
    era_enum = normalize_era(era)
    
    # Formattazione chiave grado
    g_str = str(grade).strip().lower().replace("_", ".")
    if "10" in g_str:
        benchmark_ref = "PSA_10"
        if subgrades_black_label:
            grade_key = "10.0_black_label"
        elif is_pristine or "pristine" in g_str:
            grade_key = "10.0_pristine"
        elif comp_enum == GradingCompany.CGC and ("gem" in g_str or not is_pristine):
            grade_key = "10.0_gem"
        else:
            grade_key = "10.0"
        
        lookup = EMPIRICAL_RATIOS_GRADE10.get((comp_enum, grade_key, era_enum))
        if not lookup:
            lookup = EMPIRICAL_RATIOS_GRADE10.get((comp_enum, "10.0", era_enum), (0.70, 20.0, 0.75, "Fallback Grade 10"))
    else:
        benchmark_ref = "PSA_9"
        if "9.5" in g_str or "95" in g_str:
            grade_key = "9.5"
        elif "8.5" in g_str or "85" in g_str:
            grade_key = "8.5"
        elif "8" in g_str:
            grade_key = "8.0"
        elif "7.5" in g_str or "75" in g_str:
            grade_key = "7.5"
        elif "7" in g_str:
            grade_key = "7.0"
        else:
            grade_key = "9.0"
        
        lookup = EMPIRICAL_RATIOS_GRADE9.get((comp_enum, grade_key, era_enum))
        if not lookup:
            lookup = EMPIRICAL_RATIOS_GRADE9.get((comp_enum, "9.0", era_enum), (0.85, 10.0, 0.88, "Fallback Grade 9"))

    mult, liq_pen, sniper_factor, notes = lookup
    return GradingAdjustment(
        company=comp_enum,
        grade_label=f"{comp_enum.value} {grade}",
        benchmark_ref=benchmark_ref,
        multiplier=mult,
        liquidity_penalty_pct=liq_pen,
        sniper_ceiling_factor=sniper_factor,
        era=era_enum,
        notes=notes,
    )


def get_company_relative_factor_vs_psa(
    company: str | GradingCompany,
    grade: str | float,
    era: str | Era = Era.MODERN,
    subgrades_black_label: bool = False,
    is_pristine: bool = False,
) -> float:
    """
    Ritorna il fattore di prezzo relativo tra una specifica compagnia e PSA
    per lo stesso identico grado ed era (es. BGS 8.5 vs PSA 8.5 -> ~0.92x).
    """
    comp_enum = normalize_company(company)
    if comp_enum == GradingCompany.PSA and not (subgrades_black_label or is_pristine):
        return 1.000
    
    era_enum = normalize_era(era)
    adj_comp = get_grading_adjustment(comp_enum, grade, era_enum, subgrades_black_label, is_pristine)
    adj_psa = get_grading_adjustment(GradingCompany.PSA, grade, era_enum)
    if adj_psa.multiplier > 0:
        return round(adj_comp.multiplier / adj_psa.multiplier, 3)
    return round(adj_comp.multiplier, 3)


_LADDER_CACHE_DATA: Optional[Dict[str, Any]] = None


def load_grade_ladder_cache() -> Dict[str, Any]:
    """Carica in memoria la cache locale grade_ladder_prices.json."""
    global _LADDER_CACHE_DATA
    if _LADDER_CACHE_DATA is None:
        ladder_file = Path(__file__).resolve().parent.parent.parent / "data_cache" / "grade_ladder_prices.json"
        if ladder_file.exists():
            try:
                _LADDER_CACHE_DATA = json.loads(ladder_file.read_text(encoding="utf-8"))
            except Exception:
                _LADDER_CACHE_DATA = {}
        else:
            _LADDER_CACHE_DATA = {}
    return _LADDER_CACHE_DATA


def get_grade_benchmarks_ladder(
    base_psa9_eur: float,
    era: Era | str,
    item_id: Optional[str] = None,
    game_slug: Optional[str] = None,
    item_slug: Optional[str] = None,
    pc_ladder: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Genera la scala completa dei prezzi benchmark PSA (da 10.0 fino a 7.0 con mezzi voti).
    Privilegia SEMPRE i dati reali di PriceCharting estratti da grade_ladder_prices.json o live,
    e calcola stime algoritmiche calibrate per i gradi mancanti.
    
    Ritorna un dizionario con chiavi '10.0', '9.5', '9.0', '8.5', '8.0', '7.5', '7.0'.
    """
    era_enum = normalize_era(era)
    real_tiers: Dict[str, float] = {}

    # Se pc_ladder è fornito esplicitamente, usalo
    if pc_ladder:
        real_tiers = dict(pc_ladder)
    elif item_id or (game_slug and item_slug):
        ladder_cache = load_grade_ladder_cache()
        target_entry = ladder_cache.get(item_id, {}) if item_id else {}
        if not target_entry and item_slug:
            clean_slug = item_slug.replace("-", "_")
            for k, v in ladder_cache.items():
                if clean_slug in k or (item_id and item_id.lower() == k.lower()):
                    target_entry = v
                    break
        for tier_key in ["psa10", "grade9_5", "grade9", "grade8", "grade7"]:
            if tier_key in target_entry and target_entry[tier_key]:
                sorted_dates = sorted(target_entry[tier_key].keys())
                real_tiers[tier_key] = float(target_entry[tier_key][sorted_dates[-1]])

    ladder: Dict[str, Dict[str, Any]] = {}

    # 1. Grado 10.0
    if "psa10" in real_tiers:
        p10 = real_tiers["psa10"]
        s10 = "PriceCharting Reale PSA 10"
        r10 = True
    else:
        p10 = estimate_psa10_from_psa9(base_psa9_eur, era_enum)
        s10 = f"Stima Algoritmica ({ERA_PSA10_TO_PSA9_RATIO.get(era_enum, 2.80):.2f}x PSA 9)"
        r10 = False
    ladder["10.0"] = {
        "price_eur": p10,
        "label": "PSA 10 Gem Mint",
        "source": s10,
        "is_real": r10,
        "ratio_vs_psa9": round(p10 / base_psa9_eur, 2) if base_psa9_eur > 0 else 2.80,
    }

    # 2. Grado 9.5
    if "grade9_5" in real_tiers:
        p95 = real_tiers["grade9_5"]
        s95 = "PriceCharting Reale Grado 9.5"
        r95 = True
    else:
        p95 = estimate_grade95_from_psa9(base_psa9_eur, era_enum)
        s95 = f"Stima Algoritmica ({ERA_BGS95_TO_PSA9_RATIO.get(era_enum, 1.65):.2f}x PSA 9)"
        r95 = False
    ladder["9.5"] = {
        "price_eur": p95,
        "label": "PSA 9.5 / BGS 9.5 Gem Mint",
        "source": s95,
        "is_real": r95,
        "ratio_vs_psa9": round(p95 / base_psa9_eur, 2) if base_psa9_eur > 0 else 1.65,
    }

    # 3. Grado 9.0 (Benchmark di riferimento)
    if "grade9" in real_tiers:
        p9 = real_tiers["grade9"]
        s9 = "PriceCharting Reale Grado 9"
        r9 = True
    else:
        p9 = base_psa9_eur
        s9 = "Database PokeQuant (Benchmark PSA 9)"
        r9 = True
    ladder["9.0"] = {
        "price_eur": p9,
        "label": "PSA 9.0 Mint",
        "source": s9,
        "is_real": r9,
        "ratio_vs_psa9": 1.00,
    }

    # 4. Grado 8.5
    if "grade8_5" in real_tiers:
        p85 = real_tiers["grade8_5"]
        s85 = "PriceCharting Reale Grado 8.5"
        r85 = True
    elif "grade8" in real_tiers and "grade9" in real_tiers:
        p85 = round((real_tiers["grade8"] + real_tiers["grade9"]) / 2, 2)
        s85 = "PriceCharting Reale Interpolato (G8-G9)"
        r85 = True
    elif "grade8" in real_tiers:
        p85 = round(real_tiers["grade8"] * 1.25, 2)
        s85 = "PriceCharting G8 + Premio Mezzo Voto (+25%)"
        r85 = True
    else:
        mult_85 = EMPIRICAL_RATIOS_GRADE9[(GradingCompany.PSA, "8.5", era_enum)][0]
        p85 = round(base_psa9_eur * mult_85, 2)
        s85 = f"Stima Algoritmica ({mult_85:.2f}x PSA 9)"
        r85 = False
    ladder["8.5"] = {
        "price_eur": p85,
        "label": "PSA 8.5 NM-Mint+",
        "source": s85,
        "is_real": r85,
        "ratio_vs_psa9": round(p85 / base_psa9_eur, 2) if base_psa9_eur > 0 else 0.78,
    }

    # 5. Grado 8.0
    if "grade8" in real_tiers:
        p80 = real_tiers["grade8"]
        s80 = "PriceCharting Reale Grado 8"
        r80 = True
    else:
        mult_80 = EMPIRICAL_RATIOS_GRADE9[(GradingCompany.PSA, "8.0", era_enum)][0]
        p80 = round(base_psa9_eur * mult_80, 2)
        s80 = f"Stima Algoritmica ({mult_80:.2f}x PSA 9)"
        r80 = False
    ladder["8.0"] = {
        "price_eur": p80,
        "label": "PSA 8.0 NM-Mint",
        "source": s80,
        "is_real": r80,
        "ratio_vs_psa9": round(p80 / base_psa9_eur, 2) if base_psa9_eur > 0 else 0.65,
    }

    # 6. Grado 7.5
    if "grade7_5" in real_tiers:
        p75 = real_tiers["grade7_5"]
        s75 = "PriceCharting Reale Grado 7.5"
        r75 = True
    elif "grade7" in real_tiers and "grade8" in real_tiers:
        p75 = round((real_tiers["grade7"] + real_tiers["grade8"]) / 2, 2)
        s75 = "PriceCharting Reale Interpolato (G7-G8)"
        r75 = True
    elif "grade7" in real_tiers:
        p75 = round(real_tiers["grade7"] * 1.18, 2)
        s75 = "PriceCharting G7 + Premio Mezzo Voto (+18%)"
        r75 = True
    else:
        mult_75 = EMPIRICAL_RATIOS_GRADE9[(GradingCompany.PSA, "7.5", era_enum)][0]
        p75 = round(base_psa9_eur * mult_75, 2)
        s75 = f"Stima Algoritmica ({mult_75:.2f}x PSA 9)"
        r75 = False
    ladder["7.5"] = {
        "price_eur": p75,
        "label": "PSA 7.5 Near Mint+",
        "source": s75,
        "is_real": r75,
        "ratio_vs_psa9": round(p75 / base_psa9_eur, 2) if base_psa9_eur > 0 else 0.55,
    }

    # 7. Grado 7.0
    if "grade7" in real_tiers:
        p70 = real_tiers["grade7"]
        s70 = "PriceCharting Reale Grado 7"
        r70 = True
    else:
        mult_70 = EMPIRICAL_RATIOS_GRADE9[(GradingCompany.PSA, "7.0", era_enum)][0]
        p70 = round(base_psa9_eur * mult_70, 2)
        s70 = f"Stima Algoritmica ({mult_70:.2f}x PSA 9)"
        r70 = False
    ladder["7.0"] = {
        "price_eur": p70,
        "label": "PSA 7.0 Near Mint",
        "source": s70,
        "is_real": r70,
        "ratio_vs_psa9": round(p70 / base_psa9_eur, 2) if base_psa9_eur > 0 else 0.48,
    }

    # Calcolo percentuale di sconto vs PSA 9
    for k, v in ladder.items():
        if base_psa9_eur > 0:
            v["discount_vs_psa9_pct"] = round((v["price_eur"] / base_psa9_eur - 1.0) * 100.0, 1)
        else:
            v["discount_vs_psa9_pct"] = 0.0

    return ladder


def adjust_price_for_grading(
    base_psa_price_eur: float,
    company: str | GradingCompany,
    grade: str | float,
    era: str | Era = Era.MODERN,
    subgrades_black_label: bool = False,
    is_pristine: bool = False,
    variant: str = "standard",
    is_grade_benchmark_price: bool = False,
) -> Tuple[float, float, GradingAdjustment]:
    """
    Ricalibra un prezzo benchmark PSA (Grado 9 o Grado 10 o Grado specifico) per la compagnia e l'eventuale variante speciale desiderata.
    
    Parametri:
      base_psa_price_eur: Prezzo benchmark PSA (se is_grade_benchmark_price=False: PSA 9 per gradi <=9.5, PSA 10 per grado 10).
      company: Compagnia di gradazione (PSA, BGS, CGC, SGC, TAG, GRAAD, PCA, CCC, AiGrading, ACE).
      grade: Voto (10.0, 9.5, 9.0, 8.5, 8.0, 7.5, 7.0).
      is_grade_benchmark_price: Se True, indica che base_psa_price_eur è GIÀ il prezzo di mercato reale PSA di quel grado specifico.
    
    Ritorna:
      (fair_value_calibrato_eur, max_edge_sniper_ceiling_eur, adjustment_obj)
    """
    v_mult, v_desc = get_variant_multiplier(variant)
    effective_base_psa = base_psa_price_eur * v_mult

    adj = get_grading_adjustment(
        company=company,
        grade=grade,
        era=era,
        subgrades_black_label=subgrades_black_label,
        is_pristine=is_pristine,
    )

    if is_grade_benchmark_price:
        # Se effective_base_psa è già il prezzo PSA di quel grado (es. estratto da PriceCharting),
        # si applica solo il fattore relativo della compagnia vs PSA.
        comp_rel = get_company_relative_factor_vs_psa(
            company=company,
            grade=grade,
            era=era,
            subgrades_black_label=subgrades_black_label,
            is_pristine=is_pristine,
        )
        fair_value = effective_base_psa * comp_rel
        sniper_ceiling = (effective_base_psa * 1.05) * adj.sniper_ceiling_factor
    else:
        fair_value = effective_base_psa * adj.multiplier
        sniper_ceiling = effective_base_psa * adj.sniper_ceiling_factor

    return round(fair_value, 2), round(sniper_ceiling, 2), adj

