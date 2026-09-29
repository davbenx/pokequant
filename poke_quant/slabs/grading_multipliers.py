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
import urllib.parse
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List


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
        sniper_ceiling = round(round(fair_value, 2) * 1.05, 2)
    else:
        fair_value = effective_base_psa * adj.multiplier
        sniper_ceiling = effective_base_psa * adj.sniper_ceiling_factor

    return round(fair_value, 2), round(sniper_ceiling, 2), adj


_POP_PRESSURE_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


def load_pop_pressure_cache() -> Dict[str, Dict[str, Any]]:
    """Carica e calcola in cache le metriche di pressione demografica relativa per carta ed era."""
    global _POP_PRESSURE_CACHE
    if _POP_PRESSURE_CACHE is not None:
        return _POP_PRESSURE_CACHE

    pop_file = Path(__file__).resolve().parent.parent.parent / "data_cache" / "population_history.csv"
    if not pop_file.exists():
        _POP_PRESSURE_CACHE = {}
        return _POP_PRESSURE_CACHE

    try:
        import pandas as pd
        import numpy as np
        from poke_quant.data.storage import load_metadata

        meta = load_metadata()
        df = pd.read_csv(pop_file)
        piv = df.pivot_table(index="item_id", columns="grade", values="total_pop", aggfunc="first")
        piv.columns = [int(c) if str(c).isdigit() else c for c in piv.columns]

        era_ratios: Dict[str, list] = {"vintage": [], "mid_era": [], "modern": []}
        card_raw: Dict[str, Dict[str, Any]] = {}

        for iid in piv.index:
            m = meta.get(iid, {})
            era_val = m.get("era")
            if not era_val:
                rd = str(m.get("release_date", "2020"))[:4]
                yr = int(rd) if rd.isdigit() else 2020
                era_val = "vintage" if yr <= 2003 else ("mid_era" if yr <= 2016 else "modern")
            norm_e = normalize_era(era_val).value
            if norm_e not in era_ratios:
                norm_e = "modern"

            p8 = piv.loc[iid].get(8, np.nan)
            p9 = piv.loc[iid].get(9, np.nan)
            p10 = piv.loc[iid].get(10, np.nan)

            if pd.notna(p8) and pd.notna(p9) and p9 > 0:
                r = float(p8 / p9)
                era_ratios[norm_e].append(r)
                card_raw[iid] = {
                    "era": norm_e,
                    "ratio_8_9": r,
                    "pop_8": float(p8),
                    "pop_9": float(p9),
                    "pop_10": float(p10) if pd.notna(p10) else None,
                }
            elif pd.notna(p9) and (pd.isna(p8) or p8 == 0):
                r = 0.0
                era_ratios[norm_e].append(r)
                card_raw[iid] = {
                    "era": norm_e,
                    "ratio_8_9": r,
                    "pop_8": 0.0,
                    "pop_9": float(p9),
                    "pop_10": float(p10) if pd.notna(p10) else None,
                }

        sorted_arrays = {k: np.sort(np.array(v)) for k, v in era_ratios.items() if len(v) > 0}

        cache: Dict[str, Dict[str, Any]] = {}
        for iid, data in card_raw.items():
            norm_e = data["era"]
            arr = sorted_arrays.get(norm_e)
            r = data["ratio_8_9"]
            if arr is not None and len(arr) > 0:
                idx = np.searchsorted(arr, r, side="right")
                pct = float((idx / len(arr)) * 100.0)
            else:
                pct = 50.0

            if pct <= 50.0:
                tier = "low"
                badge_html = f'<span title="Percentile P{pct:.0f}: Popolazione equilibrata (il rapporto Pop8/Pop9 è inferiore alla media dell\'era)" style="background:rgba(16,185,129,0.15); color:#10b981; border:1px solid #10b981; border-radius:4px; padding:1px 6px; font-size:11px; font-weight:600;">🟢 Pop Equilibrata (P{pct:.0f})</span>'
                is_overcrowded = False
            elif pct <= 75.0:
                tier = "fisiologica"
                badge_html = f'<span title="Percentile P{pct:.0f}: Distribuzione fisiologica rispetto all\'era" style="background:rgba(245,158,11,0.15); color:#fbbf24; border:1px solid #fbbf24; border-radius:4px; padding:1px 6px; font-size:11px; font-weight:600;">🟡 Pop Fisiologica (P{pct:.0f})</span>'
                is_overcrowded = False
            elif pct <= 90.0:
                tier = "moderata"
                badge_html = f'<span title="Percentile P{pct:.0f}: Diluizione moderata (rapporto Pop8/Pop9 superiore al 75% delle carte dell\'era)" style="background:rgba(249,115,22,0.15); color:#fb923c; border:1px solid #fb923c; border-radius:4px; padding:1px 6px; font-size:11px; font-weight:600;">🟠 Diluizione Moderata (P{pct:.0f})</span>'
                is_overcrowded = False
            else:
                tier = "overcrowded"
                badge_html = f'<span title="Percentile P{pct:.0f}: Sovraffollamento (Top {100-pct:.0f}% per accumulo di grado 8 rispetto a 9 nell\'era)" style="background:rgba(244,63,94,0.15); color:#f43f5e; border:1px solid #f43f5e; border-radius:4px; padding:1px 6px; font-size:11px; font-weight:600;">🔴 Sovraffollamento Pop (P{pct:.0f})</span>'
                is_overcrowded = True

            cache[iid] = {
                "ratio_8_9": round(r, 2),
                "percentile": round(pct, 1),
                "tier": tier,
                "badge_html": badge_html,
                "is_overcrowded": is_overcrowded,
                "pop_8": data["pop_8"],
                "pop_9": data["pop_9"],
                "pop_10": data["pop_10"],
            }
        _POP_PRESSURE_CACHE = cache
    except Exception:
        _POP_PRESSURE_CACHE = {}

    return _POP_PRESSURE_CACHE


def get_card_pop_pressure(item_id: Optional[str], era: Era | str) -> Dict[str, Any]:
    """Ritorna le metriche relative di pressione demografica per una carta."""
    if not item_id:
        return {
            "ratio_8_9": None,
            "percentile": None,
            "tier": "unknown",
            "badge_html": '<span style="color:#64748b; font-size:11px;">Pop N/D</span>',
            "is_overcrowded": False,
        }
    cache = load_pop_pressure_cache()
    if item_id in cache:
        return cache[item_id]

    clean_id = item_id.lower().replace("-", "_")
    for k, v in cache.items():
        if k.lower() in clean_id or clean_id in k.lower():
            return v

    return {
        "ratio_8_9": None,
        "percentile": None,
        "tier": "unknown",
        "badge_html": '<span style="color:#64748b; font-size:11px;">Pop N/D</span>',
        "is_overcrowded": False,
    }


def get_recommended_grade_targets(
    base_psa9_eur: float,
    era: Era | str,
    item_id: Optional[str] = None,
    game_slug: Optional[str] = None,
    item_slug: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Identifica la migliore gradazione consigliata (Target Primario) e le alternative
    minori consigliate (punti di ingresso a sconto ad alta liquidità) per una specifica carta ed era,
    arricchito con indicatori di pressione demografica relativa (Pop Pressure Index).
    """
    norm_era = normalize_era(era)
    ladder = get_grade_benchmarks_ladder(
        base_psa9_eur=base_psa9_eur,
        era=norm_era,
        item_id=item_id,
        game_slug=game_slug,
        item_slug=item_slug,
    )
    pop_info = get_card_pop_pressure(item_id, norm_era)

    if norm_era == Era.MODERN:
        target_grade = "PSA 10"
        target_price = ladder["10.0"]["price_eur"]
        target_label = "PSA 10 Gem Mint (Target Primario Moderno)"
        target_badge = "🎯 Target: PSA 10 (Gem Mint)"
        badge_color = "#f59e0b"
        max_edge = round(target_price * 1.05, 2)

        p_95 = ladder["9.5"]["price_eur"]
        disc_vs_10 = round((1.0 - (p_95 / target_price)) * 100.0, 1) if target_price > 0 else 0.0
        minor_alts = [
            {
                "grade": "9.5",
                "company": "BGS",
                "label": "BGS 9.5 Gem Mint",
                "price_eur": p_95,
                "discount_vs_target_pct": -disc_vs_10,
                "is_real": ladder["9.5"]["is_real"],
                "source": ladder["9.5"]["source"],
                "rationale": "Unica alternativa minore consigliata sul Moderno (-35/40% vs PSA 10, alta qualità costruttiva Beckett e potenziale regrade)",
            }
        ]
        minor_str = f"BGS 9.5 ~{p_95:.0f}€ (-{disc_vs_10:.0f}% vs 10) · ⚠️ Sconsigliati gradi ≤ 9.0"
        advice = "Nel Moderno il Pop Report è saturo di Gem Mint (>70-80%). Punta a PSA 10 o come alternativa minore a BGS 9.5. I gradi ≤ 9.0 scambiano sotto il costo di gradazione e distruggono valore rispetto al Raw."
        is_grade9_viable = False
    elif norm_era == Era.VINTAGE:
        target_grade = "PSA 9"
        target_price = ladder["9.0"]["price_eur"]
        target_label = "PSA 9 Mint (Sweet Spot Istituzionale)"
        target_badge = "🎯 Target: PSA 9 (Mint)"
        badge_color = "#10b981"
        max_edge = round(target_price * 1.05, 2)

        p_85 = ladder["8.5"]["price_eur"]
        p_80 = ladder["8.0"]["price_eur"]
        p_70 = ladder["7.0"]["price_eur"]
        d85 = ladder["8.5"]["discount_vs_psa9_pct"]
        d80 = ladder["8.0"]["discount_vs_psa9_pct"]
        d70 = ladder["7.0"]["discount_vs_psa9_pct"]

        minor_alts = [
            {
                "grade": "8.5",
                "company": "PSA",
                "label": "PSA 8.5 NM-Mint+",
                "price_eur": p_85,
                "discount_vs_target_pct": d85,
                "is_real": ladder["8.5"]["is_real"],
                "source": ladder["8.5"]["source"],
                "rationale": "Near Mint+ solido, ottimo compromesso qualitativo ed estetico a sconto",
            },
            {
                "grade": "8.0",
                "company": "PSA",
                "label": "PSA 8.0 NM-Mint",
                "price_eur": p_80,
                "discount_vs_target_pct": d80,
                "is_real": ladder["8.0"]["is_real"],
                "source": ladder["8.0"]["source"],
                "rationale": "Soglia di ingresso ad alta rotazione e liquidità retail su mercato EU",
            },
            {
                "grade": "7.0",
                "company": "PSA",
                "label": "PSA 7.0 Near Mint",
                "price_eur": p_70,
                "discount_vs_target_pct": d70,
                "is_real": ladder["7.0"]["is_real"],
                "source": ladder["7.0"]["source"],
                "rationale": "Entry-level per carte rare e costose (WotC Holo, 1st Edition, Shinings)",
            },
        ]
        minor_str = f"PSA 8.5 ~{p_85:.0f}€ ({d85:+.0f}%) · PSA 8.0 ~{p_80:.0f}€ ({d80:+.0f}%) · PSA 7.0 ~{p_70:.0f}€ ({d70:+.0f}%)"
        if pop_info.get("is_overcrowded"):
            minor_str += f" · ⚠️ Pop G8 abbondante (P{pop_info['percentile']:.0f}): preferire Target PSA 9"
        advice = "Nel Vintage PSA 9 è lo Sweet Spot Istituzionale di massima liquidità. Se il budget è limitato o la carta supera i 150-200€, PSA 8.5, 8.0 e 7.0 offrono ottimi ingressi a forte sconto (-22%/-52%)."
        is_grade9_viable = True
    else:  # MID_ERA
        target_grade = "PSA 9"
        target_price = ladder["9.0"]["price_eur"]
        target_label = "PSA 9 Mint (Target Bilanciato)"
        target_badge = "🎯 Target: PSA 9 (Mint)"
        badge_color = "#10b981"
        max_edge = round(target_price * 1.05, 2)

        p_85 = ladder["8.5"]["price_eur"]
        p_80 = ladder["8.0"]["price_eur"]
        d85 = ladder["8.5"]["discount_vs_psa9_pct"]
        d80 = ladder["8.0"]["discount_vs_psa9_pct"]

        minor_alts = [
            {
                "grade": "8.5",
                "company": "PSA",
                "label": "PSA 8.5 NM-Mint+",
                "price_eur": p_85,
                "discount_vs_target_pct": d85,
                "is_real": ladder["8.5"]["is_real"],
                "source": ladder["8.5"]["source"],
                "rationale": "Near Mint+ a sconto (-25%)",
            },
            {
                "grade": "8.0",
                "company": "PSA",
                "label": "PSA 8.0 NM-Mint",
                "price_eur": p_80,
                "discount_vs_target_pct": d80,
                "is_real": ladder["8.0"]["is_real"],
                "source": ladder["8.0"]["source"],
                "rationale": "Soglia liquida per carte EX/Lv.X rare a sconto (-40%)",
            },
        ]
        minor_str = f"PSA 8.5 ~{p_85:.0f}€ ({d85:+.0f}%) · PSA 8.0 ~{p_80:.0f}€ ({d80:+.0f}%)"
        if pop_info.get("is_overcrowded"):
            minor_str += f" · ⚠️ Pop G8 abbondante (P{pop_info['percentile']:.0f}): preferire Target PSA 9"
        advice = "Nel Mid-Era PSA 9 è la scelta principale; PSA 8.5 e 8.0 offrono valide entrate secondarie a sconto con solida conservazione del valore."
        is_grade9_viable = True

    return {
        "era": norm_era.value,
        "target_grade": target_grade,
        "target_price_eur": target_price,
        "target_label": target_label,
        "target_badge": target_badge,
        "badge_color": badge_color,
        "max_edge_eur": max_edge,
        "minor_alternatives": minor_alts,
        "minor_alternatives_str": minor_str,
        "advice": advice,
        "is_grade9_viable": is_grade9_viable,
        "ladder": ladder,
        "pop_pressure": pop_info,
    }


def get_card_strategy_and_pop_details(
    item_id: Optional[str] = None,
    game_slug: Optional[str] = None,
    item_slug: Optional[str] = None,
    card_name: Optional[str] = None,
    era: Optional[Era | str] = None,
    rarity: Optional[str] = None,
    release_date: Optional[str] = None,
    mode: str = "production",
) -> Dict[str, Any]:
    """
    Recupera i dati completi di popolazione multi-fonte (PSA, CGC, Totale),
    i fattori di scarsità (pull rate atteso da box, rigidità offerta, età) e
    il posizionamento Value nella strategia quantitativa PokeQuant (Core BUY,
    Panchina, Alternativa, Neutra, AVOID, o Custom).
    """
    from poke_quant.data.population_fetcher import fetch_pricecharting_population_cached
    from poke_quant.engine.strategies.scarcity_value_factor import EXPECTED_COPIES_PER_BOX
    from poke_quant.data.storage import load_metadata

    resolved_id = item_id
    meta = {}
    try:
        meta = load_metadata()
    except Exception:
        pass

    if not resolved_id and game_slug and item_slug:
        for iid, info in meta.items():
            if info.get("game_slug") == game_slug and info.get("item_slug") == item_slug:
                resolved_id = iid
                break

    card_meta = meta.get(resolved_id, {}) if resolved_id else {}
    final_card_name = card_name or card_meta.get("name") or (item_slug.replace("-", " ").title() if item_slug else "Carta")
    final_game_slug = game_slug or card_meta.get("game_slug") or ""
    final_item_slug = item_slug or card_meta.get("item_slug") or ""
    final_rarity = rarity or card_meta.get("rarity")
    final_rel_date = release_date or card_meta.get("release_date")
    norm_era = normalize_era(era or card_meta.get("era") or "vintage")

    # 1. Recupero Censimento Popolazione
    pop_rows = fetch_pricecharting_population_cached(
        game_slug=final_game_slug,
        item_slug=final_item_slug,
        item_id=resolved_id,
    )

    psa_by_grade: Dict[str, int] = {}
    cgc_by_grade: Dict[str, int] = {}
    total_by_grade: Dict[str, int] = {}
    psa_total = 0
    cgc_total = 0
    market_total = 0

    for r in pop_rows:
        g = str(r.get("grade", "")).strip()
        p = r.get("psa_pop")
        c = r.get("cgc_pop")
        t = r.get("total_pop")
        if p is not None:
            psa_by_grade[g] = int(p)
            psa_total += int(p)
        if c is not None:
            cgc_by_grade[g] = int(c)
            cgc_total += int(c)
        if t is not None:
            total_by_grade[g] = int(t)
            market_total += int(t)

    psa_10 = psa_by_grade.get("10", 0)
    psa_9 = psa_by_grade.get("9", 0)
    psa_8 = psa_by_grade.get("8", 0)
    psa_7 = psa_by_grade.get("7", 0)

    cgc_10 = cgc_by_grade.get("10", 0)
    cgc_95 = cgc_by_grade.get("9.5", 0)
    cgc_9 = cgc_by_grade.get("9", 0)
    cgc_85 = cgc_by_grade.get("8.5", 0)

    gem_rate_psa = round((psa_10 / psa_total) * 100.0, 1) if psa_total > 0 else None
    has_pop_report = len(pop_rows) > 0 and (psa_total > 0 or market_total > 0)

    pop_pressure = get_card_pop_pressure(resolved_id, norm_era)

    pc_pop_url = f"https://www.pricecharting.com/pop/item/{final_game_slug}/{final_item_slug}" if (final_game_slug and final_item_slug) else None

    search_q_parts = [final_card_name]
    if final_game_slug:
        clean_set = final_game_slug.replace("pokemon-", "").replace("-", " ")
        search_q_parts.append(clean_set)
    psa_search_url = f"https://www.psacard.com/pop/search?q={urllib.parse.quote_plus(' '.join(search_q_parts))}"

    # 2. Fattore Scarsità
    copies_per_box = EXPECTED_COPIES_PER_BOX.get(final_rarity) if final_rarity else None
    if copies_per_box is not None:
        if copies_per_box < 1.0:
            boxes_per_copy = round(1.0 / copies_per_box, 1)
            pull_rate_desc = f"~1 copia ogni {boxes_per_copy} booster box (~{round(36 * boxes_per_copy)} bustine)"
            rarity_tier_label = "Rarità Estrema (Secret / Ultra Chase)"
        elif copies_per_box <= 2.0:
            packs_per_copy = round(36.0 / copies_per_box)
            pull_rate_desc = f"~{copies_per_box:.1f} copie a box (~1 ogni {packs_per_copy} bustine)"
            rarity_tier_label = "Molto Rara (Special Illustration / Alt Art)"
        elif copies_per_box <= 9.0:
            packs_per_copy = round(36.0 / copies_per_box, 1)
            pull_rate_desc = f"~{copies_per_box:.0f} copie a box (~1 ogni {packs_per_copy} bustine)"
            rarity_tier_label = "Rara (Illustration / Ultra Rare / EX)"
        else:
            packs_per_copy = round(36.0 / copies_per_box, 1)
            pull_rate_desc = f"~{copies_per_box:.0f} copie a box (~1 ogni {packs_per_copy} bustine)"
            rarity_tier_label = "Rara Olografica Regolare"
    else:
        pull_rate_desc = "Frequenza bustine non standardizzata da tabella booster"
        rarity_tier_label = final_rarity or "Non catalogata"

    # Età e offerta
    if final_rel_date:
        yr_str = str(final_rel_date)[:4]
        rel_yr = int(yr_str) if yr_str.isdigit() else 2020
    elif norm_era == Era.VINTAGE:
        rel_yr = 2000
    elif norm_era == Era.MID_ERA:
        rel_yr = 2010
    else:
        rel_yr = 2022

    age_years = max(0, 2026 - rel_yr)

    if rel_yr <= 2003:
        supply_status = "🔒 Fuori Stampa (Vintage 20+ anni)"
        supply_elasticity = "Offerta rigidamente anelastica: produzione chiusa da oltre due decenni, stock sigillato esaurito. Il censimento delle copie mint è un tetto massimo invalicabile."
    elif rel_yr <= 2016:
        supply_status = "🔒 Fuori Stampa (Mid-Era)"
        supply_elasticity = "Offerta anelastica: set archiviato da anni, aperture di box sigillati rare ed estremamente costose."
    elif age_years >= 3:
        supply_status = "📦 Fuori Stampa / Fine Ciclo"
        supply_elasticity = "Offerta quasi-fissa: distribuzione primaria conclusa da oltre 2 anni, afflusso di nuove gradazioni in progressiva stabilizzazione."
    else:
        supply_status = "🔄 In Stampa / Distribuzione Attiva"
        supply_elasticity = "Offerta elastica: set moderno ancora reperibile a scaffale. Rischio di espansione continua del censimento delle copie gradate."

    # Scarsità di censimento
    if psa_total > 0:
        if psa_10 <= 50:
            census_verdict = f"💎 Scarsità Assoluta: appena {psa_10} copie Gem Mint PSA 10 al mondo su {psa_total:,} censite complessivamente."
        elif psa_10 <= 300:
            census_verdict = f"🛡️ Popolazione Ristretta: {psa_10:,} copie PSA 10 al mondo ({gem_rate_psa:.1f}% di Gem Rate)."
        elif psa_10 <= 2500:
            census_verdict = f"📊 Popolazione Liquida: {psa_10:,} copie PSA 10 al mondo ({gem_rate_psa:.1f}% di Gem Rate)."
        else:
            census_verdict = f"🌊 Iper-Abbondanza Demografica: ben {psa_10:,} copie PSA 10 censite ({gem_rate_psa:.1f}% di Gem Rate, forte diluizione)."
    else:
        census_verdict = "Censimento di mercato non disponibile nel database locale."

    # 3. Fattore Value (Modello Edonico PokeQuant)
    dashboard_cache_file = Path(__file__).resolve().parent.parent.parent / "data_cache" / "precomputed_dashboard_data.json"
    strat_key = "production" if mode == "production" else "dac7"
    strategy_data: Dict[str, Any] = {}
    if dashboard_cache_file.exists():
        try:
            with open(dashboard_cache_file, "r", encoding="utf-8") as f:
                d_all = json.load(f)
                strategy_data = d_all.get("singles_signals", {}).get(strat_key, {})
        except Exception:
            strategy_data = {}

    buy_map = {r["item_id"]: r for r in strategy_data.get("buy_rows", [])}
    alt_map = {r["item_id"]: r for r in strategy_data.get("alt_rows", [])}
    avoid_map = {r["item_id"]: r for r in strategy_data.get("avoid_rows", [])}

    if resolved_id and resolved_id in buy_map:
        b_row = buy_map[resolved_id]
        is_core = b_row.get("tier") == "core"
        strat_tier = "CORE_BUY" if is_core else "BENCH_BUY"
        strat_badge = "💎 Tier 1: Core Conviction (BUY)" if is_core else "🛡️ Tier 2: Panchina & Riserve (BUY)"
        strat_badge_color = "#10b981"
        res_val = b_row.get("residual")
        disc_val = b_row.get("discount_pct", 0.0)
        value_verdict = (
            f"La carta è selezionata dal modello quantitativo istituzionale nel portafoglio BUY: "
            f"scambia a uno sconto edonico del {abs(disc_val):.1f}% rispetto a carte comparabili con pari scarsità ed età (residuo ε={res_val:.2f}). "
            f"Rappresenta un ingresso primario ad altissima asimmetria statistica."
        )
    elif resolved_id and resolved_id in alt_map:
        a_row = alt_map[resolved_id]
        strat_tier = "ALT_BUY"
        strat_badge = "🔄 Alternativa Quantile BUY (Top 20%)"
        strat_badge_color = "#38bdf8"
        res_val = a_row.get("residual")
        disc_val = a_row.get("discount_pct", 0.0)
        value_verdict = (
            f"La carta si colloca nel miglior 20% del mercato (Top Quantile per fattore Scarsità-Valore) "
            f"con uno sconto del {abs(disc_val):.1f}% sui comparabili (residuo ε={res_val:.2f}). "
            f"Ottima alternativa di ripiego per diversificare o superare vincoli di reperibilità del Core."
        )
    elif resolved_id and resolved_id in avoid_map:
        av_row = avoid_map[resolved_id]
        strat_tier = "AVOID"
        strat_badge = "🚫 Quantile AVOID (Sopravvalutata vs Fondamentali)"
        strat_badge_color = "#f43f5e"
        res_val = av_row.get("residual")
        disc_val = av_row.get("discount_pct", 0.0)
        value_verdict = (
            f"Attenzione: il modello rileva che il prezzo attuale incorpora un forte premio speculativo "
            f"(+{abs(disc_val):.1f}% sopra i fondamentali, residuo ε={res_val:.2f}) rispetto a quanto giustificato dalla scarsità e tiratura. "
            f"Alto rischio di contrazione o sottoperformance rispetto ai peer."
        )
    elif resolved_id and resolved_id in meta:
        strat_tier = "NEUTRAL"
        strat_badge = "⚖️ Fascia Neutra (Fair Value di Mercato)"
        strat_badge_color = "#94a3b8"
        res_val = None
        disc_val = 0.0
        value_verdict = (
            "Il prezzo di mercato è perfettamente in linea con i fondamentali del modello edonico PokeQuant "
            "(scarsità oggettiva per box, anzianità temporale e serie storica). Nessuna anomalia o disallineamento statistico rilevato."
        )
    else:
        strat_tier = "CUSTOM"
        strat_badge = "✏️ Carta Custom / Fuori Catalogo Quantitativo"
        strat_badge_color = "#64748b"
        res_val = None
        disc_val = 0.0
        value_verdict = (
            "Carta personalizzata o fuori dal paniere dei 1.000+ asset monitorati in continuo. "
            "Il Fair Value è calcolato empiricamente tramite la matrice cross-sezionale delle case di gradazione."
        )

    return {
        "item_id": resolved_id,
        "card_name": final_card_name,
        "game_slug": final_game_slug,
        "item_slug": final_item_slug,
        "era": norm_era.value,
        # Pop report
        "has_pop_report": has_pop_report,
        "psa_census": {
            "10": psa_10,
            "9": psa_9,
            "8": psa_8,
            "7": psa_7,
            "total": psa_total,
        },
        "cgc_census": {
            "10": cgc_10,
            "9.5": cgc_95,
            "9": cgc_9,
            "8.5": cgc_85,
            "total": cgc_total,
        },
        "market_total": market_total,
        "gem_rate_psa": gem_rate_psa,
        "pop_pressure": pop_pressure,
        "pricecharting_pop_url": pc_pop_url,
        "psa_search_url": psa_search_url,
        "psa_portal_url": "https://www.psacard.com/pop",
        # Scarcity factor
        "rarity": final_rarity or "Standard",
        "rarity_tier_label": rarity_tier_label,
        "pull_rate_desc": pull_rate_desc,
        "release_year": rel_yr,
        "age_years": age_years,
        "supply_status": supply_status,
        "supply_elasticity": supply_elasticity,
        "census_verdict": census_verdict,
        # Value factor
        "strategy_tier": strat_tier,
        "strategy_badge": strat_badge,
        "strategy_badge_color": strat_badge_color,
        "residual": res_val,
        "discount_pct": disc_val,
        "value_verdict": value_verdict,
    }



