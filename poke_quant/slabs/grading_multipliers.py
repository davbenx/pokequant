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
from dataclasses import dataclass
from enum import Enum
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


def get_variant_multiplier(variant: str) -> Tuple[float, str]:
    """Ritorna (moltiplicatore, descrizione) per la variante richiesta."""
    v = str(variant).strip().lower()
    if "base" in v and ("1" in v or "first" in v):
        return SPECIAL_VARIANTS["first_edition_base"]
    if "1" in v or "first" in v:
        return SPECIAL_VARIANTS["first_edition_wotc"]
    if "no_symbol" in v or "no symbol" in v or "senza logo" in v:
        return SPECIAL_VARIANTS["no_symbol"]
    if "shadowless" in v:
        return SPECIAL_VARIANTS["shadowless"]
    if "reverse" in v or "firework" in v:
        return SPECIAL_VARIANTS["reverse_holo_lc"]
    return SPECIAL_VARIANTS["standard"]


# =============================================================================
# MATRICE CALIBRATA DEI MOLTIPLICATORI EMPIRICI (scripts/grading_company_multiplier_research.py)
# =============================================================================

# Formato: (Company, Grade_Key, Era) -> (Multiplier vs Ref, Liquidity Penalty %, Sniper Factor)
# Dove:
#   - Se Grade_Key appartiene alla famiglia 9 (9.0, 9.5), il Benchmark di Riferimento è PSA 9 = 1.00x
#   - Se Grade_Key appartiene alla famiglia 10, il Benchmark di Riferimento è PSA 10 = 1.00x

EMPIRICAL_RATIOS_GRADE9: Dict[Tuple[GradingCompany, str, Era], Tuple[float, float, float, str]] = {
    # --- PSA (Benchmark) ---
    (GradingCompany.PSA, "9.0", Era.VINTAGE): (1.000, 0.0, 1.000, "Benchmark base del modello"),
    (GradingCompany.PSA, "9.0", Era.MID_ERA): (1.000, 0.0, 1.000, "Benchmark base del modello"),
    (GradingCompany.PSA, "9.0", Era.MODERN): (1.000, 0.0, 1.000, "Benchmark base del modello"),

    # --- BGS (Beckett) 9.5 Gem Mint ---
    # Posizionato tra PSA 9 e PSA 10: forte premio nel Vintage, premio su Modern
    (GradingCompany.BGS, "9.5", Era.VINTAGE): (1.809, 0.0, 1.250, "BGS 9.5 Gem Mint vintage: premio forte vs PSA 9"),
    (GradingCompany.BGS, "9.5", Era.MID_ERA): (1.552, 0.0, 1.200, "BGS 9.5 Gem Mint mid-era: premio strutturale"),
    (GradingCompany.BGS, "9.5", Era.MODERN): (1.611, 0.0, 1.150, "BGS 9.5 Gem Mint moderno: posizionato a 0.65x di PSA 10"),

    # --- BGS (Beckett) 9.0 Mint ---
    (GradingCompany.BGS, "9.0", Era.VINTAGE): (0.884, 5.0, 0.900, "BGS 9.0 Mint vintage: lieve sconto di liquidità vs PSA 9"),
    (GradingCompany.BGS, "9.0", Era.MID_ERA): (0.887, 5.0, 0.900, "BGS 9.0 Mint mid-era"),
    (GradingCompany.BGS, "9.0", Era.MODERN): (0.903, 3.0, 0.920, "BGS 9.0 Mint moderno"),

    # --- CGC Cards 9.0 Mint ---
    (GradingCompany.CGC, "9.0", Era.VINTAGE): (0.901, 7.0, 0.920, "CGC 9.0 Mint vintage: sconto liquidità medio 8-10%"),
    (GradingCompany.CGC, "9.0", Era.MID_ERA): (0.913, 6.0, 0.930, "CGC 9.0 Mint mid-era"),
    (GradingCompany.CGC, "9.0", Era.MODERN): (0.931, 5.0, 0.940, "CGC 9.0 Mint moderno: molto vicino a PSA 9"),

    # --- CGC Cards 9.5 Gem Mint (Old Blue Label / Pristine conversion) ---
    (GradingCompany.CGC, "9.5", Era.VINTAGE): (1.350, 0.0, 1.150, "CGC 9.5 Gem Mint vintage"),
    (GradingCompany.CGC, "9.5", Era.MID_ERA): (1.250, 0.0, 1.120, "CGC 9.5 Gem Mint mid-era"),
    (GradingCompany.CGC, "9.5", Era.MODERN): (1.180, 0.0, 1.100, "CGC 9.5 Gem Mint moderno"),

    # --- TAG Grading (USA, Digital 1000 DPI) 9.0 Mint ---
    (GradingCompany.TAG, "9.0", Era.VINTAGE): (0.800, 15.0, 0.830, "TAG 9.0 vintage: diffidenza collezionisti storici"),
    (GradingCompany.TAG, "9.0", Era.MID_ERA): (0.840, 12.0, 0.860, "TAG 9.0 mid-era: community emergente"),
    (GradingCompany.TAG, "9.0", Era.MODERN): (0.880, 8.0, 0.900, "TAG 9.0 moderno: forte interesse tech-oriented USA"),

    # --- SGC (Tuxedo) 9.0 Mint ---
    (GradingCompany.SGC, "9.0", Era.VINTAGE): (0.785, 15.0, 0.820, "SGC 9.0 Mint vintage: sconto estero significativo"),
    (GradingCompany.SGC, "9.0", Era.MID_ERA): (0.800, 14.0, 0.830, "SGC 9.0 Mint mid-era"),
    (GradingCompany.SGC, "9.0", Era.MODERN): (0.836, 12.0, 0.850, "SGC 9.0 Mint moderno"),

    # --- GRAAD (Italia) 9.0 Mint ---
    # Mercato locale IT ~0.80x, ma su liquidità internazionale (Cardmarket EU / eBay) sconta forte penalità
    (GradingCompany.GRAAD, "9.0", Era.VINTAGE): (0.681, 28.0, 0.720, "GRAAD 9.0 vintage: sconto pesante fuori dall'Italia"),
    (GradingCompany.GRAAD, "9.0", Era.MID_ERA): (0.711, 25.0, 0.740, "GRAAD 9.0 mid-era"),
    (GradingCompany.GRAAD, "9.0", Era.MODERN): (0.750, 20.0, 0.770, "GRAAD 9.0 moderno"),

    # --- AiGrading (Italia) 9.0 Mint ---
    (GradingCompany.AIGRADING, "9.0", Era.VINTAGE): (0.680, 28.0, 0.710, "AiGrading 9.0 vintage: ente regionale italiano (stile GRAAD)"),
    (GradingCompany.AIGRADING, "9.0", Era.MID_ERA): (0.700, 25.0, 0.730, "AiGrading 9.0 mid-era"),
    (GradingCompany.AIGRADING, "9.0", Era.MODERN): (0.740, 20.0, 0.760, "AiGrading 9.0 moderno: liquidità limitata all'Italia"),

    # --- PCA (Francia) 9.0 Mint ---
    (GradingCompany.PCA, "9.0", Era.VINTAGE): (0.726, 25.0, 0.740, "PCA 9.0 vintage: liquido principalmente in Francia"),
    (GradingCompany.PCA, "9.0", Era.MID_ERA): (0.737, 23.0, 0.750, "PCA 9.0 mid-era"),
    (GradingCompany.PCA, "9.0", Era.MODERN): (0.776, 18.0, 0.780, "PCA 9.0 moderno"),

    # --- CCC (Classic Card Collector, DACH) 9.0 Mint ---
    (GradingCompany.CCC, "9.0", Era.VINTAGE): (0.690, 28.0, 0.720, "CCC 9.0 vintage: ente regionale DACH/Germania"),
    (GradingCompany.CCC, "9.0", Era.MID_ERA): (0.710, 25.0, 0.740, "CCC 9.0 mid-era"),
    (GradingCompany.CCC, "9.0", Era.MODERN): (0.750, 20.0, 0.770, "CCC 9.0 moderno: scambi prevalentemente Germania/Austria"),

    # --- ACE Grading (UK) 9.0 Mint ---
    (GradingCompany.ACE, "9.0", Era.VINTAGE): (0.740, 22.0, 0.760, "ACE 9.0 vintage: mercato prevalentemente UK"),
    (GradingCompany.ACE, "9.0", Era.MID_ERA): (0.760, 20.0, 0.780, "ACE 9.0 mid-era"),
    (GradingCompany.ACE, "9.0", Era.MODERN): (0.800, 16.0, 0.810, "ACE 9.0 moderno"),
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


def get_grading_adjustment(
    company: str | GradingCompany,
    grade: str | float,
    era: str | Era = Era.MODERN,
    subgrades_black_label: bool = False,
    is_pristine: bool = False,
) -> GradingAdjustment:
    """
    Ritorna la rettifica quantitativa calibrata per una combinazione (Compagnia, Grado, Era).
    """
    comp_enum = normalize_company(company)
    era_enum = normalize_era(era)
    
    # Formattazione chiave grado
    g_str = str(grade).strip().lower()
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
            # Fallback generico per grado 10
            lookup = EMPIRICAL_RATIOS_GRADE10.get((comp_enum, "10.0", era_enum), (0.70, 20.0, 0.75, "Fallback Grade 10"))
    else:
        benchmark_ref = "PSA_9"
        if "9.5" in g_str:
            grade_key = "9.5"
        else:
            grade_key = "9.0"
        
        lookup = EMPIRICAL_RATIOS_GRADE9.get((comp_enum, grade_key, era_enum))
        if not lookup:
            # Fallback generico per grado 9
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


def adjust_price_for_grading(
    base_psa_price_eur: float,
    company: str | GradingCompany,
    grade: str | float,
    era: str | Era = Era.MODERN,
    subgrades_black_label: bool = False,
    is_pristine: bool = False,
    variant: str = "standard",
) -> Tuple[float, float, GradingAdjustment]:
    """
    Ricalibra un prezzo benchmark PSA (Grado 9 o Grado 10) per la compagnia e l'eventuale variante speciale desiderata.
    
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
    fair_value = effective_base_psa * adj.multiplier
    # Tetto sniper: applica il fattore prudenziale per preservare l'edge ed evitare overpaying
    sniper_ceiling = effective_base_psa * adj.sniper_ceiling_factor
    return round(fair_value, 2), round(sniper_ceiling, 2), adj
