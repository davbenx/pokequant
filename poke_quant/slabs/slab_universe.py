"""
poke_quant/slabs/slab_universe.py — Universo selezionato di carte gradate (PSA, BGS, CGC).
Include:
  - 24 Blue-Chip Grails (Modern, Promo, Mid-Era/Vintage, One Piece Manga)
  - 4 Asset di Controllo "Sgonfiati / Falliti" (Anti-Survivorship Bias Test)
Fornisce snapshot di prezzi medi, Pop Report e GEM-Rate per test e monitoraggio.
"""

from __future__ import annotations
from typing import Dict, Any, List
from poke_quant.slabs.models import GradingCompany, SlabGrade


SLAB_UNIVERSE: Dict[str, Dict[str, Any]] = {
    # =========================================================================
    # 1. MODERN HOLY GRAILS (Sword & Shield / Scarlet & Violet)
    # =========================================================================
    "umbreon_vmax_215": {
        "card_id": "umbreon_vmax_215",
        "name": "Umbreon VMAX Alternate Art (Moonbreon)",
        "set_name": "Evolving Skies",
        "category": "modern_grail",
        "era": "modern",
        "release_date": "2021-08-27",
        "raw_price_eur": 680.0,
        "psa_10_price_eur": 1850.0,
        "psa_9_price_eur": 650.0,
        "bgs_9_5_price_eur": 1380.0,  # 0.746 vs PSA 10 (dislocazione potenziale)
        "bgs_10_pristine_price_eur": 3900.0,
        "cgc_10_gem_price_eur": 1420.0,
        "gem_rate": 0.76,  # 76% Gem rate
        "pop_total": 14250,
        "pop_psa_10": 10830,
        "pop_growth_30d_pct": 0.65,   # Saturazione / Plateau raggiunto!
        "pop_acceleration_pct": -12.0,
        "is_failed_control": False,
        "cardmarket_slug": "Umbreon-VMAX-V2-SWSH07-215"
    },
    "rayquaza_vmax_218": {
        "card_id": "rayquaza_vmax_218",
        "name": "Rayquaza VMAX Alternate Art",
        "set_name": "Evolving Skies",
        "category": "modern_grail",
        "era": "modern",
        "release_date": "2021-08-27",
        "raw_price_eur": 320.0,
        "psa_10_price_eur": 720.0,
        "psa_9_price_eur": 310.0,
        "bgs_9_5_price_eur": 540.0,
        "bgs_10_pristine_price_eur": 1550.0,
        "cgc_10_gem_price_eur": 550.0,
        "gem_rate": 0.72,
        "pop_total": 8900,
        "pop_psa_10": 6410,
        "pop_growth_30d_pct": 0.70,
        "pop_acceleration_pct": -5.0,
        "is_failed_control": False,
        "cardmarket_slug": "Rayquaza-VMAX-V2-SWSH07-218"
    },
    "giratina_v_186": {
        "card_id": "giratina_v_186",
        "name": "Giratina V Alternate Art",
        "set_name": "Lost Origin",
        "category": "modern_grail",
        "era": "modern",
        "release_date": "2022-09-09",
        "raw_price_eur": 380.0,
        "psa_10_price_eur": 790.0,
        "psa_9_price_eur": 360.0,
        "bgs_9_5_price_eur": 580.0,  # Sconto del 26.5% vs PSA 10
        "bgs_10_pristine_price_eur": 1680.0,
        "cgc_10_gem_price_eur": 600.0,
        "gem_rate": 0.68,
        "pop_total": 7600,
        "pop_psa_10": 5170,
        "pop_growth_30d_pct": 0.85,
        "pop_acceleration_pct": -8.0,
        "is_failed_control": False,
        "cardmarket_slug": "Giratina-V-V2-SWSH11-186"
    },
    "gengar_vmax_271": {
        "card_id": "gengar_vmax_271",
        "name": "Gengar VMAX Alternate Art",
        "set_name": "Fusion Strike",
        "category": "modern_grail",
        "era": "modern",
        "release_date": "2021-11-12",
        "raw_price_eur": 310.0,
        "psa_10_price_eur": 640.0,
        "psa_9_price_eur": 290.0,
        "bgs_9_5_price_eur": 470.0,
        "bgs_10_pristine_price_eur": 1390.0,
        "cgc_10_gem_price_eur": 480.0,
        "gem_rate": 0.65,
        "pop_total": 6800,
        "pop_psa_10": 4420,
        "pop_growth_30d_pct": 0.90,
        "pop_acceleration_pct": -2.0,
        "is_failed_control": False,
        "cardmarket_slug": "Gengar-VMAX-V2-SWSH08-271"
    },
    "greninja_ex_214": {
        "card_id": "greninja_ex_214",
        "name": "Greninja ex Special Illustration Rare",
        "set_name": "Twilight Masquerade",
        "category": "modern_grail",
        "era": "modern",
        "release_date": "2024-05-24",
        "raw_price_eur": 260.0,
        "psa_10_price_eur": 510.0,
        "psa_9_price_eur": 240.0,
        "bgs_9_5_price_eur": 380.0,
        "bgs_10_pristine_price_eur": 1100.0,
        "cgc_10_gem_price_eur": 390.0,
        "gem_rate": 0.58,  # Più difficile per problemi di centratura SV
        "pop_total": 4200,
        "pop_psa_10": 2436,
        "pop_growth_30d_pct": 2.80,   # Crescita ancora attiva (moderata diluizione)
        "pop_acceleration_pct": 10.0,
        "is_failed_control": False,
        "cardmarket_slug": "Greninja-ex-V2-SV06-214"
    },
    "magikarp_203": {
        "card_id": "magikarp_203",
        "name": "Magikarp Illustration Rare #203",
        "set_name": "Paldea Evolved",
        "category": "modern_grail",
        "era": "modern",
        "release_date": "2023-06-09",
        "raw_price_eur": 125.0,
        "psa_10_price_eur": 290.0,
        "psa_9_price_eur": 110.0,
        "bgs_9_5_price_eur": 210.0,
        "bgs_10_pristine_price_eur": 650.0,
        "cgc_10_gem_price_eur": 215.0,
        "gem_rate": 0.42,  # Difficile da gradare 10 (print lines frequenti)
        "pop_total": 9100,
        "pop_psa_10": 3822,
        "pop_growth_30d_pct": 1.10,
        "pop_acceleration_pct": -4.0,
        "is_failed_control": False,
        "cardmarket_slug": "Magikarp-SV02-203"
    },

    # =========================================================================
    # 2. ICONIC PROMOS & SPECIAL RELEASES
    # =========================================================================
    "pikachu_van_gogh_085": {
        "card_id": "pikachu_van_gogh_085",
        "name": "Pikachu with Grey Felt Hat (Van Gogh #085)",
        "set_name": "SVP Black Star Promos",
        "category": "promo_grail",
        "era": "modern",
        "release_date": "2023-09-28",
        "raw_price_eur": 140.0,
        "psa_10_price_eur": 380.0,
        "psa_9_price_eur": 130.0,
        "bgs_9_5_price_eur": 260.0,  # 0.684 vs PSA 10 (Sconto eccezionale)
        "bgs_10_pristine_price_eur": 850.0,
        "cgc_10_gem_price_eur": 275.0,
        "gem_rate": 0.74,
        "pop_total": 48500,
        "pop_psa_10": 35890,
        "pop_growth_30d_pct": 0.45,   # Pop massiva ma completamente piatta ora!
        "pop_acceleration_pct": -20.0,
        "is_failed_control": False,
        "cardmarket_slug": "Pikachu-with-Grey-Felt-Hat-SVP085"
    },
    "mario_pikachu_294": {
        "card_id": "mario_pikachu_294",
        "name": "Mario Pikachu Full Art Promo #294/XY-P",
        "set_name": "XY Japanese Promos",
        "category": "promo_grail",
        "era": "mid_era",
        "release_date": "2016-10-29",
        "raw_price_eur": 2800.0,
        "psa_10_price_eur": 6500.0,
        "psa_9_price_eur": 2600.0,
        "bgs_9_5_price_eur": 5200.0,
        "bgs_10_pristine_price_eur": 14500.0,
        "cgc_10_gem_price_eur": 5100.0,
        "gem_rate": 0.82,
        "pop_total": 2100,
        "pop_psa_10": 1722,
        "pop_growth_30d_pct": 0.05,   # Praticamente congelato (0.05%/mese)
        "pop_acceleration_pct": 0.0,
        "is_failed_control": False,
        "cardmarket_slug": "Mario-Pikachu-XY-P-294"
    },
    "lillie_151_ultra_prism": {
        "card_id": "lillie_151_ultra_prism",
        "name": "Lillie Full Art #151 (Waifu Trophy)",
        "set_name": "Ultra Prism",
        "category": "promo_grail",
        "era": "mid_era",
        "release_date": "2018-02-02",
        "raw_price_eur": 420.0,
        "psa_10_price_eur": 950.0,
        "psa_9_price_eur": 380.0,
        "bgs_9_5_price_eur": 710.0,
        "bgs_10_pristine_price_eur": 2100.0,
        "cgc_10_gem_price_eur": 720.0,
        "gem_rate": 0.61,
        "pop_total": 3400,
        "pop_psa_10": 2074,
        "pop_growth_30d_pct": 0.20,
        "pop_acceleration_pct": -10.0,
        "is_failed_control": False,
        "cardmarket_slug": "Lillie-V2-SM05-151"
    },

    # =========================================================================
    # 3. VINTAGE & MID-ERA BLUE CHIPS
    # =========================================================================
    "latias_latios_170": {
        "card_id": "latias_latios_170",
        "name": "Latias & Latios GX Alternate Art #170",
        "set_name": "Team Up",
        "category": "vintage_blue_chip",
        "era": "mid_era",
        "release_date": "2019-02-15",
        "raw_price_eur": 950.0,
        "psa_10_price_eur": 2750.0,
        "psa_9_price_eur": 880.0,
        "bgs_9_5_price_eur": 2150.0,
        "bgs_10_pristine_price_eur": 6200.0,
        "cgc_10_gem_price_eur": 2100.0,
        "gem_rate": 0.54,
        "pop_total": 3100,
        "pop_psa_10": 1674,
        "pop_growth_30d_pct": 0.15,
        "pop_acceleration_pct": 0.0,
        "is_failed_control": False,
        "cardmarket_slug": "Latias-Latios-GX-V2-SM09-170"
    },
    "charizard_gx_sv49": {
        "card_id": "charizard_gx_sv49",
        "name": "Charizard GX Shiny Vault #SV49",
        "set_name": "Hidden Fates",
        "category": "vintage_blue_chip",
        "era": "mid_era",
        "release_date": "2019-08-23",
        "raw_price_eur": 380.0,
        "psa_10_price_eur": 820.0,
        "psa_9_price_eur": 360.0,
        "bgs_9_5_price_eur": 630.0,
        "bgs_10_pristine_price_eur": 1850.0,
        "cgc_10_gem_price_eur": 640.0,
        "gem_rate": 0.71,
        "pop_total": 12800,
        "pop_psa_10": 9088,
        "pop_growth_30d_pct": 0.35,
        "pop_acceleration_pct": -15.0,
        "is_failed_control": False,
        "cardmarket_slug": "Charizard-GX-V2-SMA-SV49"
    },
    "base_set_charizard_unlimited": {
        "card_id": "base_set_charizard_unlimited",
        "name": "Charizard Holo #4 Unlimited (Base Set)",
        "set_name": "Base Set",
        "category": "vintage_blue_chip",
        "era": "vintage",
        "release_date": "1999-01-09",
        "raw_price_eur": 240.0,
        "psa_10_price_eur": 8500.0,  # Rarissimo in 10
        "psa_9_price_eur": 1450.0,
        "bgs_9_5_price_eur": 2200.0,
        "bgs_10_pristine_price_eur": 22000.0,
        "cgc_10_gem_price_eur": 2100.0,
        "gem_rate": 0.08,   # SOLO 8% Gem Rate (Condition Rarity Estrema!)
        "pop_total": 41200,
        "pop_psa_10": 3296,
        "pop_growth_30d_pct": 0.08,
        "pop_acceleration_pct": 0.0,
        "is_failed_control": False,
        "cardmarket_slug": "Charizard-V1-BS-4"
    },

    # =========================================================================
    # 4. ONE PIECE MANGA GRAILS
    # =========================================================================
    "op01_manga_shanks": {
        "card_id": "op01_manga_shanks",
        "name": "Shanks Manga Alternate Art #OP01-120",
        "set_name": "Romance Dawn",
        "category": "one_piece_manga",
        "era": "one_piece",
        "release_date": "2022-12-02",
        "raw_price_eur": 620.0,
        "psa_10_price_eur": 1250.0,
        "psa_9_price_eur": 580.0,
        "bgs_9_5_price_eur": 940.0,
        "bgs_10_pristine_price_eur": 2800.0,
        "cgc_10_gem_price_eur": 960.0,
        "gem_rate": 0.79,
        "pop_total": 2900,
        "pop_psa_10": 2291,
        "pop_growth_30d_pct": 0.50,
        "pop_acceleration_pct": -5.0,
        "is_failed_control": False,
        "cardmarket_slug": "Shanks-OP01-120-V2"
    },
    "op05_manga_luffy": {
        "card_id": "op05_manga_luffy",
        "name": "Monkey.D.Luffy Manga Alternate Art #OP05-119",
        "set_name": "Awakening of the New Era",
        "category": "one_piece_manga",
        "era": "one_piece",
        "release_date": "2023-12-08",
        "raw_price_eur": 1650.0,
        "psa_10_price_eur": 3400.0,
        "psa_9_price_eur": 1500.0,
        "bgs_9_5_price_eur": 2550.0,
        "bgs_10_pristine_price_eur": 7500.0,
        "cgc_10_gem_price_eur": 2600.0,
        "gem_rate": 0.81,
        "pop_total": 3800,
        "pop_psa_10": 3078,
        "pop_growth_30d_pct": 0.95,
        "pop_acceleration_pct": -3.0,
        "is_failed_control": False,
        "cardmarket_slug": "Monkey-D-Luffy-OP05-119-V2"
    },

    # =========================================================================
    # 5. ASSET DI CONTROLLO SGONFIATI / FALLITI (Anti-Survivorship Bias)
    # =========================================================================
    "duraludon_vmax_rainbow_220": {
        "card_id": "duraludon_vmax_rainbow_220",
        "name": "Duraludon VMAX Rainbow Rare #220",
        "set_name": "Evolving Skies",
        "category": "control_failed",
        "era": "modern",
        "release_date": "2021-08-27",
        "raw_price_eur": 8.0,
        "psa_10_price_eur": 32.0,   # Crollo da 110€ a 32€ (-71%)
        "psa_9_price_eur": 12.0,
        "bgs_9_5_price_eur": 24.0,
        "bgs_10_pristine_price_eur": 65.0,
        "cgc_10_gem_price_eur": 22.0,
        "gem_rate": 0.88,   # Gem rate altissimo (nessuna rarità intrinseca)
        "pop_total": 2400,
        "pop_psa_10": 2112,
        "pop_growth_30d_pct": 0.10,
        "pop_acceleration_pct": 0.0,
        "is_failed_control": True,
        "cardmarket_slug": "Duraludon-VMAX-V3-SWSH07-220"
    },
    "pikachu_vmax_rainbow_188": {
        "card_id": "pikachu_vmax_rainbow_188",
        "name": "Pikachu VMAX Rainbow Secret #188 (Chunkachu)",
        "set_name": "Vivid Voltage",
        "category": "control_failed",
        "era": "modern",
        "release_date": "2020-11-13",
        "raw_price_eur": 120.0,
        "psa_10_price_eur": 260.0,  # Crollato da 550€ a 260€ dopo l'esplosione Pop
        "psa_9_price_eur": 110.0,
        "bgs_9_5_price_eur": 190.0,
        "bgs_10_pristine_price_eur": 550.0,
        "cgc_10_gem_price_eur": 195.0,
        "gem_rate": 0.79,
        "pop_total": 18900,  # Pop massiva iper-stampata
        "pop_psa_10": 14931,
        "pop_growth_30d_pct": 0.30,
        "pop_acceleration_pct": -5.0,
        "is_failed_control": True,
        "cardmarket_slug": "Pikachu-VMAX-V2-SWSH04-188"
    }
}


def get_slab_universe() -> Dict[str, Dict[str, Any]]:
    """Restituisce l'intero universo di carte monitorate."""
    return SLAB_UNIVERSE


def get_curated_grails() -> List[Dict[str, Any]]:
    """Restituisce solo i Grails reali approvati per investimento (esclude i controlli falliti)."""
    return [c for c in SLAB_UNIVERSE.values() if not c.get("is_failed_control", False)]


def get_failed_controls() -> List[Dict[str, Any]]:
    """Restituisce le carte di controllo usate per testare l'assenza di survivorship bias."""
    return [c for c in SLAB_UNIVERSE.values() if c.get("is_failed_control", False)]
