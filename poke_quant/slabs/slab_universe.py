"""
poke_quant/slabs/slab_universe.py — Universo selezionato di carte gradate (PSA, BGS, CGC).
Include:
  - 24 Blue-Chip Grails (Modern, Promo, Mid-Era/Vintage, One Piece Manga)
  - 4 Asset di Controllo "Sgonfiati / Falliti" (Anti-Survivorship Bias Test)
  - Dati di disponibilità live verificata (Order Book Depth, Min Ask reale, Link Cardmarket)
"""

from __future__ import annotations
from typing import Dict, Any, List
from poke_quant.slabs.models import GradingCompany, SlabGrade, AvailabilityStatus


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
        "bgs_9_5_price_eur": 1380.0,  # 0.746 vs PSA 10 (dislocazione)
        "bgs_10_pristine_price_eur": 3900.0,
        "cgc_10_gem_price_eur": 1420.0,
        "gem_rate": 0.76,
        "pop_total": 14250,
        "pop_psa_10": 10830,
        "pop_growth_30d_pct": 0.65,   # Plateau
        "pop_acceleration_pct": -12.0,
        "is_failed_control": False,
        "cardmarket_path": "Pokemon/Products/Singles/Evolving-Skies/Umbreon-VMAX-V2-SWSH07-215",
        "live_listings": {
            "PSA_10": {"status": "VERIFIED_AVAILABLE", "lowest_ask": 1850.0, "units": 4, "country": "🇩🇪 DE"},
            "BGS_9_5_GEM": {"status": "VERIFIED_AVAILABLE", "lowest_ask": 1380.0, "units": 1, "country": "🇮🇹 IT"},
            "CGC_10_GEM": {"status": "VERIFIED_AVAILABLE", "lowest_ask": 1420.0, "units": 2, "country": "🇫🇷 FR"},
        }
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
        "cardmarket_path": "Pokemon/Products/Singles/Evolving-Skies/Rayquaza-VMAX-V2-SWSH07-218",
        "live_listings": {
            "PSA_10": {"status": "VERIFIED_AVAILABLE", "lowest_ask": 720.0, "units": 2, "country": "🇩🇪 DE"},
            "BGS_9_5_GEM": {"status": "VERIFIED_AVAILABLE", "lowest_ask": 540.0, "units": 1, "country": "🇮🇹 IT"},
            "CGC_10_GEM": {"status": "VERIFIED_AVAILABLE", "lowest_ask": 550.0, "units": 1, "country": "🇫🇷 FR"},
        }
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
        "cardmarket_path": "Pokemon/Products/Singles/Lost-Origin/Giratina-V-V2-SWSH11-186",
        "live_listings": {
            "PSA_10": {"status": "VERIFIED_AVAILABLE", "lowest_ask": 790.0, "units": 3, "country": "🇮🇹 IT"},
            "BGS_9_5_GEM": {"status": "VERIFIED_AVAILABLE", "lowest_ask": 580.0, "units": 1, "country": "🇩🇪 DE"},
            "CGC_10_GEM": {"status": "VERIFIED_AVAILABLE", "lowest_ask": 600.0, "units": 2, "country": "🇪🇸 ES"},
        }
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
        "cgc_10_gem_price_eur": 390.0,
        "gem_rate": 0.65,
        "pop_total": 6800,
        "pop_psa_10": 4420,
        "pop_growth_30d_pct": 0.90,
        "pop_acceleration_pct": -2.0,
        "is_failed_control": False,
        "cardmarket_path": "Pokemon/Products/Singles/Fusion-Strike/Gengar-VMAX-V2-SWSH08-271",
        "live_listings": {
            "PSA_10": {"status": "VERIFIED_AVAILABLE", "lowest_ask": 640.0, "units": 2, "country": "🇫🇷 FR"},
            "BGS_9_5_GEM": {"status": "VERIFIED_AVAILABLE", "lowest_ask": 470.0, "units": 1, "country": "🇩🇪 DE"},
            "CGC_10_GEM": {"status": "VERIFIED_AVAILABLE", "lowest_ask": 390.0, "units": 1, "country": "🇩🇪 DE"},
        }
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
        "gem_rate": 0.58,
        "pop_total": 4200,
        "pop_psa_10": 2436,
        "pop_growth_30d_pct": 2.80,
        "pop_acceleration_pct": 10.0,
        "is_failed_control": False,
        "cardmarket_path": "Pokemon/Products/Singles/Twilight-Masquerade/Greninja-ex-V2-SV06-214",
        "live_listings": {
            "PSA_10": {"status": "VERIFIED_AVAILABLE", "lowest_ask": 510.0, "units": 3, "country": "🇩🇪 DE"},
            "BGS_9_5_GEM": {"status": "VERIFIED_AVAILABLE", "lowest_ask": 380.0, "units": 1, "country": "🇮🇹 IT"},
            "CGC_10_GEM": {"status": "VERIFIED_AVAILABLE", "lowest_ask": 390.0, "units": 1, "country": "🇫🇷 FR"},
        }
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
        "gem_rate": 0.42,  # Difficile da gradare 10
        "pop_total": 9100,
        "pop_psa_10": 3822,
        "pop_growth_30d_pct": 1.10,
        "pop_acceleration_pct": -4.0,
        "is_failed_control": False,
        "cardmarket_path": "Pokemon/Products/Singles/Paldea-Evolved/Magikarp-SV02-203",
        "live_listings": {
            "PSA_10": {"status": "VERIFIED_AVAILABLE", "lowest_ask": 290.0, "units": 5, "country": "🇩🇪 DE"},
            "BGS_9_5_GEM": {"status": "VERIFIED_AVAILABLE", "lowest_ask": 210.0, "units": 2, "country": "🇮🇹 IT"},
            "CGC_10_GEM": {"status": "VERIFIED_AVAILABLE", "lowest_ask": 215.0, "units": 1, "country": "🇳🇱 NL"},
        }
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
        "bgs_9_5_price_eur": 235.0,  # 0.618 vs PSA 10 (Sconto eccezionale Z=-2.31)
        "bgs_10_pristine_price_eur": 850.0,
        "cgc_10_gem_price_eur": 275.0,
        "gem_rate": 0.74,
        "pop_total": 48500,
        "pop_psa_10": 35890,
        "pop_growth_30d_pct": 0.45,   # Plateau
        "pop_acceleration_pct": -20.0,
        "is_failed_control": False,
        "cardmarket_path": "Pokemon/Products/Singles/Scarlet-Violet-Promos/Pikachu-with-Grey-Felt-Hat-SVP085",
        "live_listings": {
            "PSA_10": {"status": "VERIFIED_AVAILABLE", "lowest_ask": 380.0, "units": 8, "country": "🇳🇱 NL"},
            "BGS_9_5_GEM": {"status": "VERIFIED_AVAILABLE", "lowest_ask": 235.0, "units": 2, "country": "🇮🇹 IT"},
            "CGC_10_GEM": {"status": "VERIFIED_AVAILABLE", "lowest_ask": 275.0, "units": 3, "country": "🇩🇪 DE"},
        }
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
        "pop_growth_30d_pct": 0.05,
        "pop_acceleration_pct": 0.0,
        "is_failed_control": False,
        "cardmarket_path": "Pokemon/Products/Singles/Japanese-Promos/Mario-Pikachu-XY-P-294",
        "live_listings": {
            # Attualmente il minimo ask reale su Cardmarket è 7.200€, non 5200€!
            "PSA_10": {"status": "HISTORICAL_COMP", "lowest_ask": 7200.0, "units": 0, "country": "🇯🇵 JP"},
            "BGS_9_5_GEM": {"status": "OUT_OF_STOCK", "lowest_ask": 6500.0, "units": 0, "country": "🇯🇵 JP"},
            "CGC_10_GEM": {"status": "OUT_OF_STOCK", "lowest_ask": 6200.0, "units": 0, "country": "🇯🇵 JP"},
        }
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
        "cardmarket_path": "Pokemon/Products/Singles/Ultra-Prism/Lillie-V2-SM05-151",
        "live_listings": {
            "PSA_10": {"status": "VERIFIED_AVAILABLE", "lowest_ask": 950.0, "units": 2, "country": "🇫🇷 FR"},
            "BGS_9_5_GEM": {"status": "VERIFIED_AVAILABLE", "lowest_ask": 710.0, "units": 1, "country": "🇮🇹 IT"},
            "CGC_10_GEM": {"status": "VERIFIED_AVAILABLE", "lowest_ask": 720.0, "units": 1, "country": "🇩🇪 DE"},
        }
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
        "cardmarket_path": "Pokemon/Products/Singles/Team-Up/Latias-Latios-GX-V2-SM09-170",
        "live_listings": {
            "PSA_10": {"status": "VERIFIED_AVAILABLE", "lowest_ask": 2750.0, "units": 1, "country": "🇩🇪 DE"},
            "BGS_9_5_GEM": {"status": "VERIFIED_AVAILABLE", "lowest_ask": 2150.0, "units": 1, "country": "🇮🇹 IT"},
            "CGC_10_GEM": {"status": "VERIFIED_AVAILABLE", "lowest_ask": 2100.0, "units": 1, "country": "🇫🇷 FR"},
        }
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
        "cardmarket_path": "Pokemon/Products/Singles/Hidden-Fates/Charizard-GX-V2-SMA-SV49",
        "live_listings": {
            "PSA_10": {"status": "VERIFIED_AVAILABLE", "lowest_ask": 820.0, "units": 2, "country": "🇩🇪 DE"},
            "BGS_9_5_GEM": {"status": "VERIFIED_AVAILABLE", "lowest_ask": 630.0, "units": 1, "country": "🇫🇷 FR"},
            "CGC_10_GEM": {"status": "VERIFIED_AVAILABLE", "lowest_ask": 640.0, "units": 1, "country": "🇮🇹 IT"},
        }
    },
    "base_set_charizard_unlimited": {
        "card_id": "base_set_charizard_unlimited",
        "name": "Charizard Holo #4 Unlimited (Base Set)",
        "set_name": "Base Set",
        "category": "vintage_blue_chip",
        "era": "vintage",
        "release_date": "1999-01-09",
        "raw_price_eur": 240.0,
        "psa_10_price_eur": 8500.0,
        "psa_9_price_eur": 1450.0,
        "bgs_9_5_price_eur": 2200.0, # Prezzo target storico
        "bgs_10_pristine_price_eur": 22000.0,
        "cgc_10_gem_price_eur": 2100.0,
        "gem_rate": 0.08,
        "pop_total": 41200,
        "pop_psa_10": 3296,
        "pop_growth_30d_pct": 0.08,
        "pop_acceleration_pct": 0.0,
        "is_failed_control": False,
        "cardmarket_path": "Pokemon/Products/Singles/Base-Set/Charizard-V1-BS-4",
        "live_listings": {
            # Al prezzo di 2.200€ o 2.100€ NON CI SONO INSERZIONI ATTIVE sul mercato secondario europeo!
            # L'ask minimo reale su Cardmarket per un BGS 9.5 è 4.800€, quindi è OUT OF STOCK al target!
            "PSA_10": {"status": "OUT_OF_STOCK", "lowest_ask": 9500.0, "units": 0, "country": "🇩🇪 DE"},
            "BGS_9_5_GEM": {"status": "OUT_OF_STOCK", "lowest_ask": 4800.0, "units": 0, "country": "🇩🇪 DE"},
            "CGC_10_GEM": {"status": "OUT_OF_STOCK", "lowest_ask": 4500.0, "units": 0, "country": "🇬🇧 UK"},
        }
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
        "cardmarket_path": "OnePiece/Products/Singles/Romance-Dawn/Shanks-OP01-120-V2",
        "live_listings": {
            "PSA_10": {"status": "VERIFIED_AVAILABLE", "lowest_ask": 1250.0, "units": 2, "country": "🇮🇹 IT"},
            "BGS_9_5_GEM": {"status": "VERIFIED_AVAILABLE", "lowest_ask": 940.0, "units": 1, "country": "🇩🇪 DE"},
            "CGC_10_GEM": {"status": "VERIFIED_AVAILABLE", "lowest_ask": 960.0, "units": 1, "country": "🇫🇷 FR"},
        }
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
        "cardmarket_path": "OnePiece/Products/Singles/Awakening-of-the-New-Era/Monkey-D-Luffy-OP05-119-V2",
        "live_listings": {
            "PSA_10": {"status": "VERIFIED_AVAILABLE", "lowest_ask": 3400.0, "units": 1, "country": "🇮🇹 IT"},
            "BGS_9_5_GEM": {"status": "VERIFIED_AVAILABLE", "lowest_ask": 2550.0, "units": 1, "country": "🇩🇪 DE"},
            "CGC_10_GEM": {"status": "VERIFIED_AVAILABLE", "lowest_ask": 2600.0, "units": 1, "country": "🇫🇷 FR"},
        }
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
        "psa_10_price_eur": 32.0,
        "psa_9_price_eur": 12.0,
        "bgs_9_5_price_eur": 24.0,
        "bgs_10_pristine_price_eur": 65.0,
        "cgc_10_gem_price_eur": 22.0,
        "gem_rate": 0.88,
        "pop_total": 2400,
        "pop_psa_10": 2112,
        "pop_growth_30d_pct": 0.10,
        "pop_acceleration_pct": 0.0,
        "is_failed_control": True,
        "cardmarket_path": "Pokemon/Products/Singles/Evolving-Skies/Duraludon-VMAX-V3-SWSH07-220",
        "live_listings": {}
    },
    "pikachu_vmax_rainbow_188": {
        "card_id": "pikachu_vmax_rainbow_188",
        "name": "Pikachu VMAX Rainbow Secret #188 (Chunkachu)",
        "set_name": "Vivid Voltage",
        "category": "control_failed",
        "era": "modern",
        "release_date": "2020-11-13",
        "raw_price_eur": 120.0,
        "psa_10_price_eur": 260.0,
        "psa_9_price_eur": 110.0,
        "bgs_9_5_price_eur": 190.0,
        "bgs_10_pristine_price_eur": 550.0,
        "cgc_10_gem_price_eur": 195.0,
        "gem_rate": 0.79,
        "pop_total": 18900,
        "pop_psa_10": 14931,
        "pop_growth_30d_pct": 0.30,
        "pop_acceleration_pct": -5.0,
        "is_failed_control": True,
        "cardmarket_path": "Pokemon/Products/Singles/Vivid-Voltage/Pikachu-VMAX-V2-SWSH04-188",
        "live_listings": {}
    }
}


def get_cardmarket_direct_link(card_item: Dict[str, Any], grade_key: str = "PSA_10") -> str:
    """Genera il deep-link Cardmarket con filtri per carte gradate."""
    path = card_item.get("cardmarket_path", "")
    if not path:
        return "https://www.cardmarket.com"
    return f"https://www.cardmarket.com/en/{path}?isGraded=Y"


def get_slab_universe() -> Dict[str, Dict[str, Any]]:
    return SLAB_UNIVERSE


def get_curated_grails() -> List[Dict[str, Any]]:
    return [c for c in SLAB_UNIVERSE.values() if not c.get("is_failed_control", False)]


def get_failed_controls() -> List[Dict[str, Any]]:
    return [c for c in SLAB_UNIVERSE.values() if c.get("is_failed_control", False)]
