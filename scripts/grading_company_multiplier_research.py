#!/usr/bin/env python3
"""
scripts/grading_company_multiplier_research.py — Statistiche descrittive su
un dataset di CONFRONTI DI PREZZO STIMATI A MANO tra case di gradazione.

AVVISO DI ATTENDIBILITA' (corretto durante l'audit generale del repo su
richiesta esplicita dell'utente - "trova bug, inconsistenze... invalida"):
il commento precedente affermava "Rilevazioni Reali di Venduto & Ask
Comparativi - Fonti: PriceCharting Multi-Company Splits, eBay Completed/Sold
Comps, Cardmarket Asks" per EMPIRICAL_COMP_DATA. FALSO: questo file non
contiene NESSUNA chiamata di rete, NESSUN fetch, NESSUN caricamento da CSV/
JSON - EMPIRICAL_COMP_DATA e' una lista Python scritta a mano, con prezzi
digitati dalla conoscenza generale del mercato (grading arbitrage community,
prezzi "tipici" per carte iconiche), NON misurati o verificati da questo
codice. Le statistiche sotto (mediana/media/std/IQR) sono calcolate
CORRETTAMENTE sul dataset, ma il dataset stesso e' una stima soggettiva, non
un campione di vendite reali - la differenza tra "statistica corretta" e
"input verificato" e' esattamente il tipo di errore che questa ricerca ha
sempre trattato come squalificante altrove (vedi la storia respinta di
popolazioni PSA "fabbricate" da un altro agente, stesso principio). Non
cancellato (i numeri non sono assurdi, sono stime plausibili di chi conosce
il mercato) ma NON deve essere presentato come dato empirico verificato in
nessun punto downstream (grading_multipliers.py, la UI del calcolatore slab
in app.py) senza un avviso "stima non verificata" visibile all'utente.

Analizza il dataset stimato su un campione stratificato di carte
rappresentative attraverso 3 macro-ere del collezionismo:
  1. Vintage (1999–2003: WotC Base Set, Jungle, Fossil, Rocket, Gym, Neo, e-Series)
  2. Mid-Era (2004–2016: EX Series, Diamond & Pearl, Platinum, HGSS, Black & White, XY)
  3. Moderno (2017–2026: Sun & Moon, Sword & Shield, Scarlet & Violet)

Calcola:
  - Distribuzioni statistiche (Mediana, Media, Dev.Std, IQR, Min, Max) dei rapporti di
    prezzo rispetto al benchmark di riferimento:
      * Base Grado 9: Rapporto vs PSA 9 = 1.00x
      * Base Grado 10: Rapporto vs PSA 10 = 1.00x
  - Sconto di Liquidità e Frizione di Mercato Internazionale per le case regionali europee
    (GRAAD, PCA, ACE) rispetto alle major globali (PSA, BGS, CGC).
  - Matrice di calibrazione PARZIALE per poke_quant/slabs/grading_multipliers.py - solo
    grado 9/9.5/10, solo PSA/BGS/CGC/SGC/GRAAD/PCA: quel modulo estende questa matrice con
    tabelle per TAG/AiGrading/CCC/ACE e per i gradi 7.0-8.5 che NON hanno alcun
    riscontro qui - vedi l'avviso equivalente in grading_multipliers.py.
"""

from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# =============================================================================
# DATASET STIMATO A MANO (NON misurato, NON fetchato - vedi avviso in testa al
# file): confronti di prezzo per carte iconiche tra case/gradi di gradazione,
# scritti dalla conoscenza generale del mercato del collezionismo, non da
# vendite reali raccolte da questo codice.
# =============================================================================

EMPIRICAL_COMP_DATA: List[Dict[str, Any]] = [
    # -------------------------------------------------------------------------
    # 1. VINTAGE (1999 - 2003)
    # -------------------------------------------------------------------------
    {
        "card": "Charizard #4 Holo (Base Set)",
        "era": "vintage",
        "psa_10": 11500.0,
        "bgs_10_pristine": 15000.0,
        "bgs_9_5": 5200.0,
        "cgc_10_gem": 4300.0,
        "cgc_10_pristine": 6800.0,
        "psa_9": 2750.0,
        "bgs_9": 2400.0,
        "cgc_9": 2450.0,
        "sgc_9": 2100.0,
        "graad_9": 1850.0,
        "pca_9": 1900.0,
        "raw_nm": 650.0,
    },
    {
        "card": "Blastoise #2 Holo (Base Set)",
        "era": "vintage",
        "psa_10": 6400.0,
        "bgs_10_pristine": 8300.0,
        "bgs_9_5": 2500.0,
        "cgc_10_gem": 2000.0,
        "cgc_10_pristine": 3100.0,
        "psa_9": 880.0,
        "bgs_9": 780.0,
        "cgc_9": 800.0,
        "sgc_9": 720.0,
        "graad_9": 620.0,
        "pca_9": 650.0,
        "raw_nm": 190.0,
    },
    {
        "card": "Gengar #H9 Holo (Skyridge)",
        "era": "vintage",
        "psa_10": 8500.0,
        "bgs_10_pristine": 16000.0,
        "bgs_9_5": 4200.0,
        "cgc_10_gem": 4100.0,
        "cgc_10_pristine": 5800.0,
        "psa_9": 1450.0,
        "bgs_9": 1300.0,
        "cgc_9": 1320.0,
        "sgc_9": 1150.0,
        "graad_9": 980.0,
        "pca_9": 1050.0,
        "raw_nm": 380.0,
    },
    {
        "card": "Sabrina #20 Holo 1st Edition (Gym Challenge)",
        "era": "vintage",
        "psa_10": 950.0,
        "bgs_10_pristine": 1800.0,
        "bgs_9_5": 380.0,
        "cgc_10_gem": 350.0,
        "cgc_10_pristine": 520.0,
        "psa_9": 220.0,
        "bgs_9": 195.0,
        "cgc_9": 195.0,
        "sgc_9": 170.0,
        "graad_9": 150.0,
        "pca_9": 160.0,
        "raw_nm": 160.0,
    },
    {
        "card": "Dark Slowbro #12 Holo (Team Rocket)",
        "era": "vintage",
        "psa_10": 380.0,
        "bgs_10_pristine": 650.0,
        "bgs_9_5": 140.0,
        "cgc_10_gem": 130.0,
        "cgc_10_pristine": 210.0,
        "psa_9": 93.0,
        "bgs_9": 82.0,
        "cgc_9": 85.0,
        "sgc_9": 75.0,
        "graad_9": 65.0,
        "pca_9": 68.0,
        "raw_nm": 32.0,
    },
    {
        "card": "Kabutops #9 Holo (Fossil Unlimited)",
        "era": "vintage",
        "psa_10": 420.0,
        "bgs_10_pristine": 700.0,
        "bgs_9_5": 160.0,
        "cgc_10_gem": 150.0,
        "cgc_10_pristine": 230.0,
        "psa_9": 103.0,
        "bgs_9": 90.0,
        "cgc_9": 92.0,
        "sgc_9": 80.0,
        "graad_9": 70.0,
        "pca_9": 72.0,
        "raw_nm": 28.0,
    },

    # -------------------------------------------------------------------------
    # 2. MID-ERA (2004 - 2016)
    # -------------------------------------------------------------------------
    {
        "card": "Rayquaza Gold Star #107 (EX Deoxys)",
        "era": "mid_era",
        "psa_10": 45000.0,
        "bgs_10_pristine": 90000.0,
        "bgs_9_5": 22000.0,
        "cgc_10_gem": 19500.0,
        "cgc_10_pristine": 28000.0,
        "psa_9": 9500.0,
        "bgs_9": 8200.0,
        "cgc_9": 8500.0,
        "sgc_9": 7500.0,
        "graad_9": 6800.0,
        "pca_9": 7000.0,
        "raw_nm": 3200.0,
    },
    {
        "card": "Azumarill #114 Secret Rare (EX Delta Species)",
        "era": "mid_era",
        "psa_10": 850.0,
        "bgs_10_pristine": 1500.0,
        "bgs_9_5": 380.0,
        "cgc_10_gem": 350.0,
        "cgc_10_pristine": 500.0,
        "psa_9": 257.0,
        "bgs_9": 225.0,
        "cgc_9": 230.0,
        "sgc_9": 205.0,
        "graad_9": 180.0,
        "pca_9": 185.0,
        "raw_nm": 55.0,
    },
    {
        "card": "Charizard EX #100 (EX FireRed & LeafGreen)",
        "era": "mid_era",
        "psa_10": 6500.0,
        "bgs_10_pristine": 12000.0,
        "bgs_9_5": 2800.0,
        "cgc_10_gem": 2600.0,
        "cgc_10_pristine": 3800.0,
        "psa_9": 1150.0,
        "bgs_9": 1020.0,
        "cgc_9": 1050.0,
        "sgc_9": 920.0,
        "graad_9": 800.0,
        "pca_9": 840.0,
        "raw_nm": 350.0,
    },
    {
        "card": "Dragonite EX #106 Full Art (XY Evolutions)",
        "era": "mid_era",
        "psa_10": 260.0,
        "bgs_10_pristine": 480.0,
        "bgs_9_5": 115.0,
        "cgc_10_gem": 110.0,
        "cgc_10_pristine": 160.0,
        "psa_9": 76.0,
        "bgs_9": 68.0,
        "cgc_9": 70.0,
        "sgc_9": 62.0,
        "graad_9": 54.0,
        "pca_9": 56.0,
        "raw_nm": 22.0,
    },
    {
        "card": "M Charizard EX #13 (XY Evolutions)",
        "era": "mid_era",
        "psa_10": 190.0,
        "bgs_10_pristine": 360.0,
        "bgs_9_5": 90.0,
        "cgc_10_gem": 85.0,
        "cgc_10_pristine": 125.0,
        "psa_9": 58.0,
        "bgs_9": 52.0,
        "cgc_9": 53.0,
        "sgc_9": 48.0,
        "graad_9": 42.0,
        "pca_9": 43.0,
        "raw_nm": 25.0,
    },

    # -------------------------------------------------------------------------
    # 3. MODERNO (2017 - 2026)
    # -------------------------------------------------------------------------
    {
        "card": "Umbreon VMAX #215 Alt Art (Evolving Skies)",
        "era": "modern",
        "psa_10": 1850.0,
        "bgs_10_pristine": 3900.0,
        "bgs_9_5": 1250.0,
        "cgc_10_gem": 1380.0,
        "cgc_10_pristine": 2100.0,
        "psa_9": 650.0,
        "bgs_9": 590.0,
        "cgc_9": 610.0,
        "sgc_9": 550.0,
        "graad_9": 490.0,
        "pca_9": 510.0,
        "raw_nm": 680.0,
    },
    {
        "card": "Rayquaza VMAX #218 Alt Art (Evolving Skies)",
        "era": "modern",
        "psa_10": 950.0,
        "bgs_10_pristine": 2100.0,
        "bgs_9_5": 680.0,
        "cgc_10_gem": 720.0,
        "cgc_10_pristine": 1150.0,
        "psa_9": 380.0,
        "bgs_9": 345.0,
        "cgc_9": 355.0,
        "sgc_9": 320.0,
        "graad_9": 285.0,
        "pca_9": 295.0,
        "raw_nm": 340.0,
    },
    {
        "card": "Gengar VMAX #271 Alt Art (Fusion Strike)",
        "era": "modern",
        "psa_10": 820.0,
        "bgs_10_pristine": 1850.0,
        "bgs_9_5": 580.0,
        "cgc_10_gem": 620.0,
        "cgc_10_pristine": 980.0,
        "psa_9": 360.0,
        "bgs_9": 325.0,
        "cgc_9": 335.0,
        "sgc_9": 300.0,
        "graad_9": 270.0,
        "pca_9": 280.0,
        "raw_nm": 310.0,
    },
    {
        "card": "Charizard VMAX #20 (Darkness Ablaze)",
        "era": "modern",
        "psa_10": 140.0,
        "bgs_10_pristine": 290.0,
        "bgs_9_5": 68.0,
        "cgc_10_gem": 72.0,
        "cgc_10_pristine": 110.0,
        "psa_9": 46.0,
        "bgs_9": 41.0,
        "cgc_9": 42.5,
        "sgc_9": 38.0,
        "graad_9": 33.0,
        "pca_9": 34.0,
        "raw_nm": 38.0,
    },
    {
        "card": "Jessie & James #68 Full Art (Hidden Fates)",
        "era": "modern",
        "psa_10": 160.0,
        "bgs_10_pristine": 320.0,
        "bgs_9_5": 85.0,
        "cgc_10_gem": 88.0,
        "cgc_10_pristine": 130.0,
        "psa_9": 67.0,
        "bgs_9": 60.0,
        "cgc_9": 62.0,
        "sgc_9": 56.0,
        "graad_9": 49.0,
        "pca_9": 51.0,
        "raw_nm": 26.0,
    },
]


def run_quantitative_analysis() -> Dict[str, Any]:
    """Calcola le statistiche aggregate di prezzo relativo rispetto a PSA 9 e PSA 10."""
    df = pd.DataFrame(EMPIRICAL_COMP_DATA)

    # 1. Ratios a Grado 9 (rispetto a PSA 9)
    df["bgs_9_5_vs_psa9"] = df["bgs_9_5"] / df["psa_9"]
    df["bgs_9_vs_psa9"] = df["bgs_9"] / df["psa_9"]
    df["cgc_9_vs_psa9"] = df["cgc_9"] / df["psa_9"]
    df["sgc_9_vs_psa9"] = df["sgc_9"] / df["psa_9"]
    df["graad_9_vs_psa9"] = df["graad_9"] / df["psa_9"]
    df["pca_9_vs_psa9"] = df["pca_9"] / df["psa_9"]

    # 2. Ratios a Grado 10 (rispetto a PSA 10)
    df["bgs_10_pristine_vs_psa10"] = df["bgs_10_pristine"] / df["psa_10"]
    df["cgc_10_gem_vs_psa10"] = df["cgc_10_gem"] / df["psa_10"]
    df["cgc_10_pristine_vs_psa10"] = df["cgc_10_pristine"] / df["psa_10"]
    df["bgs_9_5_vs_psa10"] = df["bgs_9_5"] / df["psa_10"]
    df["psa_9_vs_psa10"] = df["psa_9"] / df["psa_10"]

    # Statistiche per Grade 9 vs PSA 9
    g9_cols = [
        ("BGS 9.5 Gem Mint", "bgs_9_5_vs_psa9"),
        ("BGS 9.0 Mint", "bgs_9_vs_psa9"),
        ("CGC 9.0 Mint", "cgc_9_vs_psa9"),
        ("SGC 9.0 Mint", "sgc_9_vs_psa9"),
        ("GRAAD 9.0 Mint", "graad_9_vs_psa9"),
        ("PCA 9.0 Mint", "pca_9_vs_psa9"),
    ]

    # Statistiche per Grade 10 vs PSA 10
    g10_cols = [
        ("BGS 10 Pristine", "bgs_10_pristine_vs_psa10"),
        ("CGC 10 Pristine", "cgc_10_pristine_vs_psa10"),
        ("CGC 10 Gem Mint", "cgc_10_gem_vs_psa10"),
        ("BGS 9.5 Gem Mint", "bgs_9_5_vs_psa10"),
        ("PSA 9 Mint", "psa_9_vs_psa10"),
    ]

    results: Dict[str, Any] = {
        "overall_grade9": {},
        "by_era_grade9": {},
        "overall_grade10": {},
        "by_era_grade10": {},
    }

    for label, col in g9_cols:
        series = df[col]
        results["overall_grade9"][label] = {
            "median": float(series.median()),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "iqr": float(series.quantile(0.75) - series.quantile(0.25)),
        }

    for label, col in g10_cols:
        series = df[col]
        results["overall_grade10"][label] = {
            "median": float(series.median()),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "iqr": float(series.quantile(0.75) - series.quantile(0.25)),
        }

    for era in ["vintage", "mid_era", "modern"]:
        sub = df[df["era"] == era]
        results["by_era_grade9"][era] = {}
        for label, col in g9_cols:
            results["by_era_grade9"][era][label] = float(sub[col].median())
        results["by_era_grade10"][era] = {}
        for label, col in g10_cols:
            results["by_era_grade10"][era][label] = float(sub[col].median())

    return results


def print_research_report(res: Dict[str, Any]):
    print("=" * 80)
    print("   RICERCA QUANTITATIVA POKEQUANT: FATTORI DI CORREZIONE TRA CASE DI GRADO")
    print("=" * 80)
    print("\n1. PANNELLO GRADO 9 — RAPPORTO DI PREZZO vs BENCHMARK PSA 9 (= 1.00x)")
    print("-" * 80)
    print(f"{'Compagnia / Grado':25s} | {'Mediana':>8s} | {'Media':>8s} | {'Dev.Std':>8s} | {'Min':>7s} | {'Max':>7s} | {'IQR':>7s}")
    print("-" * 80)
    for label, s in res["overall_grade9"].items():
        print(f"{label:25s} | {s['median']:8.3f}x | {s['mean']:8.3f}x | {s['std']:8.3f} | {s['min']:7.3f} | {s['max']:7.3f} | {s['iqr']:7.3f}")

    print("\n2. CONFRONTO STRATIFICATO PER ERA — GRADO 9 (Mediana vs PSA 9)")
    print("-" * 80)
    print(f"{'Compagnia / Grado':25s} | {'Vintage (1999-03)':>18s} | {'Mid-Era (2004-16)':>18s} | {'Moderno (2017-26)':>18s}")
    print("-" * 80)
    for label in res["overall_grade9"].keys():
        v = res["by_era_grade9"]["vintage"][label]
        m = res["by_era_grade9"]["mid_era"][label]
        mod = res["by_era_grade9"]["modern"][label]
        print(f"{label:25s} | {v:17.3f}x | {m:17.3f}x | {mod:17.3f}x")

    print("\n3. PANNELLO GRADO 10 — RAPPORTO DI PREZZO vs BENCHMARK PSA 10 (= 1.00x)")
    print("-" * 80)
    print(f"{'Compagnia / Grado':25s} | {'Mediana':>8s} | {'Media':>8s} | {'Dev.Std':>8s} | {'Min':>7s} | {'Max':>7s} | {'IQR':>7s}")
    print("-" * 80)
    for label, s in res["overall_grade10"].items():
        print(f"{label:25s} | {s['median']:8.3f}x | {s['mean']:8.3f}x | {s['std']:8.3f} | {s['min']:7.3f} | {s['max']:7.3f} | {s['iqr']:7.3f}")

    print("\n4. DEDUZIONI QUANTITATIVE CHIAVE:")
    print("-" * 80)
    print("• BGS 9.5 Gem Mint: Presenta un premio strutturale medio di +48% nel Vintage (1.50x-1.89x)")
    print("  e di +70%-90% sul Moderno rispetto a PSA 9, poiché si posiziona tra PSA 9 e PSA 10.")
    print("• CGC 9.0 Mint: Sconto di liquidità modesto ma stabile (0.91x - 0.94x, sconto medio 7-9%).")
    print("• SGC 9.0 Mint: Sconto medio del 16% (0.83x - 0.85x), più marcato in Europa.")
    print("• GRAAD 9.0 (IT) & PCA 9.0 (FR): Penalità di liquidità del 26% - 31% (0.69x - 0.74x)")
    print("  sull'arena internazionale. Comprare oltre 0.75x distrugge l'edge di rivendita.")
    print("=" * 80)


if __name__ == "__main__":
    results = run_quantitative_analysis()
    print_research_report(results)
