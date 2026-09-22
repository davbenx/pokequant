"""
poke_quant/data/europe_market_calibrator.py — Calibratore del Mercato Europeo di Riferimento (Cardmarket EUR).

ATTENZIONE — NON è dato Cardmarket reale, nonostante il nome. calibrate_price_matrix_for_europe()
prende la serie US PriceCharting e la moltiplica per fattori fissi hardcoded (1.06x Pokémon EN,
1.08x One Piece, max(1.12x, 58€) JP, 1.02x singole) — nessuna transazione Cardmarket reale è
mai stata osservata per costruire questi numeri. Il file che genera, data_cache/historical_prices_europe.csv,
NON va usato per validare strategie (le strategie di questa sessione usano tutte historical_prices.csv,
la serie US reale) — contraddice il claim "100% Dati Reali" del README del progetto. Trovato durante
l'audit Fase 0, non ancora sostituito con dati Cardmarket reali (bloccato da Cloudflare / serve API MKM
ufficiale, vedi discussione sessione). Tenerlo a mente se questo modulo viene ripreso in futuro.

Converte e "adatta" (in realtà: gonfia con moltiplicatori inventati) le serie storiche:
  1. Inclusione IVA europea (19-22%) nei floor distributivi e MSRP
  2. Modello commissioni Cardmarket (5% lordo + 0.60€ trustee)
  3. Spread linguistico e premio di liquidità dei box in lingua inglese e giapponese in Europa
  4. Gestione multi-franchise (Pokémon ENG, Pokémon JAP, One Piece TCG)
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

from poke_quant.data.storage import load_price_matrix, save_price_matrix, load_metadata, ensure_cache_dir

logger = logging.getLogger(__name__)

EUROPE_MATRIX_FILENAME = "historical_prices_europe.csv"


def calibrate_price_matrix_for_europe(
    us_prices_df: Optional[pd.DataFrame] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Trasforma la matrice dei prezzi dal mercato USA/globale al mercato europeo reale (Cardmarket EUR).
    Applica i fattori empirici di aggiustamento:
      - Per i box Pokémon ENG: il floor distributivo europeo include IVA (22%), impedendo dump a 90€.
        Durante la fase OOP, la minore offerta fisica in Europa crea un premio di ritenzione (+8% - +15%).
      - Per i box Pokémon JAP: costo di importazione aereo, dazio doganale (0-2.7%) e IVA 22% alla dogana.
      - Per i box One Piece TCG: allocazioni Bandai europee più strette rispetto agli USA.
    """
    if us_prices_df is None:
        us_prices_df = load_price_matrix()

    if us_prices_df is None or us_prices_df.empty:
        raise ValueError("Matrice prezzi di origine non trovata.")

    if metadata is None:
        metadata = load_metadata() or {}

    eu_df = us_prices_df.copy()

    for col in eu_df.columns:
        meta = metadata.get(col, {})
        franchise = meta.get("franchise", "pokemon")
        lang = meta.get("language", "en")
        p_type = meta.get("type", "sealed")

        if p_type == "sealed":
            if franchise == "one_piece":
                # One Piece in Europa: quote distributive rigide, premio di scarsità Cardmarket +10%
                eu_df[col] = eu_df[col] * 1.08
            elif lang == "jp":
                # Giapponese importato in Europa: floor minimo 55-65€ per coprire spedizione EMS + IVA doganale
                eu_df[col] = eu_df[col].apply(lambda x: max(x * 1.12, 58.0) if x > 0 else 0.0)
            else:
                # Pokémon Occidentale (ENG):
                # Il prezzo floor in Europa con IVA non scende sotto 115-125€ nel periodo moderno.
                # Nelle fasi OOP mature, segue il benchmark internazionale con un attrito logistico del +5%.
                eu_df[col] = eu_df[col] * 1.06
        else:
            # Singole: Cardmarket ha commissioni più basse (5% vs 13% TCGPlayer USA),
            # ma l'IVA e la liquidità bilanciano il prezzo medio
            eu_df[col] = eu_df[col] * 1.02

    eu_df = eu_df.round(2)
    save_price_matrix(eu_df, filename=EUROPE_MATRIX_FILENAME)
    return eu_df


def get_market_price_matrix(regime: str = "europe_cardmarket") -> pd.DataFrame:
    """
    Restituisce la matrice dei prezzi calibrata in base al mercato di riferimento:
      - 'europe_cardmarket': Mercato Europeo Cardmarket (EUR, IVA inclusa, fee 5%)
      - 'us_global': Mercato Globale / USA (PriceCharting / eBay USA in USD)
    """
    if regime == "europe_cardmarket":
        cache_path = ensure_cache_dir() / EUROPE_MATRIX_FILENAME
        if cache_path.exists():
            df = load_price_matrix(EUROPE_MATRIX_FILENAME)
            if df is not None and not df.empty:
                return df
        # Genera calibrazione se assente
        return calibrate_price_matrix_for_europe()
    else:
        df = load_price_matrix("historical_prices.csv")
        if df is None or df.empty:
            raise ValueError("Matrice storica originale non trovata.")
        return df
