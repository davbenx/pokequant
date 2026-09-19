"""
poke_quant/data/catalog_fetcher.py — Interfaccia verso l'API PokemonTCG.io per anagrafica set,
immagini, carte e snapshot prezzi correnti Cardmarket (EUR) e TCGplayer (USD).
"""

from __future__ import annotations
import json
import logging
from typing import Dict, List, Any, Optional
import requests

BASE_URL = "https://api.pokemontcg.io/v2"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

logger = logging.getLogger(__name__)


def fetch_all_sets(api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """Recupera l'elenco completo di tutti i set storici e moderni con date di rilascio ufficiali."""
    headers = dict(HEADERS)
    if api_key:
        headers["X-Api-Key"] = api_key

    url = f"{BASE_URL}/sets"
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        # Ordina cronologicamente per data di rilascio decrescente
        data.sort(key=lambda s: s.get("releaseDate", "1900/01/01"), reverse=True)
        return data
    except Exception as e:
        logger.error(f"Errore nel recupero set da PokemonTCG.io: {e}")
        return []


def fetch_cards_by_set(set_id: str, api_key: Optional[str] = None, page_size: int = 250) -> List[Dict[str, Any]]:
    """Recupera le carte di un set con relative quotazioni correnti (Cardmarket EUR & TCGplayer USD)."""
    headers = dict(HEADERS)
    if api_key:
        headers["X-Api-Key"] = api_key

    url = f"{BASE_URL}/cards?q=set.id:{set_id}&pageSize={page_size}"
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        cards = resp.json().get("data", [])
        return cards
    except Exception as e:
        logger.error(f"Errore nel recupero carte per set {set_id}: {e}")
        return []
