"""
poke_quant/data/catalog_fetcher.py — Interfaccia verso l'API PokemonTCG.io per anagrafica set,
immagini, carte e snapshot prezzi correnti Cardmarket (EUR) e TCGplayer (USD).

L'API pokemontcg.io e' intermittente (500/502 osservati ripetutamente in questa sessione
durante scripts/discover_chase_cards.py, quasi sempre risolti al 2°-3° tentativo) - senza
retry, un singolo errore transitorio si traduceva nel messaggio "servizio non disponibile"
anche quando il servizio era in realta' su e giu' nell'arco di pochi secondi.
"""

from __future__ import annotations
import logging
import time
from typing import Dict, List, Any, Optional
import requests

BASE_URL = "https://api.pokemontcg.io/v2"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

logger = logging.getLogger(__name__)


def _get_with_retries(url: str, headers: dict, timeout: int, retries: int = 3, backoff_s: float = 2.0) -> Optional[dict]:
    last_error = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_error = e
            if attempt < retries - 1:
                time.sleep(backoff_s)
    logger.error(f"PokemonTCG.io non raggiungibile dopo {retries} tentativi ({url}): {last_error}")
    return None


def fetch_all_sets(api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """Recupera l'elenco completo di tutti i set storici e moderni con date di rilascio ufficiali."""
    headers = dict(HEADERS)
    if api_key:
        headers["X-Api-Key"] = api_key

    payload = _get_with_retries(f"{BASE_URL}/sets", headers, timeout=15)
    if payload is None:
        return []
    data = payload.get("data", [])
    # Ordina cronologicamente per data di rilascio decrescente
    data.sort(key=lambda s: s.get("releaseDate", "1900/01/01"), reverse=True)
    return data


def fetch_cards_by_set(set_id: str, api_key: Optional[str] = None, page_size: int = 250) -> List[Dict[str, Any]]:
    """Recupera le carte di un set con relative quotazioni correnti (Cardmarket EUR & TCGplayer USD)."""
    headers = dict(HEADERS)
    if api_key:
        headers["X-Api-Key"] = api_key

    url = f"{BASE_URL}/cards?q=set.id:{set_id}&pageSize={page_size}"
    payload = _get_with_retries(url, headers, timeout=20)
    if payload is None:
        return []
    return payload.get("data", [])
