"""
poke_quant/data/population_fetcher.py — Download e parsing dei Population Report
PSA/CGC da PriceCharting (per-grado, per-compagnia).

Trovato durante la ricerca su "possiamo trovare dati per pop report?" (2026-09-25):
PriceCharting ha lanciato a febbraio 2026 una pagina dedicata
/pop/item/<game_slug>/<item_slug> con popolazione PSA+CGC per grado, aggiornata
mensilmente (blog.pricecharting.com/2026/02/population-reports-for-psa-cgc.html) -
stesso dominio già usato per i prezzi, MAI bloccato da Cloudflare (a differenza di
psacard.com, cgccards.com, gemrate.com - tutti verificati 403 in questa ricerca).

LIMITE DICHIARATO: è uno SNAPSHOT del giorno del fetch, non un archivio storico -
PriceCharting non pubblica la popolazione passata. Per costruire una serie storica
utile a testare "la crescita della pop prediceva il prezzo" bisogna accumulare uno
snapshot per volta a partire da oggi (scripts/fetch_population_snapshot.py, pensato
per essere rilanciato con la stessa cadenza mensile dichiarata dalla fonte) - non è
utilizzabile per un backtest retroattivo 2021-2026.

Un'alternativa con vera storia (dati giornalieri dal 1 gennaio 2022, PSA+BGS+SGC+CGC
unificati) esiste - l'API di GemRate (docs.gemrate.com) - ma richiede una API key
richiesta via contatto commerciale, e l'accesso storico è esplicitamente escluso dai
piani base ("Historical data access not included in basic plans. Contact GemRate for
more information.") - non self-serve, non gratuito, non azionabile senza un passo
esplicito dell'utente. Non integrato qui per questo motivo.
"""

from __future__ import annotations
import logging
import re
import time
from typing import Dict, List, Any, Optional
import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

logger = logging.getLogger(__name__)

_ROW_RE = re.compile(
    r'<td class="grade-col">([^<]+)</td>\s*'
    r'<td class="psa-col">([^<]+)</td>\s*'
    r'<td class="cgc-col">([^<]+)</td>\s*'
    r'<td class="total-col">([^<]+)</td>\s*'
    r'<td class="price-col">([^<]+)</td>',
    re.IGNORECASE,
)


def _to_int(raw: str) -> Optional[int]:
    raw = raw.strip().replace(",", "")
    if raw in ("", "-"):
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _to_price(raw: str) -> Optional[float]:
    raw = raw.strip().replace(",", "").replace("$", "")
    if raw in ("", "-"):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def parse_population_table(html: str) -> List[Dict[str, Any]]:
    """Estrae le righe grado/PSA/CGC/Totale/Prezzo dalla tabella #population-table
    di una pagina /pop/item/... di PriceCharting. Ritorna [] se la carta non ha
    ancora un Population Report (pagina priva della tabella)."""
    rows = []
    for grade_raw, psa_raw, cgc_raw, total_raw, price_raw in _ROW_RE.findall(html):
        grade = grade_raw.strip()
        if grade.lower() == "total":
            continue
        rows.append({
            "grade": grade,
            "psa_pop": _to_int(psa_raw),
            "cgc_pop": _to_int(cgc_raw),
            "total_pop": _to_int(total_raw),
            "price_usd": _to_price(price_raw),
        })
    return rows


def fetch_pricecharting_population(game_slug: str, item_slug: str) -> List[Dict[str, Any]]:
    """Scarica e parsa il Population Report per una carta. Ritorna [] se la pagina
    non esiste o non ha ancora dati di popolazione (carta troppo nuova/poco tracciata,
    non un errore). Stesso schema di retry-on-429 di fetch_pricecharting_cover_image_url
    (poke_quant/data/price_fetcher.py) - lo stesso dominio, lo stesso rate limit."""
    url = f"https://www.pricecharting.com/pop/item/{game_slug}/{item_slug}"
    last_attempt = 2
    resp = None
    for attempt in range(last_attempt + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
        except Exception as e:
            if attempt == last_attempt:
                logger.warning(f"Impossibile scaricare {url} per la popolazione: {e}")
                return []
            continue
        if resp.status_code == 429 and attempt < last_attempt:
            wait_s = float(resp.headers.get("Retry-After", 1.0 * (attempt + 1)))
            time.sleep(min(wait_s, 5.0))
            continue
        if resp.status_code == 404:
            return []
        try:
            resp.raise_for_status()
        except Exception as e:
            logger.warning(f"Impossibile scaricare {url} per la popolazione: {e}")
            return []
        break
    return parse_population_table(resp.text)
