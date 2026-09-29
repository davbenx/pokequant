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
import json
import logging
from pathlib import Path
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


POP_JSON_CACHE_FILE = Path(__file__).resolve().parent.parent.parent / "data_cache" / "pricecharting_pop_cache.json"
POP_CSV_FILE = Path(__file__).resolve().parent.parent.parent / "data_cache" / "population_history.csv"

_POP_IN_MEMORY_CACHE: Dict[str, List[Dict[str, Any]]] = {}
_POP_CSV_DF_CACHE: Optional[Any] = None


def _get_pop_csv_records(item_id: str) -> List[Dict[str, Any]]:
    global _POP_CSV_DF_CACHE
    if _POP_CSV_DF_CACHE is None:
        if POP_CSV_FILE.exists():
            try:
                import pandas as pd
                _POP_CSV_DF_CACHE = pd.read_csv(POP_CSV_FILE)
            except Exception:
                _POP_CSV_DF_CACHE = False
        else:
            _POP_CSV_DF_CACHE = False

    if _POP_CSV_DF_CACHE is False or _POP_CSV_DF_CACHE is None:
        return []

    try:
        import pandas as pd
        sub = _POP_CSV_DF_CACHE[_POP_CSV_DF_CACHE["item_id"] == item_id]
        if sub.empty:
            return []
        rows = []
        for _, r in sub.iterrows():
            g_raw = str(r["grade"]).strip()
            if g_raw.lower() == "total":
                continue
            rows.append({
                "grade": g_raw,
                "psa_pop": int(r["psa_pop"]) if pd.notna(r["psa_pop"]) else None,
                "cgc_pop": int(r["cgc_pop"]) if pd.notna(r["cgc_pop"]) else None,
                "total_pop": int(r["total_pop"]) if pd.notna(r["total_pop"]) else None,
                "price_usd": float(r["price_usd"]) if pd.notna(r["price_usd"]) else None,
            })
        return rows
    except Exception as e:
        logger.warning(f"Errore lettura pop da CSV per {item_id}: {e}")
        return []


def _load_pop_json_cache() -> Dict[str, Any]:
    if not POP_JSON_CACHE_FILE.exists():
        return {}
    try:
        with open(POP_JSON_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_pop_json_cache(cache_data: Dict[str, Any]):
    try:
        POP_JSON_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(POP_JSON_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2)
    except Exception as e:
        logger.warning(f"Impossibile salvare la cache di popolazione in {POP_JSON_CACHE_FILE}: {e}")


def fetch_pricecharting_population_cached(
    game_slug: str,
    item_slug: str,
    item_id: Optional[str] = None,
    max_age_days: int = 7,
) -> List[Dict[str, Any]]:
    """
    Ritorna il population report (per-grado PSA/CGC/Totale) per una carta,
    utilizzando una strategia di caching gerarchica:
    1. Memoria (0ms)
    2. File JSON persistente data_cache/pricecharting_pop_cache.json (< 7 giorni)
    3. File CSV storico data_cache/population_history.csv (se carta già censita in PokeQuant)
    4. Fetch live su PriceCharting https://www.pricecharting.com/pop/item/... con retry e caching automatico
    """
    key = f"{game_slug.strip()}/{item_slug.strip()}".lower() if game_slug and item_slug else (item_id or "")
    if not key:
        return []

    # 1. Check memoria
    if key in _POP_IN_MEMORY_CACHE:
        return _POP_IN_MEMORY_CACHE[key]
    if item_id and item_id in _POP_IN_MEMORY_CACHE:
        return _POP_IN_MEMORY_CACHE[item_id]

    # 2. Check JSON cache su disco
    json_cache = _load_pop_json_cache()
    cached_entry = json_cache.get(key) or (json_cache.get(item_id) if item_id else None)
    if cached_entry and isinstance(cached_entry, dict):
        cached_ts = cached_entry.get("cached_at", 0)
        cached_rows = cached_entry.get("rows", [])
        if time.time() - cached_ts < max_age_days * 86400 and cached_rows:
            _POP_IN_MEMORY_CACHE[key] = cached_rows
            if item_id:
                _POP_IN_MEMORY_CACHE[item_id] = cached_rows
            return cached_rows

    # 3. Check CSV storico locale se abbiamo item_id
    resolved_id = item_id
    if not resolved_id and game_slug and item_slug:
        try:
            from poke_quant.data.storage import load_metadata
            meta = load_metadata()
            for iid, info in meta.items():
                if info.get("game_slug") == game_slug and info.get("item_slug") == item_slug:
                    resolved_id = iid
                    break
        except Exception:
            resolved_id = None

    if resolved_id:
        csv_rows = _get_pop_csv_records(resolved_id)
        if csv_rows:
            _POP_IN_MEMORY_CACHE[key] = csv_rows
            if resolved_id:
                _POP_IN_MEMORY_CACHE[resolved_id] = csv_rows
            # Salva anche in JSON per uniformità
            json_cache[key] = {"cached_at": int(time.time()), "rows": csv_rows}
            if resolved_id:
                json_cache[resolved_id] = {"cached_at": int(time.time()), "rows": csv_rows}
            _save_pop_json_cache(json_cache)
            return csv_rows

    # 4. Fetch live su PriceCharting se game_slug e item_slug disponibili
    if game_slug and item_slug:
        live_rows = fetch_pricecharting_population(game_slug, item_slug)
        if live_rows:
            _POP_IN_MEMORY_CACHE[key] = live_rows
            if resolved_id:
                _POP_IN_MEMORY_CACHE[resolved_id] = live_rows
            json_cache[key] = {"cached_at": int(time.time()), "rows": live_rows}
            if resolved_id:
                json_cache[resolved_id] = {"cached_at": int(time.time()), "rows": live_rows}
            _save_pop_json_cache(json_cache)
            return live_rows

    return []

