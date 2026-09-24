"""
poke_quant/data/cardmarket_bridge.py — Bridge Cardmarket per quotazioni reali, multilingua e multi-franchise.
Gestisce il monitoraggio dei prezzi minimi reali su Cardmarket per:
  - Pokémon TCG Occidentale (Inglese idLanguage=1, Italiano idLanguage=5)
  - Pokémon TCG Giapponese (JAP: High-Class & God Pack Sets)
  - One Piece TCG (OP-01 to OP-08+)
Include caching locale, validazione dei prezzi e calcolo dello scostamento rispetto al Prezzo Massimo di Acquisto.
"""

from __future__ import annotations
import json
import logging
import urllib.parse
from pathlib import Path
from typing import Dict, Any, Optional, List
import datetime

logger = logging.getLogger(__name__)

CACHE_FILE = Path(__file__).resolve().parent.parent.parent / "data_cache" / "cardmarket_live_quotes.json"

LANGUAGE_CODES = {
    "en": {"id": 1, "name": "English", "flag": "🇬🇧"},
    "fr": {"id": 2, "name": "French", "flag": "🇫🇷"},
    "de": {"id": 3, "name": "German", "flag": "🇩🇪"},
    "es": {"id": 4, "name": "Spanish", "flag": "🇪🇸"},
    "it": {"id": 5, "name": "Italian", "flag": "🇮🇹"},
    "jp": {"id": 2, "name": "Japanese", "flag": "🇯🇵"},
}

DEFAULT_CARDMARKET_SEED: Dict[str, Dict[str, Any]] = {
    # --- POKEMON TCG ENG (Scarlet & Violet Era) — Prezzi reali verificati su Cardmarket ---
    "stellar_crown_bb": {
        "name": "Stellar Crown Booster Box",
        "franchise": "pokemon",
        "language": "en",
        "last_verified_price": 249.65,
        "last_updated": "2026-09-19",
        "source": "Cardmarket Real / User Verified",
        "msrp": 160.0,
        "cardmarket_path": "Pokemon/Products/Booster-Boxes/Stellar-Crown-Booster-Box"
    },
    "temporal_forces_bb": {
        "name": "Temporal Forces Booster Box",
        "franchise": "pokemon",
        "language": "en",
        "last_verified_price": 225.00,
        "last_updated": "2026-09-19",
        "source": "Cardmarket Real / User Verified",
        "msrp": 160.0,
        "cardmarket_path": "Pokemon/Products/Booster-Boxes/Temporal-Forces-Booster-Box"
    },
    "paradox_rift_bb": {
        "name": "Paradox Rift Booster Box",
        "franchise": "pokemon",
        "language": "en",
        "last_verified_price": 225.00,
        "last_updated": "2026-09-19",
        "source": "Cardmarket Real / User Verified",
        "msrp": 160.0,
        "cardmarket_path": "Pokemon/Products/Booster-Boxes/Paradox-Rift-Booster-Box"
    },
    "scarlet_violet_base_bb": {
        "name": "Scarlet & Violet Base Booster Box",
        "franchise": "pokemon",
        "language": "en",
        "last_verified_price": 204.90,
        "last_updated": "2026-09-19",
        "source": "Cardmarket Real / User Verified",
        "msrp": 160.0,
        "cardmarket_path": "Pokemon/Products/Booster-Boxes/Scarlet-Violet-Booster-Box"
    },
    "paldea_evolved_bb": {
        "name": "Paldea Evolved Booster Box",
        "franchise": "pokemon",
        "language": "en",
        "last_verified_price": 275.00,
        "last_updated": "2026-09-19",
        "source": "Cardmarket Real",
        "msrp": 160.0,
        "cardmarket_path": "Pokemon/Products/Booster-Boxes/Paldea-Evolved-Booster-Box"
    },
    "twilight_masquerade_bb": {
        "name": "Twilight Masquerade Booster Box",
        "franchise": "pokemon",
        "language": "en",
        "last_verified_price": 235.00,
        "last_updated": "2026-09-19",
        "source": "Cardmarket Real",
        "msrp": 160.0,
        "cardmarket_path": "Pokemon/Products/Booster-Boxes/Twilight-Masquerade-Booster-Box"
    },
    "obsidian_flames_bb": {
        "name": "Obsidian Flames Booster Box",
        "franchise": "pokemon",
        "language": "en",
        "last_verified_price": 195.00,
        "last_updated": "2026-09-19",
        "source": "Cardmarket Real",
        "msrp": 160.0,
        "cardmarket_path": "Pokemon/Products/Booster-Boxes/Obsidian-Flames-Booster-Box"
    },
    "surging_sparks_bb": {
        "name": "Surging Sparks Booster Box",
        "franchise": "pokemon",
        "language": "en",
        "last_verified_price": 230.00,
        "last_updated": "2026-09-19",
        "source": "Cardmarket Real",
        "msrp": 160.0,
        "cardmarket_path": "Pokemon/Products/Booster-Boxes/Surging-Sparks-Booster-Box"
    },
    "scarlet_violet_151_bundle": {
        "name": "Scarlet & Violet 151 Booster Bundle (6 Packs)",
        "franchise": "pokemon",
        "language": "en",
        "last_verified_price": 45.00,
        "last_updated": "2026-09-19",
        "source": "Cardmarket Real",
        "msrp": 30.0,
        "cardmarket_path": "Pokemon/Products/Booster-Bundles/Scarlet-Violet-151-Booster-Bundle"
    },

    # --- POKEMON TCG JAP — High-Class & God Pack Sets ---
    "jp_vstar_universe_bb": {
        "name": "VSTAR Universe Booster Box [JP]",
        "franchise": "pokemon",
        "language": "jp",
        "last_verified_price": 115.00,
        "last_updated": "2026-09-19",
        "source": "Cardmarket Real",
        "msrp": 55.0,
        "cardmarket_path": "Pokemon/Products/Booster-Boxes/VSTAR-Universe-Booster-Box"
    },
    "jp_vmax_climax_bb": {
        "name": "VMAX Climax Booster Box [JP]",
        "franchise": "pokemon",
        "language": "jp",
        "last_verified_price": 135.00,
        "last_updated": "2026-09-19",
        "source": "Cardmarket Real",
        "msrp": 55.0,
        "cardmarket_path": "Pokemon/Products/Booster-Boxes/VMAX-Climax-Booster-Box"
    },
    "jp_shiny_treasure_bb": {
        "name": "Shiny Treasure ex Booster Box [JP]",
        "franchise": "pokemon",
        "language": "jp",
        "last_verified_price": 62.00,
        "last_updated": "2026-09-19",
        "source": "Cardmarket Real",
        "msrp": 55.0,
        "cardmarket_path": "Pokemon/Products/Booster-Boxes/Shiny-Treasure-ex-Booster-Box"
    },
    "jp_shiny_star_v_bb": {
        "name": "Shiny Star V Booster Box [JP]",
        "franchise": "pokemon",
        "language": "jp",
        "last_verified_price": 145.00,
        "last_updated": "2026-09-19",
        "source": "Cardmarket Real",
        "msrp": 55.0,
        "cardmarket_path": "Pokemon/Products/Booster-Boxes/Shiny-Star-V-Booster-Box"
    },
    "jp_tag_all_stars_bb": {
        "name": "Tag All Stars Booster Box [JP]",
        "franchise": "pokemon",
        "language": "jp",
        "last_verified_price": 480.00,
        "last_updated": "2026-09-19",
        "source": "Cardmarket Real",
        "msrp": 55.0,
        "cardmarket_path": "Pokemon/Products/Booster-Boxes/Tag-Team-GX-Tag-All-Stars-Booster-Box"
    },

    # --- ONE PIECE TCG (OP-01 to OP-08) — Satellite Asset Class ---
    "op01_romance_dawn_bb": {
        "name": "Romance Dawn Booster Box [OP-01]",
        "franchise": "one_piece",
        "language": "en",
        "last_verified_price": 380.00,
        "last_updated": "2026-09-19",
        "source": "Cardmarket Real",
        "msrp": 120.0,
        "cardmarket_path": "OnePiece/Products/Booster-Boxes/Romance-Dawn-Booster-Box"
    },
    "op02_paramount_war_bb": {
        "name": "Paramount War Booster Box [OP-02]",
        "franchise": "one_piece",
        "language": "en",
        "last_verified_price": 185.00,
        "last_updated": "2026-09-19",
        "source": "Cardmarket Real",
        "msrp": 120.0,
        "cardmarket_path": "OnePiece/Products/Booster-Boxes/Paramount-War-Booster-Box"
    },
    "op03_pillars_strength_bb": {
        "name": "Pillars of Strength Booster Box [OP-03]",
        "franchise": "one_piece",
        "language": "en",
        "last_verified_price": 175.00,
        "last_updated": "2026-09-19",
        "source": "Cardmarket Real",
        "msrp": 120.0,
        "cardmarket_path": "OnePiece/Products/Booster-Boxes/Pillars-of-Strength-Booster-Box"
    },
    "op05_new_era_bb": {
        "name": "Awakening of the New Era Booster Box [OP-05]",
        "franchise": "one_piece",
        "language": "en",
        "last_verified_price": 280.00,
        "last_updated": "2026-09-19",
        "source": "Cardmarket Real",
        "msrp": 120.0,
        "cardmarket_path": "OnePiece/Products/Booster-Boxes/Awakening-of-the-New-Era-Booster-Box"
    },
    "op06_wings_captain_bb": {
        "name": "Wings of the Captain Booster Box [OP-06]",
        "franchise": "one_piece",
        "language": "en",
        "last_verified_price": 138.00,
        "last_updated": "2026-09-19",
        "source": "Cardmarket Real",
        "msrp": 120.0,
        "cardmarket_path": "OnePiece/Products/Booster-Boxes/Flocks-of-the-Captain-Booster-Box"
    }
}


def ensure_cardmarket_cache() -> Path:
    """Assicura l'esistenza del file di cache e lo inizializza con i dati di seed se assente."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not CACHE_FILE.exists():
        save_cardmarket_quotes(DEFAULT_CARDMARKET_SEED)
    return CACHE_FILE


def load_cardmarket_quotes() -> Dict[str, Dict[str, Any]]:
    """Carica tutte le quote Cardmarket memorizzate in cache."""
    ensure_cardmarket_cache()
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Unione con i default per garantire copertura completa
            for k, v in DEFAULT_CARDMARKET_SEED.items():
                if k not in data:
                    data[k] = v
            return data
    except Exception as e:
        logger.error(f"Errore caricamento cardmarket_live_quotes.json: {e}")
        return DEFAULT_CARDMARKET_SEED.copy()


def save_cardmarket_quotes(quotes: Dict[str, Dict[str, Any]]) -> bool:
    """Salva le quote Cardmarket in cache locale persistente."""
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(quotes, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Errore salvataggio cardmarket_live_quotes.json: {e}")
        return False


def update_cardmarket_quote(
    item_id: str,
    price: float,
    language: str = "en",
    source: str = "Cardmarket Desk Override"
) -> bool:
    """Aggiorna il prezzo verificato di un singolo item in cache per la lingua indicata."""
    quotes = load_cardmarket_quotes()
    now_str = datetime.date.today().isoformat()
    if item_id not in quotes:
        quotes[item_id] = {
            "name": item_id.replace("_", " ").title(),
            "franchise": "pokemon" if "op" not in item_id else "one_piece",
            "language": language,
            "last_verified_price": float(price),
            "last_updated": now_str,
            "source": source
        }
    else:
        quotes[item_id]["last_verified_price"] = float(price)
        quotes[item_id]["language"] = language
        quotes[item_id]["last_updated"] = now_str
        quotes[item_id]["source"] = source

    return save_cardmarket_quotes(quotes)


def get_cardmarket_live_prices(
    franchise_filter: Optional[str] = None,
    language_filter: Optional[str] = None
) -> Dict[str, float]:
    """Restituisce un dizionario {item_id: live_price} filtrabile per franchise e lingua."""
    quotes = load_cardmarket_quotes()
    res = {}
    for item_id, q in quotes.items():
        if franchise_filter is not None and q.get("franchise") != franchise_filter:
            continue
        if language_filter is not None and q.get("language") != language_filter:
            continue
        res[item_id] = float(q.get("last_verified_price", 0.0))
    return res


def get_cardmarket_deep_link(
    item_name: str,
    franchise: str = "pokemon",
    language: str = "en",
    custom_path: Optional[str] = None,
    item_type: str = "sealed"
) -> str:
    """
    Costruisce l'URL diretto e preciso su Cardmarket con filtri lingua (idLanguage)
    e parametri di ricerca ottimali.

    item_type="sealed" (default, comportamento storico) -> aggiunge "Booster Box"
    al nome se non gia' presente. item_type="single" -> NON lo aggiunge (era un bug:
    per una singola carta il link finiva a cercare il box, non la carta) e aggiunge
    invece "PSA 9" come suggerimento di ricerca testuale, perche' il fattore scarsita'
    (scarcity_value_factor.py) opera solo sulla serie Grade 9 di PriceCharting - e'
    quello il prodotto da verificare, non la carta raw. Nessun filtro Cardmarket
    reale per grado esiste in questo URL: resta una ricerca testuale approssimata,
    l'utente deve comunque controllare a mano il grado dell'inserzione.
    """
    game = "OnePiece" if franchise == "one_piece" or "One Piece" in item_name or "OP-" in item_name or "OP0" in item_name else "Pokemon"
    lang_id = LANGUAGE_CODES.get(language.lower(), {}).get("id", 1)

    if custom_path:
        return f"https://www.cardmarket.com/en/{custom_path}?idLanguage={lang_id}"

    clean_name = item_name.replace("[JP]", "").replace("[OP-01]", "").replace("[OP-02]", "").replace("[OP-03]", "").replace("[OP-05]", "").replace("[OP-06]", "").replace("[OP-07]", "").replace("[OP-08]", "").strip()
    if item_type == "single":
        clean_name += " PSA 9"
    elif "box" not in clean_name.lower() and "bundle" not in clean_name.lower():
        clean_name += " Booster Box"

    encoded = urllib.parse.quote(clean_name)
    return f"https://www.cardmarket.com/en/{game}/Products/Search?searchString={encoded}&idLanguage={lang_id}"


def evaluate_cardmarket_item(
    item_id: str,
    meta: Dict[str, Any],
    live_price: float,
    current_date: Optional[datetime.date] = None
) -> Dict[str, Any]:
    """
    Valuta un asset incrociando il prezzo REALE Cardmarket con le regole quantitative:
    - Prezzo Massimo di Acquisto = MSRP * 1.15
    - Finestra d'acquisto: Mesi 4 - 14
    - Stato operativo:
        🟢 ACQUISTABILE A SCONTO
        ⚠️ BLOCCATO (Prezzo Superiore al Cap di Sicurezza)
        🔒 FINESTRA CHIUSA (Out of Print)
        🔄 TRIGGER VENDITA RAGGIUNTO (+70% o +100%)
    """
    if current_date is None:
        current_date = datetime.date.today()

    name = meta.get("name", item_id)
    franchise = meta.get("franchise", "pokemon")
    lang = meta.get("language", "en")
    tier = meta.get("set_tier", "B")
    msrp = float(meta.get("msrp", 140.0) or 140.0)
    max_buy_px = round(msrp * 1.15, 2)

    rel_str = meta.get("release_date")
    age_months = 999
    if rel_str:
        try:
            rel_dt = datetime.date.fromisoformat(rel_str)
            age_months = (current_date.year - rel_dt.year) * 12 + (current_date.month - rel_dt.month)
        except Exception:
            pass

    delta_vs_cap = round(live_price - max_buy_px, 2)
    delta_vs_cap_pct = round(((live_price - max_buy_px) / max_buy_px) * 100, 1) if max_buy_px > 0 else 0.0
    roi_vs_msrp = round(((live_price - msrp) / msrp) * 100, 1) if msrp > 0 else 0.0

    if live_price > max_buy_px:
        if age_months > 14 or roi_vs_msrp >= 70.0:
            status_code = "SELL_TARGET"
            status_label = "🔄 TARGET ROTAZIONE (+70% Raggiunto)"
            action_desc = f"Prezzo Cardmarket {live_price:.2f}€ (+{roi_vs_msrp}% vs MSRP). Liquidare Tranche 1 (40% scorte)."
            badge_color = "amber"
        else:
            status_code = "BLOCKED_OVERPRICED"
            status_label = f"⚠️ BLOCCATO (Prezzo > Cap {max_buy_px:.0f}€)"
            action_desc = f"In finestra temporale ({age_months}m) ma prezzo {live_price:.2f}€ supera il cap (+{delta_vs_cap_pct}%). NON COMPRARE."
            badge_color = "red"
    elif 4 <= age_months <= 14 and live_price <= max_buy_px and live_price > 0:
        status_code = "BUY_SIGNAL"
        status_label = "🟢 ACQUISTABILE (In Finestra a Sconto)"
        action_desc = f"Mese {age_months}/14. Prezzo reale {live_price:.2f}€ sotto il cap max ({max_buy_px:.2f}€). Margine: {-delta_vs_cap:.2f}€."
        badge_color = "emerald"
    elif 1 <= age_months < 4:
        status_code = "WATCHLIST"
        status_label = "🟡 IN AVVICINAMENTO"
        action_desc = f"Set recente ({age_months}m). Attendere la prima ondata di ristampa/sconti."
        badge_color = "yellow"
    else:
        status_code = "OOP_CLOSED"
        status_label = "🔒 FINESTRA CHIUSA (OOP)"
        action_desc = f"Set maturo ({age_months}m). Non più in finestra d'ingresso primario."
        badge_color = "slate"

    return {
        "item_id": item_id,
        "name": name,
        "franchise": franchise,
        "language": lang,
        "tier": tier,
        "age_months": age_months,
        "msrp": msrp,
        "live_price": live_price,
        "max_buy_px": max_buy_px,
        "delta_vs_cap": delta_vs_cap,
        "delta_vs_cap_pct": delta_vs_cap_pct,
        "roi_vs_msrp": roi_vs_msrp,
        "status_code": status_code,
        "status_label": status_label,
        "action_desc": action_desc,
        "badge_color": badge_color,
        "cardmarket_url": get_cardmarket_deep_link(name, franchise, lang)
    }
