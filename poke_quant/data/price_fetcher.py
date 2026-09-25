"""
poke_quant/data/price_fetcher.py — Download e parsing delle serie storiche reali dei prezzi
(mensili, basate su vendite transate reali) da PriceCharting e marketplace.
"""

from __future__ import annotations
import datetime
import json
import logging
import re
import time
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import requests

from poke_quant.config import DEFAULT_EUR_USD
from poke_quant.data.storage import (
    save_price_matrix, load_price_matrix, save_metadata, load_metadata
)
from poke_quant.data.fx_rates import rate_for_month

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

logger = logging.getLogger(__name__)


def fetch_pricecharting_cover_image_url(game_slug: str, item_slug: str) -> Optional[str]:
    """Scarica l'URL dell'immagine di copertina reale del prodotto da PriceCharting
    (non un placeholder generico). Il marcatore stabile e' <div id="product_details">
    seguito da <div class="cover"><img src=...> - le altre immagini nella pagina
    (tabelle di prodotti simili/ricerca) compaiono PRIMA di questo blocco, quindi
    cercare solo dopo id="product_details" evita di prendere la copertina di un
    prodotto diverso. Ritorna None se la pagina non ha questo blocco (slug rotto).

    Trovato verificando "molte immagini delle carte singole non le vedo": la
    dashboard richiede fino a 50-60 immagini in un solo caricamento pagina (BUY
    + AVOID box e singole), tutte sincrone senza pausa - un burst che fa scattare
    il rate limiting 429 di PriceCharting su una parte di esse (verificato
    empiricamente: sporadico, non un singolo prodotto rotto). Prima nessun retry:
    un singolo 429 diventava un "None" cacheato per 7 giorni intero
    (get_product_image in app.py) - un rate limit transitorio si trasformava in
    "niente immagine per una settimana". Aggiunto un retry breve con backoff
    solo sul 429 (rispetta Retry-After se presente, altrimenti 1s/2s) - non
    elimina il rate limit, ma recupera la stragrande maggioranza dei casi senza
    aspettare la prossima settimana."""
    url = f"https://www.pricecharting.com/game/{game_slug}/{item_slug}"
    last_attempt = 2
    resp = None
    for attempt in range(last_attempt + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
        except Exception as e:
            if attempt == last_attempt:
                logger.warning(f"Impossibile scaricare {url} per l'immagine: {e}")
                return None
            continue
        if resp.status_code == 429 and attempt < last_attempt:
            wait_s = float(resp.headers.get("Retry-After", 1.0 * (attempt + 1)))
            time.sleep(min(wait_s, 5.0))
            continue
        try:
            resp.raise_for_status()
        except Exception as e:
            logger.warning(f"Impossibile scaricare {url} per l'immagine: {e}")
            return None
        break
    text = resp.text
    marker = text.find('id="product_details"')
    if marker < 0:
        return None
    m = re.search(r"storage\.googleapis\.com/images\.pricecharting\.com/[^\"'\s]+\.(?:jpg|jpeg|png|webp)",
                  text[marker:marker + 2000])
    return f"https://{m.group(0)}" if m else None

# Catalogo di riferimento istituzionale per backtest (Box sigillati e Singole iconiche)
STANDARD_UNIVERSE = {
    # --- BOOSTER BOX SIGILLATI: SWORD & SHIELD (ERA MODERNA COMPLETA) ---
    "evolving_skies_bb": {
        "name": "Evolving Skies Booster Box",
        "type": "sealed",
        "product_type": "booster_box",
        "set_tier": "S",
        "game_slug": "pokemon-evolving-skies",
        "item_slug": "booster-box",
        "release_date": "2021-08-27",
        "msrp": 140.0,
        "era": "sword_shield",
        "chase_mascot": "Eeveelutions"
    },
    "lost_origin_bb": {
        "name": "Lost Origin Booster Box",
        "type": "sealed",
        "product_type": "booster_box",
        "set_tier": "A",
        "game_slug": "pokemon-lost-origin",
        "item_slug": "booster-box",
        "release_date": "2022-09-09",
        "msrp": 140.0,
        "era": "sword_shield",
        "chase_mascot": "Giratina"
    },
    "fusion_strike_bb": {
        "name": "Fusion Strike Booster Box",
        "type": "sealed",
        "product_type": "booster_box",
        "set_tier": "A",
        "game_slug": "pokemon-fusion-strike",
        "item_slug": "booster-box",
        "release_date": "2021-11-12",
        "msrp": 140.0,
        "era": "sword_shield",
        "chase_mascot": "Gengar"
    },
    "brilliant_stars_bb": {
        "name": "Brilliant Stars Booster Box",
        "type": "sealed",
        "product_type": "booster_box",
        "set_tier": "A",
        "game_slug": "pokemon-brilliant-stars",
        "item_slug": "booster-box",
        "release_date": "2022-02-25",
        "msrp": 140.0,
        "era": "sword_shield",
        "chase_mascot": "Charizard"
    },
    "chilling_reign_bb": {
        "name": "Chilling Reign Booster Box",
        "type": "sealed",
        "product_type": "booster_box",
        "set_tier": "B",
        "game_slug": "pokemon-chilling-reign",
        "item_slug": "booster-box",
        "release_date": "2021-06-18",
        "msrp": 140.0,
        "era": "sword_shield",
        "chase_mascot": "Galarian Birds"
    },
    "silver_tempest_bb": {
        "name": "Silver Tempest Booster Box",
        "type": "sealed",
        "product_type": "booster_box",
        "set_tier": "B",
        "game_slug": "pokemon-silver-tempest",
        "item_slug": "booster-box",
        "release_date": "2022-11-11",
        "msrp": 140.0,
        "era": "sword_shield",
        "chase_mascot": "Lugia"
    },
    "astral_radiance_bb": {
        "name": "Astral Radiance Booster Box",
        "type": "sealed",
        "product_type": "booster_box",
        "set_tier": "B",
        "game_slug": "pokemon-astral-radiance",
        "item_slug": "booster-box",
        "release_date": "2022-05-27",
        "msrp": 140.0,
        "era": "sword_shield",
        "chase_mascot": "Origin Formes"
    },
    "battle_styles_bb": {
        "name": "Battle Styles Booster Box",
        "type": "sealed",
        "product_type": "booster_box",
        "set_tier": "C",
        "game_slug": "pokemon-battle-styles",
        "item_slug": "booster-box",
        "release_date": "2021-03-19",
        "msrp": 140.0,
        "era": "sword_shield",
        "chase_mascot": "Tyranitar"
    },
    "vivid_voltage_bb": {
        "name": "Vivid Voltage Booster Box",
        "type": "sealed",
        "product_type": "booster_box",
        "set_tier": "C",
        "game_slug": "pokemon-vivid-voltage",
        "item_slug": "booster-box",
        "release_date": "2020-11-13",
        "msrp": 140.0,
        "era": "sword_shield",
        "chase_mascot": "Pikachu"
    },
    "darkness_ablaze_bb": {
        "name": "Darkness Ablaze Booster Box",
        "type": "sealed",
        "product_type": "booster_box",
        "set_tier": "C",
        "game_slug": "pokemon-darkness-ablaze",
        "item_slug": "booster-box",
        "release_date": "2020-08-14",
        "msrp": 140.0,
        "era": "sword_shield",
        "chase_mascot": "Charizard VMAX"
    },
    "rebel_clash_bb": {
        "name": "Rebel Clash Booster Box",
        "type": "sealed",
        "product_type": "booster_box",
        "set_tier": "C",
        "game_slug": "pokemon-rebel-clash",
        "item_slug": "booster-box",
        "release_date": "2020-05-01",
        "msrp": 140.0,
        "era": "sword_shield",
        "chase_mascot": "None"
    },

    # --- BOOSTER BOX SIGILLATI: SCARLET & VIOLET ---
    "paldea_evolved_bb": {
        "name": "Paldea Evolved Booster Box",
        "type": "sealed",
        "product_type": "booster_box",
        "set_tier": "B",
        "game_slug": "pokemon-paldea-evolved",
        "item_slug": "booster-box",
        "release_date": "2023-06-09",
        "msrp": 160.0,
        "era": "scarlet_violet",
        "chase_mascot": "Magikarp"
    },
    "twilight_masquerade_bb": {
        "name": "Twilight Masquerade Booster Box",
        "type": "sealed",
        "product_type": "booster_box",
        "set_tier": "B",
        "game_slug": "pokemon-twilight-masquerade",
        "item_slug": "booster-box",
        "release_date": "2024-05-24",
        "msrp": 160.0,
        "era": "scarlet_violet",
        "chase_mascot": "Greninja"
    },

    # --- BOOSTER BOX SIGILLATI: SUN & MOON (CICLO COMPLETO VINTAGE/MID-ERA) ---
    "team_up_bb": {
        "name": "Team Up Booster Box",
        "type": "sealed",
        "product_type": "booster_box",
        "set_tier": "S",
        "game_slug": "pokemon-team-up",
        "item_slug": "booster-box",
        "release_date": "2019-02-15",
        "msrp": 120.0,
        "era": "sun_moon",
        "chase_mascot": "Latias & Latios"
    },
    "cosmic_eclipse_bb": {
        "name": "Cosmic Eclipse Booster Box",
        "type": "sealed",
        "product_type": "booster_box",
        "set_tier": "S",
        "game_slug": "pokemon-cosmic-eclipse",
        "item_slug": "booster-box",
        "release_date": "2019-11-01",
        "msrp": 120.0,
        "era": "sun_moon",
        "chase_mascot": "Arceus Dialga Palkia"
    },
    "unbroken_bonds_bb": {
        "name": "Unbroken Bonds Booster Box",
        "type": "sealed",
        "product_type": "booster_box",
        "set_tier": "S",
        "game_slug": "pokemon-unbroken-bonds",
        "item_slug": "booster-box",
        "release_date": "2019-05-03",
        "msrp": 120.0,
        "era": "sun_moon",
        "chase_mascot": "Reshiram & Charizard"
    },
    "unified_minds_bb": {
        "name": "Unified Minds Booster Box",
        "type": "sealed",
        "product_type": "booster_box",
        "set_tier": "S",
        "game_slug": "pokemon-unified-minds",
        "item_slug": "booster-box",
        "release_date": "2019-08-02",
        "msrp": 120.0,
        "era": "sun_moon",
        "chase_mascot": "Mewtwo & Mew"
    },
    "ultra_prism_bb": {
        "name": "Ultra Prism Booster Box",
        "type": "sealed",
        "product_type": "booster_box",
        "set_tier": "A",
        "game_slug": "pokemon-ultra-prism",
        "item_slug": "booster-box",
        "release_date": "2018-02-02",
        "msrp": 120.0,
        "era": "sun_moon",
        "chase_mascot": "Lillie"
    },
    "crimson_invasion_bb": {
        "name": "Crimson Invasion Booster Box",
        "type": "sealed",
        "product_type": "booster_box",
        "set_tier": "C",
        "game_slug": "pokemon-crimson-invasion",
        "item_slug": "booster-box",
        "release_date": "2017-11-03",
        "msrp": 120.0,
        "era": "sun_moon",
        "chase_mascot": "None"
    },

    # --- SPECIAL SETS (NO BOOSTER BOX: ETB & BUNDLE ONLY) ---
    "crown_zenith_etb": {
        "name": "Crown Zenith Elite Trainer Box",
        "type": "sealed",
        "product_type": "specialty_etb",
        "set_tier": "A",
        "game_slug": "pokemon-crown-zenith",
        "item_slug": "elite-trainer-box",
        "release_date": "2023-01-20",
        "msrp": 55.0,
        "era": "sword_shield",
        "chase_mascot": "Giratina / Mewtwo"
    },
    "celebrations_etb": {
        "name": "Celebrations Elite Trainer Box",
        "type": "sealed",
        "product_type": "specialty_etb",
        "set_tier": "A",
        "game_slug": "pokemon-celebrations",
        "item_slug": "elite-trainer-box",
        "release_date": "2021-10-08",
        "msrp": 55.0,
        "era": "sword_shield",
        "chase_mascot": "Base Charizard Reprints"
    },
    "shining_fates_etb": {
        "name": "Shining Fates Elite Trainer Box",
        "type": "sealed",
        "product_type": "specialty_etb",
        "set_tier": "C",
        "game_slug": "pokemon-shining-fates",
        "item_slug": "elite-trainer-box",
        "release_date": "2021-02-19",
        "msrp": 55.0,
        "era": "sword_shield",
        "chase_mascot": "Shiny Charizard VMAX"
    },
    "hidden_fates_etb": {
        "name": "Hidden Fates Elite Trainer Box",
        "type": "sealed",
        "product_type": "specialty_etb",
        "set_tier": "S",
        "game_slug": "pokemon-hidden-fates",
        "item_slug": "elite-trainer-box",
        "release_date": "2019-09-20",
        "msrp": 50.0,
        "era": "sun_moon",
        "chase_mascot": "Shiny Charizard GX"
    },
    "scarlet_violet_151_etb": {
        "name": "Scarlet & Violet 151 Elite Trainer Box",
        "type": "sealed",
        "product_type": "specialty_etb",
        "set_tier": "S",
        "game_slug": "pokemon-scarlet-&-violet-151",
        "item_slug": "elite-trainer-box",
        "release_date": "2023-09-22",
        "msrp": 55.0,
        "era": "scarlet_violet",
        "chase_mascot": "Original 151 Gen 1"
    },
    "scarlet_violet_151_bundle": {
        "name": "Scarlet & Violet 151 Booster Bundle (6 Packs)",
        "type": "sealed",
        "product_type": "specialty_bundle",
        "set_tier": "S",
        "game_slug": "pokemon-scarlet-&-violet-151",
        "item_slug": "booster-bundle",
        "release_date": "2023-09-22",
        "msrp": 30.0,
        "era": "scarlet_violet",
        "chase_mascot": "Original 151 Gen 1"
    },

    # --- TOP CHASE SINGLES MODERNE (Alternate Art / SIR) ---
    "umbreon_vmax_215": {
        "name": "Umbreon VMAX Alternate Art (Moonbreon)",
        "type": "single",
        "game_slug": "pokemon-evolving-skies",
        "item_slug": "umbreon-vmax-215",
        "release_date": "2021-08-27",
        "launch_price": 280.0,
        "era": "modern"
    },
    "rayquaza_vmax_218": {
        "name": "Rayquaza VMAX Alternate Art",
        "type": "single",
        "game_slug": "pokemon-evolving-skies",
        "item_slug": "rayquaza-vmax-218",
        "release_date": "2021-08-27",
        "launch_price": 240.0,
        "era": "modern"
    },
    "giratina_v_186": {
        "name": "Giratina V Alternate Art",
        "type": "single",
        "game_slug": "pokemon-lost-origin",
        "item_slug": "giratina-v-186",
        "release_date": "2022-09-09",
        "launch_price": 220.0,
        "era": "modern"
    },
    "gengar_vmax_271": {
        "name": "Gengar VMAX Alternate Art",
        "type": "single",
        "game_slug": "pokemon-fusion-strike",
        "item_slug": "gengar-vmax-271",
        "release_date": "2021-11-12",
        "launch_price": 130.0,
        "era": "modern"
    },
    "charizard_v_154": {
        "name": "Charizard V Alternate Art",
        "type": "single",
        "game_slug": "pokemon-brilliant-stars",
        "item_slug": "charizard-v-154",
        "release_date": "2022-02-25",
        "launch_price": 180.0,
        "era": "modern"
    },

    # --- VINTAGE BLUE-CHIPS DI RIFERIMENTO ---
    "charizard_base_unlimited": {
        "name": "Charizard #4 Base Set Unlimited",
        "type": "single",
        "game_slug": "pokemon-base-set",
        "item_slug": "charizard-4",
        "release_date": "1999-01-09",
        "launch_price": 350.0,
        "era": "vintage"
    }
}


def fetch_pricecharting_series(
    game_slug: str, item_slug: str, eur_usd_series: Optional[pd.Series] = None
) -> Dict[str, pd.Series]:
    """
    Scarica la serie temporale mensile reale da PriceCharting.
    Restituisce un dict con la serie 'raw' (ungraded/box) ed eventualmente 'graded'.

    Conversione EUR: se eur_usd_series è fornita (vedi poke_quant.data.fx_rates,
    tasso storico reale Twelve Data), converte OGNI mese al proprio tasso reale
    invece della costante fissa DEFAULT_EUR_USD. Il tasso è oscillato da 1.22 a
    0.97 nel periodo 2021-2026: usare un default fisso introduce un errore
    sistematico fino al 12% in singoli mesi. Se eur_usd_series è None, mantiene
    il comportamento precedente (costante fissa) per compatibilità.
    """
    url = f"https://www.pricecharting.com/game/{game_slug}/{item_slug}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        logger.warning(f"Impossibile scaricare {url}: {e}")
        return {}

    m = re.search(r'VGPC\.chart_data\s*=\s*(\{.*?\});', resp.text, re.DOTALL)
    if not m:
        logger.warning(f"Dati chart non trovati per {url}")
        return {}

    try:
        data = json.loads(m.group(1))
    except Exception as e:
        logger.warning(f"Errore parsing JSON chart per {url}: {e}")
        return {}

    res: Dict[str, pd.Series] = {}

    # Serie raw / loose / box
    raw_points = data.get("used", [])
    if raw_points:
        dates = []
        prices_eur = []
        for p in raw_points:
            if len(p) >= 2 and p[1] > 0:
                dt_obj = datetime.datetime.fromtimestamp(p[0] / 1000.0)
                dt = dt_obj.strftime("%Y-%m-01")
                px_usd = p[1] / 100.0
                rate = rate_for_month(eur_usd_series, pd.Timestamp(dt_obj), DEFAULT_EUR_USD) if eur_usd_series is not None else DEFAULT_EUR_USD
                px_eur = round(px_usd / rate, 2)
                dates.append(dt)
                prices_eur.append(px_eur)
        if dates:
            s = pd.Series(prices_eur, index=pd.to_datetime(dates)).sort_index()
            s = s[~s.index.duplicated(keep="last")]
            res["raw"] = s

    # Serie graded (se disponibile)
    graded_points = data.get("graded", [])
    if graded_points:
        dates_g = []
        prices_g = []
        for p in graded_points:
            if len(p) >= 2 and p[1] > 0:
                dt_obj = datetime.datetime.fromtimestamp(p[0] / 1000.0)
                dt = dt_obj.strftime("%Y-%m-01")
                px_usd = p[1] / 100.0
                rate = rate_for_month(eur_usd_series, pd.Timestamp(dt_obj), DEFAULT_EUR_USD) if eur_usd_series is not None else DEFAULT_EUR_USD
                px_eur = round(px_usd / rate, 2)
                dates_g.append(dt)
                prices_g.append(px_eur)
        if dates_g:
            sg = pd.Series(prices_g, index=pd.to_datetime(dates_g)).sort_index()
            sg = sg[~sg.index.duplicated(keep="last")]
            res["graded"] = sg

    return res


def build_and_cache_universe(
    universe_dict: Optional[Dict[str, Dict[str, Any]]] = None,
    force_refresh: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Costruisce la matrice prezzi storica per tutti gli item dell'universo.
    Se presente in cache e force_refresh=False, la carica direttamente dal disco.
    """
    if universe_dict is None:
        universe_dict = STANDARD_UNIVERSE

    if not force_refresh:
        cached_df = load_price_matrix()
        cached_meta = load_metadata()
        if cached_df is not None and cached_meta is not None:
            return cached_df, cached_meta

    series_map = {}
    metadata_map = {}

    for item_id, info in universe_dict.items():
        logger.info(f"Download serie per {info['name']}...")
        s_dict = fetch_pricecharting_series(info["game_slug"], info["item_slug"])
        if "raw" in s_dict and not s_dict["raw"].empty:
            series_map[item_id] = s_dict["raw"]
            meta_copy = dict(info)
            if "graded" in s_dict and not s_dict["graded"].empty:
                meta_copy["last_psa_price"] = float(s_dict["graded"].iloc[-1])
            metadata_map[item_id] = meta_copy

    if not series_map:
        raise ValueError("Nessuna serie storica scaricata con successo.")

    # Combina in una singola matrice mensile
    price_df = pd.DataFrame(series_map).sort_index()

    # Forward fill per gestire gap o rilasci sfalsati
    price_df = price_df.ffill()

    save_price_matrix(price_df)
    save_metadata(metadata_map)

    return price_df, metadata_map
