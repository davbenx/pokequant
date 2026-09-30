"""
poke_quant/data/price_fetcher.py — Download e parsing delle serie storiche reali dei prezzi
(mensili, basate su vendite transate reali) da PriceCharting e marketplace.

LIMITE NOTO E ACCETTATO (2026-09-29, Fase 2 del piano dati - decisione esplicita
dell'utente "accetta il limite, concentrati su prezzi/gradazioni/popolazioni
realmente ottenibili"): nessun campo di LIQUIDITA'/VOLUME esiste in nessuna fonte
oggi raggiungibile da questo progetto.
  - VGPC.chart_data (parsato sotto) espone SOLO coppie [timestamp_ms, prezzo] per
    livello di gradazione - nessun conteggio di vendite/volume, verificato a mano
    ispezionando il JSON reale di piu' pagine PriceCharting prima di scrivere
    questo modulo.
  - Lo scraping automatico di Cardmarket (che mostra "N venduti" per inserzione)
    e' bloccato (verificato altrove nel repo - non ripetuto qui).
  - poke_quant/data/liquidity_filter.py compensa con un PROXY indiretto
    (stabilita' della serie prezzo: niente salti/range anomali) - non e' una
    misura di volume reale, e va trattato come tale in qualunque analisi futura
    che usi questi dati.
Non inseguito oltre per scelta esplicita dell'utente - se emerge in futuro una
fonte reale di volume/liquidita', va aggiunta qui con lo stesso standard di
verifica dal vivo usato per ogni altro dato di questo progetto (mai fabbricata).
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
from poke_quant.data.fx_rates import rate_for_month, get_current_eur_usd_rate

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


def _variant_candidate_slugs(item_slug: str, variant_type: str) -> List[str]:
    """Costruisce i possibili slug della pagina PriceCharting dedicata alla
    variante (es. '<nome>-1st-edition-<numero>' o '1st-edition-<nome>-<numero>') -
    condiviso tra fetch_pricecharting_variant_grade9 e
    fetch_pricecharting_variant_grade_tier."""
    parts = item_slug.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        name_part, num_part = parts[0], parts[1]
        return [f"{name_part}-{variant_type}-{num_part}", f"{variant_type}-{name_part}-{num_part}"]
    return [f"{item_slug}-{variant_type}", f"{variant_type}-{item_slug}"]


def fetch_pricecharting_variant_grade_tier(
    game_slug: str, item_slug: str, variant_type: str = "1st-edition", tier: str = "grade9"
) -> Optional[Tuple[float, float, str]]:
    """
    Come fetch_pricecharting_grade_tier_price, ma sulla pagina dedicata della
    VARIANTE (1st Edition/No Symbol/Shadowless) invece che sulla stampa
    standard - prova il prezzo REALE per lo specifico grado richiesto.

    TROVATO (richiesta esplicita dell'utente dopo il bug del voto slab che non
    aggiornava il prezzo per le varianti, 2026-09-29: "deve prendere i dati
    reali quanto possibile"): PriceCharting espone una colonna reale per grado
    ANCHE sulla pagina della variante (stesso schema VGPC.chart_data della
    stampa standard) - prima veniva letta SOLO la colonna Grado 9 ("graded"),
    sempre, e ogni altro grado veniva stimato algoritmicamente anche quando un
    dato reale per quel grado specifico era gia' disponibile sulla stessa
    pagina. fetch_pricecharting_variant_grade9() ora e' un alias di questa
    funzione con tier="grade9" (comportamento identico a prima per chi la
    chiama gia').

    Restituisce (prezzo_eur, prezzo_usd, url) oppure None se non disponibile.
    Consulta prima la cache locale data_cache/variant_prices_grade9.json per
    garantire funzionamento istantaneo e affidabile anche su ambienti cloud.
    """
    from pathlib import Path
    cache_path = Path(__file__).resolve().parent.parent.parent / "data_cache" / "variant_prices_grade9.json"
    fx_rate = get_current_eur_usd_rate()

    tier_name, raw_key = _normalize_grade_tier(tier)

    # Mezzi voti interpolati (8.5/7.5): stessa logica di
    # fetch_pricecharting_grade_tier_price, ma sui due gradi reali vicini
    # SULLA PAGINA DELLA VARIANTE, non su quella standard.
    if tier_name == "grade8_5":
        p8 = fetch_pricecharting_variant_grade_tier(game_slug, item_slug, variant_type, tier="grade8")
        p9 = fetch_pricecharting_variant_grade_tier(game_slug, item_slug, variant_type, tier="grade9")
        if p8 and p9:
            return round((p8[0] + p9[0]) / 2, 2), round((p8[1] + p9[1]) / 2, 2), p8[2]
        if p8:
            return round(p8[0] * 1.25, 2), round(p8[1] * 1.25, 2), p8[2]
        if p9:
            return round(p9[0] * 0.78, 2), round(p9[1] * 0.78, 2), p9[2]
        return None
    if tier_name == "grade7_5":
        p7 = fetch_pricecharting_variant_grade_tier(game_slug, item_slug, variant_type, tier="grade7")
        p8 = fetch_pricecharting_variant_grade_tier(game_slug, item_slug, variant_type, tier="grade8")
        if p7 and p8:
            return round((p7[0] + p8[0]) / 2, 2), round((p7[1] + p8[1]) / 2, 2), p7[2]
        if p7:
            return round(p7[0] * 1.18, 2), round(p7[1] * 1.18, 2), p7[2]
        if p8:
            return round(p8[0] * 0.85, 2), round(p8[1] * 0.85, 2), p8[2]
        return None

    # Grado 9 mantiene ESATTAMENTE la chiave cache originale (senza suffisso
    # tier) per restare compatibile con data_cache/variant_prices_grade9.json
    # gia' popolato da fetch_pricecharting_variant_grade9() nelle sessioni
    # precedenti - solo i tier nuovi (10/9.5/8/7) usano un suffisso dedicato.
    cache_key = (
        f"{game_slug}:{item_slug}:{variant_type}" if tier_name == "grade9"
        else f"{game_slug}:{item_slug}:{variant_type}:{tier_name}"
    )

    # 1. Verifica cache persistente locale
    if cache_path.exists():
        try:
            cached_data = json.loads(cache_path.read_text(encoding="utf-8"))
            if cache_key in cached_data:
                entry = cached_data[cache_key]
                usd = float(entry["usd"])
                eur = round(usd / fx_rate, 2)
                return eur, usd, entry["url"]
        except Exception:
            pass

    if not raw_key:
        return None

    # 2. Se non presente in cache, tenta scraping in tempo reale
    for slug in _variant_candidate_slugs(item_slug, variant_type):
        url = f"https://www.pricecharting.com/game/{game_slug}/{slug}"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=6)
            if resp.status_code == 200:
                m = re.search(r'VGPC\.chart_data\s*=\s*(\{.*?\});', resp.text, re.DOTALL)
                if m:
                    data = json.loads(m.group(1))
                    pts = data.get(raw_key, [])
                    if pts and len(pts[-1]) >= 2 and pts[-1][1] > 0:
                        usd = round(pts[-1][1] / 100.0, 2)
                        eur = round(usd / fx_rate, 2)
                        # Salva in cache per le prossime chiamate
                        try:
                            cached_data = {}
                            if cache_path.exists():
                                cached_data = json.loads(cache_path.read_text(encoding="utf-8"))
                            cached_data[cache_key] = {
                                "eur": eur,
                                "usd": usd,
                                "url": url,
                                "variant_type": variant_type,
                                "game_slug": game_slug,
                                "item_slug": item_slug,
                                "tier": tier_name,
                            }
                            cache_path.write_text(json.dumps(cached_data, indent=2), encoding="utf-8")
                        except Exception:
                            pass
                        return eur, usd, url
        except Exception:
            continue
    return None


def fetch_pricecharting_variant_grade9(
    game_slug: str, item_slug: str, variant_type: str = "1st-edition"
) -> Optional[Tuple[float, float, str]]:
    """Alias storico di fetch_pricecharting_variant_grade_tier(tier="grade9") -
    mantenuto per compatibilita' con i chiamanti/test esistenti."""
    return fetch_pricecharting_variant_grade_tier(game_slug, item_slug, variant_type, tier="grade9")


def _normalize_grade_tier(tier: str) -> Tuple[str, Optional[str]]:
    """Mappa un grado (10.0/9.5/9.0/8.5/8.0/7.5/7.0, in qualunque formattazione)
    a (tier_name interno, chiave raw del JSON VGPC.chart_data di PriceCharting).
    raw_key=None per i mezzi voti interpolati (8.5/7.5, PriceCharting non li
    traccia come colonna a se' - vedi interpolazione in
    fetch_pricecharting_grade_tier_price/fetch_pricecharting_variant_grade_tier).
    Condivisa tra la pagina standard e quella di variante: stessa identica
    mappatura, PriceCharting usa lo stesso schema di colonne su entrambe."""
    normalized_tier = tier.lower().replace(".", "_").replace(" ", "")
    if "10" in normalized_tier:
        return "psa10", "manualonly"
    if "9_5" in normalized_tier or "95" in normalized_tier:
        return "grade9_5", "boxonly"
    if "8_5" in normalized_tier or "85" in normalized_tier:
        return "grade8_5", None
    if "8" in normalized_tier:
        return "grade8", "new"
    if "7_5" in normalized_tier or "75" in normalized_tier:
        return "grade7_5", None
    if "7" in normalized_tier:
        return "grade7", "cib"
    return "grade9", "graded"


def fetch_pricecharting_grade_tier_price(
    game_slug: str, item_slug: str, tier: str = "psa10", item_id: Optional[str] = None
) -> Optional[Tuple[float, float, str, str]]:
    """
    Recupera il prezzo reale di un livello di grado specifico (psa10, grade9_5, grade9) da PriceCharting.
    Restituisce (prezzo_eur, prezzo_usd, url, fonte) oppure None se non disponibile.

    Priorità di ricerca:
      1. Cache storica locale data_cache/grade_ladder_prices.json
      2. Cache locale dedicata data_cache/pricecharting_tier_cache.json
      3. Richiesta live a PriceCharting con parsing di VGPC.chart_data
    """
    from pathlib import Path
    data_cache_dir = Path(__file__).resolve().parent.parent.parent / "data_cache"
    ladder_file = data_cache_dir / "grade_ladder_prices.json"
    tier_cache_file = data_cache_dir / "pricecharting_tier_cache.json"
    fx_rate = get_current_eur_usd_rate()
    url = f"https://www.pricecharting.com/game/{game_slug}/{item_slug}"

    tier_name, raw_key = _normalize_grade_tier(tier)

    # Gestione specifica per mezzi voti interpolati (8.5 e 7.5)
    if tier_name == "grade8_5":
        cache_key = f"{game_slug}:{item_slug}:grade8_5"
        if tier_cache_file.exists():
            try:
                cached_tiers = json.loads(tier_cache_file.read_text(encoding="utf-8"))
                if cache_key in cached_tiers:
                    entry = cached_tiers[cache_key]
                    usd = float(entry["usd"])
                    eur = round(usd / fx_rate, 2)
                    return eur, usd, entry.get("url", url), "PriceCharting Cache Reale (G8.5)"
            except Exception:
                pass
        p8 = fetch_pricecharting_grade_tier_price(game_slug, item_slug, tier="grade8", item_id=item_id)
        p9 = fetch_pricecharting_grade_tier_price(game_slug, item_slug, tier="grade9", item_id=item_id)
        if p8 and p9:
            eur = round((p8[0] + p9[0]) / 2, 2)
            usd = round((p8[1] + p9[1]) / 2, 2)
            url = p8[2]
            return eur, usd, url, "PriceCharting Reale Interpolato (G8-G9)"
        elif p8:
            return round(p8[0] * 1.25, 2), round(p8[1] * 1.25, 2), p8[2], "PriceCharting G8 + Premio Mezzo Voto (+25%)"
        elif p9:
            return round(p9[0] * 0.78, 2), round(p9[1] * 0.78, 2), p9[2], "PriceCharting G9 Rettificato G8.5 (0.78x)"

    if tier_name == "grade7_5":
        cache_key = f"{game_slug}:{item_slug}:grade7_5"
        if tier_cache_file.exists():
            try:
                cached_tiers = json.loads(tier_cache_file.read_text(encoding="utf-8"))
                if cache_key in cached_tiers:
                    entry = cached_tiers[cache_key]
                    usd = float(entry["usd"])
                    eur = round(usd / fx_rate, 2)
                    return eur, usd, entry.get("url", url), "PriceCharting Cache Reale (G7.5)"
            except Exception:
                pass
        p7 = fetch_pricecharting_grade_tier_price(game_slug, item_slug, tier="grade7", item_id=item_id)
        p8 = fetch_pricecharting_grade_tier_price(game_slug, item_slug, tier="grade8", item_id=item_id)
        if p7 and p8:
            eur = round((p7[0] + p8[0]) / 2, 2)
            usd = round((p7[1] + p8[1]) / 2, 2)
            url = p7[2]
            return eur, usd, url, "PriceCharting Reale Interpolato (G7-G8)"
        elif p7:
            return round(p7[0] * 1.18, 2), round(p7[1] * 1.18, 2), p7[2], "PriceCharting G7 + Premio Mezzo Voto (+18%)"
        elif p8:
            return round(p8[0] * 0.85, 2), round(p8[1] * 0.85, 2), p8[2], "PriceCharting G8 Rettificato G7.5 (0.85x)"

    # 1. Verifica in grade_ladder_prices.json
    if ladder_file.exists():
        try:
            ladder_data = json.loads(ladder_file.read_text(encoding="utf-8"))
            target_key = item_id if (item_id and item_id in ladder_data) else None
            if not target_key:
                clean_slug = item_slug.replace("-", "_")
                for k in ladder_data.keys():
                    if clean_slug in k or (item_id and item_id.lower() == k.lower()):
                        target_key = k
                        break
            if target_key and tier_name in ladder_data[target_key]:
                series = ladder_data[target_key][tier_name]
                if series:
                    sorted_dates = sorted(series.keys())
                    eur = float(series[sorted_dates[-1]])
                    usd = round(eur * fx_rate, 2)
                    return eur, usd, url, "PriceCharting Storico Reale"
        except Exception:
            pass

    # 2. Verifica in pricecharting_tier_cache.json
    cache_key = f"{game_slug}:{item_slug}:{tier_name}"
    if tier_cache_file.exists():
        try:
            cached_tiers = json.loads(tier_cache_file.read_text(encoding="utf-8"))
            if cache_key in cached_tiers:
                entry = cached_tiers[cache_key]
                usd = float(entry["usd"])
                eur = round(usd / fx_rate, 2)
                return eur, usd, entry.get("url", url), "PriceCharting Cache Reale"
        except Exception:
            pass

    # 3. Richiesta live e parsing VGPC.chart_data
    if raw_key:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=6)
            if resp.status_code == 200:
                m = re.search(r'VGPC\.chart_data\s*=\s*(\{.*?\});', resp.text, re.DOTALL)
                if m:
                    data = json.loads(m.group(1))
                    pts = data.get(raw_key, [])
                    if pts and len(pts[-1]) >= 2 and pts[-1][1] > 0:
                        usd = round(pts[-1][1] / 100.0, 2)
                        eur = round(usd / fx_rate, 2)
                        # Salva in cache
                        try:
                            cached_tiers = {}
                            if tier_cache_file.exists():
                                cached_tiers = json.loads(tier_cache_file.read_text(encoding="utf-8"))
                            cached_tiers[cache_key] = {
                                "eur": eur,
                                "usd": usd,
                                "url": url,
                                "tier": tier_name,
                                "game_slug": game_slug,
                                "item_slug": item_slug,
                                "date": datetime.datetime.now().strftime("%Y-%m-%d"),
                            }
                            tier_cache_file.write_text(json.dumps(cached_tiers, indent=2), encoding="utf-8")
                        except Exception:
                            pass
                        return eur, usd, url, "PriceCharting Live Reale"
        except Exception:
            pass

    return None


def parse_pricecharting_url_or_slug(text: str) -> Optional[Tuple[str, str]]:
    """
    Estrae (game_slug, item_slug) da qualsiasi formato di URL o slug PriceCharting:
    - https://www.pricecharting.com/game/pokemon-base-set/charizard-4
    - http://pricecharting.com/game/pokemon-surging-sparks/pikachu-ex-238#graded
    - /game/pokemon-team-up/gengar-&-mimikyu-gx-165?sort=price
    - pokemon-base-set/charizard-4
    Ritorna None se il testo non corrisponde a un formato PriceCharting valido.
    """
    if not text:
        return None
    s = text.strip().split("#")[0].split("?")[0].rstrip("/")
    m = re.search(r"pricecharting\.com/game/([a-zA-Z0-9_\-%&]+)/([a-zA-Z0-9_\-%&]+)", s, re.IGNORECASE)
    if m:
        return m.group(1), m.group(2)
    m2 = re.search(r"^([a-zA-Z0-9_\-%&]+)/([a-zA-Z0-9_\-%&]+)$", s)
    if m2:
        return m2.group(1), m2.group(2)
    return None


def search_metadata_card_by_query(query: str, metadata: dict) -> Optional[Tuple[str, dict]]:
    """
    Cerca nel catalogo metadata una carta singola che corrisponde al testo o alla query digitata dall'utente.
    Supporta corrispondenza esatta per slug/item_id, oppure ricerca multi-parola con scoring su nome, slug e set.
    Restituisce (item_id, item_info) oppure None.
    """
    q = query.lower().strip()
    if not q or len(q) < 2:
        return None
    words = [w for w in re.split(r"[\s\-_#]+", q) if w]
    if not words:
        return None

    clean_q = q.replace(" ", "-").replace("_", "-")
    # 1. Corrispondenza esatta per slug o item_id
    for item_id, info in metadata.items():
        if info.get("type") != "single":
            continue
        if item_id.lower() == clean_q or info.get("item_slug", "").lower() == clean_q:
            return item_id, info

    # 2. Corrispondenza a parole chiave con scoring
    best_match = None
    best_score = 0.0
    for item_id, info in metadata.items():
        if info.get("type") != "single":
            continue
        name_lower = info.get("name", "").lower()
        slug_lower = info.get("item_slug", "").lower()
        game_lower = info.get("game_slug", "").lower()
        full_text = f"{name_lower} {slug_lower} {game_lower}"
        matched_words = sum(1 for w in words if w in full_text)
        if matched_words > 0:
            ratio = matched_words / len(words)
            if ratio >= 0.5:
                # Scoring: ratio elevato, numero di parole matchate e penalizzazione per nomi troppo lunghi
                score = (ratio * 100.0) + (matched_words * 10.0) - (len(name_lower) * 0.05)
                if score > best_score:
                    best_score = score
                    best_match = (item_id, info)

    return best_match
