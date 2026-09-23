#!/usr/bin/env python3
"""
scripts/discover_promo_singles.py — Scopre carte promozionali (SVP, SWSH Black Star
Promos, ecc.) — set esclusi da discover_chase_cards.py perché non hanno mai un
booster box sigillato da cui la pipeline "bottom-up" parte (vedi SET_IDS in quel
file, derivato dagli item _bb/_etb esistenti). Le promo si distribuiscono per
eventi/preorder, non in scatole - quindi non sono mai entrate nell'elenco di ricerca.

Verificato: PriceCharting non separa le promo per era come i set normali - le
raggruppa TUTTE sotto un unico game_slug "pokemon-promo", con slug interni non
uniformi (a volte nome+numero, a volte solo nome, a volte con codice di era
incorporato come "swsh261"). slugify_card() di discover_chase_cards.py NON
funziona qui - si usa invece la ricerca sul sito (resolve_promo_slug) e si
verifica SEMPRE con un fetch storico reale prima di aggiungere qualsiasi carta,
esattamente come per i set normali.

Stessa disciplina anti-survivorship-bias delle singole normali: ogni carta trovata
sopra la soglia di prezzo va in "chase_price_filter_survivorship_biased", un
campione casuale (nessun filtro di prezzo/rarita') va in "random_control" - in
un'UNICA passata per set, per non raddoppiare le chiamate a pokemontcg.io.

Uso:
    python scripts/discover_promo_singles.py [--min-price 40] [--control-per-set 5] [--dry-run]
"""

import argparse
import random
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, save_metadata
from poke_quant.data.price_fetcher import fetch_pricecharting_series
from poke_quant.data.fx_rates import load_eur_usd_series
from scripts.discover_chase_cards import fetch_set_cards, make_item_id, CHASE_RARITIES

PROMO_GAME_SLUG = "pokemon-promo"

# 9 set promozionali reali confermati su pokemontcg.io (vedi /v2/sets, "promo" nel
# nome o nell'id) - nessun booster box esiste per nessuno di questi, per costruzione.
PROMO_SET_IDS = {
    "basep": "wizards_black_star_promos",
    "np": "nintendo_black_star_promos",
    "dpp": "dp_black_star_promos",
    "hsp": "hgss_black_star_promos",
    "bwp": "bw_black_star_promos",
    "xyp": "xy_black_star_promos",
    "smp": "sm_black_star_promos",
    "swshp": "swsh_black_star_promos",
    "svp": "sv_black_star_promos",
}

RANDOM_SEED = 11
HEADERS = {"User-Agent": "Mozilla/5.0"}


def _slugify_name(name: str) -> str:
    s = name.lower().replace("'", "").replace(".", "")
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return re.sub(r"-+", "-", s).strip("-")


def resolve_promo_slug(card_name: str, retries: int = 3) -> str | None:
    """Trova l'item_slug reale su pokemon-promo cercando per nome. Nessuno slug e'
    prevedibile qui (vedi docstring del modulo) - a differenza dei set normali."""
    query = requests.utils.quote(card_name)
    url = f"https://www.pricecharting.com/search-products?type=prices&q={query}"
    target_prefix = _slugify_name(card_name)
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            links = re.findall(r"/game/pokemon-promo/([a-z0-9-]+)", r.text)
            for slug in dict.fromkeys(links):  # dedup preservando l'ordine
                if slug.startswith(target_prefix) or target_prefix.startswith(slug):
                    return slug
            return None
        except Exception as e:
            print(f"  [retry {attempt}] ricerca '{card_name}': {e}")
            time.sleep(3)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-price", type=float, default=40.0)
    parser.add_argument("--control-per-set", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    metadata = load_metadata()
    eur_usd = load_eur_usd_series()
    existing_keys = {(v.get("game_slug"), v.get("item_slug")) for v in metadata.values()}

    chase_candidates, control_candidates = [], []
    for set_id, era in PROMO_SET_IDS.items():
        cards = fetch_set_cards(set_id)
        print(f"{set_id} ({era}): {len(cards)} carte", flush=True)
        if not cards:
            continue
        for c in cards:
            cm = c.get("cardmarket", {}).get("prices", {})
            price = max(cm.get("averageSellPrice", 0) or 0, cm.get("trendPrice", 0) or 0)
            entry = {
                "era": era, "name": c["name"], "number": c["number"],
                "rarity": c.get("rarity"), "artist": c.get("artist"), "cm_price": price,
                "release": (c.get("set", {}).get("releaseDate", "") or "").replace("/", "-"),
            }
            if price >= args.min_price:
                chase_candidates.append(entry)
        k = min(args.control_per_set, len(cards))
        for c in random.sample(cards, k):
            cm = c.get("cardmarket", {}).get("prices", {})
            price = max(cm.get("averageSellPrice", 0) or 0, cm.get("trendPrice", 0) or 0)
            control_candidates.append({
                "era": era, "name": c["name"], "number": c["number"],
                "rarity": c.get("rarity"), "artist": c.get("artist"), "cm_price": price,
                "release": (c.get("set", {}).get("releaseDate", "") or "").replace("/", "-"),
            })
        time.sleep(1.0)

    print(f"\nCandidati chase: {len(chase_candidates)} | Candidati controllo: {len(control_candidates)}")
    if args.dry_run:
        return

    added, skipped_dup, unresolved = [], [], []
    for entry, sel_method in (
        [(c, "chase_price_filter_survivorship_biased") for c in chase_candidates]
        + [(c, "random_control") for c in control_candidates]
    ):
        item_slug = resolve_promo_slug(entry["name"])
        if not item_slug:
            unresolved.append((entry["name"], entry["number"], "slug non risolto via ricerca"))
            continue
        key = (PROMO_GAME_SLUG, item_slug)
        if key in existing_keys:
            skipped_dup.append(entry["name"])
            continue
        fetched = fetch_pricecharting_series(PROMO_GAME_SLUG, item_slug, eur_usd_series=eur_usd)
        if "raw" not in fetched or fetched["raw"].empty:
            unresolved.append((entry["name"], entry["number"], f"{PROMO_GAME_SLUG}/{item_slug}"))
            time.sleep(0.15)
            continue

        item_id = make_item_id(entry["name"], entry["number"]) + "_promo"
        metadata[item_id] = {
            "name": f"{entry['name']} #{entry['number']} (Promo)", "type": "single", "product_type": "single_card",
            "game_slug": PROMO_GAME_SLUG, "item_slug": item_slug, "release_date": entry["release"] or None,
            "rarity": entry.get("rarity"), "artist": entry.get("artist"), "franchise": "pokemon", "language": "en",
            "cardmarket_ref_price_eur": round(entry["cm_price"], 2),
            "source_note": "Promo (SVP/Black Star ecc.) - slug risolto via ricerca PriceCharting, non prevedibile",
            "selection_method": sel_method,
            "is_promo": True,
        }
        added.append(item_id)
        existing_keys.add(key)
        if len(added) % 15 == 0:
            save_metadata(metadata)
            print(f"  [checkpoint] salvato dopo {len(added)} promo aggiunte", flush=True)
        time.sleep(0.15)

    print(f"\nAggiunte: {len(added)} | Duplicati: {len(skipped_dup)} | Non risolte: {len(unresolved)}")
    if unresolved:
        print("Non risolte (prime 15):")
        for name, number, why in unresolved[:15]:
            print(f"  {name} #{number} -> {why}")

    save_metadata(metadata)
    print(f"\nitems_metadata.json ora contiene {len(metadata)} item totali.")


if __name__ == "__main__":
    main()
