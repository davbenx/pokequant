#!/usr/bin/env python3
"""
scripts/discover_chase_cards.py — Scopre carte "chase" (prezzo Cardmarket alto,
rarità da hype) nei set già rappresentati nell'universo sealed, usando pokemontcg.io
(API ufficiale, gratuita, prezzi Cardmarket/TCGplayer correnti — non storici).

Per ciascuna carta trovata sopra la soglia di prezzo, costruisce e VERIFICA (fetch
reale, non per assunzione) lo slug PriceCharting corrispondente, poi la aggiunge a
items_metadata.json come "single" solo se il fetch storico riesce davvero.

Uso:
    python scripts/discover_chase_cards.py [--min-price 40] [--dry-run]

Con --dry-run stampa i candidati senza scrivere su items_metadata.json né
riscaricare i pannelli prezzi (va poi lanciato scripts/rebuild_prices_with_real_fx.py
per popolare i CSV una volta confermati i nuovi item).
"""

import argparse
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, save_metadata
from poke_quant.data.price_fetcher import fetch_pricecharting_series
from poke_quant.data.fx_rates import load_eur_usd_series

HEADERS = {"User-Agent": "Mozilla/5.0"}

# Set pokemontcg.io da cercare, mappati alla stessa "era" usata negli item _bb/_etb
# già presenti in items_metadata.json (serve per riusarne il game_slug corretto).
SET_IDS = {
    "swsh6": "chilling_reign", "swsh7": "evolving_skies", "swsh8": "fusion_strike",
    "swsh9": "brilliant_stars", "swsh10": "astral_radiance", "swsh11": "lost_origin",
    "swsh12": "silver_tempest", "sv2": "paldea_evolved", "sv3": "obsidian_flames",
    "sv4": "paradox_rift", "sv5": "temporal_forces", "sv6": "twilight_masquerade",
    "sv7": "stellar_crown", "sv8": "surging_sparks", "sv3pt5": "scarlet_violet_151",
    # Seconda ondata (copertura di TUTTI gli era sealed già in items_metadata.json,
    # non solo i 15 iniziali) — mappatura verificata su pokemontcg.io /v2/sets:
    "xy7": "ancient_origins", "swsh5": "battle_styles", "sm3": "burning_shadows",
    "cel25": "celebrations", "sm12": "cosmic_eclipse", "sm4": "crimson_invasion",
    "swsh12pt5": "crown_zenith", "swsh3": "darkness_ablaze", "xy12": "evolutions",
    "xy2": "flashfire", "sm115": "hidden_fates", "sm8": "lost_thunder",
    "swsh2": "rebel_clash", "xy6": "roaring_skies", "swsh45": "shining_fates",
    "sm9": "team_up", "sm5": "ultra_prism", "sm10": "unbroken_bonds",
    "sm11": "unified_minds", "swsh4": "vivid_voltage",
    # jp_* (VMAX Climax, VSTAR Universe, Shiny Star V, Tag All Stars, Shiny Treasure ex)
    # sono set giapponesi esclusivi: pokemontcg.io copre la stampa inglese e non li ha,
    # quindi non sono ricercabili con questo metodo. scarlet_violet_base è escluso: il
    # suo game_slug booster-box è già confermato rotto su PriceCharting (redirect a
    # ricerca generica) e non è chiaro se le singole userebbero uno slug diverso.

    # Terza ondata: tutti i restanti set confermati come box sigillato reale su
    # PriceCharting (scripts/discover_sealed_universe.py, 1999-2026), incluso il
    # vintage. Il filtro di attendibilità (poke_quant/data/liquidity_filter.py) fa
    # comunque da rete di sicurezza sulle serie troppo rumorose, quindi non c'è
    # rischio ad ampliare qui: nel peggiore dei casi la carta viene flaggata ed
    # esclusa dall'universo azionabile in automatico.
    "base2": "jungle", "base3": "fossil", "base4": "base_set_2", "base5": "team_rocket",
    "gym1": "gym_heroes", "gym2": "gym_challenge", "neo1": "neo_genesis", "neo2": "neo_discovery",
    "neo3": "neo_revelation", "neo4": "neo_destiny", "base6": "legendary_collection",
    "ecard2": "aquapolis", "ecard3": "skyridge", "ex5": "hidden_legends", "ex7": "team_rocket_returns",
    "ex8": "deoxys", "ex9": "emerald", "ex10": "unseen_forces", "ex11": "delta_species",
    "ex12": "legend_maker", "ex13": "holon_phantoms", "ex14": "crystal_guardians",
    "ex15": "dragon_frontiers", "ex16": "power_keepers", "dp2": "mysterious_treasures",
    "dp3": "secret_wonders", "dp4": "great_encounters", "dp5": "majestic_dawn",
    "dp6": "legends_awakened", "dp7": "stormfront", "pl1": "platinum", "pl2": "rising_rivals",
    "pl3": "supreme_victors", "pl4": "arceus", "col1": "call_of_legends", "bw2": "emerging_powers",
    "bw3": "noble_victories", "bw4": "next_destinies", "bw5": "dark_explorers", "bw6": "dragons_exalted",
    "bw7": "boundaries_crossed", "bw8": "plasma_storm", "bw9": "plasma_freeze", "bw10": "plasma_blast",
    "bw11": "legendary_treasures", "xy1": "xy", "xy3": "furious_fists", "xy4": "phantom_forces",
    "xy5": "primal_clash", "xy8": "breakthrough", "xy9": "breakpoint", "xy10": "fates_collide",
    "xy11": "steam_siege", "sm2": "guardians_rising", "sm6": "forbidden_light", "sm7": "celestial_storm",
    "sv9": "journey_together", "sv10": "destined_rivals", "me1": "mega_evolution",
    "me2": "phantasmal_flames", "me3": "perfect_order", "me4": "chaos_rising", "me5": "pitch_black",
}

CHASE_RARITIES = {
    "Rare Secret", "Rare Rainbow", "Rare Ultra", "Special Illustration Rare",
    "Illustration Rare", "Hyper Rare", "Rare Holo VMAX", "Rare Holo VSTAR",
}


def fetch_set_cards(set_id: str, retries: int = 3) -> list:
    url = f"https://api.pokemontcg.io/v2/cards?q=set.id:{set_id}&pageSize=250"
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=25)
            r.raise_for_status()
            return r.json().get("data", [])
        except Exception as e:
            print(f"  [retry {attempt}] {set_id}: {e}")
            time.sleep(4)
    return []


def slugify_card(name: str, number: str) -> str:
    s = name.lower().replace("'", "").replace(".", "")
    s = re.sub(r"[^a-z0-9&]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return f"{s}-{number}"


def make_item_id(name: str, number: str) -> str:
    s = name.lower().replace("'", "").replace(".", "")
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return f"{s}_{number}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-price", type=float, default=40.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    metadata = load_metadata()
    eur_usd = load_eur_usd_series()

    era_to_game_slug = {}
    for item_id, info in metadata.items():
        for suffix in ["_bb", "_etb", "_bundle"]:
            if item_id.endswith(suffix):
                era_to_game_slug[item_id[: -len(suffix)]] = info.get("game_slug")

    existing_keys = {(v.get("game_slug"), v.get("item_slug")) for v in metadata.values()}

    candidates = []
    for set_id, era in SET_IDS.items():
        cards = fetch_set_cards(set_id)
        print(f"{set_id} ({era}): {len(cards)} carte scaricate")
        for c in cards:
            cm = c.get("cardmarket", {}).get("prices", {})
            price = max(cm.get("averageSellPrice", 0) or 0, cm.get("trendPrice", 0) or 0)
            if price >= args.min_price:
                candidates.append({
                    "era": era, "name": c["name"], "number": c["number"],
                    "rarity": c.get("rarity"), "cm_price": price,
                    "release": (c.get("set", {}).get("releaseDate", "") or "").replace("/", "-"),
                })
        time.sleep(1.0)

    print(f"\nCandidati sopra {args.min_price} EUR: {len(candidates)}")
    if args.dry_run:
        for c in sorted(candidates, key=lambda x: -x["cm_price"]):
            print(f"  {c['cm_price']:>8.2f}  {c['name']:30s} #{c['number']:6s} {c['rarity']}")
        return

    added, skipped_dup, failed = [], [], []
    for c in candidates:
        game_slug = era_to_game_slug.get(c["era"])
        if not game_slug:
            continue
        item_slug = slugify_card(c["name"], c["number"])
        if (game_slug, item_slug) in existing_keys:
            skipped_dup.append(c["name"])
            continue
        fetched = fetch_pricecharting_series(game_slug, item_slug, eur_usd_series=eur_usd)
        if "raw" not in fetched or fetched["raw"].empty:
            failed.append((c["name"], c["number"], game_slug, item_slug))
            time.sleep(0.15)
            continue

        item_id = make_item_id(c["name"], c["number"])
        if item_id in metadata:
            item_id = f"{item_id}_{c['era']}"
        metadata[item_id] = {
            "name": f"{c['name']} #{c['number']}", "type": "single", "product_type": "single_card",
            "game_slug": game_slug, "item_slug": item_slug, "release_date": c["release"] or None,
            "rarity": c.get("rarity"), "franchise": "pokemon", "language": "en",
            "cardmarket_ref_price_eur": round(c["cm_price"], 2),
            "source_note": "Scoperta via pokemontcg.io (rarity/prezzo), storico da PriceCharting (raw+grade9)",
        }
        added.append(item_id)
        existing_keys.add((game_slug, item_slug))
        time.sleep(0.15)

    print(f"\nAggiunte: {len(added)} | Duplicati scartati: {len(skipped_dup)} | Slug falliti: {len(failed)}")
    if failed:
        print("Slug falliti (verificare a mano):")
        for name, number, gs, isl in failed:
            print(f"  {name} #{number} -> {gs}/{isl}")

    save_metadata(metadata)
    print(f"\nitems_metadata.json ora contiene {len(metadata)} item totali.")
    print("Esegui scripts/rebuild_prices_with_real_fx.py per popolare i pannelli prezzi.")


if __name__ == "__main__":
    main()
