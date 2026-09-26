#!/usr/bin/env python3
"""
scripts/discover_random_control_mtg_singles.py — PROGETTO PILOTA Magic: The
Gathering, campione di controllo NON selezionato sul prezzo (stesso principio
di discover_random_control_singles.py per Pokemon): per gli stessi 29 set MTG
gia' confermati, pesca un campione CASUALE di carte (qualsiasi rarita',
qualsiasi prezzo, incluse le comuni mai notate) per misurare quanto il
campione "chase" (discover_mtg_chase_singles.py) sia gonfiato da survivorship
bias.

Uso: python scripts/discover_random_control_mtg_singles.py [--per-set 15] [--dry-run]
"""
import argparse
import random
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, save_metadata
from poke_quant.data.fx_rates import load_eur_usd_series
from poke_quant.data.price_fetcher import fetch_pricecharting_series
from scripts.discover_mtg_chase_singles import (
    slugify, item_slug_for_card, fetch_set_cards, mtg_game_slugs, build_set_code_by_slug, HEADERS,
)

RANDOM_SEED = 7  # stesso seed usato per Pokemon, per riproducibilita'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-set", type=int, default=15)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    metadata = load_metadata()
    eur_usd = load_eur_usd_series()
    game_slugs = mtg_game_slugs(metadata)
    set_code_by_slug = build_set_code_by_slug()
    existing_keys = {(v.get("game_slug"), v.get("item_slug")) for v in metadata.values()}

    sampled = []
    for game_slug in game_slugs:
        set_code = set_code_by_slug.get(game_slug)
        if not set_code:
            continue
        cards = fetch_set_cards(set_code)
        cards = [c for c in cards if c.get("rarity") not in ("special", "bonus")]
        print(f"{game_slug} ({set_code}): {len(cards)} carte scaricate, campiono senza filtro di prezzo/rarita", flush=True)
        if not cards:
            continue
        k = min(args.per_set, len(cards))
        for c in random.sample(cards, k):
            eur = c.get("prices", {}).get("eur")
            sampled.append({
                "name": c["name"], "number": c["collector_number"], "rarity": c["rarity"],
                "game_slug": game_slug, "eur_price": float(eur) if eur else 0.0,
                "release": c.get("released_at"),
            })
        time.sleep(0.3)

    print(f"\nCampione casuale totale (pre-dedup, pre-verifica PriceCharting): {len(sampled)}")
    if args.dry_run:
        for c in sampled:
            print(f"  {c['eur_price']:>8.2f}  {c['name']:30s} #{c['number']:6s} {c['rarity']}")
        return

    added, skipped_dup, unresolved = [], [], []
    for c in sampled:
        item_slug = item_slug_for_card(c["name"], c["number"])
        key = (c["game_slug"], item_slug)
        if key in existing_keys:
            skipped_dup.append(c["name"])
            continue
        fetched = fetch_pricecharting_series(c["game_slug"], item_slug, eur_usd_series=eur_usd)
        if "raw" not in fetched or fetched["raw"].empty:
            unresolved.append((c["name"], c["number"], c["game_slug"], item_slug))
            time.sleep(0.15)
            continue
        item_id = f"mtg_{slugify(c['name'])}_{c['number']}"
        if item_id in metadata:
            item_id = f"{item_id}_{c['game_slug'].replace('magic-', '')}_ctrl"
        metadata[item_id] = {
            "name": f"{c['name']} #{c['number']}", "type": "single", "product_type": "single_card",
            "game_slug": c["game_slug"], "item_slug": item_slug,
            "release_date": c["release"], "rarity": c["rarity"], "franchise": "magic", "language": "en",
            "cardmarket_ref_price_eur": round(c["eur_price"], 2),
            "source_note": "Progetto pilota MTG - campione di controllo casuale (nessun filtro di "
                            "prezzo/rarita) per stima survivorship bias",
            "selection_method": "random_control",
        }
        added.append(item_id)
        existing_keys.add(key)
        time.sleep(0.15)

    resolution_rate = (len(added) / max(1, len(sampled) - len(skipped_dup))) * 100.0
    print(f"\nAggiunti (controllo): {len(added)} | Duplicati scartati: {len(skipped_dup)} | "
          f"Non risolti su PriceCharting: {len(unresolved)}")
    print(f"Tasso di risoluzione PriceCharting sul campione casuale: {resolution_rate:.1f}%")
    save_metadata(metadata)
    print(f"\nitems_metadata.json ora contiene {len(metadata)} item totali ({len(added)} nuovi item di controllo).")


if __name__ == "__main__":
    main()
