#!/usr/bin/env python3
"""
scripts/discover_random_control_singles.py — Campione di controllo NON selezionato
sul prezzo, per misurare quanto il survivorship bias di discover_chase_cards.py
gonfia i risultati della strategia Carry/Scarcity sulle singles.

discover_chase_cards.py include una carta nell'universo SOLO SE il suo prezzo
Cardmarket CORRENTE (oggi) supera una soglia — quindi ogni carta in quell'universo
è, per costruzione, una carta che sappiamo con informazione 2026 essersi rivelata
"vincente". Questo script fa l'opposto: per gli stessi set (scoperti dinamicamente
via build_set_ids(), non piu' un numero fisso), pesca un campione
CASUALE di carte (qualsiasi rarità, qualsiasi prezzo corrente, incluse le comuni
che non hanno mai fatto notizia) e verifica se PriceCharting ne traccia lo storico.

Residuo di bias non eliminabile e DICHIARATO (non nascosto): PriceCharting stessa
traccia solo le carte per cui esiste domanda di ricerca sufficiente — un campione
"casuale sulle carte esistenti" non è un campione "casuale su ciò che PriceCharting
copre". Il tasso di risoluzione (quante del campione casuale hanno davvero uno
storico PriceCharting) viene stampato esplicitamente per quantificare quanto
questo secondo filtro (indipendente da rarità Cardmarket) pesa.

Uso:
    python scripts/discover_random_control_singles.py [--per-set 5] [--dry-run]

Poi lanciare scripts/rebuild_prices_with_real_fx.py per popolare i pannelli prezzi,
e confrontare CarryScarcityFactorStrategy su:
  - universo chase-only (selection_method == "chase_price_filter_survivorship_biased")
  - universo chase + control combinato
per stimare l'inflazione da survivorship bias.
"""

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, save_metadata
from poke_quant.data.price_fetcher import fetch_pricecharting_series
from poke_quant.data.fx_rates import load_eur_usd_series
from scripts.discover_chase_cards import build_set_ids, fetch_set_cards, slugify_card, make_item_id

RANDOM_SEED = 7  # fissato per riproducibilità del campione di controllo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-set", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    random.seed(RANDOM_SEED)

    metadata = load_metadata()
    eur_usd = load_eur_usd_series()

    era_to_game_slug = {}
    for item_id, info in metadata.items():
        for suffix in ["_bb", "_etb", "_bundle"]:
            if item_id.endswith(suffix):
                era_to_game_slug[item_id[: -len(suffix)]] = info.get("game_slug")

    existing_keys = {(v.get("game_slug"), v.get("item_slug")) for v in metadata.values()}

    set_ids = build_set_ids(metadata)
    sampled = []
    for set_id, era in set_ids.items():
        cards = fetch_set_cards(set_id)
        print(f"{set_id} ({era}): {len(cards)} carte scaricate, campiono senza filtro di prezzo/rarita")
        if not cards:
            continue
        k = min(args.per_set, len(cards))
        for c in random.sample(cards, k):
            cm = c.get("cardmarket", {}).get("prices", {})
            price = max(cm.get("averageSellPrice", 0) or 0, cm.get("trendPrice", 0) or 0)
            sampled.append({
                "era": era, "name": c["name"], "number": c["number"],
                "rarity": c.get("rarity"), "cm_price": price,
                "release": (c.get("set", {}).get("releaseDate", "") or "").replace("/", "-"),
            })
        time.sleep(1.0)

    print(f"\nCampione casuale totale (pre-dedup, pre-verifica PriceCharting): {len(sampled)}")
    if args.dry_run:
        for c in sampled:
            print(f"  {c['cm_price']:>8.2f}  {c['name']:30s} #{c['number']:6s} {c['rarity']}")
        return

    added, skipped_dup, unresolved = [], [], []
    for c in sampled:
        game_slug = era_to_game_slug.get(c["era"])
        if not game_slug:
            continue
        item_slug = slugify_card(c["name"], c["number"])
        if (game_slug, item_slug) in existing_keys:
            skipped_dup.append(c["name"])
            continue
        fetched = fetch_pricecharting_series(game_slug, item_slug, eur_usd_series=eur_usd)
        if "raw" not in fetched or fetched["raw"].empty:
            unresolved.append((c["name"], c["number"], game_slug, item_slug))
            time.sleep(0.15)
            continue

        item_id = make_item_id(c["name"], c["number"])
        if item_id in metadata:
            item_id = f"{item_id}_{c['era']}_ctrl"
        metadata[item_id] = {
            "name": f"{c['name']} #{c['number']}", "type": "single", "product_type": "single_card",
            "game_slug": game_slug, "item_slug": item_slug, "release_date": c["release"] or None,
            "rarity": c.get("rarity"), "franchise": "pokemon", "language": "en",
            "cardmarket_ref_price_eur": round(c["cm_price"], 2),
            "source_note": "Campione di controllo casuale (nessun filtro di prezzo/rarita) per stima survivorship bias",
            "selection_method": "random_control",
        }
        added.append(item_id)
        existing_keys.add((game_slug, item_slug))
        time.sleep(0.15)

    resolution_rate = (len(added) / max(1, len(sampled) - len(skipped_dup))) * 100.0
    print(f"\nAggiunti (controllo): {len(added)} | Duplicati scartati: {len(skipped_dup)} | "
          f"Non risolti su PriceCharting: {len(unresolved)}")
    print(f"Tasso di risoluzione PriceCharting sul campione casuale: {resolution_rate:.1f}% "
          "(bias residuo dichiarato: PriceCharting copre solo carte con domanda di ricerca)")

    save_metadata(metadata)
    print(f"\nitems_metadata.json ora contiene {len(metadata)} item totali "
          f"({len(added)} nuovi item di controllo non biased).")
    print("Esegui scripts/rebuild_prices_with_real_fx.py per popolare i pannelli prezzi.")


if __name__ == "__main__":
    main()
