#!/usr/bin/env python3
"""
scripts/discover_sealed_universe.py — Verifica sistematicamente, per OGNI set
Pokémon noto (anagrafica pokemontcg.io, 1999-oggi), se esiste un booster box
reale su PriceCharting, e lo aggiunge a items_metadata.json se sì.

Nessun MSRP viene inventato per i set vintage (msrp=None esplicito) — le
strategie che lo usano (sealed_accumulator.py, optimal_sealed_strategy.py)
ricadono su un default solo quando la chiave è assente o None, non fabbricano
un numero storico che non conosciamo.

ATTENZIONE: PriceCharting traccia lo storico prezzi solo dagli ultimi ~68 mesi
per QUALSIASI set, anche quelli del 1999 — questo script espande l'AMPIEZZA
cross-sezionale dell'universo (più asset sulla stessa finestra), non la
PROFONDITÀ temporale. Verificato empiricamente, non assunto.

Uso: python scripts/discover_sealed_universe.py [--dry-run]
"""

import argparse
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, save_metadata
from poke_quant.data.fx_rates import load_eur_usd_series
from poke_quant.data.price_fetcher import fetch_pricecharting_series

HEADERS = {"User-Agent": "Mozilla/5.0"}


def guess_slug(name: str) -> str:
    s = name.lower().replace("&", "and")
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return re.sub(r"-+", "-", s).strip("-")


def era_id(name: str) -> str:
    s = name.lower().replace("&", "and")
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sets_resp = requests.get("https://api.pokemontcg.io/v2/sets", headers=HEADERS, timeout=20)
    sets_resp.raise_for_status()
    all_sets = sets_resp.json().get("data", [])
    print(f"Set totali in anagrafica: {len(all_sets)}")

    metadata = load_metadata()
    eur_usd = load_eur_usd_series()
    existing_slugs = {(v.get("game_slug"), v.get("item_slug")) for v in metadata.values()}

    hits = []
    for s in all_sets:
        slug = "pokemon-" + guess_slug(s["name"])
        url = f"https://www.pricecharting.com/game/{slug}/booster-box"
        try:
            r = requests.get(url, headers=HEADERS, timeout=15, allow_redirects=False)
            if r.status_code == 200:
                hits.append({"name": s["name"], "release": s.get("releaseDate"), "game_slug": slug})
        except Exception:
            pass
        time.sleep(0.4)

    print(f"Booster box confermati su PriceCharting: {len(hits)}")
    if args.dry_run:
        for h in hits:
            print(f"  {h['release']}  {h['name']} -> {h['game_slug']}")
        return

    added, skipped, failed = [], [], []
    for h in hits:
        key = (h["game_slug"], "booster-box")
        if key in existing_slugs:
            skipped.append(h["name"])
            continue
        fetched = fetch_pricecharting_series(h["game_slug"], "booster-box", eur_usd_series=eur_usd)
        if "raw" not in fetched or fetched["raw"].empty:
            failed.append(h["name"])
            continue
        item_id = f"{era_id(h['name'])}_bb"
        if item_id in metadata:
            continue
        metadata[item_id] = {
            "name": f"{h['name']} Booster Box", "type": "sealed", "product_type": "booster_box",
            "game_slug": h["game_slug"], "item_slug": "booster-box",
            "release_date": (h.get("release") or "").replace("/", "-") or None,
            "msrp": None, "set_tier": "UNASSIGNED",
            "franchise": "pokemon", "language": "en",
            "source_note": "Scoperta sistematica su tutti i set pokemontcg.io, verificata via fetch reale PriceCharting",
        }
        added.append(item_id)
        existing_slugs.add(key)

    print(f"Aggiunti: {len(added)} | già presenti: {len(skipped)} | falliti: {len(failed)}")
    save_metadata(metadata)
    print(f"items_metadata.json ora contiene {len(metadata)} item totali")


if __name__ == "__main__":
    main()
