#!/usr/bin/env python3
"""
scripts/discover_one_piece_singles.py — Scopre carte One Piece TCG di valore
usando le pagine "console" di PriceCharting (elenco completo carte per set con
slug e prezzo corrente), non pokemontcg.io (che copre solo Pokémon). Non serve
alcuna registrazione/API key.

Esclude varianti da torneo/premio a tiratura unica (Championship, Winner,
Top 8, Serial, PRB01) e le voci "Booster Box" che compaiono mescolate nella
stessa lista: sono mercati troppo illiquidi/non ripetibili per un backtest
sistematico, non un limite tecnico.

Uso: python scripts/discover_one_piece_singles.py [--min-price 45] [--dry-run]
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

# Set One Piece già confermati come sealed in items_metadata.json (op01/02/03/05/06);
# aggiungere qui nuovi game_slug quando si trovano nuovi set OP sigillati.
OP_GAME_SLUGS = [
    "one-piece-romance-dawn", "one-piece-paramount-war", "one-piece-pillars-of-strength",
    "one-piece-awakening-of-the-new-era", "one-piece-wings-of-the-captain",
]

EXCLUDE_KEYWORDS = ["championship", "winner", "top 8", "serial", "prb01", "booster box"]


def fetch_console_candidates(game_slug: str) -> list:
    url = f"https://www.pricecharting.com/console/{game_slug}"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    html = r.text
    rows = re.findall(
        r'href="/game/' + re.escape(game_slug) + r'/([a-z0-9-]+)">([^<]+)</a>.*?\$([0-9,]+\.\d{2})',
        html, re.S,
    )
    seen, out = set(), []
    for slug, title, price_str in rows:
        if slug in seen:
            continue
        seen.add(slug)
        out.append({"game_slug": game_slug, "item_slug": slug, "title": title.strip(),
                    "price_usd": float(price_str.replace(",", ""))})
    return out


def make_item_id(game_slug: str, item_slug: str) -> str:
    era = game_slug.replace("one-piece-", "op_")
    s = re.sub(r"[^a-z0-9]+", "_", item_slug).strip("_")
    return f"{era}_{s}"[:60]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-price", type=float, default=45.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    all_candidates = []
    for game_slug in OP_GAME_SLUGS:
        cards = fetch_console_candidates(game_slug)
        print(f"{game_slug}: {len(cards)} carte trovate")
        all_candidates.extend(cards)
        time.sleep(1.0)

    filtered = [
        c for c in all_candidates
        if c["price_usd"] >= args.min_price and not any(k in c["title"].lower() for k in EXCLUDE_KEYWORDS)
    ]
    print(f"\nCandidati dopo filtro liquidità/prezzo: {len(filtered)}")
    if args.dry_run:
        for c in sorted(filtered, key=lambda x: -x["price_usd"])[:40]:
            print(f"  ${c['price_usd']:>8.2f}  {c['title']}")
        return

    metadata = load_metadata()
    eur_usd = load_eur_usd_series()
    existing_slugs = {(v.get("game_slug"), v.get("item_slug")) for v in metadata.values()}

    added, skipped, failed = [], [], []
    for c in filtered:
        key = (c["game_slug"], c["item_slug"])
        if key in existing_slugs:
            skipped.append(c["title"])
            continue
        fetched = fetch_pricecharting_series(c["game_slug"], c["item_slug"], eur_usd_series=eur_usd)
        if "raw" not in fetched or fetched["raw"].empty:
            failed.append(c["title"])
            continue
        item_id = make_item_id(c["game_slug"], c["item_slug"])
        if item_id in metadata:
            continue
        metadata[item_id] = {
            "name": c["title"], "type": "single", "product_type": "single_card",
            "game_slug": c["game_slug"], "item_slug": c["item_slug"],
            "release_date": None, "franchise": "one_piece", "language": "en",
            "pricecharting_ref_price_usd": c["price_usd"],
            "source_note": "Scoperta via console page PriceCharting, storico verificato via fetch reale",
        }
        added.append(item_id)
        existing_slugs.add(key)

    print(f"Aggiunte: {len(added)} | duplicati: {len(skipped)} | falliti: {len(failed)}")
    save_metadata(metadata)
    print(f"items_metadata.json ora contiene {len(metadata)} item totali")


if __name__ == "__main__":
    main()
