#!/usr/bin/env python3
"""
scripts/discover_mtg_chase_singles.py — PROGETTO PILOTA Magic: The Gathering,
lato singole, campione "chase" (selezionato sul prezzo CORRENTE - survivorship
bias PER COSTRUZIONE, stesso schema/stessa etichetta di discover_chase_cards.py
per Pokemon: "selection_method": "chase_price_filter_survivorship_biased").
Va SEMPRE eseguito insieme a discover_random_control_mtg_singles.py prima di
fidarsi di un backtest - stesso principio metodologico usato per Pokemon.

Fonte carte: Scryfall (gratuito, nessuna API key) sui 29 set MTG "expansion"
2015-2024 confermati da discover_mtg_sealed_universe.py. Scryfall fornisce
GIA' un prezzo di riferimento EUR (aggregato da Cardmarket) in ogni oggetto
carta - non serve un lookup separato come per Pokemon (pokemontcg.io).

Soglia €3 (non €40 come Pokemon): la distribuzione di prezzo nei set MTG
moderni "expansion" e' molto piu' compressa - anche Dominaria (set con Mox
Amber, uno dei piu' cari del pilota) ha solo 4 carte sopra i 5€. Una soglia
piu' alta produrrebbe un campione chase quasi vuoto su meta' dei set.

Prezzo RAW/ungraded (non Grade 9 come Pokemon) - scelta confermata
dall'utente dopo aver verificato che la gradazione MTG su PriceCharting e'
reale solo per poche carte vintage iconiche (es. Black Lotus), quasi assente
sulle chase moderne (es. Ragavan, Modern Horizons 2 - nessun dato gradato a
nessun voto). Il pannello RAW e' molto piu' ricco su MTG.

Uso: python scripts/discover_mtg_chase_singles.py [--min-price-eur 3.0] [--dry-run]
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

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}


def slugify(name: str) -> str:
    s = name.lower().replace("&", "and")
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return re.sub(r"-+", "-", s).strip("-")


def item_slug_for_card(name: str, number: str) -> str:
    return f"{slugify(name)}-{number}"


def fetch_set_cards(set_code: str) -> list:
    cards, url = [], f"https://api.scryfall.com/cards/search?q=set:{set_code}&order=set&unique=prints"
    while url:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            break
        d = r.json()
        cards += d.get("data", [])
        url = d.get("next_page")
        time.sleep(0.1)
    return cards


def mtg_game_slugs(metadata: dict) -> list:
    """game_slug MTG gia' scoperti (discover_mtg_sealed_universe.py) - riusa
    esattamente gli stessi, verificati via fetch reale, non ne ricalcola di nuovi."""
    return sorted({v["game_slug"] for v in metadata.values()
                    if v.get("type") == "sealed" and v.get("franchise") == "magic"})


def build_set_code_by_slug() -> dict:
    """{game_slug: scryfall_set_code} - un solo fetch di /sets, poi match per nome
    slugificato allo stesso modo del game_slug PriceCharting (stessa funzione
    slugify() usata per costruire i game_slug in discover_mtg_sealed_universe.py)."""
    r = requests.get("https://api.scryfall.com/sets", headers=HEADERS, timeout=20)
    r.raise_for_status()
    out = {}
    for s in r.json().get("data", []):
        slug = "magic-" + slugify(s["name"])
        out[slug] = s["code"]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-price-eur", type=float, default=3.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    metadata = load_metadata()
    eur_usd = load_eur_usd_series()
    game_slugs = mtg_game_slugs(metadata)
    print(f"Set MTG sealed gia' confermati (universo box pilota): {len(game_slugs)}")
    set_code_by_slug = build_set_code_by_slug()

    existing_keys = {(v.get("game_slug"), v.get("item_slug")) for v in metadata.values()}

    candidates = []
    for game_slug in game_slugs:
        set_code = set_code_by_slug.get(game_slug)
        if not set_code:
            print(f"  [SALTATO] {game_slug}: set code Scryfall non trovato")
            continue
        cards = fetch_set_cards(set_code)
        n_priced = 0
        for c in cards:
            eur = c.get("prices", {}).get("eur")
            if not eur or float(eur) < args.min_price_eur:
                continue
            if c.get("rarity") in ("special", "bonus"):
                continue
            n_priced += 1
            candidates.append({
                "name": c["name"], "number": c["collector_number"], "rarity": c["rarity"],
                "game_slug": game_slug, "eur_price": float(eur),
                "release": c.get("released_at"),
            })
        print(f"  {game_slug} ({set_code}): {n_priced} carte >= {args.min_price_eur}EUR su {len(cards)}", flush=True)
        time.sleep(0.3)

    print(f"\nCandidati chase totali (pre-verifica PriceCharting): {len(candidates)}")
    if args.dry_run:
        for c in sorted(candidates, key=lambda x: -x["eur_price"])[:20]:
            print(f"  {c['eur_price']:>8.2f}EUR  {c['name']:35s} #{c['number']:6s} {c['rarity']}")
        return

    added, skipped_dup, unresolved = [], [], []
    for c in candidates:
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
            item_id = f"{item_id}_{c['game_slug'].replace('magic-', '')}"
        metadata[item_id] = {
            "name": f"{c['name']} #{c['number']}", "type": "single", "product_type": "single_card",
            "game_slug": c["game_slug"], "item_slug": item_slug,
            "release_date": c["release"], "rarity": c["rarity"], "franchise": "magic", "language": "en",
            "cardmarket_ref_price_eur": round(c["eur_price"], 2),
            "source_note": "Progetto pilota MTG - campione selezionato sul prezzo corrente "
                            "(survivorship bias PER COSTRUZIONE, vedi discover_random_control_mtg_singles.py "
                            "per il campione di controllo non biased)",
            "selection_method": "chase_price_filter_survivorship_biased",
        }
        added.append(item_id)
        existing_keys.add(key)
        time.sleep(0.15)

    print(f"\nAggiunti: {len(added)} | duplicati scartati: {len(skipped_dup)} | non risolti su PriceCharting: {len(unresolved)}")
    save_metadata(metadata)
    print(f"items_metadata.json ora contiene {len(metadata)} item totali")


if __name__ == "__main__":
    main()
