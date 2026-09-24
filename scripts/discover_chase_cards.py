#!/usr/bin/env python3
"""
scripts/discover_chase_cards.py — Scopre carte "chase" (prezzo Cardmarket alto,
rarità da hype) nei set già rappresentati nell'universo sealed, usando pokemontcg.io
(API ufficiale, gratuita, prezzi Cardmarket/TCGplayer correnti — non storici).

ATTENZIONE — SURVIVORSHIP BIAS STRUTTURALE, non accidentale: il filtro `--min-price`
(riga ~139, `if price >= args.min_price`) seleziona sul prezzo Cardmarket CORRENTE
(oggi), non sul prezzo storico al momento dell'uscita. Ogni carta che entra in
questo universo è quindi, per costruzione, una carta che SAPPIAMO con informazione
2026 essersi rivelata "vincente". Nessuna carta oggi economica/flop può mai entrare,
anche se storicamente disponibile su PriceCharting. Qualsiasi Sharpe/CAGR misurato
da CarryScarcityFactorStrategy(item_type_filter="single") su questo universo è
gonfiato rispetto a un investitore che comprava senza sapere in anticipo l'esito.
Ogni item scritto qui riceve "selection_method": "chase_price_filter_survivorship_biased"
in items_metadata.json per poter essere sempre isolato/escluso nei confronti.
Il campione di controllo NON biased (stessi set, scoperti dinamicamente via
build_set_ids() - non piu' un numero fisso -, carte scelte senza filtro di
prezzo/rarità) è in scripts/discover_random_control_singles.py — va sempre eseguito
insieme a questo prima di fidarsi di un backtest sulle singles.

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
from scripts.discover_sealed_universe import era_id

HEADERS = {"User-Agent": "Mozilla/5.0"}

# SET_IDS ({pokemontcg_set_id: era}) non e' piu' una mappa scritta a mano - vedi
# build_set_ids() sotto: si ricostruisce da sola a ogni run leggendo quali era
# hanno gia' un box sealed in metadata, quindi cattura ogni nuovo set che
# discover_sealed_universe.py aggiunge senza bisogno di toccare questo file.
# Nota: jp_* (VMAX Climax, VSTAR Universe, Shiny Star V, Tag All Stars, Shiny
# Treasure ex) sono set giapponesi esclusivi che pokemontcg.io non ha (copre
# solo la stampa inglese), quindi restano fuori da questo metodo comunque.

CHASE_RARITIES = {
    "Rare Secret", "Rare Rainbow", "Rare Ultra", "Special Illustration Rare",
    "Illustration Rare", "Hyper Rare", "Rare Holo VMAX", "Rare Holo VSTAR",
}


def build_set_ids(metadata: dict) -> dict:
    """Costruisce {pokemontcg_set_id: era} DINAMICAMENTE al posto della mappa
    scritta a mano - copre automaticamente ogni nuovo set sealed appena
    discover_sealed_universe.py lo aggiunge a metadata, senza bisogno di
    aggiornare questo file a mano. Funziona perche' discover_sealed_universe.py
    usa la STESSA era_id() per costruire l'item_id (es. "Chilling Reign" ->
    "chilling_reign_bb") - quindi un match esatto qui e' garantito per
    costruzione per ogni set scoperto da quello script, non solo per quelli
    gia' noti oggi (verificato: 97/98 delle voci storiche gia' scritte a mano
    coincidono esattamente con questo calcolo, l'unica eccezione era gia'
    coperta da una run precedente e non richiede piu' azione)."""
    known_eras = set()
    for item_id in metadata:
        for suffix in ["_bb", "_etb", "_bundle"]:
            if item_id.endswith(suffix):
                known_eras.add(item_id[: -len(suffix)])

    all_sets = []
    for attempt in range(3):
        try:
            resp = requests.get("https://api.pokemontcg.io/v2/sets", headers=HEADERS, timeout=20)
            resp.raise_for_status()
            all_sets = resp.json().get("data", [])
            break
        except Exception as e:
            print(f"  [retry {attempt}] /v2/sets: {e}")
            time.sleep(4)

    result = {}
    for s in all_sets:
        era = era_id(s["name"])
        if era in known_eras:
            result[s["id"]] = era
    return result


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

    set_ids = build_set_ids(metadata)
    print(f"Set con box sealed gia' noto trovati su pokemontcg.io: {len(set_ids)}")

    candidates = []
    for set_id, era in set_ids.items():
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
            "selection_method": "chase_price_filter_survivorship_biased",
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
