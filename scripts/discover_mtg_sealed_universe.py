#!/usr/bin/env python3
"""
scripts/discover_mtg_sealed_universe.py — PROGETTO PILOTA Magic: The Gathering
(richiesto dall'utente: "aggiungiamo... magic the gathering al nostro sistema,
sia per le singole che per i box... solo progetto pilota").

Stesso metodo di discover_sealed_universe.py (Pokemon): verifica sistematica,
non un elenco scelto a mano per il risultato. Candidati da Scryfall (catalogo
MTG ufficiale-de-facto, gratuito, nessuna API key - vedi verifica di fattibilita'
in questa stessa conversazione) con set_type=="expansion" (esclude
Masters/Commander/Jumpstart/Secret Lair/Art Series - prodotti PENSATI per
essere ristampe o senza booster box draft standard, non un filtro scelto per
escludere risultati scomodi) pubblicati 2015-2024 (finestra di eta' comparabile
a quella usata per Pokemon/One Piece). Per ognuno, verifica REALE (fetch,
non assunzione) se PriceCharting traccia un prodotto "Booster Box" con storico
prezzi utilizzabile - stesso schema game_slug delle singole
(magic-<nome-set>, vedi ricerca di fattibilita').

RISCHIO STRUTTURALE DICHIARATO (diverso da Pokemon, va tenuto a mente
valutando i risultati): Magic ha una politica di RISTAMPA sistematica e
ufficiale di alcune carte/set (Masters, "The List", Secret Lair) - qui esclusa
per costruzione (solo set_type="expansion"), ma anche i set "expansion" stessi
NON hanno la stessa garanzia di "mai piu' ristampati" delle espansioni
Pokemon - Wizards ha ristampato interi set in edizione "Remastered"
(es. Time Spiral Remastered, Innistrad Remastered, Ravnica Remastered,
Dominaria Remastered) pur trattandoli come prodotti distinti. L'assunzione
economica dietro il momentum sui box ("l'offerta sealed si riduce solo, mai
si rinnova") e' quindi meno solida su MTG che su Pokemon - da NON ignorare
nel giudicare i risultati del backtest.

Uso: python scripts/discover_mtg_sealed_universe.py [--dry-run]
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


def era_id(name: str) -> str:
    s = name.lower().replace("&", "and")
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def fetch_expansion_candidates() -> list:
    r = requests.get("https://api.scryfall.com/sets", headers=HEADERS, timeout=20)
    r.raise_for_status()
    sets = r.json().get("data", [])
    return [
        s for s in sets
        if s.get("set_type") == "expansion"
        and "2015-01-01" <= (s.get("released_at") or "0000") <= "2024-12-31"
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    candidates = fetch_expansion_candidates()
    print(f"Set MTG candidati (Scryfall, set_type=expansion, 2015-2024): {len(candidates)}")

    metadata = load_metadata()
    eur_usd = load_eur_usd_series()
    existing_slugs = {(v.get("game_slug"), v.get("item_slug")) for v in metadata.values()}

    hits = []
    for s in candidates:
        slug = "magic-" + slugify(s["name"])
        url = f"https://www.pricecharting.com/game/{slug}/booster-box"
        try:
            r = requests.get(url, headers=HEADERS, timeout=15, allow_redirects=False)
            if r.status_code == 200:
                hits.append({"name": s["name"], "release": s.get("released_at"), "game_slug": slug})
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
        item_id = f"mtg_{era_id(h['name'])}_bb"
        if item_id in metadata:
            continue
        metadata[item_id] = {
            "name": f"{h['name']} Booster Box", "type": "sealed", "product_type": "booster_box",
            "game_slug": h["game_slug"], "item_slug": "booster-box",
            "release_date": h.get("release"),
            "msrp": None, "set_tier": "UNASSIGNED",
            "franchise": "magic", "language": "en",
            "source_note": "Progetto pilota MTG - scoperta sistematica set expansion 2015-2024 (Scryfall), "
                            "verificata via fetch reale PriceCharting",
        }
        added.append(item_id)
        existing_slugs.add(key)

    print(f"Aggiunti: {len(added)} | già presenti: {len(skipped)} | falliti (nessun prezzo): {len(failed)}")
    save_metadata(metadata)
    print(f"items_metadata.json ora contiene {len(metadata)} item totali")


if __name__ == "__main__":
    main()
