#!/usr/bin/env python3
"""
scripts/discover_pokemon_chinese_sealed_universe.py — Progetto pilota Pokemon
Cinese (richiesto dall'utente: "aggiungiamo... Pokémon chinese", stesso
standard di rigore usato per il pilota MTG - vedi scripts/mtg_pilot_validation.py).

Scoperta strutturale via la categoria REALE di PriceCharting (verificata dal
vivo: curl su pricecharting.com/category/pokemon-cards, NON assunta) - 23
console page con game_slug `pokemon-chinese-<set>`. Per ognuna, verifica REALE
(fetch, non assunzione) se esiste un prodotto "Booster Box" o "Booster Pack"
con storico prezzi utilizzabile, stesso metodo di discover_one_piece_singles.py
(console page PriceCharting) e discover_mtg_sealed_universe.py (verifica prima
di aggiungere, nessun MSRP inventato).

RELEASE_DATE: PriceCharting non espone una data di rilascio per questi
prodotti (verificato: campo "Release Date" presente in pagina ma vuoto/"none").
Uso il primo mese con un prezzo REALE nella serie storica come proxy - non e'
una data fabbricata, e' un fatto vero ("non abbiamo dato precedente a questo"),
esplicitamente etichettato come proxy in source_note. Essendo tutti prodotti
dell'era Scarlet & Violet (2023+, confermato dai nomi carta nelle pagine
console: Terapagos ex, Hydrapple ex, Cyclizar ex - carte SV, non piu' vecchie),
la finestra "era moderna" (>= 2019-01-01) e' comunque rispettata senza bisogno
di MSRP per l'ammissione all'universo liquido (poke_quant/data/liquidity_filter.py::
is_liquid_sealed).

MSRP: None per costruzione (mai verificato un prezzo di lancio ufficiale in
Yuan/EUR per questi prodotti - stesso principio "mai inventare un numero che
non conosciamo" di discover_mtg_sealed_universe.py). Il tetto MAX_PRICE_TO_MSRP_RATIO
non si applica quindi qui (richiede MSRP), ma non serve per l'ammissione
essendo tutti prodotti moderni.

Uso: python scripts/discover_pokemon_chinese_sealed_universe.py [--dry-run]
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

# Verificato dal vivo il 2026-09-29 (curl su pricecharting.com/category/pokemon-cards,
# grep di href="/console/pokemon-chinese-*") - lista reale, non assunta.
CHINESE_GAME_SLUGS = [
    "pokemon-chinese-151-collect", "pokemon-chinese-30th-celebration",
    "pokemon-chinese-cs4ac", "pokemon-chinese-cs4bc", "pokemon-chinese-cs5ac",
    "pokemon-chinese-csm2ac", "pokemon-chinese-csm2bc", "pokemon-chinese-csm2cc",
    "pokemon-chinese-csv10c", "pokemon-chinese-csv4c", "pokemon-chinese-csv5c",
    "pokemon-chinese-csv6c", "pokemon-chinese-csv7c", "pokemon-chinese-csv8c",
    "pokemon-chinese-csv95c", "pokemon-chinese-csv9c",
    "pokemon-chinese-gem-pack", "pokemon-chinese-gem-pack-2", "pokemon-chinese-gem-pack-3",
    "pokemon-chinese-gem-pack-4", "pokemon-chinese-gem-pack-5", "pokemon-chinese-gem-pack-6",
    "pokemon-chinese-promo",
]

SEALED_SLUGS = ["booster-box", "booster-pack"]


def era_id(game_slug: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", game_slug.replace("pokemon-chinese-", "")).strip("_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    metadata = load_metadata()
    eur_usd = load_eur_usd_series()
    existing_keys = {(v.get("game_slug"), v.get("item_slug")) for v in metadata.values()}

    added, failed = [], []
    for game_slug in CHINESE_GAME_SLUGS:
        for item_slug in SEALED_SLUGS:
            key = (game_slug, item_slug)
            if key in existing_keys:
                continue
            url = f"https://www.pricecharting.com/game/{game_slug}/{item_slug}"
            try:
                resp = requests.get(url, headers=HEADERS, timeout=15, allow_redirects=False)
            except Exception as e:
                failed.append((game_slug, item_slug, str(e)))
                continue
            if resp.status_code in (301, 302):
                failed.append((game_slug, item_slug, "redirect (prodotto inesistente)"))
                continue
            fetched = fetch_pricecharting_series(game_slug, item_slug, eur_usd_series=eur_usd)
            if "raw" not in fetched or fetched["raw"].empty:
                failed.append((game_slug, item_slug, "nessuno storico prezzi reale"))
                continue
            first_real_date = fetched["raw"].index.min().strftime("%Y-%m-%d")
            item_id = f"pkzh_{era_id(game_slug)}_{item_slug.replace('booster-', '')}"
            print(f"  TROVATO: {game_slug}/{item_slug} - {len(fetched['raw'])} mesi, dal {first_real_date}")
            if args.dry_run:
                added.append(item_id)
                continue
            metadata[item_id] = {
                "name": f"{game_slug.replace('pokemon-chinese-', '').replace('-', ' ').title()} {item_slug.replace('-', ' ').title()}",
                "type": "sealed", "product_type": "booster_box" if item_slug == "booster-box" else "booster_pack",
                "game_slug": game_slug, "item_slug": item_slug,
                "release_date": first_real_date, "msrp": None, "set_tier": "UNASSIGNED",
                "franchise": "pokemon_chinese", "language": "zh",
                "source_note": "Progetto pilota Pokemon Cinese - release_date e' il primo mese con "
                                "prezzo reale in storico (proxy dichiarato, non una data ufficiale verificata; "
                                "PriceCharting non espone una release date per questo prodotto). MSRP mai "
                                "verificato, lasciato None per costruzione.",
            }
            existing_keys.add(key)
            added.append(item_id)
            time.sleep(0.3)

    print(f"\nAggiunti: {len(added)} | falliti/non trovati: {len(failed)}")
    for gs, isl, reason in failed:
        print(f"  fallito: {gs}/{isl} - {reason}")
    if not args.dry_run:
        save_metadata(metadata)
        print(f"\nitems_metadata.json ora contiene {len(metadata)} item totali.")


if __name__ == "__main__":
    main()
