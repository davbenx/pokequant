#!/usr/bin/env python3
"""
scripts/discover_one_piece_sealed_universe.py — Scoperta ricorrente di nuovi
box sigillati One Piece TCG (EN), stesso pattern di discover_sealed_universe.py
(Pokemon EN) e discover_pokemon_chinese_sealed_universe.py (Pokemon Cinese).

AGGIUNTO 2026-09-29 (richiesta esplicita dell'utente: "fai in modo che il
motore scarichi dati gia' su One piece"): i 5 box One Piece gia' in produzione
(op01/02/03/05/06) erano stati curati UNA VOLTA a mano da una sessione
precedente - l'ultimo risale a "Wings of the Captain" (2024-03-15), oltre
DUE ANNI E MEZZO fa rispetto a oggi. Nessuno script scopriva i box usciti
dopo, a differenza di Pokemon EN (discover_sealed_universe.py, riscarica
l'anagrafica pokemontcg.io a ogni run del motore mensile) - un franchise
"tenuto in produzione" (scripts/one_piece_pilot_validation.py) con un
universo fermo a 2 anni fa non e' davvero mantenuto. Questo script chiude
il gap: scopre dal vivo, a ogni run, tramite la categoria REALE di
PriceCharting (pricecharting.com/category/one-piece-cards, verificata dal
vivo il 2026-09-29 - NON un elenco congelato), ogni nuovo set inglese con un
prodotto "Booster Box" reale.

Esclude ESPLICITAMENTE i set `one-piece-japanese-*` (stessa categoria della
console page mescola EN e JP) - stessa decisione di Pokemon JP
(poke_quant/data/liquidity_filter.py::DEFAULT_EXCLUDED_LANGUAGES): il
mercato secondario giapponese non e' mai stato validato per questo
franchise, non si aggiunge dato non richiesto senza motivo. Se in futuro si
vuole valutare One Piece JP, va fatto con lo stesso rigore del pilota
Pokemon JP (scripts/jp_pilot_validation.py), non per default qui.

Numerazione OP-XX: i 5 set gia' in metadata usano un prefisso ufficiale
op01/op02/... (numerazione Bandai) che questo script NON puo' derivare in
modo affidabile dal solo slug (richiederebbe una fonte esterna verificata
per il numero di set ufficiale, che non ho). Per non inventare un numero
che non conosco, i nuovi item_id usano lo slug stesso (es. "op_kingdoms_of_intrigue_bb"),
non un prefisso "opNN_" - la numerazione ufficiale va aggiunta a mano in un
secondo momento se serve, senza bloccare la scoperta/raccolta dati.

MSRP: 100.0 USD per ogni box - stesso valore gia' usato per tutti e 5 i box
esistenti in metadata (MSRP ufficiale Bandai, costante su tutti i set EN,
fatto di dominio pubblico ben documentato - non una stima per-set inventata).
PriceCharting non espone un campo MSRP/release-date verificato: stesso
limite gia' documentato per Pokemon Cinese.

Uso: python scripts/discover_one_piece_sealed_universe.py [--dry-run]
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

CATEGORY_URL = "https://www.pricecharting.com/category/one-piece-cards"
SEALED_SLUGS = ["booster-box"]
KNOWN_MSRP_USD = 100.0


def fetch_one_piece_en_game_slugs() -> list:
    """Ri-scarica dal vivo la categoria one-piece-cards di PriceCharting ed
    estrae i console (game_slug) `one-piece-<set>`, escludendo esplicitamente
    `one-piece-japanese-*` (mercato mai validato, vedi docstring del modulo)."""
    resp = requests.get(CATEGORY_URL, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    all_slugs = set(re.findall(r'href="/console/(one-piece-[a-z0-9-]+)"', resp.text))
    return sorted(s for s in all_slugs if not s.startswith("one-piece-japanese-"))


def era_id(game_slug: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", game_slug.replace("one-piece-", "")).strip("_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    metadata = load_metadata()
    eur_usd = load_eur_usd_series()
    existing_keys = {(v.get("game_slug"), v.get("item_slug")) for v in metadata.values()}

    game_slugs = fetch_one_piece_en_game_slugs()
    print(f"Console 'one-piece-*' (EN, non-japanese) trovate in categoria: {len(game_slugs)}")

    added, failed = [], []
    for game_slug in game_slugs:
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
            item_id = f"op_{era_id(game_slug)}_bb"
            print(f"  TROVATO: {game_slug}/{item_slug} - {len(fetched['raw'])} mesi, dal {first_real_date}")
            if args.dry_run:
                added.append(item_id)
                continue
            metadata[item_id] = {
                "name": f"{game_slug.replace('one-piece-', '').replace('-', ' ').title()} Booster Box",
                "type": "sealed", "product_type": "booster_box",
                "game_slug": game_slug, "item_slug": item_slug,
                "release_date": first_real_date, "msrp": KNOWN_MSRP_USD, "set_tier": "UNASSIGNED",
                "era": "one_piece", "franchise": "one_piece", "language": "en",
                "source_note": "Scoperta automatica (discover_one_piece_sealed_universe.py) - "
                                "release_date e' il primo mese con prezzo reale in storico (proxy "
                                "dichiarato, PriceCharting non espone una release date verificata). "
                                "MSRP 100 USD: valore ufficiale Bandai, costante sui set EN gia' "
                                "in produzione, non stimato per questo set specifico. Numerazione "
                                "OP-XX ufficiale non nota, item_id derivato dallo slug.",
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
