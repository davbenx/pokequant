#!/usr/bin/env python3
"""
scripts/discover_pokemon_chinese_sealed_universe.py — Progetto pilota Pokemon
Cinese (richiesto dall'utente: "aggiungiamo... Pokémon chinese", stesso
standard di rigore usato per il pilota MTG - vedi scripts/mtg_pilot_validation.py).

Scoperta strutturale via la categoria REALE di PriceCharting
(pricecharting.com/category/pokemon-cards, ri-scaricata dal vivo A OGNI RUN,
non piu' un elenco congelato al 2026-09-29 - AGGIORNATO 2026-09-29 per la
riconciliazione del motore dati con signal_scanner.py, richiesta esplicita
dell'utente "fai in modo che il motore scarichi dati gia' su... pokemon
china": senza questo la lista dei game_slug sarebbe rimasta ferma allo
snapshot iniziale, non vedendo mai nuovi set cinesi usciti dopo oggi, stesso
principio dinamico di discover_sealed_universe.py che riscarica l'anagrafica
pokemontcg.io a ogni run). Per ogni console page trovata, verifica REALE
(fetch, non assunzione) se esiste un prodotto "Booster Box" o "Booster Pack"
con storico prezzi utilizzabile, stesso metodo di discover_one_piece_singles.py
(console page PriceCharting) e discover_mtg_sealed_universe.py (verifica prima
di aggiungere, nessun MSRP inventato). Idempotente: gli item gia' in metadata
(via existing_keys) vengono saltati, quindi rilanciarlo ogni mese aggiunge
SOLO i nuovi set, non ritocca quelli gia' presenti.

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

CATEGORY_URL = "https://www.pricecharting.com/category/pokemon-cards"
SEALED_SLUGS = ["booster-box", "booster-pack"]


def fetch_chinese_game_slugs() -> list:
    """Ri-scarica dal vivo la categoria pokemon-cards di PriceCharting ed estrae
    i console (game_slug) `pokemon-chinese-<set>` - stesso meccanismo di
    scoperta strutturale usato per il resto del progetto (mai un elenco
    congelato), cosi' un nuovo set cinese uscito dopo oggi viene visto dal
    prossimo run senza bisogno di aggiornare questo file a mano."""
    # BUG TROVATO (workflow mensile GitHub Action fallito, 2026-10-02, run
    # 37025114175): questa chiamata non era protetta da try/except - la
    # run fallita mostrava decine di 429 "Too Many Requests" da
    # PriceCharting su altri step nella STESSA run (rebuild_prices_with_real_fx.py,
    # che tollera i 429 per singolo item e continua) - questa pagina
    # categoria, colpita dallo stesso rate-limit, crashava invece con un
    # traceback non gestito prima di stampare qualunque output. Stesso
    # fix di discover_sealed_universe.py: catturare, loggare, uscire con
    # codice non-zero SENZA crash (il chiamante lo tratta comunque come
    # "step fallito" per design, vedi run_monthly_production_signal.py).
    try:
        resp = requests.get(CATEGORY_URL, headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[ERRORE] Categoria PriceCharting non raggiungibile o in errore ({e}) - nessuna scoperta questo mese, riprovare al prossimo run.")
        sys.exit(1)
    slugs = sorted(set(re.findall(r'href="/console/(pokemon-chinese-[a-z0-9-]+)"', resp.text)))
    return slugs


def era_id(game_slug: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", game_slug.replace("pokemon-chinese-", "")).strip("_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    metadata = load_metadata()
    eur_usd = load_eur_usd_series()
    existing_keys = {(v.get("game_slug"), v.get("item_slug")) for v in metadata.values()}

    chinese_game_slugs = fetch_chinese_game_slugs()
    print(f"Console 'pokemon-chinese-*' trovate in categoria: {len(chinese_game_slugs)}")

    added, failed = [], []
    for game_slug in chinese_game_slugs:
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
