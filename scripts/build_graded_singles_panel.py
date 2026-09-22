#!/usr/bin/env python3
"""
scripts/build_graded_singles_panel.py — Costruisce un pannello storico REALE per
singole "già gradate" (basso rischio umano: nessuna sottomissione, nessuna
valutazione soggettiva — il pannello segue il prezzo di una carta già certificata).

Fonte: PriceCharting (già in uso, conforme — vedi poke_quant/data/price_fetcher.py),
che fornisce per ogni carta singola due serie storiche mensili reali:
  - "used"   -> Ungraded / Raw
  - "graded" -> confermato empiricamente = tier "Grade 9" (verificato per confronto
    valuta: prezzo live mostrato in pagina per Grade 9 in USD / 1.08 = valore EUR
    scaricato da questa serie, al centesimo). NON è PSA 10: PriceCharting mostra
    l'intera scala di grado (Ungraded..PSA 10) solo come snapshot live corrente,
    non come storico per-grado. PSA 10 storico NON è disponibile con questo metodo.

build_and_cache_universe() in price_fetcher.py scarica già questa serie "graded"
ma la scarta subito dopo aver preso solo l'ultimo valore (mislabeled
"last_psa_price" — non è PSA, è Grade 9). Questo script salva l'intera serie.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, ensure_cache_dir
from poke_quant.data.price_fetcher import fetch_pricecharting_series

RAW_FILENAME = "historical_prices_graded_singles_raw.csv"
GRADE9_FILENAME = "historical_prices_graded_singles_grade9.csv"


def main():
    metadata = load_metadata() or {}
    singles = {k: v for k, v in metadata.items() if v.get("type") == "single"}
    print(f"Universo singole nel metadata: {len(singles)}")

    raw_series = {}
    grade9_series = {}
    for item_id, info in singles.items():
        game_slug, item_slug = info.get("game_slug"), info.get("item_slug")
        if not game_slug or not item_slug:
            print(f"  [SKIP] {item_id}: game_slug/item_slug mancanti nel metadata")
            continue
        print(f"  Scarico {item_id} ({game_slug}/{item_slug})...")
        result = fetch_pricecharting_series(game_slug, item_slug)
        if "raw" in result and not result["raw"].empty:
            raw_series[item_id] = result["raw"]
        if "graded" in result and not result["graded"].empty:
            grade9_series[item_id] = result["graded"]

    if not raw_series:
        print("Nessuna serie raw scaricata. Interrotto.")
        return

    raw_df = pd.DataFrame(raw_series).sort_index()
    grade9_df = pd.DataFrame(grade9_series).sort_index()

    path = ensure_cache_dir()
    raw_df.to_csv(path / RAW_FILENAME)
    grade9_df.to_csv(path / GRADE9_FILENAME)

    print(f"\nSalvato: data_cache/{RAW_FILENAME} ({raw_df.shape[0]} mesi x {raw_df.shape[1]} carte)")
    print(f"Salvato: data_cache/{GRADE9_FILENAME} ({grade9_df.shape[0]} mesi x {grade9_df.shape[1]} carte)")
    print("\nATTENZIONE: universo piccolo (poche carte singole nel metadata attuale).")
    print("Risultati di qualunque backtest su questo pannello vanno letti come")
    print("preliminari/esplorativi, non come validazione statisticamente potente.")


if __name__ == "__main__":
    main()
