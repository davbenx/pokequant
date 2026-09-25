#!/usr/bin/env python3
"""
scripts/fetch_population_snapshot.py — Prende UNO snapshot odierno della popolazione
PSA/CGC (PriceCharting, poke_quant/data/population_fetcher.py) per l'universo singole,
e lo accoda a data_cache/population_history.csv.

Va rilanciato con la stessa cadenza dichiarata dalla fonte ("population census updated
monthly", blog.pricecharting.com/2026/02/population-reports-for-psa-cgc.html) - ogni
run aggiunge un nuovo giorno di dati, non sovrascrive; la serie storica si costruisce
SOLO accumulando run successivi nel tempo, mai retroattivamente (non esiste un archivio
storico gratuito accessibile - vedi il docstring di population_fetcher.py per l'unica
alternativa con vera storia trovata, GemRate, a pagamento/contatto commerciale).

Uso:
    python scripts/fetch_population_snapshot.py [--limit 20] [--dry-run]
"""
import argparse
import datetime
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, append_population_snapshot
from poke_quant.data.population_fetcher import fetch_pricecharting_population


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Solo le prime N carte (per test rapidi)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    metadata = load_metadata()
    singles = [
        (item_id, info) for item_id, info in sorted(metadata.items())
        if info.get("type") == "single" and info.get("game_slug") and info.get("item_slug")
    ]
    if args.limit:
        singles = singles[: args.limit]

    today = datetime.date.today().isoformat()
    print(f"Snapshot popolazione {today} su {len(singles)} carte...", flush=True)

    n_with_data, n_empty = 0, 0
    for i, (item_id, info) in enumerate(singles, 1):
        pop_rows = fetch_pricecharting_population(info["game_slug"], info["item_slug"])
        if pop_rows:
            n_with_data += 1
            if not args.dry_run:
                append_population_snapshot([
                    {"date": today, "item_id": item_id, "grade": r["grade"],
                     "psa_pop": r["psa_pop"], "cgc_pop": r["cgc_pop"],
                     "total_pop": r["total_pop"], "price_usd": r["price_usd"]}
                    for r in pop_rows
                ])
        else:
            n_empty += 1
        if i % 50 == 0:
            print(f"  {i}/{len(singles)} ({n_with_data} con dati, {n_empty} senza)", flush=True)
        time.sleep(0.2)

    print(f"\nFatto: {n_with_data} carte con population report, {n_empty} senza "
          f"(troppo nuove/poco tracciate - non un errore).")
    if not args.dry_run:
        print("Aggiunto a data_cache/population_history.csv - rilancia mensilmente per "
              "costruire una serie storica utile a testare l'effetto della crescita della pop.")


if __name__ == "__main__":
    main()
