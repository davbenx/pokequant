#!/usr/bin/env python3
"""
scripts/integrate_psa10_history_github.py — Scarica e riconcilia il dataset esterno
gratuito github.com/samaygodika/pokemon-psa10-history con il nostro universo singole,
producendo data_cache/population_history_psa10_alt.csv (popolazione PSA10/PSA9 nel
tempo, fonte alt.xyz via quel repo - vedi poke_quant/data/external_psa10_history.py
per i limiti dichiarati: storico reale solo dall'11/9/2026, progetto hobby di terzi
senza LICENSE/SLA, solo PSA, riconciliazione non perfetta).

NON sostituisce data_cache/population_history.csv (PriceCharting, PSA+CGC) - è un
file SEPARATO, provenienza diversa, va trattato come complemento/cross-check, non
mescolato ciecamente.

Uso:
    python scripts/integrate_psa10_history_github.py [--dates N] [--dry-run]

--dates N: quanti degli ultimi N giorni disponibili scaricare (default: tutti quelli
pubblicati, provando ogni data da history/summary.json['first_daily'] a oggi e
saltando quelle assenti - alcuni giorni possono mancare per manutenzione del repo
a monte, non è un errore nostro).
"""
import argparse
import csv
import datetime
import io
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, ensure_cache_dir
from poke_quant.data.external_psa10_history import (
    find_matching_asset, build_number_index, extract_number_from_item_slug,
)

RAW_BASE = "https://raw.githubusercontent.com/samaygodika/pokemon-psa10-history/main"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}


def fetch_csv_rows(path: str) -> list:
    r = requests.get(f"{RAW_BASE}/{path}", headers=HEADERS, timeout=60)
    if r.status_code != 200:
        return []
    return list(csv.DictReader(io.StringIO(r.text)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dates", type=int, default=None, help="Solo gli ultimi N giorni disponibili")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("Scarico history/assets.csv (anagrafica asset alt.xyz)...", flush=True)
    asset_rows = fetch_csv_rows("history/assets.csv")
    if not asset_rows:
        print("Impossibile scaricare assets.csv - interrotto.")
        return
    print(f"  {len(asset_rows)} asset totali nel dataset esterno.")
    number_index = build_number_index(asset_rows)

    metadata = load_metadata()
    singles = {k: v for k, v in metadata.items() if v.get("type") == "single"}

    matches = {}  # item_id -> asset_id
    for item_id, info in singles.items():
        number = extract_number_from_item_slug(info.get("item_slug", ""))
        if not number:
            continue
        hit = find_matching_asset(info.get("name", ""), info.get("game_slug", ""),
                                   info.get("rarity"), number, number_index)
        if hit:
            matches[item_id] = hit["asset_id"]

    print(f"Riconciliate {len(matches)}/{len(singles)} carte ({len(matches)/max(1,len(singles))*100:.1f}%) "
          f"con un asset alt.xyz univoco.")
    if args.dry_run:
        return

    asset_id_to_item_id = {v: k for k, v in matches.items()}  # 1:1 per costruzione (un asset -> al più un nostro item)
    wanted_asset_ids = set(matches.values())

    first_daily = datetime.date(2026, 9, 11)
    today = datetime.date.today()
    dates = []
    d = first_daily
    while d <= today:
        dates.append(d)
        d += datetime.timedelta(days=1)
    if args.dates:
        dates = dates[-args.dates:]

    rows_out = []
    for d in dates:
        date_str = d.isoformat()
        daily_rows = fetch_csv_rows(f"history/daily/{date_str}.csv")
        if not daily_rows:
            continue  # giorno assente a monte, non un errore nostro
        n_hit = 0
        for row in daily_rows:
            asset_id = row.get("asset_id")
            if asset_id not in wanted_asset_ids:
                continue
            item_id = asset_id_to_item_id[asset_id]
            rows_out.append({
                "date": date_str, "item_id": item_id, "asset_id": asset_id,
                "psa10_pop": row.get("pop_at_grade") or "",
                "psa_total_pop": row.get("company_total_pop") or "",
            })
            n_hit += 1
        print(f"  {date_str}: {n_hit} carte nostre trovate su {len(daily_rows)} righe del giorno", flush=True)

    if not rows_out:
        print("Nessuna riga estratta - niente da salvare.")
        return

    path = ensure_cache_dir() / "population_history_psa10_alt.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "item_id", "asset_id", "psa10_pop", "psa_total_pop"])
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\nSalvato {path} - {len(rows_out)} righe, {len(set(r['item_id'] for r in rows_out))} carte, "
          f"{len(set(r['date'] for r in rows_out))} giorni.")


if __name__ == "__main__":
    main()
