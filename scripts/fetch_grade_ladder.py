#!/usr/bin/env python3
"""
scripts/fetch_grade_ladder.py — Scarica lo storico mensile per i 6 livelli di
prezzo che PriceCharting tiene per carta (verificato manualmente prima di
scrivere questo script, incrociando i valori con la tabella prezzi reale della
pagina):

  used        -> Ungraded
  cib         -> Grade 7
  new         -> Grade 8
  graded      -> Grade 9   (l'unico livello gia' usato altrove nel progetto)
  boxonly     -> Grade 9.5
  manualonly  -> PSA 10

Campione PICCOLO E CASUALE (seed fisso, non scelto a mano) di carte chase con
game_slug/item_slug noti - "piu' che sufficiente per validare l'idea", non
tutto il catalogo: l'obiettivo e' testare l'ipotesi di uno spread tra
gradazioni con rigore, non costruire un prodotto in produzione.

Salva in data_cache/grade_ladder_prices.json: {item_id: {tier: {data: prezzo}}}.
"""

import json
import random
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, ensure_cache_dir
from poke_quant.data.fx_rates import load_eur_usd_series, rate_for_month
from poke_quant.config import DEFAULT_EUR_USD
import re
import pandas as pd
import datetime

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
OUT_FILE = Path(__file__).resolve().parent.parent / "data_cache" / "grade_ladder_prices.json"
TIER_MAP = {"used": "ungraded", "cib": "grade7", "new": "grade8", "graded": "grade9",
            "boxonly": "grade9_5", "manualonly": "psa10"}
SAMPLE_SIZE = 40
SEED = 42


def fetch_ladder(game_slug: str, item_slug: str, eur_usd) -> dict:
    url = f"https://www.pricecharting.com/game/{game_slug}/{item_slug}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [fallito] {url}: {e}")
        return {}
    m = re.search(r'VGPC\.chart_data\s*=\s*(\{.*?\});', resp.text, re.DOTALL)
    if not m:
        return {}
    try:
        data = json.loads(m.group(1))
    except Exception:
        return {}
    ladder = {}
    for raw_key, tier_name in TIER_MAP.items():
        pts = data.get(raw_key, [])
        series = {}
        for p in pts:
            if len(p) >= 2 and p[1] > 0:
                dt_obj = datetime.datetime.fromtimestamp(p[0] / 1000.0)
                dt = dt_obj.strftime("%Y-%m-01")
                px_usd = p[1] / 100.0
                rate = rate_for_month(eur_usd, pd.Timestamp(dt_obj), DEFAULT_EUR_USD) if eur_usd is not None else DEFAULT_EUR_USD
                series[dt] = round(px_usd / rate, 2)
        if series:
            ladder[tier_name] = series
    return ladder


def main():
    metadata = load_metadata()
    eur_usd = load_eur_usd_series()
    chase = [
        (k, v["game_slug"], v["item_slug"]) for k, v in metadata.items()
        if v.get("type") == "single" and v.get("selection_method") == "chase_price_filter_survivorship_biased"
        and v.get("game_slug") and v.get("item_slug")
    ]
    rng = random.Random(SEED)
    sample = rng.sample(chase, min(SAMPLE_SIZE, len(chase)))
    print(f"Campione casuale (seed={SEED}): {len(sample)} carte su {len(chase)} chase disponibili")

    out = {}
    if OUT_FILE.exists():
        out = json.loads(OUT_FILE.read_text())

    for i, (item_id, game_slug, item_slug) in enumerate(sample):
        if item_id in out and len(out[item_id]) == 6:
            continue
        ladder = fetch_ladder(game_slug, item_slug, eur_usd)
        n_tiers = len(ladder)
        print(f"[{i+1}/{len(sample)}] {item_id}: {n_tiers}/6 livelli trovati")
        if ladder:
            out[item_id] = ladder
        ensure_cache_dir()
        OUT_FILE.write_text(json.dumps(out))
        time.sleep(0.5)

    print(f"\nCompletato. {len(out)} carte salvate in {OUT_FILE}")


if __name__ == "__main__":
    main()
