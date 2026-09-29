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

AGGIORNATO (2026-09-29, Fase 2 del piano "raccogliere tutti i dati possibili
per test futuri" - richiesto esplicitamente dall'utente): prima campionava
150 carte chase a caso ("piu' che sufficiente per validare l'idea, non tutto
il catalogo" - era uno script di ricerca, non una pipeline dati). Ora gira
sull'INTERO universo liquido attuale (liquid_singles_ids - rispetta le
esclusioni di produzione decise nella Fase 1: Magic e Pokemon Cinese esclusi
per franchise, Pokemon JP escluso per lingua; Pokemon EN + One Piece inclusi),
non piu' un campione - e' il primo "dato extra oltre lo stretto necessario
alle due strategie attuali" richiesto esplicitamente dall'utente per costruire
un asset di dati riusabile in futuro. --sample N resta disponibile per test
rapidi locali (campione casuale, seed fisso).

Ogni chiamata rifa' il fetch dell'intera serie storica per carta (non solo il
mese corrente) - rilanciare questo script ogni mese sovrascrive con la serie
piu' recente, stesso pattern gia' usato per historical_prices*.csv (non serve
un log append-only: la storia completa e' gia' nella risposta di PriceCharting
ad ogni fetch).

Salva in data_cache/grade_ladder_prices.json: {item_id: {tier: {data: prezzo}}}.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix, ensure_cache_dir
from poke_quant.data.liquidity_filter import liquid_singles_ids
from poke_quant.data.fx_rates import load_eur_usd_series, rate_for_month
from poke_quant.config import DEFAULT_EUR_USD
import re
import pandas as pd
import datetime

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
OUT_FILE = Path(__file__).resolve().parent.parent / "data_cache" / "grade_ladder_prices.json"
TIER_MAP = {"used": "ungraded", "cib": "grade7", "new": "grade8", "graded": "grade9",
            "boxonly": "grade9_5", "manualonly": "psa10"}
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None,
                         help="Campione casuale di N carte (seed fisso) invece dell'universo liquido completo, per test rapidi locali.")
    args = parser.parse_args()

    metadata = load_metadata()
    eur_usd = load_eur_usd_series()
    grade9_prices = load_price_matrix("historical_prices_graded_singles_grade9.csv")
    liquid_ids = set(liquid_singles_ids(metadata, grade9_prices))
    universe = [
        (k, v["game_slug"], v["item_slug"]) for k, v in metadata.items()
        if k in liquid_ids and v.get("game_slug") and v.get("item_slug")
    ]

    if args.sample is not None:
        rng = random.Random(SEED)
        sample = rng.sample(universe, min(args.sample, len(universe)))
        print(f"Campione casuale (seed={SEED}): {len(sample)} carte su {len(universe)} nell'universo liquido")
    else:
        sample = universe
        print(f"Universo liquido completo: {len(sample)} carte")

    out = {}
    if OUT_FILE.exists():
        out = json.loads(OUT_FILE.read_text())

    current_month = datetime.date.today().strftime("%Y-%m-01")

    def is_fresh(item_id: str) -> bool:
        # BUG TROVATO estendendo lo script all'universo completo per la raccolta
        # mensile (Fase 2): la vecchia condizione di skip (len(out[item_id]) == 6)
        # segnava una carta "fatta" per sempre dopo il primo fetch riuscito - una
        # pipeline mensile con questa logica non avrebbe MAI piu' rifetchato quella
        # carta, restando bloccata al mese del primo run. Fresca solo se abbiamo
        # gia' un dato per il MESE CORRENTE su almeno un livello (non serve che
        # tutti i 6 lo abbiano - alcuni livelli restano storicamente vuoti per
        # carte poco gradate, vedi fetch_ladder).
        ladder = out.get(item_id)
        if not ladder or len(ladder) != 6:
            return False
        return any(current_month in series for series in ladder.values())

    for i, (item_id, game_slug, item_slug) in enumerate(sample):
        if is_fresh(item_id):
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
