#!/usr/bin/env python3
"""
scripts/fetch_pcgs_coins.py — Scarica storico reale di vendite d'asta (Auction
Prices Realized) dall'API pubblica PCGS per un piccolo universo di monete
comuni e liquide, come base per testare TS Momentum e il premio numismatico
ex-oro (vedi conversazione).

Universo (18 monete, PCGS Number verificati via ricerca sul catalogo
PCGS CoinFacts pubblico, grado fisso MS63 per confrontabilita' cross-sezionale):
  - 10 dollari d'argento comuni (Morgan 1878-1921, Peace 1921-1935)
  - 8 monete d'oro comuni ($20 Liberty Head e St. Gaudens)

Endpoint: GET /coindetail/GetAPRByGrade?PCGSNo=..&GradeNo=63&PlusGrade=false
          &StartDate=2010-01-01&EndDate=oggi&NumberOfRecords=500
Autenticazione: Authorization: bearer $PCGS_ACCESS_TOKEN (da .env, mai committato).

Budget: 100 chiamate/giorno sull'account - questo script ne usa 18 (una per
moneta), con checkpoint dopo ogni chiamata per essere ripartibile.

Uso: python scripts/fetch_pcgs_coins.py
"""

import json
import os
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
OUT_FILE = ROOT / "data_cache" / "pcgs_coin_prices.json"

UNIVERSE = {
    # --- Dollari d'argento comuni (Morgan / Peace) ---
    "morgan_1881": {"name": "1881 $1 Morgan", "pcgs_no": "7316", "segment": "silver_dollar"},
    "morgan_1885": {"name": "1885 $1 Morgan", "pcgs_no": "7320", "segment": "silver_dollar"},
    "morgan_1887": {"name": "1887 $1 Morgan", "pcgs_no": "7322", "segment": "silver_dollar"},
    "morgan_1889": {"name": "1889 $1 Morgan", "pcgs_no": "7324", "segment": "silver_dollar"},
    "morgan_1898": {"name": "1898 $1 Morgan", "pcgs_no": "7333", "segment": "silver_dollar"},
    "morgan_1902": {"name": "1902 $1 Morgan", "pcgs_no": "7278", "segment": "silver_dollar"},
    "morgan_1921": {"name": "1921 $1 Morgan", "pcgs_no": "7296", "segment": "silver_dollar"},
    "morgan_1921_s": {"name": "1921-S $1 Morgan", "pcgs_no": "7300", "segment": "silver_dollar"},
    "peace_1923": {"name": "1923 $1 Peace", "pcgs_no": "7360", "segment": "silver_dollar"},
    "peace_1923_s": {"name": "1923-S $1 Peace", "pcgs_no": "7362", "segment": "silver_dollar"},
    # --- Monete d'oro comuni ($20 Liberty Head / St. Gaudens) ---
    "liberty20_1904": {"name": "1904 $20 Liberty", "pcgs_no": "9045", "segment": "gold"},
    "liberty20_1904_s": {"name": "1904-S $20 Liberty", "pcgs_no": "9046", "segment": "gold"},
    "stgaudens20_1924": {"name": "1924 $20 St.Gaudens", "pcgs_no": "9177", "segment": "gold"},
    "stgaudens20_1925": {"name": "1925 $20 St.Gaudens", "pcgs_no": "9180", "segment": "gold"},
    "stgaudens20_1926": {"name": "1926 $20 St.Gaudens", "pcgs_no": "9183", "segment": "gold"},
    "stgaudens20_1926_s": {"name": "1926-S $20 St.Gaudens", "pcgs_no": "9185", "segment": "gold"},
    "stgaudens20_1927": {"name": "1927 $20 St.Gaudens", "pcgs_no": "9186", "segment": "gold"},
    "stgaudens20_1928": {"name": "1928 $20 St.Gaudens", "pcgs_no": "9189", "segment": "gold"},
}

GRADE_NO = 63
START_DATE = "2010-01-01"
END_DATE = "2026-09-23"


def load_token() -> str:
    for line in ENV_FILE.read_text().splitlines():
        if line.startswith("PCGS_ACCESS_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("PCGS_ACCESS_TOKEN non trovato in .env")


def fetch_apr(token: str, pcgs_no: str, grade_no: int) -> dict:
    url = "https://api.pcgs.com/publicapi/coindetail/GetAPRByGrade"
    params = {
        "PCGSNo": pcgs_no, "GradeNo": grade_no, "PlusGrade": "false",
        "StartDate": START_DATE, "EndDate": END_DATE, "NumberOfRecords": 500,
    }
    resp = requests.get(url, headers={"Authorization": f"bearer {token}"}, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


def main():
    token = load_token()
    out = {}
    if OUT_FILE.exists():
        out = json.loads(OUT_FILE.read_text())

    for i, (item_id, info) in enumerate(UNIVERSE.items()):
        if item_id in out:
            print(f"[{i+1}/{len(UNIVERSE)}] {info['name']}: già presente, salto")
            continue
        try:
            data = fetch_apr(token, info["pcgs_no"], GRADE_NO)
        except Exception as e:
            print(f"[{i+1}/{len(UNIVERSE)}] {info['name']}: FALLITO ({e})")
            continue
        auctions = data.get("Auctions", [])
        print(f"[{i+1}/{len(UNIVERSE)}] {info['name']} (PCGS#{info['pcgs_no']}): "
              f"{len(auctions)} vendite, msg='{data.get('ServerMessage')}'")
        out[item_id] = {
            "name": info["name"], "pcgs_no": info["pcgs_no"], "segment": info["segment"],
            "grade_no": GRADE_NO, "auctions": auctions,
        }
        OUT_FILE.parent.mkdir(exist_ok=True)
        OUT_FILE.write_text(json.dumps(out))
        time.sleep(1.0)

    print(f"\nCompletato. {len(out)} monete salvate in {OUT_FILE}")


if __name__ == "__main__":
    main()
