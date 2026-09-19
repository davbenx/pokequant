#!/usr/bin/env python3
"""
scripts/sync_cardmarket.py — CLI per sincronizzazione, monitoraggio e aggiornamento quotazioni Cardmarket.
Supporta:
  - Pokémon TCG Occidentale (ENG/IT)
  - Pokémon TCG Giapponese (JAP High-Class)
  - One Piece TCG (OP-01 to OP-08+)
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.cardmarket_bridge import (
    load_cardmarket_quotes,
    update_cardmarket_quote,
    evaluate_cardmarket_item,
    get_cardmarket_deep_link
)
from poke_quant.data.storage import load_metadata
from poke_quant.data.europe_market_calibrator import calibrate_price_matrix_for_europe


def main():
    parser = argparse.ArgumentParser(description="PokeQuant Cardmarket Live Sync & Valuation Desk")
    parser.add_argument("--list", action="store_true", help="Elenca tutte le quotazioni Cardmarket salvate in cache")
    parser.add_argument("--franchise", choices=["pokemon", "one_piece", "all"], default="all", help="Filtra per franchise")
    parser.add_argument("--language", choices=["en", "it", "jp", "all"], default="all", help="Filtra per lingua")
    parser.add_argument("--update", nargs=2, metavar=("ITEM_ID", "PRICE"), help="Aggiorna il prezzo live di un set (es: --update temporal_forces_bb 225.0)")
    parser.add_argument("--calibrate", action="store_true", help="Ricalibra la matrice dei prezzi sul mercato europeo Cardmarket")

    args = parser.parse_args()

    if args.calibrate:
        print("🇪🇺 Calibrazione matrice prezzi per Mercato Europeo Cardmarket in corso...")
        df = calibrate_price_matrix_for_europe()
        print(f"✅ Calibrazione completata: {df.shape[0]} mesi x {df.shape[1]} asset salvati in data_cache/historical_prices_europe.csv")
        return

    if args.update:
        item_id, price_str = args.update
        try:
            px = float(price_str)
            update_cardmarket_quote(item_id, px)
            print(f"✅ Prezzo Cardmarket aggiornato per '{item_id}': {px:.2f} €")
        except ValueError:
            print(f"❌ Errore: '{price_str}' non è un valore numerico valido.")
            sys.exit(1)
        return

    # Default o --list: stampa report del desk
    quotes = load_cardmarket_quotes()
    metadata = load_metadata() or {}

    print("\n" + "=" * 92)
    print("📡 POKEQUANT CARDMARKET LIVE DESK — MONITORAGGIO PREZZI E FINESTRE D'ACQUISTO")
    print("=" * 92)
    print(f"{'SET NAME':<34} | {'FRANCHISE':<10} | {'LANG':<4} | {'LIVE (€)':<9} | {'CAP (€)':<8} | {'STATO OPERATIVO'}")
    print("-" * 92)

    for item_id, q in quotes.items():
        franchise = q.get("franchise", "pokemon")
        lang = q.get("language", "en")
        if args.franchise != "all" and franchise != args.franchise:
            continue
        if args.language != "all" and lang != args.language:
            continue

        meta = metadata.get(item_id, {})
        live_px = float(q.get("last_verified_price", 0.0))
        eval_res = evaluate_cardmarket_item(item_id, meta, live_px)

        status_fmt = eval_res["status_label"]
        print(f"{eval_res['name'][:34]:<34} | {franchise:<10} | {lang.upper():<4} | {live_px:>7.2f} € | {eval_res['max_buy_px']:>6.1f} € | {status_fmt}")

    print("=" * 92)
    print("💡 Per aggiornare un prezzo: python scripts/sync_cardmarket.py --update <item_id> <prezzo>")
    print("💡 Per ricalibrare il backtest europeo: python scripts/sync_cardmarket.py --calibrate\n")


if __name__ == "__main__":
    main()
