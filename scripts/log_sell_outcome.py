#!/usr/bin/env python3
"""
scripts/log_sell_outcome.py — Registra il lato VENDITA reale (poke_quant.data.
sell_execution_log): quando metti in vendita, quando/se vende e a che prezzo,
o se ritiri l'inserzione senza vendere. Nessun backtest o log di solo
acquisto puo' sostituire questo dato - richiesto esplicitamente dall'utente
("l'unico modo per fare un test e' comprare e vendere davvero").

Uso tipico:

    python scripts/log_sell_outcome.py list raichu_14 --price 260
    python scripts/log_sell_outcome.py sold raichu_14 --price 240 --list-date 2026-09-24
    python scripts/log_sell_outcome.py withdrawn raichu_14 --list-date 2026-09-24 --notes "nessun acquirente in 60gg"
    python scripts/log_sell_outcome.py --report

Ogni osservazione si accumula in data_cache/sell_execution_log.csv - NON
sovrascrive le precedenti. Un ritiro (withdrawn) e' dato utile quanto una
vendita riuscita: entra nel tasso di successo (sell-through rate).
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.sell_execution_log import (
    log_new_listing, log_sale_outcome, log_withdrawn, load_sell_log, compute_sell_stats,
)


def main():
    parser = argparse.ArgumentParser(description="Log listing/vendita/ritiro reale di un item")
    sub = parser.add_subparsers(dest="action")

    p_list = sub.add_parser("list", help="Registra una nuova inserzione in vendita")
    p_list.add_argument("item_id")
    p_list.add_argument("--price", type=float, required=True, help="Prezzo di listino")
    p_list.add_argument("--dashboard", type=float, default=None, help="Prezzo dashboard al momento del listing")
    p_list.add_argument("--date", default=None, help="Data listing (default: oggi)")
    p_list.add_argument("--notes", default="")

    p_sold = sub.add_parser("sold", help="Chiude un'inserzione come venduta")
    p_sold.add_argument("item_id")
    p_sold.add_argument("--price", type=float, required=True, help="Prezzo REALIZZATO (non il listino)")
    p_sold.add_argument("--list-date", default=None, help="Data della inserzione originale, se nota")
    p_sold.add_argument("--sold-date", default=None, help="Data vendita (default: oggi)")
    p_sold.add_argument("--notes", default="")

    p_withdrawn = sub.add_parser("withdrawn", help="Chiude un'inserzione senza vendita")
    p_withdrawn.add_argument("item_id")
    p_withdrawn.add_argument("--list-date", default=None)
    p_withdrawn.add_argument("--notes", default="")

    parser.add_argument("--report", action="store_true", help="Mostra le statistiche aggregate invece di loggare")
    args = parser.parse_args()

    if args.report or args.action is None:
        df = load_sell_log()
        if df is None:
            print("Nessuna vendita/inserzione registrata ancora.")
            return
        stats = compute_sell_stats(df)
        print(f"Inserzioni ancora aperte: {stats['n_listed_open']} | Vendute: {stats['n_sold']} | "
              f"Ritirate senza vendita: {stats['n_withdrawn']}")
        if stats["sell_through_rate"] is not None:
            print(f"Tasso di successo (venduto / (venduto+ritirato)): {stats['sell_through_rate']*100:.0f}%")
        if stats["median_days_on_market"] is not None:
            print(f"Giorni mediani sul mercato (solo vendute): {stats['median_days_on_market']:.0f}")
        if stats["median_discount_from_list_pct"] is not None:
            print(f"Sconto mediano realizzato vs. prezzo di listino: {stats['median_discount_from_list_pct']:+.1f}%")
        n_closed = stats["n_sold"] + stats["n_withdrawn"]
        print(f"\nOsservazioni chiuse: {n_closed} (soglia minima per fidarsi: 8) - "
              f"{'ATTENDIBILE' if stats['is_reliable'] else 'ANCORA INDICATIVO, non abbastanza dati'}")
        return

    if args.action == "list":
        row = log_new_listing(args.item_id, args.price, dashboard_price_eur=args.dashboard,
                               list_date=args.date, notes=args.notes)
        print(f"Registrato listing: {args.item_id} a {args.price:.2f}€ il {row['list_date']}")
    elif args.action == "sold":
        row = log_sale_outcome(args.item_id, args.price, list_date=args.list_date,
                                sold_date=args.sold_date, notes=args.notes)
        days = row.get("days_on_market", "?")
        print(f"Registrata vendita: {args.item_id} a {args.price:.2f}€ (giorni sul mercato: {days})")
    elif args.action == "withdrawn":
        row = log_withdrawn(args.item_id, list_date=args.list_date, notes=args.notes)
        print(f"Registrato ritiro senza vendita: {args.item_id}")


if __name__ == "__main__":
    main()
