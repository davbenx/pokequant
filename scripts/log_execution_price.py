#!/usr/bin/env python3
"""
scripts/log_execution_price.py — Logga lo scarto reale tra prezzo dashboard e primo ask
verificato disponibile, per calibrare empiricamente lo slippage (invece dello 1-2% flat
attuale, non supportato da alcun dato).

Uso tipico (fallo ogni volta che controlli Cardmarket/eBay per un acquisto reale):

    python scripts/log_execution_price.py evolving_skies_bb --ask 2350.00 --listings 2
    python scripts/log_execution_price.py evolving_skies_bb --ask 2350.00 --dashboard 2222.22 --source ebay
    python scripts/log_execution_price.py --report

Il prezzo dashboard, se non passato con --dashboard, viene preso automaticamente
dall'ultimo mese disponibile - dal pannello grade9 per le singole (lo stesso
che mostra davvero il dashboard), da historical_prices_europe.csv/
historical_prices.csv per i box. Ogni osservazione si accumula in
data_cache/execution_price_log.csv — NON sovrascrive le precedenti.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_price_matrix, load_metadata
from poke_quant.data.execution_gap_calibrator import (
    append_execution_observation, load_execution_log, compute_premium_stats_by_tier,
    MIN_OBS_FOR_CALIBRATION,
)


def _latest_dashboard_price(item_id: str, metadata: dict) -> float:
    """Stesso bug gia' trovato e corretto in scripts/flag_unreliable_assets.py:
    historical_prices.csv/historical_prices_europe.csv (legacy) e
    historical_prices_graded_singles_grade9.csv hanno valori DIVERSI per la
    stessa carta - il dashboard mostra SEMPRE il pannello grade9 per le
    singole (v. app.py::get_singles_prices_full), quindi e' quello da
    controllare qui, altrimenti si calibra lo scarto sul prezzo sbagliato.
    Trovato loggando Raichu #14: senza questo fix avrebbe preso 22,86€
    (legacy raw) invece di 107,60€ (grade9, il vero prezzo dashboard),
    gonfiando lo scarto calibrato a ~900% invece del ~114% reale."""
    item_type = metadata.get(item_id, {}).get("type")
    filenames = (
        ("historical_prices_graded_singles_grade9.csv", "historical_prices_europe.csv", "historical_prices.csv")
        if item_type == "single"
        else ("historical_prices_europe.csv", "historical_prices.csv")
    )
    for fname in filenames:
        df = load_price_matrix(fname)
        if df is not None and item_id in df.columns:
            series = df[item_id].dropna()
            if not series.empty and series.iloc[-1] > 0:
                return float(series.iloc[-1])
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="Log dashboard price vs verified executable ask")
    parser.add_argument("item_id", nargs="?", help="ID item (es. evolving_skies_bb)")
    parser.add_argument("--ask", type=float, help="Prezzo del listing/ask più basso realmente disponibile ORA")
    parser.add_argument("--dashboard", type=float, default=None, help="Override manuale del prezzo dashboard (default: ultimo mese in cache)")
    parser.add_argument("--source", default="cardmarket", choices=["cardmarket", "ebay", "tcgplayer", "pricecharting", "other"])
    parser.add_argument("--listings", type=int, default=1, help="Numero di inserzioni attive viste a quel prezzo o sotto")
    parser.add_argument("--notes", default="")
    parser.add_argument("--report", action="store_true", help="Mostra lo stato di calibrazione attuale invece di loggare")
    args = parser.parse_args()

    if args.report:
        log_df = load_execution_log()
        metadata = load_metadata() or {}
        if log_df is None:
            print("Nessuna osservazione loggata ancora. Il friction model resta sul default statico (non calibrato).")
            return
        print(f"Osservazioni totali: {len(log_df)}\n")
        stats = compute_premium_stats_by_tier(log_df, metadata)
        print(stats.to_string(index=False))
        print(f"\nSoglia minima per fidarsi di una media (n_obs): {MIN_OBS_FOR_CALIBRATION}")
        under_threshold = stats[stats["n_obs"] < MIN_OBS_FOR_CALIBRATION]["set_tier"].tolist()
        if under_threshold:
            print(f"Tier ancora SOTTO soglia (numeri indicativi, non affidabili): {under_threshold}")
        return

    if not args.item_id or args.ask is None:
        parser.error("servono item_id e --ask (usa --report per vedere lo stato attuale)")

    metadata = load_metadata() or {}
    dashboard_price = args.dashboard if args.dashboard is not None else _latest_dashboard_price(args.item_id, metadata)
    if dashboard_price <= 0:
        parser.error(f"prezzo dashboard non trovato per '{args.item_id}': passa --dashboard esplicitamente")

    row = append_execution_observation(
        item_id=args.item_id,
        dashboard_price_eur=dashboard_price,
        verified_lowest_ask_eur=args.ask,
        source=args.source,
        active_listing_count=args.listings,
        notes=args.notes,
    )
    sign = "+" if row["premium_pct"] >= 0 else ""
    print(f"Loggato: {args.item_id} | dashboard {dashboard_price:.2f} € vs ask verificato {args.ask:.2f} € "
          f"| scarto {sign}{row['premium_pct']:.1f}% | fonte {args.source} | {args.listings} inserzioni")


if __name__ == "__main__":
    main()
