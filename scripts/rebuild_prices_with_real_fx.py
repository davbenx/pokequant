#!/usr/bin/env python3
"""
scripts/rebuild_prices_with_real_fx.py — Ricostruisce historical_prices.csv e i
pannelli graded-singles usando il tasso EUR/USD storico REALE (data_cache/eur_usd_fx.csv,
Twelve Data) invece della costante fissa DEFAULT_EUR_USD=1.08 usata finora per ogni
conversione USD->EUR su tutto il periodo 2021-2026.

Rifà lo stesso fetch da PriceCharting già usato per costruire i pannelli attuali
(stesso metodo compliant, stessi item_id) — non introduce fonti nuove, corregge solo
la conversione valutaria applicata ai dati già raccolti.

Stampa un confronto vecchio-vs-nuovo sull'ultimo mese per quantificare l'impatto.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix, save_price_matrix, ensure_cache_dir
from poke_quant.data.price_fetcher import fetch_pricecharting_series
from poke_quant.data.fx_rates import load_eur_usd_series


def main():
    metadata = load_metadata() or {}
    eur_usd = load_eur_usd_series()
    if eur_usd is None:
        print("data_cache/eur_usd_fx.csv non trovato. Interrotto.")
        return
    print(f"Tasso EUR/USD reale caricato: {len(eur_usd)} mesi ({eur_usd.index[0].strftime('%Y-%m')} -> "
          f"{eur_usd.index[-1].strftime('%Y-%m')})")

    old_sealed_df = load_price_matrix("historical_prices.csv")

    # historical_prices.csv contiene STORICAMENTE sia sealed che single (run_backtest_cli.py
    # e app.py passano lo stesso prices_df sia a OptimalSealedStrategy/SealedAccumulator sia
    # a ChaseDipBuyerStrategy, che filtra internamente per type=="single"). Va mantenuto
    # combinato per non rompere quel contratto — i pannelli graded-singles dedicati sono
    # un'AGGIUNTA, non un sostituto.
    combined_series, grade9_series, raw_singles_series = {}, {}, {}

    for item_id, info in metadata.items():
        game_slug, item_slug = info.get("game_slug"), info.get("item_slug")
        if not game_slug or not item_slug:
            continue
        result = fetch_pricecharting_series(game_slug, item_slug, eur_usd_series=eur_usd)
        if not result:  # una singola ritentata su timeout transitorio, non su 404/redirect
            import time
            time.sleep(2)
            result = fetch_pricecharting_series(game_slug, item_slug, eur_usd_series=eur_usd)
        if not result:
            print(f"  [MANCANTE] {item_id} ({game_slug}/{item_slug}): fetch fallito anche al secondo tentativo")
            continue

        if "raw" in result and not result["raw"].empty:
            combined_series[item_id] = result["raw"]
            if info.get("type") == "single":
                raw_singles_series[item_id] = result["raw"]
        if info.get("type") == "single" and "graded" in result and not result["graded"].empty:
            grade9_series[item_id] = result["graded"]

    new_combined_df = pd.DataFrame(combined_series).sort_index().ffill()
    new_grade9_df = pd.DataFrame(grade9_series).sort_index()
    new_raw_singles_df = pd.DataFrame(raw_singles_series).sort_index()

    save_price_matrix(new_combined_df, "historical_prices.csv")
    save_price_matrix(new_grade9_df, "historical_prices_graded_singles_grade9.csv")
    save_price_matrix(new_raw_singles_df, "historical_prices_graded_singles_raw.csv")

    print(f"\nRicostruito historical_prices.csv: {new_combined_df.shape[0]} mesi x {new_combined_df.shape[1]} asset (sealed+single)")
    print(f"Ricostruito graded_singles_grade9.csv: {new_grade9_df.shape[0]} mesi x {new_grade9_df.shape[1]} carte")
    new_sealed_df = new_combined_df  # per il confronto delta sotto

    if old_sealed_df is not None:
        common_cols = [c for c in new_sealed_df.columns if c in old_sealed_df.columns]
        last_date = new_sealed_df.index[-1]
        if last_date in old_sealed_df.index:
            deltas = ((new_sealed_df.loc[last_date, common_cols] - old_sealed_df.loc[last_date, common_cols])
                      / old_sealed_df.loc[last_date, common_cols].replace(0, pd.NA)) * 100.0
            deltas = deltas.dropna().sort_values()
            print(f"\nImpatto sul prezzo dell'ultimo mese ({last_date.strftime('%Y-%m')}), vecchio vs nuovo tasso FX:")
            print(f"  Delta minimo:  {deltas.iloc[0]:+.2f}% ({deltas.index[0]})")
            print(f"  Delta mediano: {deltas.median():+.2f}%")
            print(f"  Delta massimo: {deltas.iloc[-1]:+.2f}% ({deltas.index[-1]})")


if __name__ == "__main__":
    main()
