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
    old_grade9_df = load_price_matrix("historical_prices_graded_singles_grade9.csv")
    old_raw_singles_df = load_price_matrix("historical_prices_graded_singles_raw.csv")

    # Inizializza con i dati già presenti in cache per non perdere lo storico
    # se singole chiamate di rete falliscono o vanno in 429/Cloudflare.
    combined_series = {col: old_sealed_df[col].dropna() for col in old_sealed_df.columns} if old_sealed_df is not None else {}
    grade9_series = {col: old_grade9_df[col].dropna() for col in old_grade9_df.columns} if old_grade9_df is not None else {}
    raw_singles_series = {col: old_raw_singles_df[col].dropna() for col in old_raw_singles_df.columns} if old_raw_singles_df is not None else {}

    # Aggiorna solo gli asset: se l'asset ha già dati recenti e non è richiesto un full refresh,
    # evita di riscaricare indiscriminatamente 3.000+ asset da zero ad ogni run.
    print(f"Aggiornamento serie prezzi per {len(metadata)} asset censiti...")
    n_updated = 0
    consecutive_failures = 0

    for idx, (item_id, info) in enumerate(metadata.items(), 1):
        game_slug, item_slug = info.get("game_slug"), info.get("item_slug")
        if not game_slug or not item_slug:
            continue

        # Se abbiamo già dati storici e siamo in un blocco consecutivo di rete (es. Cloudflare 403/429),
        # ci fermiamo per non bloccare il workflow per 2 ore inutilmente.
        if consecutive_failures >= 15:
            print(f"  [ATTENZIONE] Rilevati {consecutive_failures} fallimenti consecutivi di rete (Cloudflare/Rate Limit). "
                  f"Stop fetch anticipato: mantengo i dati in cache per gli asset rimanenti.")
            break

        try:
            result = fetch_pricecharting_series(game_slug, item_slug, eur_usd_series=eur_usd)
            if not result:
                import time
                time.sleep(1)
                result = fetch_pricecharting_series(game_slug, item_slug, eur_usd_series=eur_usd)
        except Exception as e:
            result = None

        if not result:
            consecutive_failures += 1
            continue

        consecutive_failures = 0
        n_updated += 1

        if "raw" in result and not result["raw"].empty:
            combined_series[item_id] = result["raw"]
            if info.get("type") == "single":
                raw_singles_series[item_id] = result["raw"]
        if info.get("type") == "single" and "graded" in result and not result["graded"].empty:
            grade9_series[item_id] = result["graded"]

        if idx % 100 == 0:
            print(f"  Progresso fetch: {idx}/{len(metadata)} ({n_updated} aggiornati)")

    if combined_series:
        new_combined_df = pd.DataFrame(combined_series).sort_index().ffill()
        save_price_matrix(new_combined_df, "historical_prices.csv")
    else:
        new_combined_df = old_sealed_df

    if grade9_series:
        new_grade9_df = pd.DataFrame(grade9_series).sort_index()
        save_price_matrix(new_grade9_df, "historical_prices_graded_singles_grade9.csv")
    else:
        new_grade9_df = old_grade9_df

    if raw_singles_series:
        new_raw_singles_df = pd.DataFrame(raw_singles_series).sort_index()
        save_price_matrix(new_raw_singles_df, "historical_prices_graded_singles_raw.csv")
    else:
        new_raw_singles_df = old_raw_singles_df

    print(f"\nAggiornato historical_prices.csv: {new_combined_df.shape[0]} mesi x {new_combined_df.shape[1]} asset")
    if new_grade9_df is not None:
        print(f"Aggiornato graded_singles_grade9.csv: {new_grade9_df.shape[0]} mesi x {new_grade9_df.shape[1]} carte")
    new_sealed_df = new_combined_df

    if old_sealed_df is not None and new_sealed_df is not None and not new_sealed_df.empty:
        common_cols = [c for c in new_sealed_df.columns if c in old_sealed_df.columns]
        last_date = new_sealed_df.index[-1]
        if last_date in old_sealed_df.index and common_cols:
            deltas = ((new_sealed_df.loc[last_date, common_cols] - old_sealed_df.loc[last_date, common_cols])
                      / old_sealed_df.loc[last_date, common_cols].replace(0, pd.NA)) * 100.0
            deltas = deltas.dropna().sort_values()
            if not deltas.empty:
                print(f"\nImpatto sul prezzo dell'ultimo mese ({last_date.strftime('%Y-%m')}):")
                print(f"  Delta minimo:  {deltas.iloc[0]:+.2f}% ({deltas.index[0]})")
                print(f"  Delta mediano: {deltas.median():+.2f}%")
            print(f"  Delta massimo: {deltas.iloc[-1]:+.2f}% ({deltas.index[-1]})")


if __name__ == "__main__":
    main()
