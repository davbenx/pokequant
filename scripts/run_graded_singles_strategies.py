#!/usr/bin/env python3
"""
scripts/run_graded_singles_strategies.py — Fase 2b: prima validazione, su dati REALI,
di strategie a basso rischio umano sul pannello "già gradate" (tier Grade 9, vedi
scripts/build_graded_singles_panel.py per la provenienza e le limitazioni).

ATTENZIONE — leggere prima di guardare i numeri:
  - Universo di soli 9 asset. Qualunque risultato qui è preliminare/esplorativo,
    non statisticamente potente (n troppo piccolo per DSR/PBO affidabili).
  - "Grade 9" non è PSA 10: PriceCharting non fornisce uno storico per-grado più
    alto. Il grading arbitrage (raw -> PSA10) resta non backtestabile con dati
    storici reali, coerente con la sua classificazione HUMAN_RISK_TIER="HIGH"
    (segnale live, non eseguito).
  - Nessuna frizione di piattaforma applicata qui in modo specifico per singole
    gradate (il Backtester usa le fee di config.py pensate per l'universo sealed).
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.engine.backtester import Backtester
from poke_quant.engine.strategies.time_series_momentum import TimeSeriesMomentumStrategy
from poke_quant.engine.strategies.carry_scarcity_factor import CarryScarcityFactorStrategy
from poke_quant.engine.strategies.equal_weight_benchmark import EqualWeightBenchmarkStrategy

GRADE9_FILENAME = "historical_prices_graded_singles_grade9.csv"


def main():
    grade9_df = load_price_matrix(GRADE9_FILENAME)
    metadata_full = load_metadata()
    singles_meta = {k: v for k, v in metadata_full.items() if v.get("type") == "single" and k in grade9_df.columns}

    print("=" * 90)
    print("  FASE 2b — SINGOLE GIA' GRADATE (tier Grade 9, dato reale PriceCharting)")
    print(f"  Universo: {len(singles_meta)} carte | Storico: {len(grade9_df)} mesi "
          f"({grade9_df.index[0].strftime('%Y-%m')} -> {grade9_df.index[-1].strftime('%Y-%m')})")
    print(f"  ATTENZIONE: n={len(singles_meta)} asset — ancora un campione ridotto per DSR/PBO affidabili,")
    print("  leggere questi risultati come esplorativi/direzionali, non come stima definitiva.")
    print("=" * 90)

    strategies = {
        "TS Momentum (12m)": TimeSeriesMomentumStrategy(grade9_df, lookback_months=12, item_type_filter="single"),
        "Carry/Scarsità (top 30% età)": CarryScarcityFactorStrategy(top_quantile=0.30, item_type_filter="single"),
        "Equal-Weight Buy & Hold": EqualWeightBenchmarkStrategy(item_type_filter="single"),
    }

    for name, strat in strategies.items():
        bt = Backtester(
            strategy=strat, historical_prices_df=grade9_df, items_metadata=singles_meta,
            initial_cash=10000.0, platform="cardmarket",
            apply_liquidity_slippage=True, apply_holding_cost=True,
        )
        res = bt.run()
        print(f"\n[{name}]")
        print(f"  Capitale Finale: {res.final_nav:,.2f} € | CAGR: {res.cagr*100:+.2f}% | "
              f"Sharpe: {res.sharpe:.2f} | MaxDD: {res.max_drawdown*100:.2f}% | Trade: {res.total_trades}")

    print("\n" + "=" * 90)
    print("Prossimo passo per rendere questo non-preliminare: espandere l'universo (decine/")
    print("centinaia di carte via pokemontcg.io + PriceCharting), non solo affinare le regole.")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
