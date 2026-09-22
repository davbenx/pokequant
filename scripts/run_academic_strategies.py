#!/usr/bin/env python3
"""
scripts/run_academic_strategies.py — Fase 1: strategie istituzionali/accademiche canoniche.

Confronta, sullo stesso universo e sulla stessa serie di prezzi REALE (historical_prices.csv,
PriceCharting USA — NON historical_prices_europe.csv, che è sintetico, vedi
poke_quant/data/europe_market_calibrator.py):

  1. Time-Series Momentum (Moskowitz-Ooi-Pedersen 2012)
  2. Cross-Sectional Momentum (Jegadeesh-Titman 1993)
  3. Carry/Scarsità (età da release come proxy di offerta irreversibile)
  4. Equal-Weight Buy & Hold (benchmark "beta dell'asset class", nessun timing/selezione)
  5. Optimal Sealed Strategy (baseline discrezionale esistente, per confronto)

Nessuno dei parametri delle strategie 1-4 è stato scelto guardando i rendimenti Pokémon/
One Piece: sono le definizioni standard di manuale, applicate as-is.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_price_matrix, load_metadata, load_macro_matrix
from poke_quant.engine.backtester import Backtester
from poke_quant.engine.strategies.optimal_sealed_strategy import OptimalSealedStrategy
from poke_quant.engine.strategies.time_series_momentum import TimeSeriesMomentumStrategy
from poke_quant.engine.strategies.cross_sectional_momentum import CrossSectionalMomentumStrategy
from poke_quant.engine.strategies.carry_scarcity_factor import CarryScarcityFactorStrategy
from poke_quant.engine.strategies.equal_weight_benchmark import EqualWeightBenchmarkStrategy
from poke_quant.validation.statistical_validation import deflated_sharpe_ratio, pbo_cscv


def main():
    prices_df = load_price_matrix()  # historical_prices.csv reale, NON la serie europea sintetica
    metadata = load_metadata()
    macro_df = load_macro_matrix()

    # Esclude asset "thin_unreliable" (scripts/flag_unreliable_assets.py): box vintage e
    # singole ultra-rare con prezzo guida PriceCharting inutilizzabile per volumi troppo
    # bassi (es. salti di migliaia di % in un mese). Vedi poke_quant/data/liquidity_filter.py.
    unreliable_ids = {k for k, v in metadata.items() if v.get("data_quality") == "thin_unreliable"}
    n_before = len(prices_df.columns)
    prices_df = prices_df[[c for c in prices_df.columns if c not in unreliable_ids]]
    print(f"Filtro attendibilità: esclusi {n_before - len(prices_df.columns)} asset thin_unreliable su {n_before}.")
    spy_series = macro_df["spy"] if macro_df is not None and "spy" in macro_df else None

    strategies = {
        "TS Momentum (12m)": TimeSeriesMomentumStrategy(prices_df, lookback_months=12),
        "Cross-Sectional Momentum (top 30%)": CrossSectionalMomentumStrategy(prices_df, lookback_months=6, top_quantile=0.30),
        "Carry/Scarsità (top 30% età)": CarryScarcityFactorStrategy(top_quantile=0.30),
        "Equal-Weight Buy & Hold": EqualWeightBenchmarkStrategy(),
        "Optimal Sealed (baseline discrezionale)": OptimalSealedStrategy(allowed_tiers=["S", "A"], min_buy_age_months=4, max_buy_age_months=14),
    }

    print("=" * 96)
    print("  FASE 1 — STRATEGIE ACCADEMICHE CANONICHE vs BASELINE DISCREZIONALE")
    print(f"  Universo: {len(metadata)} asset | Storico REALE: {len(prices_df)} mesi "
          f"({prices_df.index[0].strftime('%Y-%m')} -> {prices_df.index[-1].strftime('%Y-%m')})")
    print("=" * 96)

    results = {}
    for name, strat in strategies.items():
        bt = Backtester(
            strategy=strat, historical_prices_df=prices_df, items_metadata=metadata,
            initial_cash=10000.0, platform="cardmarket",
            benchmark_series=spy_series, apply_liquidity_slippage=True, apply_holding_cost=True,
        )
        res = bt.run()
        results[name] = res
        print(f"\n[{name}]")
        print(f"  Capitale Finale: {res.final_nav:,.2f} € | ROI Netto: {res.total_net_return*100:+.2f}% | "
              f"CAGR: {res.cagr*100:+.2f}%")
        print(f"  Sharpe: {res.sharpe:.2f} | MaxDD: {res.max_drawdown*100:.2f}% | "
              f"Alpha vs SPY: {res.alpha_annualized*100:+.2f}% | Trade Chiusi: {res.total_trades}")

    # ---------------------------------------------------------------
    # Validazione anti-overfitting su TUTTE le 5 varianti insieme
    # ---------------------------------------------------------------
    print("\n" + "-" * 96)
    print("VALIDAZIONE STATISTICA (DSR & PBO su tutte le 5 varianti confrontate in questo run)")
    n_trials = len(strategies)  # conteggio onesto delle varianti testate IN QUESTO CONFRONTO,
    # non del totale di combinazioni di parametri mai provate storicamente sul progetto (quello
    # è più alto e non tracciato: trattare questo DSR come limite superiore di affidabilità).

    common_idx = None
    for res in results.values():
        common_idx = res.monthly_returns.index if common_idx is None else common_idx.intersection(res.monthly_returns.index)

    perf_matrix = np.column_stack([results[name].monthly_returns.loc[common_idx].values for name in strategies])

    # NOTA BUG: res.sharpe è ANNUALIZZATO (metrics.sharpe() moltiplica per sqrt(12)),
    # ma deflated_sharpe_ratio() è parametrizzata su Sharpe PER-PERIODO coerente con n_obs
    # (Bailey & Lopez de Prado 2014). Passare lo Sharpe annualizzato satura la formula a 1.000
    # per qualunque strategia, come si può verificare qui sotto se si rimuove /sqrt(12). Questo
    # stesso bug è presente in run_backtest_cli.py e app.py: ogni DSR mostrato altrove nel
    # progetto va considerato invalido finché non viene corretto anche lì.
    for name, res in results.items():
        monthly_sr = res.sharpe / np.sqrt(12)
        dsr = deflated_sharpe_ratio(observed_sr=monthly_sr, n_trials=n_trials, n_obs=len(res.monthly_returns))
        print(f"  • DSR {name:42s}: {dsr:.3f} (Confidenza: {dsr*100:.1f}%)")

    t_len = len(perf_matrix)
    splits = 8 if t_len >= 32 else (4 if t_len >= 16 else 2)
    rem = t_len % splits
    perf_matrix_trimmed = perf_matrix[rem:, :] if rem > 0 else perf_matrix
    pbo = pbo_cscv(perf_matrix_trimmed, n_splits=splits)
    print(f"\n  Probability of Backtest Overfitting (PBO / CSCV, {splits} split): {pbo:.2f} ({pbo*100:.1f}%)")
    print("  PBO alto (>50%) significa che la strategia con lo Sharpe migliore in-sample NON è")
    print("  affidabilmente la migliore anche out-of-sample tra queste varianti.")
    print("=" * 96 + "\n")


if __name__ == "__main__":
    main()
