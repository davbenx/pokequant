#!/usr/bin/env python3
"""
scripts/sealed_age_window_search.py — Testa se restringere l'INGRESSO della
strategia sealed in produzione (TS Momentum, lookback=12m) a una finestra
d'eta' [min_age_months, max_age_months) da release, invece di comprare a
qualsiasi eta' (regola attuale, nessun tetto), migliora Sharpe/DSR/MaxDD.
Richiesto esplicitamente dall'utente con la finestra 4-14 mesi; qui si
verifica anche la STABILITA' su una griglia di finestre vicine, non solo il
punto singolo, con lo stesso standard usato per ogni altro test di questa
ricerca (griglia -> PBO/CSCV -> DSR -> block bootstrap -> walk-forward H1/H2),
contro il benchmark attuale in produzione (DSR 0.913, PBO 28.6%, Sharpe 1.10).

ESITO: NON MIGLIORA - la regola attuale senza tetto d'eta' resta la migliore su
ogni metrica che conta. Su 36 sealed (era 2019+): BASELINE Sharpe 1.10/CAGR
+23.5%/MaxDD -13.4%/24 trade, contro un massimo di Sharpe 1.05 (finestra 4-18m,
8 trade) tra tutte le varianti con tetto d'eta' provate (incl. la finestra 4-14m
richiesta: Sharpe 1.02, CAGR +20.1%, 6 trade). PBO sulla griglia 78.6% (7
candidati) - altissimo, segnala che su un universo di soli 36 asset restringere
la finestra d'ingresso non aggiunge segnale, solo rumore da meno trade/meno
diversificazione (24 trade -> 3-8 quando si aggiunge un tetto). Il meccanismo e'
intuitivo: ogni finestra d'eta' esclude la maggior parte dei mesi in cui un
asset e' idoneo a entrare, quindi meno posizioni indipendenti, varianza piu'
alta, Sharpe piu' basso - non un'illusione statistica, una perdita reale di
diversificazione. Walk-forward H1/H2 sul vincitore (=baseline stesso) identico
al benchmark in produzione, nessuna sorpresa. Non serve testare altre varianti:
il segnale (peggioramento monotono con qualsiasi restrizione) e' già chiaro.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.engine.strategies.time_series_momentum import TimeSeriesMomentumStrategy
from poke_quant.validation.statistical_validation import deflated_sharpe_ratio, pbo_cscv
from poke_quant.validation.bootstrap import block_bootstrap_metrics, summarize_bootstrap
from scripts.optimize_and_falsify import run_bt, MODERN_ERA_CUTOFF


def main():
    metadata = load_metadata()
    prices_full = load_price_matrix()
    sealed_ids = [
        k for k, v in metadata.items()
        if v.get("type") == "sealed" and v.get("data_quality") != "thin_unreliable"
        and k in prices_full.columns and v.get("release_date") and v["release_date"] >= MODERN_ERA_CUTOFF
    ]
    meta_sub = {k: v for k, v in metadata.items() if k in sealed_ids}
    prices_sub = prices_full[sealed_ids]
    print(f"Universo: {len(sealed_ids)} sealed (era moderna 2019+)")

    # Baseline in produzione (nessun filtro d'eta') + la finestra richiesta + vicini
    # per verificare stabilita' (non un solo punto scelto guardando il risultato).
    configs = {
        "BASELINE (no age filter)": dict(min_age_months=0, max_age_months=None),
        "age 4-14m": dict(min_age_months=4, max_age_months=14),
        "age 2-12m": dict(min_age_months=2, max_age_months=12),
        "age 6-16m": dict(min_age_months=6, max_age_months=16),
        "age 4-18m": dict(min_age_months=4, max_age_months=18),
        "age 0-14m": dict(min_age_months=0, max_age_months=14),
        "age 4-999m (solo min)": dict(min_age_months=4, max_age_months=None),
    }
    results = {}
    print("\nGriglia finestra d'eta' (mesi da release):")
    for name, kw in configs.items():
        strat = TimeSeriesMomentumStrategy(prices_sub, lookback_months=12, **kw)
        res = run_bt(strat, prices_sub, meta_sub)
        results[name] = res
        print(f"  {name:26s} | CAGR {res.cagr*100:+6.2f}% | Sharpe {res.sharpe:5.2f} | "
              f"MaxDD {res.max_drawdown*100:6.2f}% | Trade {res.total_trades:4d}")

    common_idx = None
    for res in results.values():
        common_idx = res.monthly_returns.index if common_idx is None else common_idx.intersection(res.monthly_returns.index)
    perf_matrix = np.column_stack([results[k].monthly_returns.loc[common_idx].values for k in configs])
    t_len = len(perf_matrix)
    splits = 8 if t_len >= 32 else 4
    rem = t_len % splits
    pbo = pbo_cscv(perf_matrix[rem:, :] if rem else perf_matrix, n_splits=splits)
    print(f"\nPBO sulla griglia ({len(configs)} candidati, {splits} split): {pbo:.3f}")

    best_name = max(results, key=lambda k: results[k].sharpe)
    best = results[best_name]
    dsr = deflated_sharpe_ratio(observed_sr=best.sharpe / np.sqrt(12), n_trials=len(configs), n_obs=len(best.monthly_returns))
    print(f"Migliore: '{best_name}' Sharpe {best.sharpe:.2f} | DSR (n_trials={len(configs)}): {dsr:.3f}")

    sims = block_bootstrap_metrics(best.monthly_returns, n_sims=500, block_size=6)
    print(summarize_bootstrap(sims))

    mid = len(prices_sub) // 2
    h1_dates, h2_dates = prices_sub.index[:mid], prices_sub.index[mid:]
    print(f"\nWalk-forward H1/H2 sul vincitore '{best_name}':")
    for label, dates in [("H1", h1_dates), ("H2", h2_dates)]:
        sub = prices_sub.loc[dates]
        strat = TimeSeriesMomentumStrategy(sub, lookback_months=12, **configs[best_name])
        res = run_bt(strat, sub, meta_sub)
        print(f"  {label} ({dates[0].strftime('%Y-%m')}->{dates[-1].strftime('%Y-%m')}) | Sharpe {res.sharpe:5.2f} | CAGR {res.cagr*100:+6.2f}%")

    print(f"\nWalk-forward H1/H2 sul BENCHMARK in produzione (nessun filtro d'eta') per confronto:")
    for label, dates in [("H1", h1_dates), ("H2", h2_dates)]:
        sub = prices_sub.loc[dates]
        strat = TimeSeriesMomentumStrategy(sub, lookback_months=12)
        res = run_bt(strat, sub, meta_sub)
        print(f"  {label} ({dates[0].strftime('%Y-%m')}->{dates[-1].strftime('%Y-%m')}) | Sharpe {res.sharpe:5.2f} | CAGR {res.cagr*100:+6.2f}%")


if __name__ == "__main__":
    main()
