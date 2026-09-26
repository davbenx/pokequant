#!/usr/bin/env python3
"""
scripts/box_singles_split_optimization.py — Verifica dello split di capitale
box/singole (50/50, hardcoded in app.py) con criteri quantitativi reali:
inverse-vol (risk parity) e mean-variance/Kelly (pesi proporzionali a
Sigma^-1 * mu sui rendimenti mensili reali dei due backtest di produzione).

BUG/GAP TROVATO (verificando "e' la scelta migliore?"): esiste una vecchia
analisi risk-parity reale (scripts/build_combined_portfolio.py, inverse-vol
62,9/37,1) ma calcolata per un fattore singole poi RIGETTATO (PBO 0,514,
sign-flip OOS) e la cui sleeve fu chiusa (100% sealed, commit 0041b53).
Quando il fattore Scarsita' (quello davvero in produzione oggi, DSR 0,943)
ha riaperto la sleeve singole (commit 7372756), lo split e' stato rimesso a
50/50 senza ricalcolare NULLA per la nuova coppia di strategie - un default
arrotondato, non un optimum.

METODO: usa i rendimenti mensili REALI dei due backtest di produzione
correnti (box: TimeSeriesMomentumStrategy dopo il fix di priorita' d'ordine,
scripts/box_entry_priority_order_test.py; singole: ScarcityValueFactorStrategy,
PRODUCTION_PARAMS), allineati sulle date comuni.

  1. Inverse-vol (risk parity): w_i ∝ 1/sigma_i - ognuna delle due sleeve
     contribuisce la stessa quota di RISCHIO al blend, non di capitale.
  2. Mean-variance/Kelly: w ∝ Sigma^-1 * mu (pesi growth-optimal per rendimenti
     gaussiani, la stessa formula della tangency portfolio di Markowitz) -
     usa sia il rendimento atteso che la matrice di covarianza (quindi anche
     la correlazione 0,37 gia' mostrata in dashboard), non solo la volatilita'.
     Vincolato long-only e normalizzato a somma 1 (nessuna leva, nessuno
     short: coerente con "compro cose fisiche", non un portafoglio derivati).

Nessun parametro scelto guardando il risultato: entrambe le formule sono
standard, applicate SENZA modifiche ai numeri reali dei due backtest.

Uso: python scripts/box_singles_split_optimization.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.data.liquidity_filter import liquid_sealed_ids, liquid_singles_ids, MAX_PRICE_TO_MSRP_RATIO
from poke_quant.engine.strategies.time_series_momentum import TimeSeriesMomentumStrategy
from poke_quant.engine.strategies.scarcity_value_factor import ScarcityValueFactorStrategy
from poke_quant.engine.backtester import Backtester
from scripts.generate_singles_signal import PRODUCTION_PARAMS


def run_box():
    metadata = load_metadata()
    prices_full = load_price_matrix()
    sealed_ids = liquid_sealed_ids(metadata, prices_full)
    meta_sub = {k: v for k, v in metadata.items() if k in sealed_ids}
    prices_sub = prices_full[sealed_ids]
    strat = TimeSeriesMomentumStrategy(prices_sub, lookback_months=12, max_price_msrp_ratio=MAX_PRICE_TO_MSRP_RATIO)
    bt = Backtester(strat, prices_sub, meta_sub, initial_cash=10000.0, platform="cardmarket",
                     apply_liquidity_slippage=True, apply_holding_cost=True, apply_buy_side_shipping=True)
    return bt.run()


def run_singles():
    metadata = load_metadata()
    prices_full = load_price_matrix("historical_prices_graded_singles_grade9.csv")
    singles_ids = liquid_singles_ids(metadata, prices_full)
    meta_sub = {k: v for k, v in metadata.items() if k in singles_ids}
    prices_sub = prices_full[singles_ids]
    strat = ScarcityValueFactorStrategy(**PRODUCTION_PARAMS)
    bt = Backtester(strat, prices_sub, meta_sub, initial_cash=10000.0, platform="cardmarket",
                     apply_liquidity_slippage=True, apply_holding_cost=True, apply_buy_side_shipping=True)
    return bt.run()


def main():
    res_box = run_box()
    res_singles = run_singles()

    common_idx = res_box.monthly_returns.index.intersection(res_singles.monthly_returns.index)
    r_box = res_box.monthly_returns.loc[common_idx]
    r_singles = res_singles.monthly_returns.loc[common_idx]

    vol_box, vol_singles = r_box.std(), r_singles.std()
    mu_box, mu_singles = r_box.mean(), r_singles.mean()
    corr = r_box.corr(r_singles)

    print(f"Mesi comuni: {len(common_idx)}")
    print(f"Box:     vol mensile {vol_box*100:.2f}% | rendimento medio mensile {mu_box*100:.2f}%")
    print(f"Singole: vol mensile {vol_singles*100:.2f}% | rendimento medio mensile {mu_singles*100:.2f}%")
    print(f"Correlazione: {corr:.3f}")

    # 1. Inverse-vol (risk parity)
    w_box_ivp = (1.0 / vol_box) / (1.0 / vol_box + 1.0 / vol_singles)
    w_singles_ivp = 1.0 - w_box_ivp
    print(f"\n1. Inverse-vol (risk parity): box {w_box_ivp*100:.1f}% / singole {w_singles_ivp*100:.1f}%")

    # 2. Mean-variance / Kelly: w ∝ Sigma^-1 mu, vincolato long-only, normalizzato a somma 1.
    Sigma = np.array([[vol_box**2, corr*vol_box*vol_singles],
                       [corr*vol_box*vol_singles, vol_singles**2]])
    mu = np.array([mu_box, mu_singles])
    w_raw = np.linalg.solve(Sigma, mu)
    w_raw_clipped = np.clip(w_raw, 0.0, None)
    if w_raw_clipped.sum() > 0:
        w_mv = w_raw_clipped / w_raw_clipped.sum()
    else:
        w_mv = np.array([0.5, 0.5])
    print(f"2. Mean-variance/Kelly (grezzo, prima del clip long-only): box {w_raw[0]:.2f} / singole {w_raw[1]:.2f}")
    print(f"   Mean-variance/Kelly (long-only, normalizzato):          box {w_mv[0]*100:.1f}% / singole {w_mv[1]*100:.1f}%")

    print(f"\nSplit attuale in app.py: 50,0% / 50,0%")

    blend_50 = 0.5 * r_box + 0.5 * r_singles
    blend_ivp = w_box_ivp * r_box + w_singles_ivp * r_singles
    blend_mv = w_mv[0] * r_box + w_mv[1] * r_singles
    for name, blend in (("50/50", blend_50), ("inverse-vol", blend_ivp), ("mean-variance", blend_mv)):
        sharpe = (blend.mean() / blend.std()) * np.sqrt(12) if blend.std() > 0 else 0.0
        print(f"  Blend {name:14s} -> Sharpe {sharpe:.3f} | vol mensile {blend.std()*100:.2f}%")


if __name__ == "__main__":
    main()
