#!/usr/bin/env python3
"""
scripts/liquidity_filter_backtest_comparison.py — Quantifica l'effetto del
filtro di attendibilita'/liquidita' bassa (poke_quant/data/liquidity_filter.py::
compute_reliability_flags, wired in liquid_singles_ids via
data_quality=="thin_unreliable") sul backtest singole di produzione.

DOMANDA DELL'UTENTE (2026-09-30): "Con il filtro liquidita' bassa attivo, il
backtest come sarebbe?" — Il filtro e' GIA' attivo nei numeri di produzione
mostrati in dashboard (liquid_singles_ids() e' gia' chiamato da
scripts/generate_singles_signal.py e scripts/update_monthly_cache.py, fix
precedente di questa stessa sessione - vedi commento in
generate_singles_signal.py::_liquid_universe). Questo script isola SOLO
l'effetto del controllo di attendibilita' (salti di prezzo estremi /
range max-min troppo ampio, compute_reliability_flags), tenendo fermi gli
altri criteri (esclusione franchise/lingua, pavimento costo di gradazione)
per non confondere l'effetto specifico chiesto con altri filtri gia'
adottati per altre ragioni.

METODO: stesso identico Backtester/ScarcityValueFactorStrategy/PRODUCTION_PARAMS
usati in produzione (scripts/update_monthly_cache.py), eseguito due volte
sullo stesso universo di partenza (franchise/lingua/pavimento gia' applicati
in entrambi i casi) - con e senza l'esclusione data_quality=="thin_unreliable".

ESITO: vedi output.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.data.liquidity_filter import (
    liquid_singles_ids, DEFAULT_EXCLUDED_FRANCHISES, DEFAULT_EXCLUDED_LANGUAGES,
    MIN_SINGLES_MEDIAN_PRICE_EUR, MIN_SINGLES_PRICE_WINDOW_MONTHS,
)
from poke_quant.engine.backtester import Backtester
from poke_quant.engine.strategies.scarcity_value_factor import ScarcityValueFactorStrategy
from scripts.generate_singles_signal import PRODUCTION_PARAMS


def liquid_singles_ids_no_reliability_check(metadata, prices_df):
    """Stessa funzione di liquid_singles_ids, ma SENZA il controllo
    data_quality=='thin_unreliable' - isola solo quell'effetto."""
    ids = []
    for item_id, info in metadata.items():
        if info.get("type") != "single":
            continue
        if info.get("franchise") in DEFAULT_EXCLUDED_FRANCHISES:
            continue
        if info.get("language") in DEFAULT_EXCLUDED_LANGUAGES:
            continue
        if item_id not in prices_df.columns:
            continue
        s_all = prices_df[item_id].dropna()
        s_all = s_all[s_all > 0]
        if s_all.empty:
            continue
        is_magic = info.get("franchise") == "magic"
        if not is_magic:
            s_window = s_all.iloc[-MIN_SINGLES_PRICE_WINDOW_MONTHS:]
            if s_window.median() < MIN_SINGLES_MEDIAN_PRICE_EUR:
                continue
            if s_all.iloc[-1] < MIN_SINGLES_MEDIAN_PRICE_EUR:
                continue
        ids.append(item_id)
    return ids


def run_backtest(ids, metadata, prices_full, label):
    meta_sub = {k: metadata[k] for k in ids}
    prices_sub = prices_full[ids]
    strat = ScarcityValueFactorStrategy(**PRODUCTION_PARAMS)
    bt = Backtester(
        strat, prices_sub, meta_sub, initial_cash=10000.0, platform="cardmarket",
        apply_liquidity_slippage=True, apply_holding_cost=True, apply_buy_side_shipping=True,
    )
    res = bt.run()
    print(f"{label:45s} n_universo={len(ids):4d}  Sharpe={res.sharpe:.2f}  "
          f"CAGR={res.cagr*100:+.1f}%  MaxDD={res.max_drawdown*100:.1f}%  "
          f"Trades={res.total_trades}")
    return res


def main():
    metadata = load_metadata()
    prices_full = load_price_matrix("historical_prices_graded_singles_grade9.csv")

    ids_with = liquid_singles_ids(metadata, prices_full)
    ids_without = liquid_singles_ids_no_reliability_check(metadata, prices_full)

    print("1. Effetto del controllo di attendibilita' (filtro liquidita' bassa)")
    print("   isolato dagli altri criteri (franchise/lingua/pavimento gradazione,")
    print("   applicati in entrambi i casi):\n")

    res_with = run_backtest(ids_with, metadata, prices_full, "CON filtro liquidita' bassa (produzione)")
    res_without = run_backtest(ids_without, metadata, prices_full, "SENZA filtro liquidita' bassa")

    readmitted = sorted(set(ids_without) - set(ids_with))
    print(f"\n2. Carte riammesse rimuovendo il filtro: {len(readmitted)}")
    print(f"   Delta Sharpe: {res_with.sharpe - res_without.sharpe:+.2f}")
    print(f"   Delta CAGR:   {(res_with.cagr - res_without.cagr)*100:+.1f} punti percentuali")

    if readmitted:
        print("\n3. Campione (prime 10) delle carte riammesse - motivo del flag:")
        from poke_quant.data.liquidity_filter import compute_reliability_flags
        flags = compute_reliability_flags(prices_full[readmitted])
        for item_id in readmitted[:10]:
            name = metadata.get(item_id, {}).get("name", item_id)
            ok, reason = flags.get(item_id, (True, ""))
            print(f"   {name:35s} {reason}")


if __name__ == "__main__":
    main()
