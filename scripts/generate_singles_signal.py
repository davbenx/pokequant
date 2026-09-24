#!/usr/bin/env python3
"""
scripts/generate_singles_signal.py — Segnale mensile per il fattore scarsita'
sulle singole (poke_quant/engine/strategies/scarcity_value_factor.py, DSR
0,980 corretto per l'intera ricerca sulle singole - vedi
scripts/scarcity_value_singles_test.py). Stesso schema di
scripts/generate_monthly_signal.py per i box, riusato dalla dashboard.

COSA SIGNIFICA "TARDI" PER QUESTA STRATEGIA (richiesto esplicitamente
dall'utente, non un numero a caso): nel backtest validato (rebalance_every_months=3,
vedi get_singles_backtest_results() in app.py) una carta viene comprata SOLO la
prima volta che appare nel quantile BUY - una volta comprata, generate_signals()
la salta (e' gia' in portfolio.positions) finche' non esce dal quantile. Quindi
l'INTERO edge misurato nel backtest viene dall'acquisto al primo ingresso nel
quantile, controllato ogni 3 mesi - non esiste, nel backtest, il concetto di
"comprare una carta che e' nel quantile da 8 mesi", perche' sarebbe gia' stata
comprata al mese 1. Se il segnale live mostra oggi una carta che e' nel quantile
BUY da PIU' di 3 mesi consecutivi (la cadenza di controllo validata), comprarla
oggi non e' equivalente a quello che e' stato validato: e' un possibile value
trap (residuo negativo persistente che il mercato non corregge), non un ingresso
fresco. Sotto SIGNAL_FRESHNESS_MONTHS = 3, quelle carte vengono ESCLUSE dalla
lista BUY mostrata, non solo segnalate.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.engine.strategies.scarcity_value_factor import ScarcityValueFactorStrategy

PRODUCTION_PARAMS = dict(top_quantile=0.20, min_age_months=6, max_positions=60, min_cross_section=20)

# Cadenza di ribilanciamento del backtest validato (get_singles_backtest_results,
# rebalance_every_months=3) - vedi spiegazione nel docstring del modulo.
SIGNAL_FRESHNESS_MONTHS = 3
# Mesi di storico da controllare a ritroso: basta SIGNAL_FRESHNESS_MONTHS+1 per
# distinguere "fresca" (<=3 mesi consecutivi) da "tardiva" (>3), senza dover
# calcolare l'intero streak per le carte che verranno comunque escluse.
_CHECK_MONTHS = SIGNAL_FRESHNESS_MONTHS + 1


def _snapshot_for_date(prices_full: pd.DataFrame, metadata: dict, date) -> dict:
    price_row = prices_full.loc[date]
    snap = {}
    for item_id, info in metadata.items():
        if item_id in price_row.index and price_row[item_id] > 0 and not pd.isna(price_row[item_id]):
            s = dict(info)
            s["current_price"] = float(price_row[item_id])
            snap[item_id] = s
    return snap


def _quantile_membership(strat: ScarcityValueFactorStrategy, cur_dt: pd.Timestamp, snap: dict):
    """Ritorna (set carte nel quantile BUY, dict residui) per un singolo mese."""
    residuals = strat._fit_residuals(cur_dt, snap)
    if not residuals:
        return set(), {}
    n_buy = max(1, int(len(residuals) * strat.top_quantile))
    ranked = sorted(residuals.items(), key=lambda x: x[1])[:n_buy][: strat.max_positions]
    return {item_id for item_id, _ in ranked}, residuals


def _signal_streak(item_id: str, membership_by_month: dict, check_dates: list):
    """Streak di mesi CONSECUTIVI (a ritroso da check_dates[-1]) in cui item_id e'
    nel quantile BUY, e la data di inizio di quello streak. Si interrompe al primo
    mese di assenza (o dati mancanti quel mese) - un rientro dopo un'uscita conta
    come nuovo streak, non prosegue quello vecchio."""
    streak = 0
    start_date = check_dates[-1]
    for d in reversed(check_dates):
        if item_id in membership_by_month.get(d, set()):
            streak += 1
            start_date = d
        else:
            break
    return streak, start_date


def compute_singles_signal_rows():
    """Ritorna (rows, latest_date). rows contiene solo il quantile BUY (residuo
    piu' negativo) CON almeno SIGNAL_FRESHNESS_MONTHS+1 di storico disponibile
    per giudicare la freschezza, ESCLUSE le carte nel quantile da piu' di
    SIGNAL_FRESHNESS_MONTHS mesi consecutivi (vedi docstring del modulo)."""
    metadata = load_metadata()
    prices_full = load_price_matrix("historical_prices_graded_singles_grade9.csv")
    check_dates = list(prices_full.index[-_CHECK_MONTHS:])
    latest_date = check_dates[-1]

    strat = ScarcityValueFactorStrategy(**PRODUCTION_PARAMS)
    membership_by_month, residuals_by_month = {}, {}
    for d in check_dates:
        snap = _snapshot_for_date(prices_full, metadata, d)
        elig, residuals = _quantile_membership(strat, pd.to_datetime(d), snap)
        membership_by_month[d] = elig
        residuals_by_month[d] = residuals

    latest_snap = _snapshot_for_date(prices_full, metadata, latest_date)
    latest_eligible = membership_by_month[latest_date]

    rows = []
    for item_id in latest_eligible:
        streak, start_date = _signal_streak(item_id, membership_by_month, check_dates)
        if streak > SIGNAL_FRESHNESS_MONTHS:
            continue

        info = metadata[item_id]
        residual = residuals_by_month[latest_date][item_id]
        rows.append({
            "item_id": item_id,
            "name": info.get("name", item_id),
            "current_price_eur": latest_snap[item_id]["current_price"],
            "residual": residual,
            "discount_pct": (np.exp(residual) - 1.0) * 100.0,
            "signal_start_date": start_date,
            "months_in_signal": streak,
            "rarity": info.get("rarity"),
            "franchise": info.get("franchise", "pokemon"),
            "language": info.get("language", "en"),
        })
    rows.sort(key=lambda r: r["residual"])
    return rows, latest_date


def compute_singles_avoid_rows():
    """Ritorna (rows, latest_date) per il quantile OPPOSTO (residuo piu'
    positivo = sopravvalutata rispetto ai pari) - specchio del quantile BUY,
    stesso ruolo informativo della sezione 'Uscite' dei box (segnala lo stato
    dell'asset secondo il modello, non un portafoglio reale che il segnale live
    non traccia). Nessun filtro di freschezza qui: un sovrapprezzo persistente
    NON e' un value trap nello stesso senso del BUY, resta un'informazione
    valida indipendentemente da quanto dura."""
    metadata = load_metadata()
    prices_full = load_price_matrix("historical_prices_graded_singles_grade9.csv")
    latest_date = prices_full.index[-1]
    snap = _snapshot_for_date(prices_full, metadata, latest_date)

    strat = ScarcityValueFactorStrategy(**PRODUCTION_PARAMS)
    residuals = strat._fit_residuals(pd.to_datetime(latest_date), snap)
    if not residuals:
        return [], latest_date

    n_avoid = max(1, int(len(residuals) * strat.top_quantile))
    ranked = sorted(residuals.items(), key=lambda x: -x[1])[:n_avoid][: strat.max_positions]

    rows = []
    for item_id, residual in ranked:
        info = metadata[item_id]
        rows.append({
            "item_id": item_id,
            "name": info.get("name", item_id),
            "current_price_eur": snap[item_id]["current_price"],
            "residual": residual,
            "discount_pct": (np.exp(residual) - 1.0) * 100.0,
            "rarity": info.get("rarity"),
            "franchise": info.get("franchise", "pokemon"),
            "language": info.get("language", "en"),
        })
    rows.sort(key=lambda r: -r["residual"])
    return rows, latest_date


def main():
    rows, latest_date = compute_singles_signal_rows()
    print(f"Data segnale: {latest_date} | {len(rows)} carte nel quantile BUY fresche "
          f"(<= {SIGNAL_FRESHNESS_MONTHS} mesi, fattore scarsita')\n")
    for r in rows[:20]:
        start = r["signal_start_date"]
        start_str = start.strftime("%Y-%m") if hasattr(start, "strftime") else str(start)
        print(f"  {r['name']:38s} {r['current_price_eur']:8.2f}€ | sconto {r['discount_pct']:+6.1f}% | "
              f"da {start_str} ({r['months_in_signal']}m) | {r['rarity']}")

    avoid_rows, _ = compute_singles_avoid_rows()
    print(f"\nCarte sopravvalutate vs pari (quantile opposto): {len(avoid_rows)}\n")
    for r in avoid_rows[:20]:
        print(f"  {r['name']:38s} {r['current_price_eur']:8.2f}€ | sovrapprezzo {r['discount_pct']:+6.1f}% | {r['rarity']}")


if __name__ == "__main__":
    main()
