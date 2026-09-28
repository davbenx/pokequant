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
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.data.liquidity_filter import liquid_singles_ids
from poke_quant.engine.strategies.scarcity_value_factor import ScarcityValueFactorStrategy
from poke_quant.data.cardmarket_bridge import WOTC_FIRST_EDITION_SETS
from poke_quant.slabs.grading_multipliers import (
    get_recommended_grade_for_card,
    estimate_psa10_from_psa9,
    ERA_PSA10_TO_PSA9_RATIO,
    normalize_era,
)


def _liquid_universe(metadata: dict, prices_full: pd.DataFrame):
    """Restringe metadata/prices_full all'universo investibile (liquidity_filter.py::
    liquid_singles_ids) PRIMA di generare qualunque segnale. BUG TROVATO (l'utente:
    "Mantine e' consigliata a 11EUR, ma la gradazione costa di piu'"): get_singles_backtest_results()
    in app.py (Sharpe validato 1,57/1,58) filtra correttamente data_quality="thin_unreliable",
    ma questo modulo - che genera le liste BUY/alternative/avoid REALMENTE mostrate in
    dashboard - non filtrava MAI nulla: caricava metadata/prices_full grezzi. Risultato
    verificato: 2 carte su 15 nella lista BUY e 11 su 107 nelle alternative erano gia'
    flaggate thin_unreliable, escluse dal backtest che produce lo Sharpe mostrato ma
    proposte comunque come acquisto. Aggiunto anche il pavimento di costo di gradazione
    (vedi liquidity_filter.py::MIN_SINGLES_MEDIAN_PRICE_EUR) qui, non solo nel backtest."""
    ids = liquid_singles_ids(metadata, prices_full)
    return {k: metadata[k] for k in ids}, prices_full[ids]


# Frazione della distanza tra prezzo attuale e confine teorico del quantile
# che puo' essere concessa all'acquisto preservando un Edge istituzionale (Sharpe >= 1.0).
# Validato empiricamente su 69 mesi in scripts/max_edge_preservation_test.py:
# a 0.10 lo Sharpe resta 1.01 e il CAGR +19.42% (a 0.25 si tocca il pareggio Sharpe=0.0,
# mentre il vecchio confine teorico 1.0 crollava a Sharpe -0.87 e CAGR -31.3%).
PRESERVE_EDGE_ALPHA = 0.10

# Nomi di set con abbreviazioni note che title() rende male ("pokemon-xy" ->
# "Xy" invece di "XY") - lista corretta a mano, non un algoritmo generico.
_SET_NAME_OVERRIDES = {"pokemon-xy": "XY"}


def _set_label(info: dict) -> Optional[str]:
    """Nome leggibile del set/espansione da game_slug (es. "pokemon-jungle" ->
    "Jungle"), con "(Unlimited)" aggiunto per i set WOTC 1999-2000 (Base Set,
    Jungle, Fossil, Team Rocket, Gym, Base Set 2): esiste anche una stampa
    "1st Edition" della stessa carta, spesso 2-4x+ piu' cara, e il nostro
    pannello grade9 traccia sempre la Unlimited - trovato indagando uno
    scarto di prezzo reale (Raichu #14 Fossil: 107,60€ Unlimited mostrati vs
    250€ trovati dall'utente, quasi esattamente il prezzo reale della
    variante 1st Edition, non un errore nel dato).

    Richiesto esplicitamente dall'utente ("Clefable: e' jungle o prima
    edizione o Unlimited? Fai in modo che io non possa sbagliare mai la
    carta da comprare"): PRIMA questa informazione esisteva SOLO per i set
    WOTC (come suffisso "(Unlimited)" appeso al nome, senza mai dire QUALE
    set) - un nome carta come "Clefable #1" da solo non basta a identificare
    univocamente la carta da cercare, va sempre affiancato dal set. Ora
    ogni carta mostra il proprio set esplicitamente, non solo le WOTC."""
    slug = info.get("game_slug")
    if not slug:
        return None
    if slug in _SET_NAME_OVERRIDES:
        label = _SET_NAME_OVERRIDES[slug]
    else:
        for prefix in ("pokemon-", "one-piece-", "magic-"):
            if slug.startswith(prefix):
                slug = slug[len(prefix):]
                break
        label = slug.replace("-", " ").title()
    if info.get("game_slug") in WOTC_FIRST_EDITION_SETS:
        label += " (Unlimited)"
    return label

# STRATEGIA UNIFICATA PRODUZIONE:
# Optimum vincolato verificato empiricamente in scripts/dac7_turnover_search.py:
# rebalance_every_months=3, max_positions=20.
# Unifica la modalita' standard e la conformita' DAC7 (< 30 vendite annue):
# Sharpe 2.386 (DSR 0.999), CAGR +67.68%, MaxDD -9.70%, 28.6 vendite/anno reali.
# Supera la vecchia configurazione a 60 posizioni (Sharpe 2.115, CAGR +44.41%, 61.2 vendite/anno)
# eliminando frizioni, concentrando il capitale solo sui residui a massimo sconto (-80%/-81%)
# e garantendo di default la conformita' fiscale senza modalita' separate.
PRODUCTION_PARAMS = dict(rebalance_every_months=3, top_quantile=0.20, min_age_months=6, max_positions=20, min_cross_section=20)
DAC7_SINGLES_PARAMS = PRODUCTION_PARAMS  # Alias per retrocompatibilita'

# Cadenza di ribilanciamento del backtest validato - vedi spiegazione nel
# docstring del modulo. Non piu' una costante fissa: deriva dal parametro
# rebalance_every_months di qualunque config venga passata, cosi' la finestra
# di "freschezza" resta coerente con la cadenza REALMENTE usata nel backtest
# (3 mesi in produzione, 12 in modalita' DAC7 a turnover ridotto).
def _signal_freshness_months(params: dict) -> int:
    return params.get("rebalance_every_months", 3)


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
    """Ritorna (set carte nel quantile BUY, dict residui, residuo di confine
    del quantile) per un singolo mese. Il residuo di confine e' quello della
    carta piu' marginale ancora dentro al quantile BUY - serve a calcolare
    quanto puo' salire il prezzo di una carta prima che esca dal quantile
    (vedi max_edge_price in compute_singles_signal_rows)."""
    residuals = strat._fit_residuals(cur_dt, snap)
    if not residuals:
        return set(), {}, None
    n_buy = max(1, int(len(residuals) * strat.top_quantile))
    ranked = sorted(residuals.items(), key=lambda x: x[1])[:n_buy][: strat.max_positions]
    cutoff_residual = ranked[-1][1] if ranked else None
    return {item_id for item_id, _ in ranked}, residuals, cutoff_residual


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


def compute_singles_signal_rows(params: dict = None):
    """Ritorna (rows, latest_date). rows contiene solo il quantile BUY (residuo
    piu' negativo) CON almeno freshness_months+1 di storico disponibile per
    giudicare la freschezza, ESCLUSE le carte nel quantile da piu' di
    freshness_months mesi consecutivi (vedi docstring del modulo) - dove
    freshness_months = params['rebalance_every_months'], la cadenza REALE del
    backtest per questa config (3 in produzione, 12 in modalita' DAC7)."""
    params = params or PRODUCTION_PARAMS
    freshness_months = _signal_freshness_months(params)
    check_months = freshness_months + 1

    metadata = load_metadata()
    prices_full = load_price_matrix("historical_prices_graded_singles_grade9.csv")
    metadata, prices_full = _liquid_universe(metadata, prices_full)
    check_dates = list(prices_full.index[-check_months:])
    latest_date = check_dates[-1]

    strat = ScarcityValueFactorStrategy(**params)
    membership_by_month, residuals_by_month, cutoff_by_month = {}, {}, {}
    for d in check_dates:
        snap = _snapshot_for_date(prices_full, metadata, d)
        elig, residuals, cutoff = _quantile_membership(strat, pd.to_datetime(d), snap)
        membership_by_month[d] = elig
        residuals_by_month[d] = residuals
        cutoff_by_month[d] = cutoff

    latest_snap = _snapshot_for_date(prices_full, metadata, latest_date)
    latest_eligible = membership_by_month[latest_date]
    latest_cutoff = cutoff_by_month[latest_date]

    rows = []
    for item_id in latest_eligible:
        streak, start_date = _signal_streak(item_id, membership_by_month, check_dates)
        if streak > freshness_months:
            continue

        info = metadata[item_id]
        residual = residuals_by_month[latest_date][item_id]
        current_price = latest_snap[item_id]["current_price"]
        # Prezzo massimo per MANTENERE L'EDGE:
        # Il vecchio confine teorico (current_price * exp(latest_cutoff - residual))
        # rappresentava la frontiera lorda del quantile BUY. Acquistare a quel
        # prezzo distruggeva l'intero edge netto (Sharpe -0.87, CAGR -31.3%) a causa
        # delle frizioni reali (fee 5-13%, spedizione, slippage e vendite forzate al ribilanciamento).
        # Per mantenere un Edge statisticamente solido (Sharpe >= 1.01, CAGR +19.4%),
        # l'investitore puo' concedere al massimo il 10% (PRESERVE_EDGE_ALPHA = 0.10) della distanza
        # tra il prezzo attuale e il cutoff teorico (vedi scripts/max_edge_preservation_test.py).
        if latest_cutoff is not None:
            theoretical_cutoff_price = current_price * np.exp(latest_cutoff - residual)
            max_edge_price_eur = current_price + PRESERVE_EDGE_ALPHA * (theoretical_cutoff_price - current_price)
        else:
            max_edge_price_eur = None

        rel_year = int(str(info.get("release_date", "2020"))[:4]) if info.get("release_date") else 2020
        rec = get_recommended_grade_for_card(rel_year=rel_year)

        is_target_psa10 = (rec["target_grade"] == "PSA 10")
        if is_target_psa10:
            target_price_eur = estimate_psa10_from_psa9(current_price, rec["era"])
            era_ratio = ERA_PSA10_TO_PSA9_RATIO.get(normalize_era(rec["era"]), 2.80)
            target_max_edge_eur = round(max_edge_price_eur * era_ratio, 2) if max_edge_price_eur is not None else None
        else:
            target_price_eur = current_price
            target_max_edge_eur = max_edge_price_eur

        rows.append({
            "item_id": item_id,
            "name": info.get("name", item_id),
            "set_name": _set_label(info),
            "current_price_eur": current_price,
            "residual": residual,
            "discount_pct": (np.exp(residual) - 1.0) * 100.0,
            "max_edge_price_eur": max_edge_price_eur,
            "is_target_psa10": is_target_psa10,
            "target_price_eur": target_price_eur,
            "target_max_edge_price_eur": target_max_edge_eur,
            "signal_start_date": start_date,
            "months_in_signal": streak,
            "rarity": info.get("rarity"),
            "franchise": info.get("franchise", "pokemon"),
            "language": info.get("language", "en"),
            "era": rec["era"],
            "era_label": rec["era_label"],
            "recommended_grade": rec["target_grade"],
            "target_badge": rec["target_badge"],
            "badge_color": rec["badge_color"],
            "grade_advice": rec["rationale"],
            "short_grade_advice": rec["short_advice"],
            "short_advice": rec["short_advice"],
            "is_grade9_viable": rec["is_grade9_viable"],
            "warning_modern_g9": rec["warning_modern_g9"],
        })
    rows.sort(key=lambda r: r["residual"])
    for idx, r in enumerate(rows):
        r["tier"] = "core" if idx < 8 else "bench"
    return rows, latest_date


# Trovato indagando "se non trovo tutte le copie consigliate di una carta,
# cosa mi conviene comprare al suo posto?" (scripts/singles_diversify_when_capped_test.py):
# col tetto realistico di 1 copia per acquisto, usare il budget liberato per
# comprare PIU' carte diverse (rank 61-173, ancora dentro al quantile 20% piu'
# sottovalutato, solo fuori dalle prime max_positions=60 per rank) recupera
# gran parte dell'edge perso - Sharpe 0,30->0,80 - ma non tutto: MaxDD peggiora
# (-13%->-21%) e il DSR resta sotto la soglia usata per validare le altre
# strategie di questa dashboard (0,29 contro 0,87-0,95). E' un ripiego
# empiricamente migliore di lasciare il capitale fermo, NON una strategia a se'
# validata - va mostrato come tale, non come un secondo elenco BUY equivalente.
# Sentinella ampia invece di un numero calibrato su una dimensione universo
# specifica (che cambia col tempo e col nuovo pavimento di costo di gradazione
# - vedi liquid_singles_ids): lo slicing ranked[max_positions:max_positions+N]
# restituisce comunque solo cio' che esiste, anche se N supera la lunghezza
# del quantile - l'intento e' "mostra TUTTO il resto del quantile 20%", non
# un tetto di per se' significativo.
EXTRA_ALTERNATIVES = 1000


def compute_singles_alternative_rows(params: dict = None, extra_positions: int = EXTRA_ALTERNATIVES):
    """Carte nel quantile 20% piu' sottovalutato del mese corrente ESCLUSE
    quelle GIA' MOSTRATE come BUY principale (compute_singles_signal_rows) -
    NON allarga il quantile stesso (leva diversa, testata separatamente e
    piu' rischiosa, vedi scripts/singles_diversify_when_capped_test.py).
    Nessun filtro di freschezza/streak qui: sono suggerimenti di ripiego per
    il mese corrente, non un elenco BUY testato a se'.

    TROVATO verificando un cambio reale in dashboard (l'utente: "vedevo
    raichu, aerodactyl, dragonite - adesso vedo moltres, dark charizard"):
    l'esclusione qui usava SOLO il rank (prime max_positions=60), non
    l'elenco REALMENTE mostrato in compute_singles_signal_rows (che applica
    ANCHE il filtro di freschezza/value-trap). Risultato: una carta ancora
    nel quantile top-60 per rank ma esclusa dal BUY principale perche' li'
    da piu' di 3 mesi (es. raichu_14, rank 9, streak 4 mesi) non appariva
    NE' nel BUY (freschezza) NE' nelle alternative (rank<60) - invisibile
    ovunque, anche se il modello la considera ancora sottovalutata. Corretto
    usando l'elenco EFFETTIVAMENTE mostrato come esclusione, non il rank
    grezzo - ora quella carta appare come alternativa."""
    params = params or PRODUCTION_PARAMS
    shown_rows, _ = compute_singles_signal_rows(params)
    already_shown = {r["item_id"] for r in shown_rows}

    metadata = load_metadata()
    prices_full = load_price_matrix("historical_prices_graded_singles_grade9.csv")
    metadata, prices_full = _liquid_universe(metadata, prices_full)
    latest_date = prices_full.index[-1]

    strat = ScarcityValueFactorStrategy(**params)
    snap = _snapshot_for_date(prices_full, metadata, latest_date)
    residuals = strat._fit_residuals(pd.to_datetime(latest_date), snap)
    if not residuals:
        return [], latest_date

    n_buy = max(1, int(len(residuals) * strat.top_quantile))
    ranked = sorted(residuals.items(), key=lambda x: x[1])[:n_buy]
    extra = [(iid, res) for iid, res in ranked if iid not in already_shown][:extra_positions]
    # Confine del quantile 20% INTERO (non quello troncato a max_positions usato
    # per la lista principale): per un'alternativa, la domanda e' "quanto puo'
    # salire il prezzo prima che la carta esca dal quantile piu' sottovalutato
    # in assoluto", non prima di uscire dalle prime 60 per rank - sono due confini
    # diversi, e usare quello sbagliato darebbe un "massimo" piu' basso del
    # prezzo attuale per carte gia' oltre le prime 60 (matematicamente corretto
    # ma fuorviante da leggere in dashboard).
    quantile_cutoff_residual = ranked[-1][1] if ranked else None

    rows = []
    for item_id, residual in extra:
        info = metadata[item_id]
        current_price = snap[item_id]["current_price"]
        if quantile_cutoff_residual is not None:
            theoretical_cutoff_price = current_price * np.exp(quantile_cutoff_residual - residual)
            max_edge_price_eur = current_price + PRESERVE_EDGE_ALPHA * (theoretical_cutoff_price - current_price)
        else:
            max_edge_price_eur = None

        rel_year = int(str(info.get("release_date", "2020"))[:4]) if info.get("release_date") else 2020
        rec = get_recommended_grade_for_card(rel_year=rel_year)

        is_target_psa10 = (rec["target_grade"] == "PSA 10")
        if is_target_psa10:
            target_price_eur = estimate_psa10_from_psa9(current_price, rec["era"])
            era_ratio = ERA_PSA10_TO_PSA9_RATIO.get(normalize_era(rec["era"]), 2.80)
            target_max_edge_eur = round(max_edge_price_eur * era_ratio, 2) if max_edge_price_eur is not None else None
        else:
            target_price_eur = current_price
            target_max_edge_eur = max_edge_price_eur

        rows.append({
            "item_id": item_id,
            "name": info.get("name", item_id),
            "set_name": _set_label(info),
            "current_price_eur": current_price,
            "residual": residual,
            "discount_pct": (np.exp(residual) - 1.0) * 100.0,
            "max_edge_price_eur": max_edge_price_eur,
            "is_target_psa10": is_target_psa10,
            "target_price_eur": target_price_eur,
            "target_max_edge_price_eur": target_max_edge_eur,
            "rarity": info.get("rarity"),
            "franchise": info.get("franchise", "pokemon"),
            "language": info.get("language", "en"),
            "era": rec["era"],
            "era_label": rec["era_label"],
            "recommended_grade": rec["target_grade"],
            "target_badge": rec["target_badge"],
            "badge_color": rec["badge_color"],
            "grade_advice": rec["rationale"],
            "short_grade_advice": rec["short_advice"],
            "short_advice": rec["short_advice"],
            "is_grade9_viable": rec["is_grade9_viable"],
            "warning_modern_g9": rec["warning_modern_g9"],
            "tier": "alternative",
        })
    rows.sort(key=lambda r: r["residual"])
    return rows, latest_date


def compute_singles_avoid_rows(params: dict = None):
    """Ritorna (rows, latest_date) per il quantile OPPOSTO (residuo piu'
    positivo = sopravvalutata rispetto ai pari) - specchio del quantile BUY,
    stesso ruolo informativo della sezione 'Uscite' dei box (segnala lo stato
    dell'asset secondo il modello, non un portafoglio reale che il segnale live
    non traccia). Nessun filtro di freschezza qui: un sovrapprezzo persistente
    NON e' un value trap nello stesso senso del BUY, resta un'informazione
    valida indipendentemente da quanto dura."""
    params = params or PRODUCTION_PARAMS
    metadata = load_metadata()
    prices_full = load_price_matrix("historical_prices_graded_singles_grade9.csv")
    metadata, prices_full = _liquid_universe(metadata, prices_full)
    latest_date = prices_full.index[-1]
    snap = _snapshot_for_date(prices_full, metadata, latest_date)

    strat = ScarcityValueFactorStrategy(**params)
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
            "set_name": _set_label(info),
            "current_price_eur": snap[item_id]["current_price"],
            "residual": residual,
            "discount_pct": (np.exp(residual) - 1.0) * 100.0,
            "rarity": info.get("rarity"),
            "franchise": info.get("franchise", "pokemon"),
            "language": info.get("language", "en"),
            "tier": "avoid",
        })
    rows.sort(key=lambda r: -r["residual"])
    return rows, latest_date


NON_HOLO_BULK_RARITIES = {
    "Common", "common", "Uncommon", "uncommon", "Rare", "rare", "Rare ACE", "ACE SPEC Rare"
}


def filter_singles_rows(
    rows: list[dict],
    min_price: float = 0.0,
    max_price: float = 0.0,
    only_holo: bool = False,
    pokemon_only: bool = True,
    retag_tiers: bool = True,
) -> list[dict]:
    """Filtra una lista di segnali singole per rimuovere frizione di spedizione,
    carte bulk/non-holo poco liquide ed eventuali TCG non desiderati."""
    filtered = []
    for r in rows:
        p = float(r.get("target_price_eur") or r.get("current_price_eur", 0.0))
        if min_price > 0 and p < min_price:
            continue
        if max_price > 0 and p > max_price:
            continue
        if pokemon_only and r.get("franchise") != "pokemon":
            continue
        if only_holo and str(r.get("rarity")) in NON_HOLO_BULK_RARITIES:
            continue
        filtered.append(dict(r))

    if retag_tiers and any(r.get("tier") in ("core", "bench") for r in filtered):
        for idx, r in enumerate(filtered):
            if r.get("tier") in ("core", "bench"):
                r["tier"] = "core" if idx < 8 else "bench"

    return filtered


def main():
    rows, latest_date = compute_singles_signal_rows()
    print(f"Data segnale: {latest_date} | {len(rows)} carte nel quantile BUY fresche "
          f"(<= {_signal_freshness_months(PRODUCTION_PARAMS)} mesi, fattore scarsita')\n")
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
