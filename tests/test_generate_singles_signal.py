"""
tests/test_generate_singles_signal.py — Copertura per la logica di freschezza
del segnale live sulle singole (scripts/generate_singles_signal.py): una carta
va mostrata come BUY solo se e' entrata nel quantile scarsita' negli ultimi
N mesi consecutivi (N = rebalance_every_months della config usata - la cadenza
del backtest validato, 3 in produzione) - oltre, non e' un ingresso fresco ma
un possibile value trap, ed e' esclusa dalla lista mostrata all'utente.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_singles_signal import _signal_streak, PRODUCTION_PARAMS, compute_singles_avoid_rows

SIGNAL_FRESHNESS_MONTHS = PRODUCTION_PARAMS["rebalance_every_months"]  # 3, la cadenza di produzione


def _dates(n: int):
    return list(pd.date_range("2026-01-01", periods=n, freq="MS"))


def test_fresh_entry_this_month_has_streak_one():
    dates = _dates(4)
    membership = {dates[0]: set(), dates[1]: set(), dates[2]: set(), dates[3]: {"card_x"}}
    streak, start = _signal_streak("card_x", membership, dates)
    assert streak == 1
    assert start == dates[3]


def test_entry_exactly_at_freshness_threshold_is_kept():
    dates = _dates(4)
    membership = {dates[0]: set(), dates[1]: {"card_x"}, dates[2]: {"card_x"}, dates[3]: {"card_x"}}
    streak, start = _signal_streak("card_x", membership, dates)
    assert streak == SIGNAL_FRESHNESS_MONTHS
    assert start == dates[1]


def test_entry_older_than_threshold_is_flagged_as_late():
    dates = _dates(4)
    membership = {dates[0]: {"card_x"}, dates[1]: {"card_x"}, dates[2]: {"card_x"}, dates[3]: {"card_x"}}
    streak, _ = _signal_streak("card_x", membership, dates)
    assert streak > SIGNAL_FRESHNESS_MONTHS


def test_gap_then_reentry_counts_as_a_fresh_streak_not_the_old_one():
    dates = _dates(5)
    # dentro al mese 0, fuori al mese 1-2, di nuovo dentro al mese 3-4 - deve
    # contare solo lo streak recente (2), non il totale storico (3).
    membership = {dates[0]: {"card_x"}, dates[1]: set(), dates[2]: set(),
                  dates[3]: {"card_x"}, dates[4]: {"card_x"}}
    streak, start = _signal_streak("card_x", membership, dates)
    assert streak == 2
    assert start == dates[3]


def test_missing_month_data_breaks_the_streak_conservatively():
    dates = _dates(4)
    # card_x non ha dati validi al mese 2 (assente dal dict di quel mese) -
    # non possiamo provare continuita', quindi lo streak si interrompe li'.
    membership = {dates[0]: {"card_x"}, dates[1]: {"card_x"}, dates[3]: {"card_x"}}
    streak, start = _signal_streak("card_x", membership, dates)
    assert streak == 1
    assert start == dates[3]


def test_avoid_rows_puts_the_most_overpriced_card_first():
    """compute_singles_avoid_rows e' lo specchio del quantile BUY: la carta con
    il residuo piu' POSITIVO (sovrapprezzata vs pari per rarita'/eta') deve
    comparire prima."""
    dates = pd.date_range("2024-01-01", periods=1, freq="MS")
    ids = [f"card_{i}" for i in range(25)]
    prices = pd.DataFrame({i: [100.0] for i in ids}, index=dates)
    prices["card_0"] = 500.0  # molto piu' costosa delle sue pari, stessa rarita'/eta'
    metadata = {
        i: {"name": i, "type": "single", "release_date": "2019-01-01", "rarity": "Rare Holo",
            "franchise": "pokemon", "language": "en", "selection_method": "random_control"}
        for i in ids
    }
    with patch("scripts.generate_singles_signal.load_metadata", return_value=metadata), \
         patch("scripts.generate_singles_signal.load_price_matrix", return_value=prices):
        rows, latest_date = compute_singles_avoid_rows()
    assert rows[0]["item_id"] == "card_0"
    assert rows[0]["residual"] > 0
    assert latest_date == dates[-1]
