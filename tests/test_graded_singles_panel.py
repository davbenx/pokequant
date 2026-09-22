"""
tests/test_graded_singles_panel.py — Smoke test per il fetch e l'uso del pannello
"già gradate" (tier Grade 9, dato reale PriceCharting). Non fa richieste di rete:
verifica solo che, dato un pannello già salvato in data_cache/, il motore di
backtest lo consumi correttamente con il filtro item_type_filter="single".
"""

import pandas as pd
import pytest

from poke_quant.data.storage import load_price_matrix, load_metadata
from poke_quant.engine.backtester import Backtester
from poke_quant.engine.strategies.equal_weight_benchmark import EqualWeightBenchmarkStrategy

GRADE9_FILENAME = "historical_prices_graded_singles_grade9.csv"


def test_grade9_panel_exists_and_is_nonempty():
    grade9_df = load_price_matrix(GRADE9_FILENAME)
    assert grade9_df is not None
    assert len(grade9_df) > 12
    assert grade9_df.shape[1] > 0


def test_backtester_runs_on_grade9_panel_with_single_filter():
    grade9_df = load_price_matrix(GRADE9_FILENAME)
    metadata_full = load_metadata()
    singles_meta = {k: v for k, v in metadata_full.items() if v.get("type") == "single" and k in grade9_df.columns}
    assert len(singles_meta) > 0

    strat = EqualWeightBenchmarkStrategy(item_type_filter="single")
    bt = Backtester(strategy=strat, historical_prices_df=grade9_df, items_metadata=singles_meta, initial_cash=10000.0)
    res = bt.run()
    assert res.final_nav > 0
    assert res.total_trades == 0  # buy & hold puro, nessuna vendita per costruzione
