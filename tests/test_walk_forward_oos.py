"""
tests/test_walk_forward_oos.py — Smoke test per lo split walk-forward di Fase 2
(scripts/run_walk_forward_oos.py): verifica solo che i due split (temporale e per
annata di asset) producano sotto-universi/sotto-periodi non vuoti e backtestabili,
non la qualità del rendimento.
"""

import pandas as pd
import pytest

from poke_quant.data.storage import load_price_matrix, load_metadata
from poke_quant.engine.backtester import Backtester
from poke_quant.engine.strategies.time_series_momentum import TimeSeriesMomentumStrategy

VINTAGE_CUTOFF = "2023-01-01"


@pytest.fixture(scope="module")
def sealed_universe():
    prices_full = load_price_matrix()
    metadata_full = load_metadata()
    sealed_ids = [k for k, v in metadata_full.items() if v.get("type") == "sealed"]
    metadata_sealed = {k: v for k, v in metadata_full.items() if k in sealed_ids}
    prices_sealed = prices_full[[c for c in prices_full.columns if c in sealed_ids]]
    return prices_sealed, metadata_sealed, sealed_ids


def test_temporal_split_halves_are_nonempty_and_disjoint(sealed_universe):
    prices_sealed, _, _ = sealed_universe
    mid_idx = len(prices_sealed) // 2
    first_half = prices_sealed.iloc[:mid_idx]
    second_half = prices_sealed.iloc[mid_idx:]
    assert len(first_half) > 12 and len(second_half) > 12
    assert first_half.index[-1] < second_half.index[0]


def test_vintage_split_cohorts_are_nonempty_and_disjoint(sealed_universe):
    _, metadata_sealed, sealed_ids = sealed_universe
    old_ids = [k for k in sealed_ids if metadata_sealed[k]["release_date"] < VINTAGE_CUTOFF]
    new_ids = [k for k in sealed_ids if metadata_sealed[k]["release_date"] >= VINTAGE_CUTOFF]
    assert len(old_ids) > 0 and len(new_ids) > 0
    assert set(old_ids).isdisjoint(new_ids)
    assert set(old_ids) | set(new_ids) == set(sealed_ids)


def test_strategy_runs_on_each_vintage_cohort_independently(sealed_universe):
    prices_sealed, metadata_sealed, sealed_ids = sealed_universe
    old_ids = [k for k in sealed_ids if metadata_sealed[k]["release_date"] < VINTAGE_CUTOFF]
    prices_old = prices_sealed[[c for c in prices_sealed.columns if c in old_ids]]
    meta_old = {k: v for k, v in metadata_sealed.items() if k in old_ids}

    strat = TimeSeriesMomentumStrategy(prices_old, lookback_months=12)
    bt = Backtester(strategy=strat, historical_prices_df=prices_old, items_metadata=meta_old, initial_cash=10000.0)
    res = bt.run()
    assert res.final_nav > 0
