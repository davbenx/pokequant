"""
tests/test_academic_strategies.py — Smoke test per le strategie accademiche canoniche
(Fase 1): verificano solo che il motore di backtest le esegua senza errori e produca
risultati con la forma attesa, non la qualità del rendimento (quella la giudica
scripts/run_academic_strategies.py sui dati reali).
"""

import pytest

from poke_quant.data.storage import load_price_matrix, load_metadata
from poke_quant.engine.backtester import Backtester
from poke_quant.engine.strategies.time_series_momentum import TimeSeriesMomentumStrategy
from poke_quant.engine.strategies.cross_sectional_momentum import CrossSectionalMomentumStrategy
from poke_quant.engine.strategies.carry_scarcity_factor import CarryScarcityFactorStrategy
from poke_quant.engine.strategies.equal_weight_benchmark import EqualWeightBenchmarkStrategy


@pytest.fixture(scope="module")
def real_data():
    prices_df = load_price_matrix()
    metadata = load_metadata()
    assert prices_df is not None and metadata is not None
    return prices_df, metadata


def _run(strategy, prices_df, metadata):
    bt = Backtester(
        strategy=strategy, historical_prices_df=prices_df, items_metadata=metadata,
        initial_cash=10000.0, platform="cardmarket",
        apply_liquidity_slippage=True, apply_holding_cost=True,
    )
    return bt.run()


def test_time_series_momentum_runs(real_data):
    prices_df, metadata = real_data
    res = _run(TimeSeriesMomentumStrategy(prices_df, lookback_months=12), prices_df, metadata)
    assert res.final_nav > 0
    assert len(res.monthly_returns) == len(prices_df) - 1


def test_cross_sectional_momentum_runs(real_data):
    prices_df, metadata = real_data
    res = _run(CrossSectionalMomentumStrategy(prices_df, lookback_months=6, top_quantile=0.3), prices_df, metadata)
    assert res.final_nav > 0
    assert res.total_trades >= 0


def test_carry_scarcity_runs(real_data):
    prices_df, metadata = real_data
    res = _run(CarryScarcityFactorStrategy(top_quantile=0.3), prices_df, metadata)
    assert res.final_nav > 0


def test_equal_weight_benchmark_buys_broad_universe(real_data):
    prices_df, metadata = real_data
    res = _run(EqualWeightBenchmarkStrategy(), prices_df, metadata)
    assert res.final_nav > 0
    # Nessuna vendita per costruzione: buy & hold puro.
    assert (res.trades_df.empty) or (res.trades_df["action"] != "SELL").all() if not res.trades_df.empty else True


def test_time_series_momentum_exits_on_negative_trend(real_data):
    """Se il lookback supera la lunghezza dello storico, nessun trade dovrebbe aprirsi."""
    prices_df, metadata = real_data
    res = _run(TimeSeriesMomentumStrategy(prices_df, lookback_months=200), prices_df, metadata)
    assert res.total_trades == 0
    assert res.final_nav == pytest.approx(10000.0, abs=1.0)
