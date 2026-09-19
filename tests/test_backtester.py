"""
tests/test_backtester.py — Test di integrazione del motore di backtest su dati storici reali.
"""

import pytest
from poke_quant.data.storage import load_price_matrix, load_metadata
from poke_quant.engine.backtester import Backtester
from poke_quant.engine.strategies.sealed_accumulator import SealedAccumulatorStrategy
from poke_quant.engine.strategies.chase_dip_buyer import ChaseDipBuyerStrategy


def test_sealed_accumulator_backtest():
    prices_df = load_price_matrix()
    metadata = load_metadata()
    assert prices_df is not None and not prices_df.empty
    assert metadata is not None

    strat = SealedAccumulatorStrategy(
        max_allocation_per_set_pct=0.30,
        max_buy_age_months=14,
        min_hold_months=18,
        target_profit_roi=0.80
    )

    bt = Backtester(
        strategy=strat,
        historical_prices_df=prices_df,
        items_metadata=metadata,
        initial_cash=10000.0,
        platform="cardmarket",
        benchmark_cagr=0.10
    )

    res = bt.run()
    assert res.final_nav > 0
    assert res.cagr > 0  # I sealed box moderni hanno avuto performance storiche positive
    assert res.max_drawdown <= 0.0
    assert len(res.monthly_returns) > 10
    assert not res.nav_history.empty


def test_chase_dip_buyer_backtest():
    prices_df = load_price_matrix()
    metadata = load_metadata()
    assert prices_df is not None and not prices_df.empty

    strat = ChaseDipBuyerStrategy(
        min_dip_months=4,
        max_dip_months=12,
        min_drop_from_launch_pct=0.10,
        target_profit_roi=0.50
    )

    bt = Backtester(
        strategy=strat,
        historical_prices_df=prices_df,
        items_metadata=metadata,
        initial_cash=5000.0,
        platform="cardmarket",
        benchmark_cagr=0.10
    )

    res = bt.run()
    assert res.final_nav > 0
    assert res.cagr is not None
    assert not res.nav_history.empty


def test_optimal_sealed_strategy_backtest():
    from poke_quant.engine.strategies.optimal_sealed_strategy import OptimalSealedStrategy
    prices_df = load_price_matrix()
    metadata = load_metadata()
    assert prices_df is not None and metadata is not None

    strat = OptimalSealedStrategy(
        allowed_tiers=["S", "A"],
        min_buy_age_months=4,
        max_buy_age_months=14,
        max_msrp_multiplier=1.15,
        min_hold_months=30,
        target_roi=1.50
    )

    bt = Backtester(
        strategy=strat,
        historical_prices_df=prices_df,
        items_metadata=metadata,
        initial_cash=10000.0,
        platform="cardmarket",
        benchmark_cagr=0.10
    )

    res = bt.run()
    assert res.final_nav > 10000.0
    assert res.cagr > 0.15          # CAGR oltre il 15%
    assert res.sharpe > 1.0         # Sharpe superiore a 1.0
    assert res.max_drawdown > -0.15 # Drawdown contenuto entro il -15%
    assert res.win_rate == 1.0      # 100% win rate storico

