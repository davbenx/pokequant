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


def test_dynamic_capital_rotation():
    """Verifica che la rotazione dinamica del capitale (Tranche 1) sblocchi liquidità e generi segnali continui."""
    from poke_quant.engine.strategies.optimal_sealed_strategy import OptimalSealedStrategy
    prices_df = load_price_matrix()
    metadata = load_metadata()

    strat = OptimalSealedStrategy(
        allowed_tiers=["S", "A", "B"],
        enable_dynamic_rotation=True,
        tranche1_roi=0.70,
        tranche1_min_hold_months=18,
        max_allocation_pct=0.12
    )

    bt = Backtester(
        strategy=strat,
        historical_prices_df=prices_df,
        items_metadata=metadata,
        initial_cash=10000.0,
        platform="cardmarket"
    )

    res = bt.run()
    assert res.final_nav > 30000.0
    assert res.cagr > 0.25
    assert res.rotation_trades_count >= 10
    assert len(res.signals_history) > 40
    assert res.turnover_ratio > 1.0


def test_capital_scaling_and_discrete_sizing():
    """Verifica lo scaling proporzionale del capitale con gestione dei box discreti indivisibili."""
    from poke_quant.engine.strategies.optimal_sealed_strategy import OptimalSealedStrategy
    prices_df = load_price_matrix()
    metadata = load_metadata()

    # 1. Piccolo capitale: 1.000 € (deve comprare lotti minimi di 1 box intero)
    strat_small = OptimalSealedStrategy(allowed_tiers=["S", "A", "B"], enable_dynamic_rotation=True, max_allocation_pct=0.12)
    bt_small = Backtester(strategy=strat_small, historical_prices_df=prices_df, items_metadata=metadata, initial_cash=1000.0)
    res_small = bt_small.run()
    assert res_small.final_nav > 1000.0
    assert len(res_small.open_positions) >= 3
    for p in res_small.open_positions:
        assert isinstance(p["quantity"], int)
        assert p["quantity"] >= 1

    # 2. Grande capitale: 50.000 € (deve scalare le quantità proporzionalmente)
    strat_large = OptimalSealedStrategy(allowed_tiers=["S", "A", "B"], enable_dynamic_rotation=True, max_allocation_pct=0.12)
    bt_large = Backtester(strategy=strat_large, historical_prices_df=prices_df, items_metadata=metadata, initial_cash=50000.0)
    res_large = bt_large.run()
    assert res_large.final_nav > 150000.0
    # Il capitale grande deve detenere molte più unità per set
    small_units = sum(p["quantity"] for p in res_small.open_positions)
    large_units = sum(p["quantity"] for p in res_large.open_positions)
    assert large_units > small_units * 10



