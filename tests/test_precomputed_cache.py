"""
tests/test_precomputed_cache.py — Unit test per il caricamento della cache precomputata
della dashboard di PokeQuant e per la classe PrecomputedBacktestResult.
"""

from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import pytest

import app
from app import (
    PrecomputedBacktestResult,
    load_precomputed_dashboard_data,
    load_persistent_image_cache,
    annualized_turnover,
    get_signal,
    get_singles_signal,
    get_singles_avoid_signal,
    get_singles_alternatives,
    get_market_indices,
    get_backtest_results,
    get_singles_backtest_results,
    get_box_singles_split,
    get_product_image,
    PRECOMPUTED_FILE,
    IMAGE_CACHE_FILE,
)
from poke_quant.engine.backtester import BacktestResult


@pytest.fixture(autouse=True)
def reset_streamlit_caches():
    """Pulisce le cache streamlit prima e dopo ogni test per isolamento perfetto."""
    try:
        app.load_precomputed_dashboard_data.clear()
        app.get_signal.clear()
        app.get_singles_signal.clear()
        app.get_singles_avoid_signal.clear()
        app.get_singles_alternatives.clear()
        app.get_market_indices.clear()
        app.get_backtest_results.clear()
        app.get_singles_backtest_results.clear()
        app.get_box_singles_split.clear()
    except Exception:
        pass
    yield
    try:
        app.load_precomputed_dashboard_data.clear()
        app.get_signal.clear()
        app.get_singles_signal.clear()
        app.get_singles_avoid_signal.clear()
        app.get_singles_alternatives.clear()
        app.get_market_indices.clear()
        app.get_backtest_results.clear()
        app.get_singles_backtest_results.clear()
        app.get_box_singles_split.clear()
    except Exception:
        pass


def test_precomputed_backtest_result_wrapper():
    sample_data = {
        "strategy_name": "TestStrategy",
        "total_trades": 10,
        "win_rate": 0.60,
        "profit_factor": 1.75,
        "sharpe": 1.45,
        "cagr": 0.22,
        "max_drawdown": -0.15,
        "monthly_returns": {
            "2023-01-01": 0.02,
            "2023-02-01": -0.01,
            "2023-03-01": 0.05,
        },
        "nav_history": [
            {"date": "2023-01-01", "nav": 10200.0, "cash": 5000.0, "portfolio_value": 5200.0},
            {"date": "2023-02-01", "nav": 10100.0, "cash": 4000.0, "portfolio_value": 6100.0},
        ],
        "trades": [
            {
                "item_id": "test_box",
                "item_name": "Test Box",
                "buy_date": "2023-01-01",
                "sell_date": "2023-06-01",
                "holding_months": 5.0,
                "buy_price_unit": 100.0,
                "sell_price_unit": 150.0,
                "quantity": 2,
                "net_roi": 0.45,
                "net_pnl": 90.0,
            }
        ],
    }

    res = PrecomputedBacktestResult(sample_data)
    assert res.strategy_name == "TestStrategy"
    assert res.total_trades == 10
    assert res.win_rate == 0.60
    assert res.profit_factor == 1.75
    assert res.sharpe == 1.45
    assert res.cagr == 0.22
    assert res.max_drawdown == -0.15

    # monthly_returns verification
    assert isinstance(res.monthly_returns, pd.Series)
    assert len(res.monthly_returns) == 3
    assert pd.to_datetime("2023-01-01") in res.monthly_returns.index

    # nav_history verification
    assert isinstance(res.nav_history, pd.DataFrame)
    assert len(res.nav_history) == 2
    assert "nav" in res.nav_history.columns
    assert res.nav_history.loc[pd.to_datetime("2023-01-01"), "nav"] == 10200.0

    # trades_df verification & annualized_turnover
    assert isinstance(res.trades_df, pd.DataFrame)
    assert len(res.trades_df) == 1
    assert pd.api.types.is_datetime64_any_dtype(res.trades_df["buy_date"])
    assert pd.api.types.is_datetime64_any_dtype(res.trades_df["sell_date"])

    trades_yr, eur_yr = annualized_turnover(res.trades_df)
    assert trades_yr > 0
    assert eur_yr > 0


def test_load_precomputed_data_exists_and_valid():
    assert PRECOMPUTED_FILE.exists(), f"File non trovato: {PRECOMPUTED_FILE}"
    data = load_precomputed_dashboard_data()
    assert data is not None
    assert "box_signals" in data
    assert "singles_signals" in data
    assert "market_indices" in data
    assert "backtest_results" in data
    assert "risk_parity" in data
    assert len(data["box_signals"]) > 0


def test_load_precomputed_data_fallback_when_missing(monkeypatch, tmp_path):
    fake_path = tmp_path / "non_existent.json"
    monkeypatch.setattr(app, "PRECOMPUTED_FILE", fake_path)

    app.load_precomputed_dashboard_data.clear()
    res = app.load_precomputed_dashboard_data()
    assert res is None


def test_app_get_signal_with_cache():
    signals, latest_date = get_signal()
    assert isinstance(signals, list)
    assert len(signals) > 0
    assert isinstance(latest_date, str)
    assert "item_id" in signals[0]
    assert "name" in signals[0]


def test_app_get_singles_signal_with_cache():
    buy_rows, latest_date = get_singles_signal("production")
    assert isinstance(buy_rows, list)
    assert len(buy_rows) > 0
    assert "item_id" in buy_rows[0]
    assert "current_price_eur" in buy_rows[0]

    alt_rows, _ = get_singles_alternatives("production")
    assert isinstance(alt_rows, list)

    avoid_rows, _ = get_singles_avoid_signal("production")
    assert isinstance(avoid_rows, list)


def test_app_get_market_indices_with_cache():
    overall, segments, counts, breadth = get_market_indices()
    assert isinstance(overall, pd.Series)
    assert len(overall) > 0
    assert isinstance(segments, dict)
    assert len(segments) >= 2
    assert isinstance(counts, dict)
    assert isinstance(breadth, pd.Series)
    assert len(breadth) > 0


def test_app_get_backtest_results_with_cache():
    res_box, n_box = get_backtest_results()
    assert isinstance(res_box, (PrecomputedBacktestResult, BacktestResult))
    assert res_box.total_trades > 0
    assert res_box.sharpe > 0
    assert n_box > 0

    res_singles, n_s = get_singles_backtest_results("production")
    assert isinstance(res_singles, (PrecomputedBacktestResult, BacktestResult))
    assert res_singles.total_trades > 0
    assert res_singles.sharpe > 0
    assert n_s > 0


def test_app_get_box_singles_split_with_cache():
    w_box, w_singles = get_box_singles_split()
    assert round(w_box + w_singles, 2) == 1.00
    assert 0.40 <= w_box <= 0.65
    assert 0.35 <= w_singles <= 0.60


def test_persistent_image_cache():
    cache = load_persistent_image_cache()
    assert isinstance(cache, dict)
    if IMAGE_CACHE_FILE.exists():
        assert len(cache) > 0
