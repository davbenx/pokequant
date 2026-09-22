"""
tests/test_cross_sectional_factor.py — Copertura per il motore fattoriale generalizzato
e le tre funzioni di scoring (momentum skip-month, prossimita' al massimo, bassa
volatilita'), nuove per la ricerca di un segnale sulle singole richiesta dall'utente.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.engine.portfolio import Portfolio
from poke_quant.engine.strategies.cross_sectional_factor import (
    CrossSectionalFactorStrategy, momentum_factor, proximity_to_high_factor, low_volatility_factor,
    zscore_factor
)


def _snapshot(prices: dict, meta: dict) -> dict:
    snap = {}
    for item_id, px in prices.items():
        info = dict(meta[item_id])
        info["current_price"] = px
        snap[item_id] = info
    return snap


def test_momentum_factor_skips_the_most_recent_month():
    dates = pd.date_range("2021-01-01", periods=14, freq="MS")
    # idx0=idx1=100 (ancora 12m-fa per skip=0 e skip=1), idx2..12=200, idx13=90 (crollo ultimo mese)
    values = [100.0, 100.0] + [200.0] * 11 + [90.0]
    s = pd.Series(values, index=dates)
    now_date = dates[-1]
    mom_no_skip = momentum_factor(s, now_date, lookback=12, skip=0)
    mom_skip1 = momentum_factor(s, now_date, lookback=12, skip=1)
    assert mom_no_skip == pytest.approx((90.0 - 100.0) / 100.0), "senza skip il crollo dell'ultimo mese domina"
    assert mom_skip1 == pytest.approx((200.0 - 100.0) / 100.0), (
        "con skip=1 il momentum deve ignorare il crollo dell'ultimo mese e guardare 12m fa -> 1m fa"
    )


def test_proximity_to_high_factor_is_one_at_the_peak():
    dates = pd.date_range("2021-01-01", periods=13, freq="MS")
    values = [100.0] * 11 + [150.0, 150.0]  # massimo raggiunto e mantenuto
    s = pd.Series(values, index=dates)
    prox = proximity_to_high_factor(s, dates[-1], lookback=12, skip=0)
    assert prox == pytest.approx(1.0)


def test_proximity_to_high_factor_below_one_when_off_the_peak():
    dates = pd.date_range("2021-01-01", periods=13, freq="MS")
    values = [100.0] * 10 + [200.0, 150.0, 100.0]  # sceso a meta' dal massimo
    s = pd.Series(values, index=dates)
    prox = proximity_to_high_factor(s, dates[-1], lookback=12, skip=0)
    assert prox == pytest.approx(0.5)


def test_zscore_factor_is_very_negative_on_a_sudden_drop():
    dates = pd.date_range("2021-01-01", periods=14, freq="MS")
    # finestra di 13 punti con oscillazione moderata (media~100, std>0), poi crollo
    values = [95.0, 105.0, 98.0, 102.0, 96.0, 104.0, 99.0, 101.0, 97.0, 103.0, 100.0, 100.0, 100.0, 40.0]
    s = pd.Series(values, index=dates)
    z = zscore_factor(s, dates[-1], lookback=12, skip=0)
    assert z is not None and z < -2.0, "un crollo improvviso deve dare uno z-score molto negativo"


def test_zscore_factor_near_zero_when_stable():
    dates = pd.date_range("2021-01-01", periods=14, freq="MS")
    values = [99.0, 101.0, 100.0, 102.0, 98.0, 101.0, 99.0, 100.0, 101.0, 99.0, 100.0, 101.0, 99.0, 100.0]
    s = pd.Series(values, index=dates)
    z = zscore_factor(s, dates[-1], lookback=12, skip=0)
    assert z is not None and abs(z) < 1.5


def test_low_volatility_factor_ranks_stable_series_below_volatile_one():
    dates = pd.date_range("2021-01-01", periods=14, freq="MS")
    stable = pd.Series([100.0] * 14, index=dates)
    volatile = pd.Series([100, 130, 80, 140, 70, 150, 90, 160, 85, 155, 95, 145, 100, 100.0], index=dates)
    v_stable = low_volatility_factor(stable, dates[-1], lookback=12, skip=0)
    v_volatile = low_volatility_factor(volatile, dates[-1], lookback=12, skip=0)
    assert v_stable < v_volatile


def test_strategy_buys_top_quantile_by_momentum_deterministically():
    dates = pd.date_range("2021-01-01", periods=14, freq="MS")
    prices_df = pd.DataFrame({
        "strong": [100.0] * 13 + [200.0],
        "weak": [100.0] * 13 + [90.0],
    }, index=dates)
    meta = {
        "strong": {"type": "single", "release_date": "2019-01-01"},
        "weak": {"type": "single", "release_date": "2019-01-01"},
    }
    strat = CrossSectionalFactorStrategy(
        prices_df, factor_fn=momentum_factor, lookback_months=12, top_quantile=0.5,
        rebalance_every_months=1, min_age_months=0
    )
    portfolio = Portfolio(initial_cash=10000.0)
    snap = _snapshot({"strong": 200.0, "weak": 90.0}, meta)
    signals = strat.generate_signals("2022-02-01", portfolio, snap)
    buys = {s.item_id for s in signals if s.action == "BUY"}
    assert buys == {"strong"}, "il quantile top per momentum deve selezionare solo la carta col rendimento migliore"


def test_ascending_true_selects_lowest_score_quantile():
    dates = pd.date_range("2021-01-01", periods=14, freq="MS")
    prices_df = pd.DataFrame({
        "low_vol": [100.0] * 14,
        "high_vol": [100, 130, 80, 140, 70, 150, 90, 160, 85, 155, 95, 145, 100, 100.0],
    }, index=dates)
    meta = {
        "low_vol": {"type": "single", "release_date": "2019-01-01"},
        "high_vol": {"type": "single", "release_date": "2019-01-01"},
    }
    strat = CrossSectionalFactorStrategy(
        prices_df, factor_fn=low_volatility_factor, lookback_months=12, top_quantile=0.5,
        ascending=True, rebalance_every_months=1, min_age_months=0
    )
    portfolio = Portfolio(initial_cash=10000.0)
    snap = _snapshot({"low_vol": 100.0, "high_vol": 100.0}, meta)
    signals = strat.generate_signals("2022-02-01", portfolio, snap)
    buys = {s.item_id for s in signals if s.action == "BUY"}
    assert buys == {"low_vol"}, "ascending=True deve comprare la carta a volatilita' piu' bassa"


def test_two_separate_process_runs_give_identical_signals():
    """Regressione anti-nondeterminismo (stesso bug corretto per le altre strategie):
    esegue lo stesso backtest in due sottoprocessi con hash-seed diverso e verifica
    l'identita' bit-a-bit dell'output."""
    import subprocess
    code = """
import sys
sys.path.insert(0, '.')
from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.engine.backtester import Backtester
from poke_quant.engine.strategies.cross_sectional_factor import CrossSectionalFactorStrategy, momentum_factor
metadata = load_metadata()
prices_full = load_price_matrix('historical_prices_graded_singles_grade9.csv')
ids = [k for k, v in metadata.items() if v.get('type') == 'single' and k in prices_full.columns][:150]
meta_sub = {k: v for k, v in metadata.items() if k in ids}
prices_sub = prices_full[ids]
strat = CrossSectionalFactorStrategy(prices_sub, factor_fn=momentum_factor, lookback_months=12, skip_months=1, item_type_filter='single')
bt = Backtester(strat, prices_sub, meta_sub, initial_cash=10000.0, platform='cardmarket', apply_liquidity_slippage=True, apply_holding_cost=True)
res = bt.run()
print(f"{res.cagr!r}|{res.sharpe!r}|{res.total_trades!r}")
"""
    repo_root = str(Path(__file__).resolve().parent.parent)
    outs = []
    for seed in ["1", "99999"]:
        r = subprocess.run([sys.executable, "-c", code], cwd=repo_root,
                            capture_output=True, text=True, env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
                            timeout=120)
        assert r.returncode == 0, r.stderr[-2000:]
        outs.append(r.stdout.strip().splitlines()[-1])
    assert outs[0] == outs[1]
