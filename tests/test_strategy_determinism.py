"""
tests/test_strategy_determinism.py — Regressione anti-nondeterminismo.

Bug trovato in questa sessione: CarryScarcityFactorStrategy e CrossSectionalMomentumStrategy
costruivano top_ids come un `set` di stringhe e ci iteravano sopra per emettere le BUY.
L'iterazione di un set di stringhe dipende dall'hash-seed del processo (PYTHONHASHSEED,
randomizzato per default da Python 3.3+): a corto di cassa, quale carta riceve budget
prima cambiava run-to-run pur con codice e input identici - CAGR/Sharpe non riproducibili.
Il fix usa una lista ordinata per rank (pareggi rotti per item_id) per l'iterazione.

Questo test lancia lo stesso backtest in DUE SOTTOPROCESSI SEPARATI (ognuno con il proprio
hash-seed casuale, PYTHONHASHSEED non fissato) e verifica bit-per-bit che il risultato
sia identico - l'unico modo per far riemergere il bug se qualcuno reintroduce un set
nell'ordine di iterazione.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_SNIPPET = """
import sys
sys.path.insert(0, {repo!r})
from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.engine.backtester import Backtester
from poke_quant.engine.strategies.{module} import {cls}

metadata = load_metadata()
prices_full = load_price_matrix({price_file!r})
ids = [
    k for k, v in metadata.items()
    if v.get("type") == "single" and v.get("data_quality") != "thin_unreliable"
    and k in prices_full.columns
][:200]
meta_sub = {{k: v for k, v in metadata.items() if k in ids}}
prices_sub = prices_full[ids]

strat = {ctor}
bt = Backtester(strat, prices_sub, meta_sub, initial_cash=10000.0, platform="cardmarket",
                 apply_liquidity_slippage=True, apply_holding_cost=True)
res = bt.run()
print(f"{{res.cagr!r}}|{{res.sharpe!r}}|{{res.total_trades!r}}")
"""


def _run_in_fresh_process(module: str, cls: str, ctor: str, price_file: str, hashseed: str) -> str:
    code = _SNIPPET.format(repo=str(REPO_ROOT), module=module, cls=cls, ctor=ctor, price_file=price_file)
    env = {"PYTHONHASHSEED": hashseed, "PATH": "/usr/bin:/bin"}
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=str(REPO_ROOT),
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert result.returncode == 0, f"subprocess failed: {result.stderr[-2000:]}"
    return result.stdout.strip().splitlines()[-1]


def test_carry_scarcity_is_reproducible_across_random_hash_seeds():
    out_a = _run_in_fresh_process(
        "carry_scarcity_factor", "CarryScarcityFactorStrategy",
        'CarryScarcityFactorStrategy(top_quantile=0.30, item_type_filter="single")',
        "historical_prices_graded_singles_grade9.csv", "1",
    )
    out_b = _run_in_fresh_process(
        "carry_scarcity_factor", "CarryScarcityFactorStrategy",
        'CarryScarcityFactorStrategy(top_quantile=0.30, item_type_filter="single")',
        "historical_prices_graded_singles_grade9.csv", "12345",
    )
    assert out_a == out_b, (
        "risultato del backtest dipende dall'hash-seed del processo - "
        "reintrodotto un set non ordinato nell'iterazione dei segnali"
    )


def test_cross_sectional_momentum_is_reproducible_across_random_hash_seeds():
    out_a = _run_in_fresh_process(
        "cross_sectional_momentum", "CrossSectionalMomentumStrategy",
        'CrossSectionalMomentumStrategy(prices_sub, item_type_filter="single")',
        "historical_prices_graded_singles_grade9.csv", "1",
    )
    out_b = _run_in_fresh_process(
        "cross_sectional_momentum", "CrossSectionalMomentumStrategy",
        'CrossSectionalMomentumStrategy(prices_sub, item_type_filter="single")',
        "historical_prices_graded_singles_grade9.csv", "12345",
    )
    assert out_a == out_b, (
        "risultato del backtest dipende dall'hash-seed del processo - "
        "reintrodotto un set non ordinato nell'iterazione dei segnali"
    )
