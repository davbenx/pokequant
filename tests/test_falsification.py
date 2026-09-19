"""
tests/test_falsification.py — Test di regressione per l'audit di falsificazione popperiana.
Verifica che la strategia mantenga Alpha anche dopo l'esclusione degli outlier e con fee severe.
"""

from poke_quant.data.storage import load_price_matrix, load_metadata
from poke_quant.engine.backtester import Backtester
from poke_quant.engine.strategies.optimal_sealed_strategy import OptimalSealedStrategy


def test_falsification_anti_outlier():
    """Verifica che l'Alpha sopravviva alla rimozione di Evolving Skies e Team Up."""
    prices_df = load_price_matrix()
    metadata = load_metadata()

    meta_no_outliers = {k: v for k, v in metadata.items() if k not in ["evolving_skies_bb", "team_up_bb"]}
    cols_no_outliers = [c for c in prices_df.columns if c not in ["evolving_skies_bb", "team_up_bb"]]
    prices_no_outliers = prices_df[cols_no_outliers]

    strat = OptimalSealedStrategy(allowed_tiers=["S", "A"])
    bt = Backtester(strat, prices_no_outliers, meta_no_outliers, initial_cash=10000.0, platform="cardmarket")
    res = bt.run()

    assert res.cagr > 0.15          # CAGR > 15% anche senza outlier
    assert res.alpha_annualized > 0.05 # Alpha > 5% vs S&P 500


def test_falsification_tier_c_underperformance():
    """Verifica che i set Tier C performino nettamente peggio confermando l'importanza del filtro qualitativo."""
    prices_df = load_price_matrix()
    metadata = load_metadata()

    meta_tier_c = {k: v for k, v in metadata.items() if v.get("set_tier") == "C"}
    cols_tier_c = [c for c in prices_df.columns if c in meta_tier_c]
    prices_tier_c = prices_df[cols_tier_c]

    strat = OptimalSealedStrategy(allowed_tiers=["C"])
    bt = Backtester(strat, prices_tier_c, meta_tier_c, initial_cash=10000.0, platform="cardmarket")
    res = bt.run()

    # Tier C non deve generare l'alpha di Tier S/A
    assert res.cagr < 0.15


def test_falsification_cross_tcg_one_piece():
    """Verifica che la strategia mantenga rendimenti positivi anche sul franchise One Piece TCG."""
    prices_df = load_price_matrix()
    metadata = load_metadata()

    meta_op = {k: v for k, v in metadata.items() if v.get("franchise") == "one_piece"}
    cols_op = [c for c in prices_df.columns if c in meta_op]
    prices_op = prices_df[cols_op]

    strat = OptimalSealedStrategy(allowed_tiers=["S", "A"], target_roi=1.0)
    bt = Backtester(strat, prices_op, meta_op, initial_cash=10000.0, platform="cardmarket")
    res = bt.run()

    assert res.cagr > 0.10  # One Piece genera rendimento superiore all'inflazione/benchmark
    assert res.final_nav > 10000.0


def test_falsification_macro_benchmark_correlation():
    """Verifica che la correlazione con S&P 500 reale sia bassa (decorrelazione asset alternativo)."""
    from poke_quant.data.storage import load_macro_matrix
    prices_df = load_price_matrix()
    metadata = load_metadata()
    macro_df = load_macro_matrix()

    if macro_df is not None and "spy" in macro_df:
        strat = OptimalSealedStrategy()
        bt = Backtester(strat, prices_df, metadata, initial_cash=10000.0, benchmark_series=macro_df["spy"])
        res = bt.run()

        assert res.beta < 0.80  # Beta significativamente inferiore a 1
        assert abs(res.correlation_benchmark) < 0.60  # Decorrelazione confermata

