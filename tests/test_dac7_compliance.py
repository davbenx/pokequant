"""
tests/test_dac7_compliance.py — Copertura per annualized_turnover() (app.py):
converte un trades_df del backtest in (vendite/anno, EUR/anno) - la base di
calcolo per la "Modalità conforme DAC7" (soglie UE 2.000€/30 vendite annue,
sopra le quali le piattaforme segnalano il venditore come commerciale).
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import annualized_turnover


def _trades_df(sell_dates, prices, quantities=None):
    quantities = quantities or [1] * len(sell_dates)
    return pd.DataFrame({
        "sell_date": sell_dates, "sell_price_unit": prices, "quantity": quantities,
    })


def test_empty_trades_df_gives_zero():
    trades_per_year, eur_per_year = annualized_turnover(pd.DataFrame(columns=["sell_date", "sell_price_unit", "quantity"]))
    assert trades_per_year == 0.0
    assert eur_per_year == 0.0


def test_exactly_one_year_of_trades_annualizes_to_itself():
    df = _trades_df(["2024-01-01", "2024-07-01", "2025-01-01"], [100.0, 100.0, 100.0])
    trades_per_year, eur_per_year = annualized_turnover(df)
    assert trades_per_year == pytest.approx(3.0, rel=0.01)
    assert eur_per_year == pytest.approx(300.0, rel=0.01)


def test_two_years_of_trades_halves_the_annual_rate():
    df = _trades_df(["2022-01-01", "2024-01-01"], [1000.0, 1000.0])
    trades_per_year, eur_per_year = annualized_turnover(df)
    assert trades_per_year == pytest.approx(1.0, rel=0.02)
    assert eur_per_year == pytest.approx(1000.0, rel=0.02)


def test_quantity_multiplies_into_eur_volume():
    df = _trades_df(["2024-01-01", "2025-01-01"], [100.0, 100.0], quantities=[3, 1])
    trades_per_year, eur_per_year = annualized_turnover(df)
    assert eur_per_year == pytest.approx(400.0, rel=0.02), "3*100 + 1*100 = 400 su 1 anno -> 400/anno"
