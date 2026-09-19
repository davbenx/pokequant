import pytest
import pandas as pd
from poke_quant.data.europe_market_calibrator import (
    calibrate_price_matrix_for_europe,
    get_market_price_matrix
)
from poke_quant.data.storage import load_price_matrix


def test_europe_market_matrix():
    df_eu = get_market_price_matrix("europe_cardmarket")
    assert df_eu is not None
    assert not df_eu.empty
    assert "stellar_crown_bb" in df_eu.columns
    assert "temporal_forces_bb" in df_eu.columns
    assert "jp_vstar_universe_bb" in df_eu.columns
    assert "op06_wings_captain_bb" in df_eu.columns

    df_us = get_market_price_matrix("us_global")
    assert df_us is not None
    assert not df_us.empty

    # Verifica che il mercato europeo rifletta i rincari IVA e floor distributivo
    eu_tf = df_eu["temporal_forces_bb"].iloc[-1]
    us_tf = df_us["temporal_forces_bb"].iloc[-1]
    assert eu_tf >= us_tf
