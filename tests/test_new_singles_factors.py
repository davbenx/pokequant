"""
tests/test_new_singles_factors.py — Copertura per i due nuovi fattori sulle singole
(DipMeanReversionStrategy, RarityTierFactorStrategy), aggiunti per la ricerca
sistematica di un segnale alternativo a Carry/Scarsita' (fallita in validazione).
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.engine.portfolio import Portfolio
from poke_quant.engine.strategies.dip_mean_reversion import DipMeanReversionStrategy
from poke_quant.engine.strategies.rarity_tier_factor import RarityTierFactorStrategy, PREMIUM_RARITIES


def _snapshot(prices: dict, meta: dict) -> dict:
    snap = {}
    for item_id, px in prices.items():
        info = dict(meta[item_id])
        info["current_price"] = px
        snap[item_id] = info
    return snap


def test_dip_mean_reversion_buys_the_deepest_discount_not_the_priciest():
    dates = pd.date_range("2021-01-01", periods=13, freq="MS")
    # card_a: oscilla 95-105 (media~100, std>0), poi crolla a 40 nell'ultimo mese -> z molto negativo
    # card_b: oscilla 95-105 nello stesso modo ma resta a 100 nell'ultimo mese -> z~0, nessun dip
    oscillation = [95.0, 105.0, 98.0, 102.0, 96.0, 104.0, 99.0, 101.0, 97.0, 103.0, 100.0, 100.0]
    prices_df = pd.DataFrame({
        "card_a": oscillation + [40.0],
        "card_b": oscillation + [100.0],
    }, index=dates)

    meta = {
        "card_a": {"type": "single", "release_date": "2019-01-01", "rarity": "Common"},
        "card_b": {"type": "single", "release_date": "2019-01-01", "rarity": "Common"},
    }

    strat = DipMeanReversionStrategy(prices_df, lookback_months=12, bottom_quantile=0.60,
                                      rebalance_every_months=1, min_age_months=0)
    portfolio = Portfolio(initial_cash=10000.0)
    snap = _snapshot({"card_a": 40.0, "card_b": 100.0}, meta)
    signals = strat.generate_signals("2022-02-01", portfolio, snap)

    buys = [s for s in signals if s.action == "BUY"]
    assert any(s.item_id == "card_a" for s in buys), "il dip piu' profondo deve generare un BUY"
    # bottom_quantile=0.60 su 2 carte -> n_bottom=1, quindi SOLO card_a (z minimo) deve essere comprata
    assert not any(s.item_id == "card_b" for s in buys), "la carta senza dip non deve entrare nel quantile"


def test_dip_mean_reversion_respects_min_age_months():
    dates = pd.date_range("2021-01-01", periods=14, freq="MS")
    prices_df = pd.DataFrame({"newborn": [100.0] * 13 + [40.0]}, index=dates)
    meta = {"newborn": {"type": "single", "release_date": "2022-01-01", "rarity": "Common"}}

    strat = DipMeanReversionStrategy(prices_df, lookback_months=12, bottom_quantile=1.0,
                                      rebalance_every_months=1, min_age_months=6)
    portfolio = Portfolio(initial_cash=10000.0)
    snap = _snapshot({"newborn": 40.0}, meta)
    # 2022-02-01 rispetto a release 2022-01-01: eta' 1 mese < min_age_months=6 -> escluso
    signals = strat.generate_signals("2022-02-01", portfolio, snap)
    assert signals == [], "una carta troppo giovane non deve generare segnali di acquisto"


def test_dip_mean_reversion_sells_when_price_reverts_out_of_quantile():
    dates = pd.date_range("2021-01-01", periods=14, freq="MS")
    oscillation = [95.0, 105.0, 98.0, 102.0, 96.0, 104.0, 99.0, 101.0, 97.0, 103.0, 100.0, 100.0]
    prices_df = pd.DataFrame({
        "card_a": oscillation + [40.0, 100.0],  # dip poi risale
        "card_b": oscillation + [100.0, 100.0],
    }, index=dates)
    meta = {
        "card_a": {"type": "single", "release_date": "2019-01-01", "rarity": "Common"},
        "card_b": {"type": "single", "release_date": "2019-01-01", "rarity": "Common"},
    }
    strat = DipMeanReversionStrategy(prices_df, lookback_months=12, bottom_quantile=0.60,
                                      rebalance_every_months=1, min_age_months=0)
    portfolio = Portfolio(initial_cash=10000.0)
    from poke_quant.engine.portfolio import Position
    portfolio.positions = {
        "card_a": Position(item_id="card_a", item_name="Card A", item_type="single",
                            quantity=1, buy_date="2022-02-01", buy_price_unit=40.0, total_cost=40.0)
    }
    snap = _snapshot({"card_a": 100.0, "card_b": 100.0}, meta)
    signals = strat.generate_signals("2022-03-01", portfolio, snap)
    sells = [s for s in signals if s.action == "SELL"]
    assert any(s.item_id == "card_a" for s in sells), "una posizione risalita fuori dal quantile va liquidata"


def test_rarity_tier_only_buys_premium_rarity_cards():
    meta = {
        "chase_card": {"type": "single", "release_date": "2019-01-01", "rarity": "Illustration Rare"},
        "common_card": {"type": "single", "release_date": "2019-01-01", "rarity": "Common"},
    }
    strat = RarityTierFactorStrategy(rebalance_every_months=1, min_age_months=0)
    portfolio = Portfolio(initial_cash=10000.0)
    snap = _snapshot({"chase_card": 50.0, "common_card": 5.0}, meta)
    signals = strat.generate_signals("2022-01-01", portfolio, snap)

    buys = {s.item_id for s in signals if s.action == "BUY"}
    assert buys == {"chase_card"}, "solo le rarita' nella whitelist premium devono essere acquistate"


def test_rarity_tier_respects_max_positions_cap():
    meta = {
        f"card_{i}": {"type": "single", "release_date": "2019-01-01", "rarity": "Hyper Rare"}
        for i in range(10)
    }
    strat = RarityTierFactorStrategy(rebalance_every_months=1, min_age_months=0, max_positions=3)
    portfolio = Portfolio(initial_cash=100000.0)
    snap = _snapshot({k: 10.0 for k in meta}, meta)
    signals = strat.generate_signals("2022-01-01", portfolio, snap)
    buys = {s.item_id for s in signals if s.action == "BUY"}
    assert len(buys) == 3, "il tetto max_positions deve limitare il numero di acquisti in un singolo ribilanciamento"


def test_premium_rarities_constant_matches_chase_cards_definition():
    # Sanity check di consistenza: se questo elenco diverge silenziosamente da
    # CHASE_RARITIES in discover_chase_cards.py, il caveat sulla sovrapposizione
    # col campione biased (documentato in optimize_and_falsify.py) diventa falso.
    from scripts.discover_chase_cards import CHASE_RARITIES
    assert PREMIUM_RARITIES == CHASE_RARITIES
