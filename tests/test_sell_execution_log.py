"""
tests/test_sell_execution_log.py — Unit test per il registro delle vendite
reali (poke_quant.data.sell_execution_log), aggiunto perche' nessun backtest
o log di solo acquisto puo' sostituire un dato reale su SE/quando/a che
prezzo un'inserzione trova un compratore - osservazione diretta dell'utente.
"""

import pytest

from poke_quant.data.sell_execution_log import (
    log_new_listing, log_sale_outcome, log_withdrawn, load_sell_log, compute_sell_stats,
)


@pytest.fixture(autouse=True)
def isolate_cache(tmp_path, monkeypatch):
    import poke_quant.data.storage as storage
    monkeypatch.setattr(storage, "CACHE_DIR", tmp_path)
    yield tmp_path


def test_empty_log_returns_no_stats():
    stats = compute_sell_stats()
    assert stats["n_sold"] == 0 and stats["n_withdrawn"] == 0
    assert stats["sell_through_rate"] is None
    assert stats["is_reliable"] is False


def test_new_listing_then_sale_computes_days_on_market():
    log_new_listing("card_a", list_price_eur=100.0, dashboard_price_eur=80.0, list_date="2026-01-01")
    row = log_sale_outcome("card_a", realized_price_eur=90.0, list_date="2026-01-01", sold_date="2026-01-11")
    assert row["status"] == "sold"
    assert row["days_on_market"] == 10
    assert float(row["realized_price_eur"]) == 90.0

    df = load_sell_log()
    assert len(df) == 1
    assert df.iloc[0]["status"] == "sold"


def test_sale_without_prior_listing_creates_zero_day_record():
    """Vendita registrata senza aver loggato prima il listing - non si
    inventa un tempo sul mercato mai osservato, resta a 0 giorni."""
    row = log_sale_outcome("card_b", realized_price_eur=50.0)
    assert row["status"] == "sold"
    assert row["days_on_market"] == 0


def test_withdrawn_listing_counts_toward_sell_through_rate():
    log_new_listing("card_c", list_price_eur=200.0, list_date="2026-01-01")
    log_withdrawn("card_c", list_date="2026-01-01", notes="nessun acquirente in 60gg")

    df = load_sell_log()
    assert df.iloc[0]["status"] == "withdrawn"

    stats = compute_sell_stats(df)
    assert stats["n_withdrawn"] == 1
    assert stats["n_sold"] == 0
    assert stats["sell_through_rate"] == 0.0  # 0 vendute su 1 chiusa


def test_withdrawn_without_open_listing_raises():
    with pytest.raises(ValueError):
        log_withdrawn("card_does_not_exist")


def test_sell_through_rate_mixes_sold_and_withdrawn():
    log_new_listing("card_d", list_price_eur=100.0, list_date="2026-01-01")
    log_sale_outcome("card_d", realized_price_eur=95.0, list_date="2026-01-01", sold_date="2026-01-05")
    log_new_listing("card_e", list_price_eur=100.0, list_date="2026-01-01")
    log_withdrawn("card_e", list_date="2026-01-01")

    stats = compute_sell_stats()
    assert stats["n_sold"] == 1
    assert stats["n_withdrawn"] == 1
    assert stats["sell_through_rate"] == pytest.approx(0.5)


def test_discount_from_list_price_is_negative_when_sold_below_ask():
    log_new_listing("card_f", list_price_eur=100.0, list_date="2026-01-01")
    log_sale_outcome("card_f", realized_price_eur=90.0, list_date="2026-01-01", sold_date="2026-01-05")
    stats = compute_sell_stats()
    assert stats["median_discount_from_list_pct"] == pytest.approx(-10.0)


def test_reliable_only_above_min_observations():
    for i in range(8):
        log_new_listing(f"card_g{i}", list_price_eur=100.0, list_date="2026-01-01")
        log_sale_outcome(f"card_g{i}", realized_price_eur=95.0, list_date="2026-01-01", sold_date="2026-01-05")
    stats = compute_sell_stats()
    assert stats["is_reliable"] is True
