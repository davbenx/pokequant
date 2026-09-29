"""
tests/test_signal_scanner.py — Test per lo scanner segnali di mercato e
portafoglio. Riconciliato (2026-09-29, richiesta esplicita dell'utente
"Riconcilia signal_scanner.py con la strategia validata"): scan_signals()
non ricalcola piu' una strategia indipendente (finestra eta'/MSRP) - usa le
stesse righe di compute_signal_rows()/compute_singles_signal_rows()/
compute_singles_avoid_rows() gia' validate e mostrate dalla dashboard e
dall'orchestratore mensile. Qui iniettate esplicitamente (box_rows/
singles_buy_rows/singles_avoid_rows) per un test deterministico, non
ricalcolate dal disco a ogni run di pytest.
"""

import datetime
from poke_quant.signal_scanner import scan_signals, format_telegram_alert


def _fake_box_rows():
    return [
        {"item_id": "test_bb", "name": "Test Booster Box", "current_price_eur": 100.0,
         "trailing_12m_return_pct": 25.0, "max_price_eur": 2160.0, "signal": "BUY/HOLD",
         "franchise": "pokemon", "language": "en", "tier": "core"},
        {"item_id": "reversed_bb", "name": "Reversed Box", "current_price_eur": 80.0,
         "trailing_12m_return_pct": -10.0, "max_price_eur": None, "signal": "AVOID/SELL",
         "franchise": "pokemon", "language": "en", "tier": None},
        {"item_id": "excessive_bb", "name": "Excessive Box", "current_price_eur": 5000.0,
         "trailing_12m_return_pct": 40.0, "max_price_eur": 2160.0,
         "signal": "PREZZO ECCESSIVO (oltre tetto MSRP)", "franchise": "pokemon", "language": "en", "tier": None},
    ]


def _fake_singles_buy_rows():
    return [
        {"item_id": "test_card", "name": "Test Card", "set_name": "Test Set",
         "current_price_eur": 50.0, "discount_pct": -20.0, "target_grade": "PSA 9"},
    ]


def _fake_singles_avoid_rows():
    return [
        {"item_id": "overpriced_card", "name": "Overpriced Card", "current_price_eur": 200.0,
         "discount_pct": 300.0},
    ]


def test_scan_signals_uses_validated_engine_rows():
    res = scan_signals(
        current_prices={},
        today_dt=datetime.date(2026, 9, 1),
        box_rows=_fake_box_rows(),
        singles_buy_rows=_fake_singles_buy_rows(),
        singles_avoid_rows=_fake_singles_avoid_rows(),
    )
    assert [r["item_id"] for r in res["box_buy_signals"]] == ["test_bb"]
    assert [r["item_id"] for r in res["box_reversal_signals"]] == ["reversed_bb"]
    assert [r["item_id"] for r in res["box_watchlist"]] == ["excessive_bb"]
    assert res["singles_buy_signals"] == _fake_singles_buy_rows()
    assert res["singles_avoid_signals"] == _fake_singles_avoid_rows()
    assert "holdings_sell_signals" in res
    assert "user_holdings_count" in res


def test_format_telegram_alert_box_and_singles():
    res = scan_signals(
        current_prices={},
        today_dt=datetime.date(2026, 9, 1),
        box_rows=_fake_box_rows(),
        singles_buy_rows=_fake_singles_buy_rows(),
        singles_avoid_rows=_fake_singles_avoid_rows(),
    )
    msg = format_telegram_alert(res)
    assert "*POKEQUANT · SEGNALI DI MERCATO*" in msg
    assert "Test Booster Box" in msg
    assert "Test Card" in msg
    assert "Overpriced Card" in msg
    # AVOID/SELL (momentum invertito) e PREZZO ECCESSIVO non sono un BUY -
    # non devono comparire come raccomandazione d'acquisto nel messaggio.
    assert "Reversed Box" not in msg
    assert "Excessive Box" not in msg


def test_format_telegram_alert_empty_when_no_signals():
    res = scan_signals(
        current_prices={},
        today_dt=datetime.date(2026, 9, 1),
        box_rows=[],
        singles_buy_rows=[],
        singles_avoid_rows=[],
    )
    msg = format_telegram_alert(res)
    assert "Nessun segnale operativo attivo oggi." in msg
