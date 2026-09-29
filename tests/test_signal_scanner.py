"""
tests/test_signal_scanner.py — Test per lo scanner automatico di segnali di mercato e portafoglio.
"""

import datetime
from poke_quant.signal_scanner import scan_signals, format_telegram_alert


def test_scan_signals_structure():
    sample_prices = {
        "evolving_skies_bb": 1500.0,
        "lost_origin_bb": 650.0,
        "fusion_strike_bb": 800.0,
        "scarlet_violet_151_bundle": 120.0
    }
    sample_meta = {
        "lost_origin_bb": {
            "name": "Lost Origin Booster Box",
            "type": "sealed",
            "product_type": "booster_box",
            "set_tier": "A",
            "release_date": "2022-09-09",
            "msrp": 140.0
        },
        "new_test_bb": {
            "name": "Test New Booster Box",
            "type": "sealed",
            "product_type": "booster_box",
            "set_tier": "S",
            "release_date": "2026-03-01",  # ~6 mesi fa
            "msrp": 140.0
        }
    }
    sample_prices["new_test_bb"] = 125.0  # Sotto MSRP a 6 mesi -> BUY!

    res = scan_signals(
        current_prices=sample_prices,
        metadata=sample_meta,
        today_dt=datetime.date(2026, 9, 1)
    )

    assert "buy_signals" in res
    assert "sell_signals" in res
    assert "watchlist" in res
    assert len(res["buy_signals"]) >= 1
    assert res["buy_signals"][0]["item_id"] == "new_test_bb"

    # Test formattazione Telegram
    msg = format_telegram_alert(res)
    assert "*POKEQUANT · SEGNALI DI MERCATO*" in msg
    assert "Test New Booster Box" in msg


def test_scan_signals_excludes_rejected_franchises_and_languages():
    """BUG TROVATO (Fase 4, 2026-09-29): questo scanner alimenta un alert Telegram
    settimanale REALE (poke_signals.yml) e non applicava mai le esclusioni di
    produzione (Magic, Pokemon Cinese per franchise; Pokemon JP per lingua) - poteva
    mandare un "COMPRA SUBITO" su un franchise che la dashboard ha rigettato."""
    sample_prices = {"magic_test_bb": 100.0, "jp_test_bb": 100.0, "en_test_bb": 100.0}
    sample_meta = {
        "magic_test_bb": {
            "name": "Magic Test Box", "type": "sealed", "product_type": "booster_box",
            "franchise": "magic", "set_tier": "S", "release_date": "2026-03-01", "msrp": 140.0,
        },
        "jp_test_bb": {
            "name": "JP Test Box", "type": "sealed", "product_type": "booster_box",
            "franchise": "pokemon", "language": "jp", "set_tier": "S",
            "release_date": "2026-03-01", "msrp": 140.0,
        },
        "en_test_bb": {
            "name": "EN Test Box", "type": "sealed", "product_type": "booster_box",
            "franchise": "pokemon", "language": "en", "set_tier": "S",
            "release_date": "2026-03-01", "msrp": 140.0,
        },
    }
    res = scan_signals(current_prices=sample_prices, metadata=sample_meta,
                        today_dt=datetime.date(2026, 9, 1))
    buy_ids = {b["item_id"] for b in res["buy_signals"]}
    assert "magic_test_bb" not in buy_ids
    assert "jp_test_bb" not in buy_ids
    assert "en_test_bb" in buy_ids
