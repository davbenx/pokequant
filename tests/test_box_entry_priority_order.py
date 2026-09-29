"""
tests/test_box_entry_priority_order.py — Regressione per il bug di priorita'
d'ordine sugli acquisti box (2026-09-26, riconfermato nell'audit generale del
2026-09-29 - "trova bug... aggiungi test").

TimeSeriesMomentumStrategy.generate_signals() iterava i candidati d'ingresso
in market_snapshot.items() - l'ordine di INSERIMENTO in items_metadata.json
(arbitrario), non per momentum. Portfolio.buy() (poke_quant/engine/portfolio.py)
rifiuta un trade SENZA fill parziale se il costo supera la cassa residua, e
Backtester esegue le compravendite nell'ordine esatto in cui la strategia le
ha restituite - quindi quando la cassa non basta per tutti i segnali dello
stesso mese (confermato: succede in 24/24 mesi storici con 2+ segnali BUY box,
vedi scripts/box_entry_priority_order_test.py), chi viene comprato dipendeva
da quell'ordine arbitrario. Fix: i candidati sono ora ordinati per momentum
trailing DECRESCENTE prima di essere restituiti - la stessa priorita' che
scripts/box_entry_priority_order_test.py misura sul backtest storico completo
(Sharpe 1,14->1,23, DSR 0,646->0,714), qui isolata a livello di singola
chiamata generate_signals(), senza dipendere da dati storici reali.

Prima di questo fix, questo test esisteva SOLO come script standalone -
zero copertura in tests/, quindi nessuna regressione futura sull'ordine del
loop sarebbe stata rilevata da `pytest`.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.engine.portfolio import Portfolio
from poke_quant.engine.strategies.time_series_momentum import TimeSeriesMomentumStrategy


def _make_prices_df(item_momentum: dict) -> pd.DataFrame:
    """Costruisce 13 mesi di prezzi mensili per ogni item, con un rendimento
    trailing 12m pari al valore richiesto (prezzo iniziale 100, finale
    100*(1+mom))."""
    idx = pd.date_range("2024-01-01", periods=13, freq="MS")
    data = {}
    for item_id, mom in item_momentum.items():
        start, end = 100.0, 100.0 * (1.0 + mom)
        data[item_id] = pd.Series(
            [start + (end - start) * i / 12 for i in range(13)], index=idx
        )
    return pd.DataFrame(data)


def test_entry_candidates_sorted_by_momentum_descending_regardless_of_snapshot_order():
    """market_snapshot e' costruito in ordine ALFABETICO (z_low_momentum prima
    di a_high_momentum, deliberatamente il contrario dell'ordine per momentum)
    - se il bug fosse tornato (iterazione in ordine di inserimento invece che
    per momentum), il segnale per z_low_momentum uscirebbe prima."""
    prices_df = _make_prices_df({"z_low_momentum": 0.05, "a_high_momentum": 0.50})
    strat = TimeSeriesMomentumStrategy(prices_df, lookback_months=12)
    portfolio = Portfolio(initial_cash=10000.0)

    market_snapshot = {
        "z_low_momentum": {"type": "sealed", "current_price": 100.0 * 1.05, "name": "Low"},
        "a_high_momentum": {"type": "sealed", "current_price": 100.0 * 1.50, "name": "High"},
    }
    signals = strat.generate_signals("2025-01-01", portfolio, market_snapshot)

    buy_signals = [s for s in signals if s.action == "BUY"]
    assert len(buy_signals) == 2
    assert buy_signals[0].item_id == "a_high_momentum", (
        "il candidato a momentum piu' alto deve essere restituito PRIMO, "
        "indipendentemente dall'ordine di market_snapshot - altrimenti quando la "
        "cassa non basta per tutti (vedi Backtester/Portfolio.buy) verrebbe "
        "comprato per primo il segnale peggiore solo perche' arriva prima nel dict."
    )
    assert buy_signals[1].item_id == "z_low_momentum"


def test_insufficient_cash_fills_highest_momentum_first_not_snapshot_order():
    """Riproduce esattamente il meccanismo del bug storico: cassa insufficiente
    per entrambi i segnali BUY dello stesso mese. Il candidato a momentum piu'
    alto deve risultare comprabile per primo (stesso ordine di signals);
    Portfolio.buy() rifiuta senza fill parziale chi arriva dopo che la cassa
    e' finita - quindi l'ordine dei segnali DETERMINA chi viene davvero comprato."""
    prices_df = _make_prices_df({"z_low_momentum": 0.05, "a_high_momentum": 0.50})
    strat = TimeSeriesMomentumStrategy(prices_df, lookback_months=12, max_allocation_pct=1.0)
    portfolio = Portfolio(initial_cash=150.0)  # basta per un solo box da ~105-150€, non per entrambi

    market_snapshot = {
        "z_low_momentum": {"type": "sealed", "current_price": 100.0 * 1.05, "name": "Low"},
        "a_high_momentum": {"type": "sealed", "current_price": 100.0 * 1.50, "name": "High"},
    }
    signals = strat.generate_signals("2025-01-01", portfolio, market_snapshot)
    buy_signals = [s for s in signals if s.action == "BUY"]
    assert buy_signals[0].item_id == "a_high_momentum"

    # Simula l'esecuzione sequenziale reale del Backtester: compra nell'ordine
    # dei segnali, la cassa residua decide chi viene rifiutato.
    filled = []
    for sig in buy_signals:
        ok = portfolio.buy(item_id=sig.item_id, item_name=sig.item_name, item_type=sig.item_type,
                            quantity=sig.quantity, unit_price=sig.target_price, date="2025-01-01")
        if ok:
            filled.append(sig.item_id)
    assert filled == ["a_high_momentum"], (
        "il segnale a momentum piu' alto deve essere quello davvero comprato "
        "quando la cassa non basta per entrambi - se questo fallisce, l'ordine "
        "di priorita' e' regredito all'ordine arbitrario di inserimento."
    )
