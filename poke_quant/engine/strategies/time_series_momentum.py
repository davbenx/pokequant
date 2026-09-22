"""
poke_quant/engine/strategies/time_series_momentum.py — Time-Series Momentum
(Moskowitz, Ooi & Pedersen, "Time Series Momentum", JFE 2012).

Regola canonica e NON discrezionale: per ciascun asset sealed, indipendentemente
dagli altri, posizione lunga se il rendimento trailing di N mesi è positivo,
flat altrimenti. Nessun filtro di tier, MSRP o finestra d'acquisto discrezionale.

Serve come benchmark "pre-registrato": una regola presa identica dalla letteratura
accademica (la stessa logica di trend-following con cui ApexEngine tratta SPY/GLD/BTC),
applicata senza alcun parametro scelto guardando i dati Pokémon/One Piece.

min_age_months (opzionale, default 0 = nessun filtro): esclude dall'INGRESSO gli
asset più giovani di N mesi da release. Motivazione empirica, non discrezionale:
in Fase 2 (walk-forward per annata) la coorte di asset più recenti ha mostrato
drawdown molto più ampi con strategie che non li escludono (Equal-Weight: -49%
sulla coorte 2023+) rispetto a chi li esclude (Carry/Scarsità: -9.7%). Qui si
testa se la stessa esclusione, applicata SOLO come filtro di ingresso a monte
del segnale di momentum (non come strategia a sé), migliora il profilo di
rischio senza perdere l'edge di TSMOM.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
import pandas as pd
from poke_quant.engine.strategies import Signal
from poke_quant.engine.portfolio import Portfolio


class TimeSeriesMomentumStrategy:
    def __init__(
        self,
        prices_df: pd.DataFrame,
        lookback_months: int = 12,
        max_allocation_pct: float = 0.12,
        item_type_filter: str = "sealed",
        min_age_months: int = 0,
    ):
        self.prices_df = prices_df.sort_index()
        self.lookback_months = lookback_months
        self.max_allocation_pct = max_allocation_pct
        self.item_type_filter = item_type_filter
        self.min_age_months = min_age_months

    def reset(self):
        pass

    def _trailing_return(self, item_id: str, current_date: pd.Timestamp) -> Optional[float]:
        if item_id not in self.prices_df.columns:
            return None
        series = self.prices_df[item_id]
        series = series[series.index <= current_date].dropna()
        series = series[series > 0]
        if len(series) < self.lookback_months + 1:
            return None
        past = float(series.iloc[-(self.lookback_months + 1)])
        now = float(series.iloc[-1])
        if past <= 0:
            return None
        return (now - past) / past

    def generate_signals(
        self,
        current_date: str,
        portfolio: Portfolio,
        market_snapshot: Dict[str, Dict[str, Any]],
    ) -> List[Signal]:
        signals: List[Signal] = []
        cur_dt = pd.to_datetime(current_date)
        total_nav = portfolio.get_total_nav({k: v["current_price"] for k, v in market_snapshot.items()})
        max_item_budget = total_nav * self.max_allocation_pct

        # Uscita: il segnale di momentum diventa negativo o nullo -> liquida tutto.
        for item_id, pos in list(portfolio.positions.items()):
            if item_id not in market_snapshot:
                continue
            mom = self._trailing_return(item_id, cur_dt)
            if mom is not None and mom <= 0:
                signals.append(Signal(
                    action="SELL", item_id=item_id, item_name=pos.item_name,
                    item_type=pos.item_type, quantity=pos.quantity,
                    target_price=market_snapshot[item_id]["current_price"],
                    reason=f"TSMOM: rendimento trailing {self.lookback_months}m negativo ({mom*100:+.1f}%)"
                ))

        # Ingresso: segnale positivo e posizione non già aperta.
        for item_id, info in market_snapshot.items():
            if info.get("type") != self.item_type_filter or item_id in portfolio.positions:
                continue
            cur_price = info["current_price"]
            if cur_price <= 0:
                continue
            if self.min_age_months > 0 and info.get("release_date"):
                rel_dt = pd.to_datetime(info["release_date"])
                age_m = (cur_dt.year - rel_dt.year) * 12 + (cur_dt.month - rel_dt.month)
                if age_m < self.min_age_months:
                    continue
            mom = self._trailing_return(item_id, cur_dt)
            if mom is None or mom <= 0:
                continue

            available_cash = portfolio.cash
            budget = min(available_cash, max_item_budget)
            qty = int(budget // cur_price)
            if qty < 1 and available_cash >= cur_price and cur_price <= total_nav * 0.35:
                qty = 1  # lotto minimo indivisibile per conti piccoli, come nelle altre strategie
            if qty >= 1:
                signals.append(Signal(
                    action="BUY", item_id=item_id, item_name=info.get("name", item_id),
                    item_type=self.item_type_filter, quantity=qty, target_price=cur_price,
                    reason=f"TSMOM: rendimento trailing {self.lookback_months}m positivo (+{mom*100:.1f}%)"
                ))
        return signals
