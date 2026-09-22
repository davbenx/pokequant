"""
poke_quant/engine/strategies/cross_sectional_momentum.py — Cross-Sectional Momentum
(Jegadeesh & Titman, "Returns to Buying Winners and Selling Losers", JF 1993).

Regola canonica: a ogni ribilanciamento, classifica tutti gli asset sealed eleggibili
per rendimento trailing e va lungo sul quantile superiore, equamente pesato. Nessuna
selezione manuale di "quali set" — solo il rendimento passato relativo decide.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
import pandas as pd
from poke_quant.engine.strategies import Signal
from poke_quant.engine.portfolio import Portfolio


class CrossSectionalMomentumStrategy:
    def __init__(
        self,
        prices_df: pd.DataFrame,
        lookback_months: int = 6,
        top_quantile: float = 0.30,
        rebalance_every_months: int = 3,
        max_allocation_pct: float = 0.15,
        item_type_filter: str = "sealed",
    ):
        self.prices_df = prices_df.sort_index()
        self.lookback_months = lookback_months
        self.top_quantile = top_quantile
        self.rebalance_every_months = rebalance_every_months
        self.max_allocation_pct = max_allocation_pct
        self.item_type_filter = item_type_filter
        self._call_count = 0

    def reset(self):
        self._call_count = 0

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
        is_rebalance_month = (self._call_count % self.rebalance_every_months) == 0
        self._call_count += 1

        if not is_rebalance_month:
            return signals

        eligible = {
            item_id: info for item_id, info in market_snapshot.items()
            if info.get("type") == self.item_type_filter and info.get("current_price", 0) > 0
        }
        ranked = []
        for item_id, info in eligible.items():
            mom = self._trailing_return(item_id, cur_dt)
            if mom is not None:
                ranked.append((item_id, mom))
        if not ranked:
            return signals

        ranked.sort(key=lambda x: (x[1], x[0]), reverse=True)
        n_top = max(1, int(round(len(ranked) * self.top_quantile)))
        # top_ranked (lista, non set) preserva l'ordine di rank per l'iterazione di
        # acquisto sotto - un set di stringhe itera in ordine dipendente dall'hash-seed
        # del processo (PYTHONHASHSEED), rendendo non riproducibile quale carta riceve
        # budget prima che finisca la cassa. Stesso bug trovato e corretto in
        # carry_scarcity_factor.py in questa sessione.
        top_ranked = ranked[:n_top]
        top_ids = {item_id for item_id, _ in top_ranked}

        # Uscita: chi non è più nel quantile top viene liquidato.
        for item_id, pos in list(portfolio.positions.items()):
            if item_id in market_snapshot and item_id not in top_ids:
                signals.append(Signal(
                    action="SELL", item_id=item_id, item_name=pos.item_name,
                    item_type=pos.item_type, quantity=pos.quantity,
                    target_price=market_snapshot[item_id]["current_price"],
                    reason="Cross-Sectional Momentum: uscito dal quantile top"
                ))

        # Ingresso equamente pesato sul quantile top non ancora posseduto.
        total_nav = portfolio.get_total_nav({k: v["current_price"] for k, v in market_snapshot.items()})
        target_per_position = total_nav * min(self.max_allocation_pct, 1.0 / max(1, n_top))

        for item_id, _ in top_ranked:
            if item_id in portfolio.positions:
                continue
            info = market_snapshot[item_id]
            cur_price = info["current_price"]
            available_cash = portfolio.cash
            budget = min(available_cash, target_per_position)
            qty = int(budget // cur_price)
            if qty < 1 and available_cash >= cur_price and cur_price <= total_nav * 0.35:
                qty = 1
            if qty >= 1:
                mom = dict(ranked)[item_id]
                signals.append(Signal(
                    action="BUY", item_id=item_id, item_name=info.get("name", item_id),
                    item_type=self.item_type_filter, quantity=qty, target_price=cur_price,
                    reason=f"Cross-Sectional Momentum: top {int(self.top_quantile*100)}% (rend. {self.lookback_months}m {mom*100:+.1f}%)"
                ))
        return signals
