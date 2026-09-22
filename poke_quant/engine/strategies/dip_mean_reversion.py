"""
poke_quant/engine/strategies/dip_mean_reversion.py — Mean-Reversion / Dip-Buying Factor.

Ipotesi opposta al momentum: se il mercato dei collezionabili sotto-reagisce (momentum)
o sovra-reagisce (mean-reversion) al rumore di breve periodo e' una domanda empirica,
non un assunto. Time-Series/Cross-Sectional Momentum hanno gia' mostrato un edge sui
box sigillati - qui si testa la falsificazione diretta sulle singole: a ogni
ribilanciamento, calcola lo z-score del prezzo corrente rispetto a media/std mobile
di lookback_months, e va lungo sul quantile con lo SCONTO PIU' PROFONDO (z minimo),
equamente pesato. Nessuna selezione per rarita' o prezzo attuale - solo la deviazione
statistica dalla propria media storica decide.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
import pandas as pd
from poke_quant.engine.strategies import Signal
from poke_quant.engine.portfolio import Portfolio


class DipMeanReversionStrategy:
    def __init__(
        self,
        prices_df: pd.DataFrame,
        lookback_months: int = 12,
        bottom_quantile: float = 0.20,
        rebalance_every_months: int = 3,
        max_allocation_pct: float = 0.15,
        min_age_months: int = 6,
        item_type_filter: str = "single",
    ):
        self.prices_df = prices_df.sort_index()
        self.lookback_months = lookback_months
        self.bottom_quantile = bottom_quantile
        self.rebalance_every_months = rebalance_every_months
        self.max_allocation_pct = max_allocation_pct
        self.min_age_months = min_age_months
        self.item_type_filter = item_type_filter
        self._call_count = 0

    def reset(self):
        self._call_count = 0

    def _zscore(self, item_id: str, current_date: pd.Timestamp) -> Optional[float]:
        if item_id not in self.prices_df.columns:
            return None
        series = self.prices_df[item_id]
        series = series[series.index <= current_date].dropna()
        series = series[series > 0]
        if len(series) < self.lookback_months + 1:
            return None
        window = series.iloc[-(self.lookback_months + 1):-1]
        mean_p = float(window.mean())
        std_p = float(window.std())
        if std_p <= 1e-9 or mean_p <= 0:
            return None
        current_price = float(series.iloc[-1])
        return (current_price - mean_p) / std_p

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

        ranked = []
        for item_id, info in market_snapshot.items():
            if info.get("type") != self.item_type_filter or info.get("current_price", 0) <= 0:
                continue
            if self.min_age_months > 0 and info.get("release_date"):
                rel_dt = pd.to_datetime(info["release_date"])
                age_m = (cur_dt.year - rel_dt.year) * 12 + (cur_dt.month - rel_dt.month)
                if age_m < self.min_age_months:
                    continue
            z = self._zscore(item_id, cur_dt)
            if z is not None:
                ranked.append((item_id, z))
        if not ranked:
            return signals

        # Ordine ascendente per z-score (sconto piu' profondo prima); pareggi rotti
        # per item_id - lista, non set, per un'esecuzione deterministica a corto di cassa.
        ranked.sort(key=lambda x: (x[1], x[0]))
        n_bottom = max(1, int(round(len(ranked) * self.bottom_quantile)))
        bottom_ranked = ranked[:n_bottom]
        bottom_ids = {item_id for item_id, _ in bottom_ranked}

        for item_id, pos in list(portfolio.positions.items()):
            if item_id in market_snapshot and item_id not in bottom_ids:
                signals.append(Signal(
                    action="SELL", item_id=item_id, item_name=pos.item_name,
                    item_type=pos.item_type, quantity=pos.quantity,
                    target_price=market_snapshot[item_id]["current_price"],
                    reason="Dip Mean-Reversion: risalito fuori dal quantile di sconto"
                ))

        total_nav = portfolio.get_total_nav({k: v["current_price"] for k, v in market_snapshot.items()})
        target_per_position = total_nav * min(self.max_allocation_pct, 1.0 / max(1, n_bottom))

        for item_id, z in bottom_ranked:
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
                signals.append(Signal(
                    action="BUY", item_id=item_id, item_name=info.get("name", item_id),
                    item_type=self.item_type_filter, quantity=qty, target_price=cur_price,
                    reason=f"Dip Mean-Reversion: z-score {z:+.2f} vs media {self.lookback_months}m"
                ))
        return signals
