"""
poke_quant/engine/strategies/topdown_scarcity_factor.py — Selezione Top-Down:
Mercato -> Set -> Singola.

Idea dell'utente: invece di applicare il fattore scarsita' (scarcity_value_factor.py,
validato, DSR 0,943 sull'intera ricerca) su TUTTO l'universo delle singole,
restringere l'eleggibilita' ai soli set il cui BOX ha oggi momentum positivo
(stessa regola della strategia sealed in produzione) - "il value da solo puo'
essere una trappola (a buon mercato per un motivo reale), il momentum del box
confermerebbe che c'e' vera domanda dietro il set".

Meccanismo: ad ogni ribilanciamento, calcola il momentum trailing 12m di ogni
box nell'universo sealed; le singole eleggibili per la regressione di scarsita'
sono solo quelle mappate (single_to_box_map.json) a un box con momentum > 0.
Stessa regressione, stesso quantile, esecuzione sempre sulla singola (non sul
box).

ESITO VALIDAZIONE: NON VALIDATO. Peggiora rispetto al fattore scarsita' non
filtrato (Sharpe 0,63 contro 1,74, DSR 0,200 contro 0,943 corretto per
l'intera ricerca). Restringere l'eleggibilita' ai set con box in momentum
riduce la diversificazione senza aggiungere selettivita' sufficiente a
compensare - vedi scripts/topdown_scarcity_test.py per i numeri completi e
il confronto con l'alternativa che funziona (blend indipendente box+scarsita',
non nidificazione).
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from poke_quant.engine.strategies import Signal
from poke_quant.engine.portfolio import Portfolio
from poke_quant.engine.strategies.scarcity_value_factor import EXPECTED_COPIES_PER_BOX, EXCLUDED_RARITIES


class TopDownScarcityFactorStrategy:
    def __init__(
        self,
        box_prices_df: pd.DataFrame,
        single_to_box: Dict[str, str],
        box_momentum_lookback: int = 12,
        rebalance_every_months: int = 6,
        top_quantile: float = 0.20,
        max_positions: int = 60,
        max_allocation_pct: float = 0.06,
        min_age_months: int = 6,
        item_type_filter: str = "single",
        min_cross_section: int = 15,
    ):
        self.box_prices_df = box_prices_df.sort_index()
        self.single_to_box = single_to_box
        self.box_momentum_lookback = box_momentum_lookback
        self.rebalance_every_months = rebalance_every_months
        self.top_quantile = top_quantile
        self.max_positions = max_positions
        self.max_allocation_pct = max_allocation_pct
        self.min_age_months = min_age_months
        self.item_type_filter = item_type_filter
        self.min_cross_section = min_cross_section
        self._call_count = 0

    def reset(self):
        self._call_count = 0

    def _hot_box_ids(self, cur_dt: pd.Timestamp) -> set:
        hot = set()
        for box_id in self.box_prices_df.columns:
            series = self.box_prices_df[box_id][self.box_prices_df.index <= cur_dt].dropna()
            series = series[series > 0]
            lb = self.box_momentum_lookback
            if len(series) < lb + 1:
                continue
            past, now = float(series.iloc[-(lb + 1)]), float(series.iloc[-1])
            if past > 0 and (now - past) / past > 0:
                hot.add(box_id)
        return hot

    def _fit_residuals(self, cur_dt: pd.Timestamp, market_snapshot: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        hot_boxes = self._hot_box_ids(cur_dt)
        rows, ids = [], []
        for item_id, info in market_snapshot.items():
            if info.get("type") != self.item_type_filter or info.get("current_price", 0) <= 0:
                continue
            if self.single_to_box.get(item_id) not in hot_boxes:
                continue
            rarity = info.get("rarity")
            if rarity in EXCLUDED_RARITIES or rarity not in EXPECTED_COPIES_PER_BOX:
                continue
            rel_dt_raw = info.get("release_date")
            if not rel_dt_raw:
                continue
            rel_dt = pd.to_datetime(rel_dt_raw)
            age_m = (cur_dt.year - rel_dt.year) * 12 + (cur_dt.month - rel_dt.month)
            if age_m < self.min_age_months:
                continue
            scarcity = 1.0 / EXPECTED_COPIES_PER_BOX[rarity]
            is_op = 1.0 if info.get("franchise") == "one_piece" else 0.0
            is_jp = 1.0 if info.get("language") == "jp" else 0.0
            is_chase = 1.0 if info.get("selection_method") == "chase_price_filter_survivorship_biased" else 0.0
            log_price = np.log(info["current_price"])
            rows.append([1.0, np.log(scarcity), is_op, is_jp, float(age_m), is_chase, log_price])
            ids.append(item_id)

        if len(rows) < self.min_cross_section:
            return {}

        arr = np.array(rows)
        X, y = arr[:, :6], arr[:, 6]
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        residual = y - X @ coef
        return dict(zip(ids, residual))

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

        residuals = self._fit_residuals(cur_dt, market_snapshot)
        if not residuals:
            return signals

        n_buy = max(1, int(len(residuals) * self.top_quantile))
        eligible = [item_id for item_id, _ in sorted(residuals.items(), key=lambda x: x[1])[:n_buy]]
        eligible = eligible[: self.max_positions]
        eligible_set = set(eligible)

        for item_id, pos in list(portfolio.positions.items()):
            if item_id in market_snapshot and item_id not in eligible_set:
                signals.append(Signal(
                    action="SELL", item_id=item_id, item_name=pos.item_name,
                    item_type=pos.item_type, quantity=pos.quantity,
                    target_price=market_snapshot[item_id]["current_price"],
                    reason="Top-Down Scarsita': fuori dal set caldo o dal quantile"
                ))

        total_nav = portfolio.get_total_nav({k: v["current_price"] for k, v in market_snapshot.items()})
        target_per_position = total_nav * min(self.max_allocation_pct, 1.0 / max(1, len(eligible)))

        for item_id in eligible:
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
                    reason=f"Top-Down: set con box in momentum + residuo {residuals[item_id]:+.2f}"
                ))
        return signals
