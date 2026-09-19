"""
poke_quant/engine/strategies/chase_dip_buyer.py — Strategia Chase Card Dip Buyer.
Sfrutta il ciclo di hype delle singole "chase card" moderne (SIR, Alternate Art).
Compra durante la saturazione dell'offerta (mesi 4-10) e vende nella fase di consolidamento e recupero.
"""

from __future__ import annotations
from typing import Dict, List, Any
import pandas as pd
from poke_quant.engine.strategies import Signal
from poke_quant.engine.portfolio import Portfolio


class ChaseDipBuyerStrategy:
    """
    Strategia quantitativa per singole carte rare (Chase cards):
    Identifica il minimo ciclico post-lancio dovuto all'over-supply di box opening.
    """

    def __init__(
        self,
        min_dip_months: int = 4,                  # Inizia a monitorare dopo 4 mesi dal lancio
        max_dip_months: int = 10,                 # Compra entro il 10° mese dal lancio
        min_drop_from_launch_pct: float = 0.20,   # Richiede un calo di almeno il 20% rispetto al prezzo iniziale
        target_profit_roi: float = 0.60,          # Target di vendita a +60% netto
        max_allocation_per_card_pct: float = 0.15,# Max 15% del capitale su una singola carta
        max_hold_months: int = 24                 # Uscita massima a 24 mesi
    ):
        self.min_dip_months = min_dip_months
        self.max_dip_months = max_dip_months
        self.min_drop_from_launch_pct = min_drop_from_launch_pct
        self.target_profit_roi = target_profit_roi
        self.max_allocation_per_card_pct = max_allocation_per_card_pct
        self.max_hold_months = max_hold_months

    def generate_signals(
        self,
        current_date: str,
        portfolio: Portfolio,
        market_snapshot: Dict[str, Dict[str, Any]]
    ) -> List[Signal]:
        signals: List[Signal] = []
        cur_dt = pd.to_datetime(current_date)
        total_nav = portfolio.get_total_nav({k: v["current_price"] for k, v in market_snapshot.items()})

        # 1. Valutazione Uscite
        for item_id, pos in list(portfolio.positions.items()):
            if pos.item_type != "single" or item_id not in market_snapshot:
                continue

            current_price = market_snapshot[item_id]["current_price"]
            cost_basis = pos.buy_price_unit
            buy_dt = pd.to_datetime(pos.buy_date)
            holding_months = max(1, (cur_dt.year - buy_dt.year) * 12 + (cur_dt.month - buy_dt.month))
            unrealized_roi = (current_price - cost_basis) / cost_basis if cost_basis > 0 else 0.0

            # Uscita su target di profitto
            if unrealized_roi >= self.target_profit_roi:
                signals.append(Signal(
                    action="SELL",
                    item_id=item_id,
                    item_name=pos.item_name,
                    item_type="single",
                    quantity=pos.quantity,
                    target_price=current_price,
                    reason=f"Target ROI raggiunto (+{unrealized_roi*100:.1f}%) dopo {holding_months} mesi"
                ))
            # Uscita per time-stop
            elif holding_months >= self.max_hold_months:
                signals.append(Signal(
                    action="SELL",
                    item_id=item_id,
                    item_name=pos.item_name,
                    item_type="single",
                    quantity=pos.quantity,
                    target_price=current_price,
                    reason=f"Holding period massimo raggiunto ({holding_months} mesi)"
                ))

        # 2. Valutazione Ingressi
        max_card_budget = total_nav * self.max_allocation_per_card_pct
        for item_id, info in market_snapshot.items():
            if info.get("type") != "single":
                continue

            current_price = info["current_price"]
            launch_price = info.get("launch_price", current_price)
            if current_price <= 0 or launch_price <= 0:
                continue

            rel_dt = pd.to_datetime(info["release_date"])
            age_months = (cur_dt.year - rel_dt.year) * 12 + (cur_dt.month - rel_dt.month)

            # Finestra temporale del dip
            if not (self.min_dip_months <= age_months <= self.max_dip_months):
                continue

            # Verificare che il prezzo sia sceso rispetto al picco di lancio
            drawdown_from_launch = (launch_price - current_price) / launch_price
            if drawdown_from_launch < self.min_drop_from_launch_pct:
                continue

            # Posizione non già sovradimensionata
            current_pos_cost = portfolio.positions[item_id].total_cost if item_id in portfolio.positions else 0.0
            if current_pos_cost >= max_card_budget:
                continue

            available_cash = portfolio.cash
            budget = min(available_cash, max_card_budget - current_pos_cost)
            qty = int(budget // current_price)

            if qty >= 1:
                signals.append(Signal(
                    action="BUY",
                    item_id=item_id,
                    item_name=info["name"],
                    item_type="single",
                    quantity=qty,
                    target_price=current_price,
                    reason=f"Dip buying a {age_months} mesi da release (-{drawdown_from_launch*100:.1f}% da lancio)"
                ))

        return signals
