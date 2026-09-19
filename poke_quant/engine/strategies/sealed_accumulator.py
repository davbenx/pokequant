"""
poke_quant/engine/strategies/sealed_accumulator.py — Strategia Sealed Box Accumulator.
Sfrutta lo shock di offerta da fine ciclo di stampa (Out-of-Print) dei booster box.
"""

from __future__ import annotations
from typing import Dict, List, Any
import pandas as pd
from poke_quant.engine.strategies import Signal
from poke_quant.engine.portfolio import Portfolio


class SealedAccumulatorStrategy:
    """
    Strategia di accumulo sistematico di Booster Box sigillati a prezzo retail/MSRP,
    con detenzione fino al completamento del ciclo di stampa e discesa dell'offerta circolante.
    """

    def __init__(
        self,
        max_allocation_per_set_pct: float = 0.25, # Max 25% del capitale su un singolo set
        max_buy_age_months: int = 14,             # Compra solo entro i primi 14 mesi dal lancio
        min_hold_months: int = 24,                # Tieni almeno 24 mesi prima di considerare l'uscita
        target_profit_roi: float = 1.0,           # Target di uscita a +100% netto (raddoppio)
        max_hold_months: int = 48,                # Uscita forzata dopo 4 anni (rotazione capitale)
        msrp_max_multiplier: float = 1.20         # Compra a max +20% sopra MSRP
    ):
        self.max_allocation_per_set_pct = max_allocation_per_set_pct
        self.max_buy_age_months = max_buy_age_months
        self.min_hold_months = min_hold_months
        self.target_profit_roi = target_profit_roi
        self.max_hold_months = max_hold_months
        self.msrp_max_multiplier = msrp_max_multiplier

    def generate_signals(
        self,
        current_date: str,
        portfolio: Portfolio,
        market_snapshot: Dict[str, Dict[str, Any]]
    ) -> List[Signal]:
        """
        market_snapshot: {
            item_id: {
                "name": str,
                "type": "sealed",
                "current_price": float,
                "release_date": str,
                "msrp": float
            }
        }
        """
        signals: List[Signal] = []
        cur_dt = pd.to_datetime(current_date)
        total_nav = portfolio.get_total_nav({k: v["current_price"] for k, v in market_snapshot.items()})

        # 1. Valutazione Uscite (Vendite)
        for item_id, pos in list(portfolio.positions.items()):
            if pos.item_type != "sealed" or item_id not in market_snapshot:
                continue

            item_info = market_snapshot[item_id]
            current_price = item_info["current_price"]
            cost_basis = pos.buy_price_unit

            buy_dt = pd.to_datetime(pos.buy_date)
            holding_months = max(1, (cur_dt.year - buy_dt.year) * 12 + (cur_dt.month - buy_dt.month))
            unrealized_roi = (current_price - cost_basis) / cost_basis if cost_basis > 0 else 0.0

            # Condizione di uscita A: Raggiunto target di profitto dopo holding minimo
            if holding_months >= self.min_hold_months and unrealized_roi >= self.target_profit_roi:
                signals.append(Signal(
                    action="SELL",
                    item_id=item_id,
                    item_name=pos.item_name,
                    item_type="sealed",
                    quantity=pos.quantity,
                    target_price=current_price,
                    reason=f"Target ROI raggiunto (+{unrealized_roi*100:.1f}%) dopo {holding_months} mesi"
                ))
            # Condizione di uscita B: Superato tempo massimo di detenzione (ribilanciamento)
            elif holding_months >= self.max_hold_months:
                signals.append(Signal(
                    action="SELL",
                    item_id=item_id,
                    item_name=pos.item_name,
                    item_type="sealed",
                    quantity=pos.quantity,
                    target_price=current_price,
                    reason=f"Scadenza orizzonte temporale massimo ({holding_months} mesi)"
                ))

        # 2. Valutazione Ingressi (Acquisti)
        max_item_budget = total_nav * self.max_allocation_per_set_pct
        for item_id, info in market_snapshot.items():
            if info.get("type") != "sealed":
                continue

            current_price = info["current_price"]
            if current_price <= 0:
                continue

            rel_dt = pd.to_datetime(info["release_date"])
            age_months = (cur_dt.year - rel_dt.year) * 12 + (cur_dt.month - rel_dt.month)

            # Verifica vincoli di ingresso:
            # 1. Set rilasciato e non troppo vecchio (fase di stampa attiva)
            if not (0 <= age_months <= self.max_buy_age_months):
                continue

            # 2. Prezzo non gonfiatosi eccessivamente sopra MSRP
            msrp = info.get("msrp", 140.0)
            if current_price > msrp * self.msrp_max_multiplier:
                continue

            # 3. Posizione non già sovradimensionata
            current_pos_cost = portfolio.positions[item_id].total_cost if item_id in portfolio.positions else 0.0
            if current_pos_cost >= max_item_budget:
                continue

            # Calcola quantità acquistabile
            available_cash = portfolio.cash
            budget_to_use = min(available_cash, max_item_budget - current_pos_cost)
            qty_to_buy = int(budget_to_use // current_price)

            # Regola del Lotto Minimo Indivisibile
            if qty_to_buy < 1 and current_pos_cost == 0 and available_cash >= current_price:
                if current_price <= total_nav * 0.35:
                    qty_to_buy = 1

            if qty_to_buy >= 1:
                signals.append(Signal(
                    action="BUY",
                    item_id=item_id,
                    item_name=info["name"],
                    item_type="sealed",
                    quantity=qty_to_buy,
                    target_price=current_price,
                    reason=f"Set in stampa attiva ({age_months} mesi) a {current_price:.1f}€ (MSRP: {msrp:.1f}€)"
                ))

        return signals
