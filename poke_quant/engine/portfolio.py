"""
poke_quant/engine/portfolio.py — Gestione dello stato del portafoglio e inventario fisico.
Tiene traccia di liquidità, posizioni aperte, vendite concluse, costi sostenuti e NAV.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd
from poke_quant.engine.friction import calculate_sale_friction


@dataclass
class Position:
    item_id: str
    item_name: str
    item_type: str        # "sealed" o "single"
    quantity: int
    buy_date: str
    buy_price_unit: float
    total_cost: float


@dataclass
class ClosedTrade:
    item_id: str
    item_name: str
    item_type: str
    quantity: int
    buy_date: str
    sell_date: str
    buy_price_unit: float
    sell_price_unit: float
    gross_proceeds: float
    fees_paid: float
    net_proceeds: float
    net_pnl: float
    net_roi: float
    holding_months: float


class Portfolio:
    """Gestisce il capitale liquido e l'inventario fisico di carte/box collezionabili."""

    def __init__(self, initial_cash: float = 10000.0):
        self.initial_cash: float = initial_cash
        self.cash: float = initial_cash
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[ClosedTrade] = []
        self.history_records: List[dict] = []

    def buy(
        self,
        item_id: str,
        item_name: str,
        item_type: str,
        quantity: int,
        unit_price: float,
        date: str
    ) -> bool:
        """Esegue l'acquisto di un articolo scalando la liquidità."""
        total_cost = unit_price * quantity
        if total_cost > self.cash:
            return False  # Liquidità insufficiente

        self.cash -= total_cost

        if item_id in self.positions:
            # Aggiorna posizione esistente (prezzo medio di carico ponderato)
            pos = self.positions[item_id]
            new_qty = pos.quantity + quantity
            new_total = pos.total_cost + total_cost
            pos.buy_price_unit = new_total / new_qty
            pos.quantity = new_qty
            pos.total_cost = new_total
        else:
            self.positions[item_id] = Position(
                item_id=item_id,
                item_name=item_name,
                item_type=item_type,
                quantity=quantity,
                buy_date=date,
                buy_price_unit=unit_price,
                total_cost=total_cost
            )
        return True

    def sell(
        self,
        item_id: str,
        quantity: int,
        unit_gross_price: float,
        date: str,
        platform: str = "cardmarket",
        seller_absorbs_shipping: bool = False,
        slippage_pct: float = 0.0,
        carrying_cost: float = 0.0
    ) -> Optional[ClosedTrade]:
        """Esegue la vendita parziale o totale di una posizione, calcolando fee e incasso netto."""
        if item_id not in self.positions or self.positions[item_id].quantity < quantity:
            return None

        pos = self.positions[item_id]
        gross_total = unit_gross_price * quantity
        friction = calculate_sale_friction(
            gross_price=gross_total,
            item_type=pos.item_type,
            platform=platform,
            seller_absorbs_shipping=seller_absorbs_shipping,
            slippage_pct=slippage_pct,
            carrying_cost=carrying_cost
        )

        cost_basis = pos.buy_price_unit * quantity
        net_proceeds = friction.net_proceeds
        fees_paid = gross_total - net_proceeds
        net_pnl = net_proceeds - cost_basis
        net_roi = net_pnl / cost_basis if cost_basis > 0 else 0.0

        # Calcolo approssimativo holding period in mesi
        try:
            d_buy = pd.to_datetime(pos.buy_date)
            d_sell = pd.to_datetime(date)
            holding_months = max(1.0, (d_sell.year - d_buy.year) * 12 + (d_sell.month - d_buy.month))
        except Exception:
            holding_months = 1.0

        trade = ClosedTrade(
            item_id=item_id,
            item_name=pos.item_name,
            item_type=pos.item_type,
            quantity=quantity,
            buy_date=pos.buy_date,
            sell_date=date,
            buy_price_unit=pos.buy_price_unit,
            sell_price_unit=unit_gross_price,
            gross_proceeds=gross_total,
            fees_paid=fees_paid,
            net_proceeds=net_proceeds,
            net_pnl=net_pnl,
            net_roi=net_roi,
            holding_months=float(holding_months)
        )

        self.cash += net_proceeds
        self.closed_trades.append(trade)

        if pos.quantity == quantity:
            del self.positions[item_id]
        else:
            pos.quantity -= quantity
            pos.total_cost -= cost_basis

        return trade

    def get_market_value(self, current_prices: Dict[str, float]) -> float:
        """Calcola il valore totale di mercato dell'inventario aperto (mark-to-market)."""
        inventory_value = 0.0
        for item_id, pos in self.positions.items():
            price = current_prices.get(item_id, pos.buy_price_unit)
            inventory_value += price * pos.quantity
        return inventory_value

    def get_total_nav(self, current_prices: Dict[str, float]) -> float:
        """Valore liquidativo totale (Net Asset Value = Cassa + Valore Inventario)."""
        return self.cash + self.get_market_value(current_prices)

    def record_snapshot(self, date: str, current_prices: Dict[str, float]):
        """Registra lo snapshot periodico di patrimonio netto per l'equity curve."""
        inv_val = self.get_market_value(current_prices)
        total_nav = self.cash + inv_val
        self.history_records.append({
            "date": date,
            "cash": self.cash,
            "inventory_value": inv_val,
            "nav": total_nav,
            "open_positions": len(self.positions)
        })

    def get_history_df(self) -> pd.DataFrame:
        if not self.history_records:
            return pd.DataFrame(columns=["date", "cash", "inventory_value", "nav", "open_positions"])
        df = pd.DataFrame(self.history_records)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        return df

    def get_trades_df(self) -> pd.DataFrame:
        if not self.closed_trades:
            return pd.DataFrame()
        return pd.DataFrame([vars(t) for t in self.closed_trades])
