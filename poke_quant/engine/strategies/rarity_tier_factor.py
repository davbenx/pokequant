"""
poke_quant/engine/strategies/rarity_tier_factor.py — Fattore Rarita' Ex-Ante.

A differenza del filtro di prezzo di discover_chase_cards.py (selezione sull'ESITO,
survivorship bias), qui l'eleggibilita' e' decisa dalla RARITA' TIPOGRAFICA assegnata
da Nintendo/TPCi al momento della stampa - un'informazione disponibile ex-ante,
identica per chi comprava il giorno dell'uscita e per chi guarda lo storico oggi.
Ipotesi: le rarita' con stampa limitata per definizione (1 per booster box o meno)
portano un premio di scarsita' strutturale indipendente dall'esito di prezzo.
A ogni ribilanciamento, equal-weight su tutte le carte eleggibili di rarita' premium
tra quelle correntemente prezzate (non un ranking - la rarita' e' binaria, nota,
fissa nel tempo), fino a un tetto di posizioni per restare diversificati.
"""

from __future__ import annotations
from typing import Dict, List, Any, FrozenSet
import pandas as pd
from poke_quant.engine.strategies import Signal
from poke_quant.engine.portfolio import Portfolio

PREMIUM_RARITIES: FrozenSet[str] = frozenset({
    "Rare Secret", "Rare Rainbow", "Rare Ultra", "Special Illustration Rare",
    "Illustration Rare", "Hyper Rare", "Rare Holo VMAX", "Rare Holo VSTAR",
})


class RarityTierFactorStrategy:
    def __init__(
        self,
        rebalance_every_months: int = 6,
        max_positions: int = 40,
        max_allocation_pct: float = 0.08,
        min_age_months: int = 6,
        item_type_filter: str = "single",
        premium_rarities: FrozenSet[str] = PREMIUM_RARITIES,
    ):
        self.rebalance_every_months = rebalance_every_months
        self.max_positions = max_positions
        self.max_allocation_pct = max_allocation_pct
        self.min_age_months = min_age_months
        self.item_type_filter = item_type_filter
        self.premium_rarities = premium_rarities
        self._call_count = 0

    def reset(self):
        self._call_count = 0

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

        # Lista ordinata per item_id (non set) - eleggibilita' deterministica,
        # esecuzione a corto di cassa sempre nello stesso ordine riproducibile.
        eligible: List[str] = []
        for item_id, info in sorted(market_snapshot.items()):
            if info.get("type") != self.item_type_filter or info.get("current_price", 0) <= 0:
                continue
            if info.get("rarity") not in self.premium_rarities:
                continue
            if self.min_age_months > 0 and info.get("release_date"):
                rel_dt = pd.to_datetime(info["release_date"])
                age_m = (cur_dt.year - rel_dt.year) * 12 + (cur_dt.month - rel_dt.month)
                if age_m < self.min_age_months:
                    continue
            eligible.append(item_id)
        if not eligible:
            return signals

        eligible = eligible[: self.max_positions]
        eligible_set = set(eligible)  # solo per test di appartenenza, mai iterato

        for item_id, pos in list(portfolio.positions.items()):
            if item_id in market_snapshot and item_id not in eligible_set:
                signals.append(Signal(
                    action="SELL", item_id=item_id, item_name=pos.item_name,
                    item_type=pos.item_type, quantity=pos.quantity,
                    target_price=market_snapshot[item_id]["current_price"],
                    reason="Rarita' Ex-Ante: uscita dal set di rarita' premium eleggibili"
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
                    reason=f"Rarita' Ex-Ante: {info.get('rarity')} (premium tipografica)"
                ))
        return signals
