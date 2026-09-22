"""
poke_quant/engine/strategies/equal_weight_benchmark.py — Benchmark Equal-Weight Buy & Hold.

Il vero benchmark "beta dell'asset class": compra ogni asset sealed disponibile appena
entra nell'universo (equamente pesato sul capitale residuo), non vende mai. Nessuna
selezione, nessun timing, nessun tier. Serve a rispondere alla domanda posta dal Test 9
di falsification_suite.py: quanto delle altre strategie è vero stock-picking/timing
e quanto è semplicemente "aver posseduto la asset class"?
"""

from __future__ import annotations
from typing import Dict, List, Any
from poke_quant.engine.strategies import Signal
from poke_quant.engine.portfolio import Portfolio


class EqualWeightBenchmarkStrategy:
    def __init__(self, item_type_filter: str = "sealed"):
        self.item_type_filter = item_type_filter
        self.bought: set = set()

    def reset(self):
        self.bought = set()

    def generate_signals(
        self,
        current_date: str,
        portfolio: Portfolio,
        market_snapshot: Dict[str, Dict[str, Any]],
    ) -> List[Signal]:
        signals: List[Signal] = []
        new_ids = [
            item_id for item_id, info in market_snapshot.items()
            if info.get("type") == self.item_type_filter
            and info.get("current_price", 0) > 0
            and item_id not in self.bought
        ]
        if not new_ids:
            return signals

        # Riserva capitale in parti uguali per ogni nuovo asset che entra nell'universo oggi.
        budget_per_item = portfolio.cash / len(new_ids)
        for item_id in new_ids:
            info = market_snapshot[item_id]
            cur_price = info["current_price"]
            qty = int(budget_per_item // cur_price)
            if qty < 1 and portfolio.cash >= cur_price:
                qty = 1
            if qty >= 1:
                self.bought.add(item_id)
                signals.append(Signal(
                    action="BUY", item_id=item_id, item_name=info.get("name", item_id),
                    item_type=self.item_type_filter, quantity=qty, target_price=cur_price,
                    reason="Equal-Weight Benchmark: acquisto unico all'ingresso nell'universo"
                ))
            else:
                self.bought.add(item_id)  # niente budget: non ritentare ogni mese
        return signals
