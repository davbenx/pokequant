"""
poke_quant/engine/strategies — Strategie quantitative di investimento in carte e sealed Pokémon.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Signal:
    action: str            # "BUY", "SELL", "HOLD"
    item_id: str
    item_name: str
    item_type: str        # "sealed", "single"
    quantity: int
    target_price: float
    reason: str
