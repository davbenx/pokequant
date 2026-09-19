"""
poke_quant/engine/strategies/grading_arbitrage.py — Strategia Stat-Arb di Grading.
Identifica opportunità matematiche con Valore Atteso netto (EV) positivo sul grading PSA/BGS.
Modella i costi fissi del servizio, i tempi di fermo capitale e la distribuzione probabilistica dei voti.
"""

from __future__ import annotations
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from poke_quant.engine.strategies import Signal
from poke_quant.engine.portfolio import Portfolio
from poke_quant.engine.friction import evaluate_grading_arbitrage


class GradingArbitrageStrategy:
    """
    Strategia di arbitraggio statistico: acquisto di carte grezze (Raw Near Mint)
    con sottomissione a grading PSA e vendita al voto ottenuto.
    """

    def __init__(
        self,
        min_expected_roi: float = 0.30,         # Richiede almeno il +30% di EV netto per procedere
        max_allocation_per_sub_pct: float = 0.20,# Max 20% del capitale per singola sottomissione
        platform: str = "cardmarket",
        turnaround_months: int = 2              # Mesi di fermo prima di poter vendere la carta gradata
    ):
        self.min_expected_roi = min_expected_roi
        self.max_allocation_per_sub_pct = max_allocation_per_sub_pct
        self.platform = platform
        self.turnaround_months = turnaround_months

    def screen_opportunities(
        self,
        market_data: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filtra l'universo delle carte alla ricerca di anomalie di prezzo tra Raw e PSA 10.
        """
        opportunities = []
        for card_id, data in market_data.items():
            if data.get("type") != "single":
                continue

            raw_price = data.get("raw_price", data.get("current_price", 0.0))
            psa10_price = data.get("psa10_price", 0.0)
            psa9_price = data.get("psa9_price", None)
            gem_rate = data.get("gem_rate", None)
            item_era = data.get("era", "modern")

            if raw_price <= 0 or psa10_price <= 0:
                continue

            res = evaluate_grading_arbitrage(
                raw_price=raw_price,
                psa10_price=psa10_price,
                psa9_price=psa9_price,
                gem_rate=gem_rate,
                item_era=item_era,
                platform=self.platform
            )

            if res.expected_net_roi >= self.min_expected_roi:
                opportunities.append({
                    "card_id": card_id,
                    "card_name": data.get("name", card_id),
                    "raw_price": raw_price,
                    "psa10_price": psa10_price,
                    "gem_rate": res.gem_rate,
                    "expected_net_profit": res.expected_net_profit,
                    "expected_net_roi": res.expected_net_roi,
                    "multiplier_10_vs_raw": psa10_price / raw_price if raw_price > 0 else 0.0
                })

        # Ordina per miglior rendimento atteso
        opportunities.sort(key=lambda x: x["expected_net_roi"], reverse=True)
        return opportunities
