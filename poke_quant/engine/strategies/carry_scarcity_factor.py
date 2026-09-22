"""
poke_quant/engine/strategies/carry_scarcity_factor.py — Fattore di Carry/Scarsità.

Analogia accademica: come il "carry" nei futures/FX è remunerato dalla struttura a termine,
qui il "carry" è il tempo trascorso dalla release (proxy di irreversibilità dell'offerta:
niente ristampe = l'offerta fisica è fissa da quel momento in poi). A ogni ribilanciamento,
va lungo sul quantile di asset PIÙ VECCHI (quindi presumibilmente più out-of-print) tra
quelli correntemente prezzati, pesati equamente. Nessuna etichetta di qualità/tier coinvolta:
solo età anagrafica, calcolabile in modo identico ex-ante ed ex-post.
"""

from __future__ import annotations
from typing import Dict, List, Any
import pandas as pd
from poke_quant.engine.strategies import Signal
from poke_quant.engine.portfolio import Portfolio


class CarryScarcityFactorStrategy:
    def __init__(
        self,
        top_quantile: float = 0.30,
        rebalance_every_months: int = 3,
        max_allocation_pct: float = 0.15,
        min_age_months: int = 6,
        item_type_filter: str = "sealed",
    ):
        self.top_quantile = top_quantile
        self.rebalance_every_months = rebalance_every_months
        self.max_allocation_pct = max_allocation_pct
        self.min_age_months = min_age_months
        self.item_type_filter = item_type_filter
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

        ranked = []
        for item_id, info in market_snapshot.items():
            if info.get("type") != self.item_type_filter or info.get("current_price", 0) <= 0:
                continue
            rel_dt = pd.to_datetime(info["release_date"])
            age_m = (cur_dt.year - rel_dt.year) * 12 + (cur_dt.month - rel_dt.month)
            if age_m < self.min_age_months:
                continue
            ranked.append((item_id, age_m))
        if not ranked:
            return signals

        ranked.sort(key=lambda x: (x[1], x[0]), reverse=True)
        n_top = max(1, int(round(len(ranked) * self.top_quantile)))
        # NON usare un set per iterare: un set di stringhe itera in un ordine dipendente
        # dall'hash-seed del processo (PYTHONHASHSEED, randomizzato per default da Python
        # 3.3+). Trovato in questa sessione: con capitale limitato, l'ordine in cui le BUY
        # vengono emesse/eseguite decide quale carta riceve budget prima che finisca la
        # cassa - rendendo l'intero backtest NON riproducibile run-to-run (stesso codice,
        # stesso input, CAGR/Sharpe diversi). top_ranked preserva l'ordine di rank (pareggi
        # rotti per item_id, deterministico) sia per il test di appartenenza sia per
        # l'iterazione di acquisto, cosi' che a corto di cassa vinca sempre il segnale piu'
        # forte (piu' vecchio), non la fortuna dell'hash.
        top_ranked = ranked[:n_top]
        top_ids = {item_id for item_id, _ in top_ranked}

        for item_id, pos in list(portfolio.positions.items()):
            if item_id in market_snapshot and item_id not in top_ids:
                signals.append(Signal(
                    action="SELL", item_id=item_id, item_name=pos.item_name,
                    item_type=pos.item_type, quantity=pos.quantity,
                    target_price=market_snapshot[item_id]["current_price"],
                    reason="Carry/Scarsità: uscito dal quantile top per età"
                ))

        total_nav = portfolio.get_total_nav({k: v["current_price"] for k, v in market_snapshot.items()})
        target_per_position = total_nav * min(self.max_allocation_pct, 1.0 / max(1, n_top))

        for item_id, age_m in top_ranked:
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
                    reason=f"Carry/Scarsità: quantile top per età ({age_m} mesi da release)"
                ))
        return signals
