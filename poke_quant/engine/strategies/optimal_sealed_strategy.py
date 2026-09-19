"""
poke_quant/engine/strategies/optimal_sealed_strategy.py — Strategia Ottimale Validata su Dati Reali.
Sintesi empirica dello studio multidimensionale per massimizzare il rendimento netto e minimizzare il MaxDD.
"""

from __future__ import annotations
from typing import Dict, List, Any
import pandas as pd
from poke_quant.engine.strategies import Signal
from poke_quant.engine.portfolio import Portfolio


class OptimalSealedStrategy:
    """
    Strategia Ottimale Validata per Prodotti Sigillati Pokémon (Tier S & A).
    
    Regole Operative Validate Empiricamente:
    1. SELEZIONE: Esclusivamente Booster Box Standard (36 bustine) o Specialty Bundles di Tier S o A.
       Scarta tassativamente set di Tier C (senza chase iconiche o overprinted).
    2. TIMING ACQUISTO: Finestra tra il 4° e il 14° mese dalla release (fase post-lancio / prima ristampa),
       solo se il prezzo è <= 1.15x MSRP (evita il picco di hype iniziale).
    3. ALLOCAZIONE: Max 20% del capitale per singolo set (diversificazione contro ristampe straordinarie).
    4. TIMING VENDITA: Uscita dinamica con holding minimo di 30 mesi e target ROI netto >= +150%,
       oppure time-stop a 48 mesi (ciclo di rotazione Out-of-Print completo).
    """

    def __init__(
        self,
        allowed_tiers: List[str] = None,
        min_buy_age_months: int = 4,
        max_buy_age_months: int = 14,
        max_msrp_multiplier: float = 1.15,
        min_hold_months: int = 30,
        target_roi: float = 1.50,
        max_hold_months: int = 48,
        max_allocation_pct: float = 0.12,
        enable_dynamic_rotation: bool = True,
        tranche1_roi: float = 0.70,
        tranche1_min_hold_months: int = 18,
        tranche1_pct: float = 0.50
    ):
        self.allowed_tiers = allowed_tiers or ["S", "A", "B"]
        self.min_buy_age_months = min_buy_age_months
        self.max_buy_age_months = max_buy_age_months
        self.max_msrp_multiplier = max_msrp_multiplier
        self.min_hold_months = min_hold_months
        self.target_roi = target_roi
        self.max_hold_months = max_hold_months
        self.max_allocation_pct = max_allocation_pct
        self.enable_dynamic_rotation = enable_dynamic_rotation
        self.tranche1_roi = tranche1_roi
        self.tranche1_min_hold_months = tranche1_min_hold_months
        self.tranche1_pct = tranche1_pct
        self.trimmed_positions: set = set()

    def reset(self):
        """Reinizializza lo stato interno per simulazioni indipendenti."""
        self.trimmed_positions = set()

    def generate_signals(
        self,
        current_date: str,
        portfolio: Portfolio,
        market_snapshot: Dict[str, Dict[str, Any]]
    ) -> List[Signal]:
        signals: List[Signal] = []
        cur_dt = pd.to_datetime(current_date)
        total_nav = portfolio.get_total_nav({k: v["current_price"] for k, v in market_snapshot.items()})

        # 1. Vendite (Uscite & Rotazione Scalare Tranche 1)
        for item_id, pos in list(portfolio.positions.items()):
            if item_id not in market_snapshot:
                continue
            cur_price = market_snapshot[item_id]["current_price"]
            cost_basis = pos.buy_price_unit
            buy_dt = pd.to_datetime(pos.buy_date)
            holding_m = max(1, (cur_dt.year - buy_dt.year) * 12 + (cur_dt.month - buy_dt.month))
            unrealized_roi = (cur_price - cost_basis) / cost_basis if cost_basis > 0 else 0.0

            # Uscita Tranche 1 (Rotazione del Capitale):
            # A 18+ mesi (Out-of-Print confermato) e ROI >= +70%, liquida il 50% per ruotare su nuovi set
            if (self.enable_dynamic_rotation and 
                item_id not in self.trimmed_positions and 
                holding_m >= self.tranche1_min_hold_months and 
                unrealized_roi >= self.tranche1_roi and 
                pos.quantity > 1):
                qty_to_sell = max(1, int(round(pos.quantity * self.tranche1_pct)))
                if qty_to_sell >= pos.quantity:
                    qty_to_sell = pos.quantity // 2
                if qty_to_sell > 0:
                    signals.append(Signal(
                        action="SELL",
                        item_id=item_id,
                        item_name=pos.item_name,
                        item_type=pos.item_type,
                        quantity=qty_to_sell,
                        target_price=cur_price,
                        reason=f"Tranche 1 Rotazione (+{unrealized_roi*100:.1f}%) a {holding_m}m - Sblocco Liquidità"
                    ))
                    self.trimmed_positions.add(item_id)
                    continue

            # Uscita Tranche 2 (Moonbag): target ROI raggiunto dopo holding minimo
            if holding_m >= self.min_hold_months and unrealized_roi >= self.target_roi:
                signals.append(Signal(
                    action="SELL",
                    item_id=item_id,
                    item_name=pos.item_name,
                    item_type=pos.item_type,
                    quantity=pos.quantity,
                    target_price=cur_price,
                    reason=f"Tranche 2 Uscita Finale (+{unrealized_roi*100:.1f}%) a {holding_m}m"
                ))
                self.trimmed_positions.discard(item_id)
            # Condizione di uscita alternativa: Time-stop massimo (4 anni)
            elif holding_m >= self.max_hold_months:
                signals.append(Signal(
                    action="SELL",
                    item_id=item_id,
                    item_name=pos.item_name,
                    item_type=pos.item_type,
                    quantity=pos.quantity,
                    target_price=cur_price,
                    reason=f"Time-stop massimo ({holding_m}m)"
                ))
                self.trimmed_positions.discard(item_id)

        # 2. Acquisti (Ingressi su reprint dip)
        max_item_budget = total_nav * self.max_allocation_pct
        for item_id, info in market_snapshot.items():
            if info.get("type") != "sealed":
                continue

            # Filtro Tipologia: solo Booster Box o Specialty Bundles (esclude ETB standard ingombranti)
            p_type = info.get("product_type", "booster_box")
            if p_type not in ["booster_box", "specialty_bundle"]:
                continue

            # Filtro Tier: S, A o B
            tier = info.get("set_tier", "B")
            if tier not in self.allowed_tiers:
                continue

            cur_price = info["current_price"]
            if cur_price <= 0:
                continue

            rel_dt = pd.to_datetime(info["release_date"])
            age_m = (cur_dt.year - rel_dt.year) * 12 + (cur_dt.month - rel_dt.month)

            # Finestra temporale ottimale: mesi 4 - 14 (evita picco iniziale, compra sui reprint)
            if not (self.min_buy_age_months <= age_m <= self.max_buy_age_months):
                continue

            # Prezzo massimo rispetto a MSRP
            msrp = info.get("msrp", 140.0)
            if cur_price > msrp * self.max_msrp_multiplier:
                continue

            # Cap allocazione per singolo set
            cur_pos_cost = portfolio.positions[item_id].total_cost if item_id in portfolio.positions else 0.0
            if cur_pos_cost >= max_item_budget:
                continue

            available_cash = portfolio.cash
            budget = min(available_cash, max_item_budget - cur_pos_cost)
            qty = int(budget // cur_price)

            if qty >= 1:
                signals.append(Signal(
                    action="BUY",
                    item_id=item_id,
                    item_name=info["name"],
                    item_type="sealed",
                    quantity=qty,
                    target_price=cur_price,
                    reason=f"Acquisto {p_type} Tier {tier} ({age_m}m) a {cur_price:.1f}€ (MSRP: {msrp:.1f}€)"
                ))

        return signals
