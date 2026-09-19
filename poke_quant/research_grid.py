"""
poke_quant/research_grid.py — Studio quantitativo estensivo su griglia di parametri.
Analizza:
  1. Timing di acquisto (Mesi dal lancio, Prezzo vs MSRP)
  2. Tipologia di prodotto (Booster Box vs Specialty ETB vs Bundles vs Singole)
  3. Criterio di selezione qualitativa (Tier S/A vs Tier C)
  4. Regole di uscita e vendita (Holding fisso vs Target ROI dinamico)
  5. Concentrazione di portafoglio (Allocazione per set)
"""

from __future__ import annotations
import itertools
from typing import Dict, List, Any
import numpy as np
import pandas as pd

from poke_quant.data.storage import load_price_matrix, load_metadata
from poke_quant.engine.portfolio import Portfolio
from poke_quant.engine.backtester import Backtester
from poke_quant.engine.strategies import Signal
from poke_quant.validation.metrics import cagr, sharpe, max_drawdown, calmar, sortino_ratio


class FlexibleSealedStrategy:
    """Strategia parametrizzata per simulazioni su griglia multidimensionale."""

    def __init__(
        self,
        allowed_product_types: List[str] = None, # ["booster_box", "specialty_etb", "specialty_bundle"]
        allowed_tiers: List[str] = None,         # ["S", "A", "B", "C"]
        min_buy_age_months: int = 0,             # Mese minimo dal lancio per comprare
        max_buy_age_months: int = 14,            # Mese massimo dal lancio per comprare
        max_msrp_mult: float = 1.15,             # Massimo multiplo rispetto a MSRP per comprare
        min_hold_months: int = 24,               # Mesi minimi di holding prima di valutare vendita
        target_roi: float = 1.00,                # Target di guadagno netto per uscire
        max_hold_months: int = 48,               # Uscita forzata massima (mesi)
        max_alloc_pct: float = 0.25              # Max % capitale per singolo prodotto
    ):
        self.allowed_product_types = allowed_product_types or ["booster_box", "specialty_etb", "specialty_bundle"]
        self.allowed_tiers = allowed_tiers or ["S", "A", "B", "C"]
        self.min_buy_age_months = min_buy_age_months
        self.max_buy_age_months = max_buy_age_months
        self.max_msrp_mult = max_msrp_mult
        self.min_hold_months = min_hold_months
        self.target_roi = target_roi
        self.max_hold_months = max_hold_months
        self.max_alloc_pct = max_alloc_pct

    def generate_signals(
        self,
        current_date: str,
        portfolio: Portfolio,
        market_snapshot: Dict[str, Dict[str, Any]]
    ) -> List[Signal]:
        signals: List[Signal] = []
        cur_dt = pd.to_datetime(current_date)
        total_nav = portfolio.get_total_nav({k: v["current_price"] for k, v in market_snapshot.items()})

        # 1. Vendite
        for item_id, pos in list(portfolio.positions.items()):
            if item_id not in market_snapshot:
                continue
            cur_price = market_snapshot[item_id]["current_price"]
            cost_basis = pos.buy_price_unit
            buy_dt = pd.to_datetime(pos.buy_date)
            holding_m = max(1, (cur_dt.year - buy_dt.year) * 12 + (cur_dt.month - buy_dt.month))
            unrealized_roi = (cur_price - cost_basis) / cost_basis if cost_basis > 0 else 0.0

            # Uscita A: target raggiunto dopo holding minimo
            if holding_m >= self.min_hold_months and unrealized_roi >= self.target_roi:
                signals.append(Signal(
                    action="SELL",
                    item_id=item_id,
                    item_name=pos.item_name,
                    item_type=pos.item_type,
                    quantity=pos.quantity,
                    target_price=cur_price,
                    reason=f"Target ROI raggiunto (+{unrealized_roi*100:.1f}%)"
                ))
            # Uscita B: time-stop massimo
            elif holding_m >= self.max_hold_months:
                signals.append(Signal(
                    action="SELL",
                    item_id=item_id,
                    item_name=pos.item_name,
                    item_type=pos.item_type,
                    quantity=pos.quantity,
                    target_price=cur_price,
                    reason=f"Holding massimo ({holding_m}m)"
                ))

        # 2. Acquisti
        max_item_budget = total_nav * self.max_alloc_pct
        for item_id, info in market_snapshot.items():
            if info.get("type") != "sealed":
                continue

            # Filtro tipologia prodotto
            p_type = info.get("product_type", "booster_box")
            if p_type not in self.allowed_product_types:
                continue

            # Filtro Tier qualitativo
            tier = info.get("set_tier", "B")
            if tier not in self.allowed_tiers:
                continue

            cur_price = info["current_price"]
            if cur_price <= 0:
                continue

            rel_dt = pd.to_datetime(info["release_date"])
            age_m = (cur_dt.year - rel_dt.year) * 12 + (cur_dt.month - rel_dt.month)

            # Filtro finestra d'acquisto
            if not (self.min_buy_age_months <= age_m <= self.max_buy_age_months):
                continue

            # Filtro prezzo vs MSRP
            msrp = info.get("msrp", 140.0)
            if cur_price > msrp * self.max_msrp_mult:
                continue

            # Filtro cap capitale
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
                    reason=f"Acquisto {p_type} Tier {tier} a {cur_price:.1f}€ (MSRP: {msrp:.1f}€)"
                ))

        return signals


def run_comprehensive_study():
    prices_df = load_price_matrix()
    metadata = load_metadata()

    # Arricchiamo metadata per items che non avevano product_type o set_tier
    for k, v in metadata.items():
        if "product_type" not in v:
            if "bb" in k or "booster_box" in v.get("item_slug", ""):
                v["product_type"] = "booster_box"
            elif "etb" in k:
                v["product_type"] = "specialty_etb"
            elif "bundle" in k:
                v["product_type"] = "specialty_bundle"
            else:
                v["product_type"] = "single"
        if "set_tier" not in v:
            v["set_tier"] = "B"

    print("=" * 80)
    print("  POKEQUANT — STUDIO QUANTITATIVO SU GRIGLIA MULTIDIMENSIONALE")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # STUDIO 1: EFFETTO SELEZIONE QUALITATIVA (TIER S/A vs TUTTI vs TIER C)
    # -------------------------------------------------------------------------
    print("\n--- 1. STUDIO SELEZIONE QUALITATIVA (TIERS) ---")
    tier_tests = [
        ("Tier S Only (Solo Superstar)", ["S"]),
        ("Tier S + A (Qualità Superiore)", ["S", "A"]),
        ("Tutti i Tier (S + A + B + C)", ["S", "A", "B", "C"]),
        ("Tier C Only (Set Deboli/Overprinted)", ["C"])
    ]
    tier_results = []
    for label, tiers in tier_tests:
        strat = FlexibleSealedStrategy(
            allowed_product_types=["booster_box"],
            allowed_tiers=tiers,
            min_buy_age_months=0,
            max_buy_age_months=14,
            max_msrp_mult=1.20,
            min_hold_months=24,
            target_roi=1.00,
            max_hold_months=48,
            max_alloc_pct=0.25
        )
        bt = Backtester(strat, prices_df, metadata, initial_cash=10000.0, platform="cardmarket")
        res = bt.run()
        tier_results.append({
            "Selezione": label,
            "Capitale Finale": f"{res.final_nav:,.2f} €",
            "CAGR Netto": f"{res.cagr*100:+.2f}%",
            "Sharpe": f"{res.sharpe:.2f}",
            "MaxDD": f"{res.max_drawdown*100:.2f}%",
            "Trades": res.total_trades,
            "Win Rate": f"{res.win_rate*100:.0f}%"
        })
    print(pd.DataFrame(tier_results).to_string(index=False))

    # -------------------------------------------------------------------------
    # STUDIO 2: TIPOLOGIA DI PRODOTTO (BOOSTER BOX vs SPECIALTY ETB vs BUNDLE)
    # -------------------------------------------------------------------------
    print("\n--- 2. STUDIO TIPOLOGIA PRODOTTO ---")
    type_tests = [
        ("Solo Booster Box Standard (36 bustine)", ["booster_box"]),
        ("Solo Specialty Sets (ETB & Bundles)", ["specialty_etb", "specialty_bundle"]),
        ("Mix Diversificato (Booster Box + Specialty)", ["booster_box", "specialty_etb", "specialty_bundle"])
    ]
    type_results = []
    for label, ptypes in type_tests:
        strat = FlexibleSealedStrategy(
            allowed_product_types=ptypes,
            allowed_tiers=["S", "A", "B"],
            min_buy_age_months=0,
            max_buy_age_months=14,
            max_msrp_mult=1.25,
            min_hold_months=20,
            target_roi=0.80,
            max_hold_months=48,
            max_alloc_pct=0.25
        )
        bt = Backtester(strat, prices_df, metadata, initial_cash=10000.0, platform="cardmarket")
        res = bt.run()
        type_results.append({
            "Tipologia": label,
            "Capitale Finale": f"{res.final_nav:,.2f} €",
            "CAGR Netto": f"{res.cagr*100:+.2f}%",
            "Sharpe": f"{res.sharpe:.2f}",
            "MaxDD": f"{res.max_drawdown*100:.2f}%",
            "Trades": res.total_trades
        })
    print(pd.DataFrame(type_results).to_string(index=False))

    # -------------------------------------------------------------------------
    # STUDIO 3: TIMING DI ACQUISTO (FINESTRA DI INGRESSO RISPETTO AL LANCIO)
    # -------------------------------------------------------------------------
    print("\n--- 3. STUDIO TIMING DI ACQUISTO (BUY WINDOW) ---")
    timing_tests = [
        ("Lancio Immediato (Mesi 0-3)", 0, 3, 1.10),
        ("Fase Post-Lancio / 1° Ristampa (Mesi 4-8)", 4, 8, 1.15),
        ("Fondo Fisiologico / Reprints (Mesi 9-16)", 9, 16, 1.20),
        ("Ampia Finestra Attiva (Mesi 0-14)", 0, 14, 1.20),
        ("Tardivo / Pre-OOP (Mesi 15-24)", 15, 24, 1.40)
    ]
    timing_results = []
    for label, min_m, max_m, mult in timing_tests:
        strat = FlexibleSealedStrategy(
            allowed_product_types=["booster_box"],
            allowed_tiers=["S", "A"],
            min_buy_age_months=min_m,
            max_buy_age_months=max_m,
            max_msrp_mult=mult,
            min_hold_months=24,
            target_roi=1.00,
            max_hold_months=48,
            max_alloc_pct=0.25
        )
        bt = Backtester(strat, prices_df, metadata, initial_cash=10000.0, platform="cardmarket")
        res = bt.run()
        timing_results.append({
            "Finestra Acquisto": label,
            "Capitale Finale": f"{res.final_nav:,.2f} €",
            "CAGR Netto": f"{res.cagr*100:+.2f}%",
            "Sharpe": f"{res.sharpe:.2f}",
            "MaxDD": f"{res.max_drawdown*100:.2f}%",
            "Trades": res.total_trades
        })
    print(pd.DataFrame(timing_results).to_string(index=False))

    # -------------------------------------------------------------------------
    # STUDIO 4: REGOLE DI VENDITA & USCITA (TARGET ROI vs HOLDING FISSO)
    # -------------------------------------------------------------------------
    print("\n--- 4. STUDIO REGOLE DI VENDITA & USCITA ---")
    exit_tests = [
        ("Uscita Rapida (Min 18m, Target +60%)", 18, 0.60, 36),
        ("Uscita Standard (Min 24m, Target +100%)", 24, 1.00, 48),
        ("Uscita Ambiziosa (Min 30m, Target +150%)", 30, 1.50, 48),
        ("Holding Puro 3 Anni (Min 36m, Target +200%)", 36, 2.00, 48),
        ("Holding Puro 4 Anni (Min 48m, Target +300%)", 48, 3.00, 60),
    ]
    exit_results = []
    for label, min_h, tgt_roi, max_h in exit_tests:
        strat = FlexibleSealedStrategy(
            allowed_product_types=["booster_box"],
            allowed_tiers=["S", "A"],
            min_buy_age_months=0,
            max_buy_age_months=14,
            max_msrp_mult=1.20,
            min_hold_months=min_h,
            target_roi=tgt_roi,
            max_hold_months=max_h,
            max_alloc_pct=0.25
        )
        bt = Backtester(strat, prices_df, metadata, initial_cash=10000.0, platform="cardmarket")
        res = bt.run()
        exit_results.append({
            "Regola di Uscita": label,
            "Capitale Finale": f"{res.final_nav:,.2f} €",
            "CAGR Netto": f"{res.cagr*100:+.2f}%",
            "Sharpe": f"{res.sharpe:.2f}",
            "MaxDD": f"{res.max_drawdown*100:.2f}%",
            "Trades": res.total_trades,
            "Win Rate": f"{res.win_rate*100:.0f}%"
        })
    print(pd.DataFrame(exit_results).to_string(index=False))

    # -------------------------------------------------------------------------
    # STUDIO 5: CONCENTRAZIONE & DIVERSIFICAZIONE (MAX ALLOCATION PER SET)
    # -------------------------------------------------------------------------
    print("\n--- 5. STUDIO CONCENTRAZIONE DI PORTAFOGLIO ---")
    alloc_tests = [
        ("Alta Diversificazione (Max 15% per Set)", 0.15),
        ("Bilanciato (Max 25% per Set)", 0.25),
        ("Concentrato (Max 35% per Set)", 0.35),
        ("Aggressivo (Max 50% per Set)", 0.50),
    ]
    alloc_results = []
    for label, alloc in alloc_tests:
        strat = FlexibleSealedStrategy(
            allowed_product_types=["booster_box"],
            allowed_tiers=["S", "A"],
            min_buy_age_months=0,
            max_buy_age_months=14,
            max_msrp_mult=1.20,
            min_hold_months=24,
            target_roi=1.00,
            max_hold_months=48,
            max_alloc_pct=alloc
        )
        bt = Backtester(strat, prices_df, metadata, initial_cash=10000.0, platform="cardmarket")
        res = bt.run()
        alloc_results.append({
            "Allocazione": label,
            "Capitale Finale": f"{res.final_nav:,.2f} €",
            "CAGR Netto": f"{res.cagr*100:+.2f}%",
            "Sharpe": f"{res.sharpe:.2f}",
            "MaxDD": f"{res.max_drawdown*100:.2f}%",
            "Trades": res.total_trades
        })
    print(pd.DataFrame(alloc_results).to_string(index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_comprehensive_study()
