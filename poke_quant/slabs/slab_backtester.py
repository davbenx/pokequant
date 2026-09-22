"""
poke_quant/slabs/slab_backtester.py — Motore di Backtesting Storico per Carte Gradate (PSA, BGS, CGC).
Simula l'esecuzione della strategia nel periodo 2021-2026 con:
  - Frizioni reali: Cardmarket 5% + 0.60€, spedizione assicurata 12€/lastra, slippage 2.5%-3.5%
  - Dimensionamento discreto (acquisto di lastre intere)
  - Prese di beneficio a 2 tranche (+70% capitale recuperato, Z > 2.5 esaurimento)
  - Rotazione del capitale su differenziale d'Alpha
  - Separazione temporale t <= T (Zero Look-Ahead Bias)
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from poke_quant.slabs.models import GradingCompany, SlabGrade, EdgeType
from poke_quant.slabs.slab_universe import get_curated_grails
from poke_quant.slabs.edge_calculator import (
    calc_cross_grading_spread,
    calc_pop_saturation_edge,
    calc_gem_scarcity_edge,
    calc_parabolic_exhaustion_sell,
    calc_opportunity_cost_rotation
)


@dataclass
class TradeRecord:
    card_id: str
    card_name: str
    grade: SlabGrade
    buy_date: str
    sell_date: str
    buy_price_eur: float
    sell_price_eur: float
    quantity: int
    gross_pnl_eur: float
    total_friction_eur: float
    net_pnl_eur: float
    net_roi_pct: float
    holding_months: int
    exit_reason: str


@dataclass
class SlabBacktestResult:
    initial_capital: float
    final_nav: float
    total_return_pct: float
    cagr_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    max_drawdown_duration_months: int
    total_trades: int
    winning_trades: int
    win_rate_pct: float
    profit_factor: float
    total_fees_paid_eur: float
    equity_curve: pd.DataFrame
    trades_history: List[TradeRecord]
    active_positions_end: List[Dict[str, Any]]


class SlabBacktester:
    """
    Backtester quantitativo per la strategia di trading e rotazione su carte gradate.
    """

    def __init__(
        self,
        initial_capital: float = 5000.0,
        max_card_allocation_pct: float = 0.30, # Max 30% per singola lastra
        risk_free_rate: float = 0.03,
        cardmarket_fee_pct: float = 0.05,
        cardmarket_fixed_fee_eur: float = 0.60,
        shipping_insured_eur: float = 12.0,
        buy_slippage_pct: float = 0.025,
        sell_slippage_pct: float = 0.035,
    ):
        self.initial_capital = initial_capital
        self.max_card_allocation_pct = max_card_allocation_pct
        self.risk_free_rate = risk_free_rate
        self.cardmarket_fee_pct = cardmarket_fee_pct
        self.cardmarket_fixed_fee_eur = cardmarket_fixed_fee_eur
        self.shipping_insured_eur = shipping_insured_eur
        self.buy_slippage_pct = buy_slippage_pct
        self.sell_slippage_pct = sell_slippage_pct

    def run(
        self,
        price_matrix: Optional[pd.DataFrame] = None,
        universe: Optional[List[Dict[str, Any]]] = None,
        include_synthetic_demo_data: bool = False,
    ) -> SlabBacktestResult:
        """
        Esegue la simulazione cronologica mensile.

        include_synthetic_demo_data=False (default): usa solo le carte con storico
        prezzi realmente osservato (historical_prices.csv). Le carte senza dato reale
        vengono escluse, non finte. Passare True SOLO per una demo UI esplicitamente
        etichettata come tale - mai come evidenza di performance validata (vedi
        warning in _load_or_build_historical_matrix).
        """
        if universe is None:
            universe = get_curated_grails()

        # Carica o costruisce la matrice storica dei prezzi delle lastre
        if price_matrix is None:
            price_matrix = self._load_or_build_historical_matrix(
                universe, exclude_synthetic=not include_synthetic_demo_data
            )

        dates = list(price_matrix.index)
        cash = self.initial_capital
        holdings: Dict[str, Dict[str, Any]] = {}  # card_id -> {units, buy_px, buy_date, tranche_1_sold, etc.}
        trades_history: List[TradeRecord] = []
        equity_records = []
        total_fees_paid = 0.0

        for t_idx, current_date in enumerate(dates):
            current_prices = price_matrix.loc[current_date].to_dict()

            # 1. VALUTAZIONE ASSET IN PORTAFOGLIO (SELL & TAKE-PROFIT)
            to_remove = []
            for card_id, h in holdings.items():
                cur_px = current_prices.get(card_id, h["buy_px"])
                h_months = t_idx - h["buy_idx"]

                # Calcolo estensione su media mobile a 6 mesi (t <= t_idx)
                past_prices = price_matrix[card_id].iloc[max(0, t_idx - 6):t_idx + 1]
                ema_6m = past_prices.mean()
                std_1y = price_matrix[card_id].iloc[max(0, t_idx - 12):t_idx + 1].std()
                mean_1y = price_matrix[card_id].iloc[max(0, t_idx - 12):t_idx + 1].mean()

                sell_eval = calc_parabolic_exhaustion_sell(
                    current_price=cur_px,
                    buy_price=h["buy_px"],
                    ema_180=ema_6m,
                    price_1y_mean=mean_1y,
                    price_1y_std=std_1y,
                    tranche_1_already_sold=h["tranche_1_sold"]
                )

                if sell_eval["edge_active"]:
                    sell_units = h["units"]
                    if sell_eval["tranche"] == "TRANCHE_1_CAPITAL_RECOVERY" and h["units"] > 1:
                        sell_units = math.ceil(h["units"] * 0.5)
                        h["units"] -= sell_units
                        h["tranche_1_sold"] = True
                    else:
                        to_remove.append(card_id)

                    # Esecuzione della vendita
                    effective_sell_px = cur_px * (1.0 - self.sell_slippage_pct)
                    gross_proceeds = effective_sell_px * sell_units
                    fee = (gross_proceeds * self.cardmarket_fee_pct) + self.cardmarket_fixed_fee_eur + (self.shipping_insured_eur * sell_units)
                    net_proceeds = gross_proceeds - fee
                    total_fees_paid += fee
                    cash += net_proceeds

                    cost_basis = h["buy_px"] * sell_units
                    net_pnl = net_proceeds - cost_basis
                    net_roi = (net_pnl / cost_basis) * 100.0 if cost_basis > 0 else 0.0

                    trades_history.append(TradeRecord(
                        card_id=card_id,
                        card_name=h["card_name"],
                        grade=h["grade"],
                        buy_date=h["buy_date"],
                        sell_date=str(current_date),
                        buy_price_eur=round(h["buy_px"], 2),
                        sell_price_eur=round(cur_px, 2),
                        quantity=sell_units,
                        gross_pnl_eur=round(gross_proceeds - cost_basis, 2),
                        total_friction_eur=round(fee, 2),
                        net_pnl_eur=round(net_pnl, 2),
                        net_roi_pct=round(net_roi, 1),
                        holding_months=h_months,
                        exit_reason=sell_eval["reason"]
                    ))

            for cid in to_remove:
                del holdings[cid]

            # 2. SCANSIONE ROTAZIONE DEL CAPITALE (Edge 9)
            # Se abbiamo una posizione stagnante da >= 6 mesi e liquidità bassa
            if holdings and cash < 500.0:
                for card_id, h in list(holdings.items()):
                    h_months = t_idx - h["buy_idx"]
                    cur_px = current_prices.get(card_id, h["buy_px"])
                    cur_roi = ((cur_px - h["buy_px"]) / h["buy_px"]) * 100.0

                    # Se stagnante (ROI < 8% dopo 6+ mesi)
                    if h_months >= 6 and cur_roi < 8.0:
                        # Cerca se c'è un'altra carta con prezzo crollato a sconto (>20% sotto media storica)
                        for other_card in universe:
                            oc_id = other_card["card_id"]
                            if oc_id in holdings or oc_id not in current_prices:
                                continue
                            oc_cur_px = current_prices[oc_id]
                            oc_mean = price_matrix[oc_id].iloc[max(0, t_idx - 12):t_idx + 1].mean()
                            if oc_cur_px <= 0.78 * oc_mean:  # Opportunità a forte sconto
                                # Esegui rotazione!
                                sell_units = h["units"]
                                effective_sell_px = cur_px * (1.0 - self.sell_slippage_pct)
                                gross_proceeds = effective_sell_px * sell_units
                                fee = (gross_proceeds * self.cardmarket_fee_pct) + self.cardmarket_fixed_fee_eur + (self.shipping_insured_eur * sell_units)
                                net_proceeds = gross_proceeds - fee
                                total_fees_paid += fee
                                cash += net_proceeds
                                cost_basis = h["buy_px"] * sell_units
                                net_pnl = net_proceeds - cost_basis

                                trades_history.append(TradeRecord(
                                    card_id=card_id,
                                    card_name=h["card_name"],
                                    grade=h["grade"],
                                    buy_date=h["buy_date"],
                                    sell_date=str(current_date),
                                    buy_price_eur=round(h["buy_px"], 2),
                                    sell_price_eur=round(cur_px, 2),
                                    quantity=sell_units,
                                    gross_pnl_eur=round(gross_proceeds - cost_basis, 2),
                                    total_friction_eur=round(fee, 2),
                                    net_pnl_eur=round(net_pnl, 2),
                                    net_roi_pct=round((net_pnl / cost_basis) * 100, 1),
                                    holding_months=h_months,
                                    exit_reason=f"ROTAZIONE DEL CAPITALE su {other_card['name']} (Sconto -22% vs 1y)"
                                ))
                                del holdings[card_id]
                                break

            # 3. SCANSIONE OPPORTUNITÀ D'ACQUISTO (BUY)
            port_nav = cash + sum(h["units"] * current_prices.get(cid, h["buy_px"]) for cid, h in holdings.items())
            max_alloc_per_card = port_nav * self.max_card_allocation_pct

            candidates = []
            for card in universe:
                cid = card["card_id"]
                if cid in holdings:
                    continue  # Già in portafoglio
                cur_px = current_prices.get(cid, 0.0)
                if cur_px <= 0:
                    continue

                # Calcolo Z-Score del prezzo rispetto alla storia passata disponibile (t <= t_idx)
                past_series = price_matrix[cid].iloc[:t_idx + 1]
                if len(past_series) < 3:
                    continue
                mean_p = past_series.mean()
                std_p = past_series.std() if past_series.std() > 0 else 1.0
                z_px = (cur_px - mean_p) / std_p
                # Calcolo sconto rispetto alla media storica mobile
                discount_vs_mean = (mean_p - cur_px) / mean_p if mean_p > 0 else 0.0

                # Valutazione Edge 1 (Cross-Grader Dislocation) e Edge 2 (Pop Saturation)
                # In regime normale BGS 9.5 scambia al 76% di PSA 10 (sconto 24%)
                is_cross_grader_discount = (discount_vs_mean >= 0.08) or (z_px <= -0.5)
                is_dip_opportunity = discount_vs_mean >= 0.15 or z_px <= -1.2
                is_pop_plateau = (t_idx >= 8) and (z_px <= 0.5) # Fuori stampa e non euforico

                if is_cross_grader_discount or is_dip_opportunity or is_pop_plateau:
                    # Se l'edge è cross-grader, acquistiamo BGS 9.5 a premio ridotto (0.75x PSA 10)
                    buy_grade = SlabGrade.BGS_9_5_GEM if is_cross_grader_discount else SlabGrade.PSA_10
                    exec_px = cur_px * 0.76 if is_cross_grader_discount else cur_px
                    effective_discount = (mean_p - exec_px) / mean_p if mean_p > 0 else 0.15

                    candidates.append({
                        "card_id": cid,
                        "card_name": card["name"],
                        "grade": buy_grade,
                        "current_price": exec_px,
                        "discount_pct": effective_discount * 100.0,
                        "z_score": z_px
                    })

            # Ordina candidati per maggior sconto
            candidates.sort(key=lambda x: x["discount_pct"], reverse=True)

            for cand in candidates:
                cand_px = cand["current_price"]
                effective_buy_px = cand_px * (1.0 + self.buy_slippage_pct) + self.shipping_insured_eur

                # Budget di allocazione flessibile: permette l'acquisto di almeno 1 lastra se cassa sufficiente (max 45% NAV)
                max_budget = max(max_alloc_per_card, min(cash, port_nav * 0.45))
                alloc_budget = min(cash, max_budget)
                units_to_buy = int(alloc_budget // effective_buy_px)

                # Se non possiamo permetterci le unità teoriche ma abbiamo cassa per 1 lastra entro il 45% di NAV
                if units_to_buy == 0 and cash >= effective_buy_px and effective_buy_px <= port_nav * 0.45:
                    units_to_buy = 1

                if units_to_buy >= 1:
                    total_cost = effective_buy_px * units_to_buy
                    cash -= total_cost
                    holdings[cand["card_id"]] = {
                        "card_name": cand["card_name"],
                        "grade": cand["grade"],
                        "buy_px": cand_px,
                        "units": units_to_buy,
                        "buy_date": str(current_date),
                        "buy_idx": t_idx,
                        "tranche_1_sold": False
                    }

            # 4. AGGIORNAMENTO EQUITY E METRICHE TEMPORALI
            holdings_val = sum(h["units"] * current_prices.get(cid, h["buy_px"]) for cid, h in holdings.items())
            nav = cash + holdings_val
            equity_records.append({
                "date": current_date,
                "cash_eur": round(cash, 2),
                "holdings_val_eur": round(holdings_val, 2),
                "nav_eur": round(nav, 2),
                "active_slabs_count": sum(h["units"] for h in holdings.values())
            })

        # Costruzione DataFrame Equity Curve
        eq_df = pd.DataFrame(equity_records)
        eq_df["peak_nav"] = eq_df["nav_eur"].cummax()
        eq_df["drawdown_pct"] = ((eq_df["nav_eur"] - eq_df["peak_nav"]) / eq_df["peak_nav"]) * 100.0

        final_nav = eq_df["nav_eur"].iloc[-1]
        total_ret = ((final_nav - self.initial_capital) / self.initial_capital) * 100.0
        n_years = len(dates) / 12.0
        cagr = ((final_nav / self.initial_capital) ** (1.0 / n_years) - 1.0) * 100.0 if n_years > 0 else 0.0

        # Calcolo Sharpe & Sortino Ratio
        monthly_returns = eq_df["nav_eur"].pct_change().dropna()
        rf_monthly = (1.0 + self.risk_free_rate) ** (1.0 / 12.0) - 1.0
        excess_returns = monthly_returns - rf_monthly

        std_m = monthly_returns.std()
        sharpe = (excess_returns.mean() / std_m) * np.sqrt(12.0) if std_m > 0 else 0.0

        downside = monthly_returns[monthly_returns < rf_monthly] - rf_monthly
        downside_std = downside.std() if len(downside) > 1 else std_m
        sortino = (excess_returns.mean() / downside_std) * np.sqrt(12.0) if downside_std > 0 else 0.0

        max_dd = eq_df["drawdown_pct"].min()

        # Trade Stats
        total_trades = len(trades_history)
        winning_trades = len([t for t in trades_history if t.net_pnl_eur > 0])
        win_rate = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0

        gross_gains = sum(t.net_pnl_eur for t in trades_history if t.net_pnl_eur > 0)
        gross_losses = abs(sum(t.net_pnl_eur for t in trades_history if t.net_pnl_eur < 0))
        profit_factor = (gross_gains / gross_losses) if gross_losses > 0 else (99.0 if gross_gains > 0 else 0.0)

        active_positions_end = [
            {
                "card_id": cid,
                "card_name": h["card_name"],
                "grade": h["grade"].value if hasattr(h["grade"], "value") else str(h["grade"]),
                "units": h["units"],
                "buy_price": round(h["buy_px"], 2),
                "current_price": round(price_matrix.iloc[-1].get(cid, h["buy_px"]), 2),
                "unrealized_pnl": round((price_matrix.iloc[-1].get(cid, h["buy_px"]) - h["buy_px"]) * h["units"], 2)
            }
            for cid, h in holdings.items()
        ]

        return SlabBacktestResult(
            initial_capital=self.initial_capital,
            final_nav=round(final_nav, 2),
            total_return_pct=round(total_ret, 1),
            cagr_pct=round(cagr, 2),
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=round(sortino, 2),
            max_drawdown_pct=round(max_dd, 2),
            max_drawdown_duration_months=self._calc_max_dd_duration(eq_df),
            total_trades=total_trades,
            winning_trades=winning_trades,
            win_rate_pct=round(win_rate, 1),
            profit_factor=round(profit_factor, 2),
            total_fees_paid_eur=round(total_fees_paid, 2),
            equity_curve=eq_df,
            trades_history=trades_history,
            active_positions_end=active_positions_end
        )

    def _calc_max_dd_duration(self, eq_df: pd.DataFrame) -> int:
        """Calcola la massima durata del drawdown in mesi."""
        max_duration = 0
        current_duration = 0
        for dd in eq_df["drawdown_pct"]:
            if dd < -0.01:
                current_duration += 1
                if current_duration > max_duration:
                    max_duration = current_duration
            else:
                current_duration = 0
        return max_duration

    def _load_or_build_historical_matrix(
        self, universe: List[Dict[str, Any]], exclude_synthetic: bool = True
    ) -> pd.DataFrame:
        """
        Costruisce la matrice storica dei prezzi delle lastre (mensile 2021-08 -> 2026-09).

        ATTENZIONE — DATO FABBRICATO, non solo survivorship bias: per qualsiasi carta
        NON presente in historical_prices.csv (verificato: 6 delle 14 "curated grails"
        attualmente backtestate, cioè il 43%), il ramo else sotto NON stima una
        traiettoria plausibile da dati osservati — INVENTA una curva con np.linspace
        che sale matematicamente da 0.45x a 1.0x del prezzo PSA-10 di OGGI. Il prezzo
        sale per costruzione, non perché osservato: qualsiasi CAGR/Sharpe/drawdown
        calcolato includendo queste carte misura la performance di una strategia che
        compra dip e vende euforia su un percorso disegnato per salire. Non è
        evidenza di un edge reale.

        Con exclude_synthetic=True (default) queste carte vengono escluse dal
        backtest: i risultati riportati da SlabBacktester.run() usano SOLO le 8
        carte con storico realmente osservato in historical_prices.csv. Passare
        exclude_synthetic=False per includerle comunque (solo a scopo dimostrativo/
        UI, mai come evidenza di validazione).
        """
        from pathlib import Path
        csv_path = Path(__file__).resolve().parent.parent.parent / "data_cache" / "historical_prices.csv"
        df_raw = None
        if csv_path.exists():
            try:
                df_raw = pd.read_csv(csv_path, index_col=0)
            except Exception:
                pass

        if df_raw is not None and not df_raw.empty:
            dates = df_raw.index
        else:
            dates = pd.date_range(start="2021-08-01", end="2026-09-01", freq="MS").strftime("%Y-%m-%d")

        matrix_data = {}
        for card in universe:
            cid = card["card_id"]
            has_real_data = df_raw is not None and cid in df_raw.columns
            if not has_real_data and exclude_synthetic:
                continue
            if has_real_data:
                # Usa la serie storica reale (convertita in EUR)
                matrix_data[cid] = df_raw[cid].values / 1.08
            else:
                # FABBRICATO (vedi warning sopra) - traiettoria inventata, non osservata
                psa_10_target = card.get("psa_10_price_eur", 500.0)
                rel_date = card.get("release_date", "2021-08-27")
                
                # Generazione curva log-normale con cicli di rotazione
                n_steps = len(dates)
                base_start = psa_10_target * 0.45  # Inizia a sconto post-uscita
                growth_curve = np.linspace(base_start, psa_10_target, n_steps)
                # Aggiunge cicli di volatilità realistica dei collezionabili (+/- 8%)
                np.random.seed(42 + hash(cid) % 1000)
                noise = np.random.normal(1.0, 0.05, n_steps)
                matrix_data[cid] = np.maximum(base_start * 0.6, growth_curve * noise)

        return pd.DataFrame(matrix_data, index=dates)
