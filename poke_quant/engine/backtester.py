"""
poke_quant/engine/backtester.py — Motore di simulazione e backtesting a eventi discreti.
Esegue la strategia su serie temporali storiche reali, simulando la corretta sequenza di
acquisti, vendite, incassi al netto delle commissioni, calcolo del NAV e metriche di rischio.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from poke_quant.engine.portfolio import Portfolio
from poke_quant.validation.metrics import (
    cagr, sharpe, max_drawdown, calmar, sortino_ratio, ulcer_index, compute_trade_metrics
)


@dataclass
class BacktestResult:
    strategy_name: str
    initial_cash: float
    final_nav: float
    total_net_pnl: float
    total_net_return: float
    cagr: float
    sharpe: float
    max_drawdown: float
    calmar: float
    sortino: float
    ulcer_index: float
    total_trades: int
    win_rate: float
    profit_factor: float
    total_fees_paid: float
    monthly_returns: pd.Series
    nav_history: pd.DataFrame
    trades_df: pd.DataFrame
    benchmark_monthly_returns: pd.Series
    benchmark_nav: pd.Series
    alpha_annualized: float
    beta: float = 0.0
    correlation_benchmark: float = 0.0
    regime_performance: Dict[str, float] = field(default_factory=dict)
    signals_history: List[Dict[str, Any]] = field(default_factory=list)
    open_positions: List[Dict[str, Any]] = field(default_factory=list)
    turnover_ratio: float = 0.0
    rotation_trades_count: int = 0
    monthly_cash_injection: float = 0.0


class Backtester:
    """Motore di backtesting per strategie di investimento su collezionabili Pokémon."""

    def __init__(
        self,
        strategy: Any,
        historical_prices_df: pd.DataFrame,
        items_metadata: Dict[str, Dict[str, Any]],
        initial_cash: float = 10000.0,
        platform: str = "cardmarket",
        seller_absorbs_shipping: bool = False,
        benchmark_cagr: float = 0.10,  # Fallback a 10% annuo
        benchmark_series: Optional[pd.Series] = None,  # Serie reale S&P 500 / Gold / BTC
        apply_liquidity_slippage: bool = False,
        apply_holding_cost: bool = False,
        monthly_cash_injection: float = 0.0
    ):
        self.strategy = strategy
        self.historical_prices = historical_prices_df.sort_index()
        self.items_metadata = items_metadata
        self.initial_cash = initial_cash
        self.platform = platform
        self.seller_absorbs_shipping = seller_absorbs_shipping
        self.benchmark_cagr = benchmark_cagr
        self.benchmark_series = benchmark_series
        self.apply_liquidity_slippage = apply_liquidity_slippage
        self.apply_holding_cost = apply_holding_cost
        self.monthly_cash_injection = monthly_cash_injection

    def run(self) -> BacktestResult:
        if hasattr(self.strategy, "reset"):
            self.strategy.reset()

        portfolio = Portfolio(initial_cash=self.initial_cash)
        dates = self.historical_prices.index
        all_signals: List[Dict[str, Any]] = []

        for date in dates:
            date_str = pd.to_datetime(date).strftime("%Y-%m-%d")
            price_row = self.historical_prices.loc[date]

            # Iniezione periodica di liquidità (PAC / DCA opzionale)
            if self.monthly_cash_injection > 0:
                portfolio.cash += self.monthly_cash_injection

            # Costruisce lo snapshot di mercato per la data corrente
            market_snapshot: Dict[str, Dict[str, Any]] = {}
            current_prices: Dict[str, float] = {}

            for item_id, meta in self.items_metadata.items():
                if item_id in price_row and not np.isnan(price_row[item_id]) and price_row[item_id] > 0:
                    px = float(price_row[item_id])
                    current_prices[item_id] = px
                    snap = dict(meta)
                    snap["current_price"] = px
                    market_snapshot[item_id] = snap

            nav_before = portfolio.get_total_nav(current_prices)
            cash_before = portfolio.cash

            # 1. Genera segnali di trading dalla strategia
            signals = self.strategy.generate_signals(date_str, portfolio, market_snapshot)

            # Registra la cronologia completa dei segnali generati
            for sig in signals:
                meta_sig = self.items_metadata.get(sig.item_id, {})
                msrp_sig = float(meta_sig.get("msrp") or 140.0)
                max_buy_px = round(msrp_sig * 1.15, 2)
                all_signals.append({
                    "date": date_str,
                    "action": sig.action,
                    "item_id": sig.item_id,
                    "item_name": sig.item_name,
                    "item_type": sig.item_type,
                    "quantity": sig.quantity,
                    "price": sig.target_price,
                    "msrp": msrp_sig,
                    "max_buy_price": max_buy_px,
                    "total_value": sig.quantity * sig.target_price,
                    "reason": sig.reason,
                    "portfolio_cash_before": cash_before,
                    "portfolio_nav_before": nav_before
                })

            # 2. Esegue prima le VENDITE (per liberare liquidità prima degli acquisti)
            for sig in signals:
                if sig.action == "SELL":
                    meta = self.items_metadata.get(sig.item_id, {})
                    slip = meta.get("slippage_pct", 0.0) if self.apply_liquidity_slippage else 0.0
                    carry = 0.0
                    if self.apply_holding_cost and sig.item_id in portfolio.positions:
                        pos = portfolio.positions[sig.item_id]
                        try:
                            d_buy = pd.to_datetime(pos.buy_date)
                            d_sell = pd.to_datetime(date_str)
                            m_held = max(1.0, (d_sell.year - d_buy.year) * 12 + (d_sell.month - d_buy.month))
                        except Exception:
                            m_held = 1.0
                        annual_carry = meta.get("holding_cost_annual_pct", 0.005)
                        carry = (pos.buy_price_unit * sig.quantity) * (annual_carry / 12.0) * m_held

                    portfolio.sell(
                        item_id=sig.item_id,
                        quantity=sig.quantity,
                        unit_gross_price=sig.target_price,
                        date=date_str,
                        platform=self.platform,
                        seller_absorbs_shipping=self.seller_absorbs_shipping,
                        slippage_pct=slip,
                        carrying_cost=carry
                    )

            # 3. Esegue poi gli ACQUISTI
            for sig in signals:
                if sig.action == "BUY":
                    portfolio.buy(
                        item_id=sig.item_id,
                        item_name=sig.item_name,
                        item_type=sig.item_type,
                        quantity=sig.quantity,
                        unit_price=sig.target_price,
                        date=date_str
                    )

            # 4. Registra lo stato di fine periodo
            portfolio.record_snapshot(date_str, current_prices)

        # Calcolo serie storiche NAV e rendimenti
        history_df = portfolio.get_history_df()
        if history_df.empty or len(history_df) < 2:
            raise ValueError("Cronologia del backtest insufficiente per calcolare metriche.")

        monthly_returns = history_df["nav"].pct_change().dropna()
        final_nav = float(history_df["nav"].iloc[-1])
        total_pnl = final_nav - self.initial_cash
        total_return = total_pnl / self.initial_cash

        # Benchmark: serie reale se fornita, altrimenti tasso fisso
        if self.benchmark_series is not None:
            aligned_bench = self.benchmark_series.reindex(history_df.index).ffill().bfill()
            bench_returns = aligned_bench.pct_change().dropna()
            bench_nav = (aligned_bench / aligned_bench.iloc[0]) * self.initial_cash
            bench_cagr = cagr(bench_returns, periods_per_year=12)
        else:
            bench_monthly_rate = (1.0 + self.benchmark_cagr) ** (1.0 / 12.0) - 1.0
            bench_returns = pd.Series(bench_monthly_rate, index=monthly_returns.index)
            bench_nav = self.initial_cash * (1.0 + bench_returns).cumprod()
            bench_cagr = self.benchmark_cagr

        # Metriche quantitative istituzionali
        strat_cagr = cagr(monthly_returns, periods_per_year=12)
        strat_sharpe = sharpe(monthly_returns, rf_annual=0.03, periods_per_year=12)
        strat_max_dd = max_drawdown(monthly_returns)
        strat_calmar = calmar(monthly_returns, periods_per_year=12)
        strat_sortino = sortino_ratio(monthly_returns, rf_annual=0.03, periods_per_year=12)
        strat_ulcer = ulcer_index(monthly_returns)

        # Alpha netto annualizzato rispetto al benchmark
        alpha_annual = strat_cagr - bench_cagr

        # Statistiche sui singoli trade
        trades_df = portfolio.get_trades_df()
        trade_stats = compute_trade_metrics(trades_df)

        # Calcolo Beta e Correlazione vs Benchmark
        common_idx = monthly_returns.index.intersection(bench_returns.index)
        if len(common_idx) >= 4 and bench_returns.loc[common_idx].var() > 0:
            cov = float(np.cov(monthly_returns.loc[common_idx], bench_returns.loc[common_idx])[0, 1])
            var_b = float(bench_returns.loc[common_idx].var())
            beta = cov / var_b if var_b > 0 else 0.0
            corr_mat = np.corrcoef(monthly_returns.loc[common_idx], bench_returns.loc[common_idx])
            corr = float(corr_mat[0, 1]) if not np.isnan(corr_mat[0, 1]) else 0.0
        else:
            beta = 0.0
            corr = 0.0

        # Calcolo Rendimenti per Regime Macro
        regimes = {
            "BULL_HYP (2021)": ("2021-02-01", "2021-11-30"),
            "BEAR_MACRO (2022)": ("2021-12-01", "2022-12-31"),
            "ACCUMULATION (2023)": ("2023-01-01", "2023-12-31"),
            "SELECTIVE_EXPANSION (2024-2026)": ("2024-01-01", "2026-09-30")
        }
        regime_perf: Dict[str, float] = {}
        for r_name, (start_r, end_r) in regimes.items():
            sub_rets = monthly_returns.loc[start_r:end_r]
            if not sub_rets.empty:
                cum_ret = float((1.0 + sub_rets).prod() - 1.0)
                regime_perf[r_name] = cum_ret

        # Calcolo Posizioni Aperte (Inventario Attivo in Detenzione a Fine Simulazione)
        open_positions: List[Dict[str, Any]] = []
        last_price_row = self.historical_prices.iloc[-1]
        last_date_str = pd.to_datetime(self.historical_prices.index[-1]).strftime("%Y-%m-%d")
        last_dt = pd.to_datetime(last_date_str)
        for item_id, pos in portfolio.positions.items():
            cur_px = float(last_price_row.get(item_id, pos.buy_price_unit))
            mkt_val = cur_px * pos.quantity
            unrealized_pnl = mkt_val - pos.total_cost
            unrealized_roi = unrealized_pnl / pos.total_cost if pos.total_cost > 0 else 0.0
            try:
                b_dt = pd.to_datetime(pos.buy_date)
                held_m = max(1, (last_dt.year - b_dt.year) * 12 + (last_dt.month - b_dt.month))
            except Exception:
                held_m = 1

            # Informazioni su MSRP, Prezzo Massimo d'Acquisto e Finestra di Acquisto
            meta = self.items_metadata.get(item_id, {})
            msrp = float(meta.get("msrp") or 140.0)
            max_buy_price = round(msrp * 1.15, 2)
            rel_str = meta.get("release_date")
            if rel_str:
                try:
                    rel_dt = pd.to_datetime(rel_str)
                    set_age_m = max(0, (last_dt.year - rel_dt.year) * 12 + (last_dt.month - rel_dt.month))
                except Exception:
                    set_age_m = held_m + 6
            else:
                set_age_m = held_m + 6

            months_left = max(0, 14 - set_age_m)
            is_in_window = (4 <= set_age_m <= 14) and (cur_px <= max_buy_price)

            if is_in_window:
                if set_age_m <= 11:
                    w_status = f"🟢 IN FINESTRA ({months_left}m rimasti)"
                else:
                    w_status = f"⏳ IN CHIUSURA ({months_left}m rimasti)"
                w_verdict = f"Accumulabile: Prezzo {cur_px:.2f}€ <= Max {max_buy_price:.2f}€ (MSRP {msrp:.2f}€)"
            elif set_age_m < 4:
                w_status = f"🟡 RECENTE ({4 - set_age_m}m a reprint)"
                w_verdict = "Set in lancio, attendere finestra ristampe (Mesi 4-14)"
            elif set_age_m <= 14 and cur_px > max_buy_price:
                w_status = "⚠️ SOPRA PREZZO MAX"
                w_verdict = f"Prezzo {cur_px:.2f}€ > Limite Max {max_buy_price:.2f}€ (+15% MSRP)"
            else:
                w_status = "🔒 FINESTRA CHIUSA (OOP)"
                w_verdict = f"Età {set_age_m}m > 14m: Set Out-of-Print. Solo holding/vendita."

            open_positions.append({
                "item_id": item_id,
                "item_name": pos.item_name,
                "item_type": pos.item_type,
                "quantity": pos.quantity,
                "buy_date": pos.buy_date,
                "buy_price_unit": pos.buy_price_unit,
                "total_cost": pos.total_cost,
                "current_price": cur_px,
                "current_value": mkt_val,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_roi": unrealized_roi,
                "holding_months": held_m,
                "msrp": msrp,
                "max_buy_price": max_buy_price,
                "set_age_months": set_age_m,
                "is_in_buy_window": is_in_window,
                "months_left_in_window": months_left,
                "window_status": w_status,
                "window_verdict": w_verdict
            })

        # Metriche di Rotazione del Capitale e Turnover
        total_gross_sales = float(sum(t.gross_proceeds for t in portfolio.closed_trades))
        avg_nav = float(history_df["nav"].mean())
        turnover_ratio = total_gross_sales / avg_nav if avg_nav > 0 else 0.0
        rot_count = sum(1 for s in all_signals if "Tranche 1" in s.get("reason", ""))

        return BacktestResult(
            strategy_name=self.strategy.__class__.__name__,
            initial_cash=self.initial_cash,
            final_nav=final_nav,
            total_net_pnl=total_pnl,
            total_net_return=total_return,
            cagr=strat_cagr,
            sharpe=strat_sharpe,
            max_drawdown=strat_max_dd,
            calmar=strat_calmar,
            sortino=strat_sortino,
            ulcer_index=strat_ulcer,
            total_trades=trade_stats["total_trades"],
            win_rate=trade_stats["win_rate"],
            profit_factor=trade_stats["profit_factor"],
            total_fees_paid=trade_stats["total_fees_paid"],
            monthly_returns=monthly_returns,
            nav_history=history_df,
            trades_df=trades_df,
            benchmark_monthly_returns=bench_returns,
            benchmark_nav=bench_nav,
            alpha_annualized=alpha_annual,
            beta=beta,
            correlation_benchmark=corr,
            regime_performance=regime_perf,
            signals_history=all_signals,
            open_positions=open_positions,
            turnover_ratio=turnover_ratio,
            rotation_trades_count=rot_count,
            monthly_cash_injection=self.monthly_cash_injection
        )
