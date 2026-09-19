"""
poke_quant/engine/backtester.py — Motore di simulazione e backtesting a eventi discreti.
Esegue la strategia su serie temporali storiche reali, simulando la corretta sequenza di
acquisti, vendite, incassi al netto delle commissioni, calcolo del NAV e metriche di rischio.
"""

from __future__ import annotations
from dataclasses import dataclass
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
        benchmark_cagr: float = 0.10  # Benchmark di confronto (es. 10% annuo S&P 500)
    ):
        self.strategy = strategy
        self.historical_prices = historical_prices_df.sort_index()
        self.items_metadata = items_metadata
        self.initial_cash = initial_cash
        self.platform = platform
        self.seller_absorbs_shipping = seller_absorbs_shipping
        self.benchmark_cagr = benchmark_cagr

    def run(self) -> BacktestResult:
        portfolio = Portfolio(initial_cash=self.initial_cash)
        dates = self.historical_prices.index

        for date in dates:
            date_str = pd.to_datetime(date).strftime("%Y-%m-%d")
            price_row = self.historical_prices.loc[date]

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

            # 1. Genera segnali di trading dalla strategia
            signals = self.strategy.generate_signals(date_str, portfolio, market_snapshot)

            # 2. Esegue prima le VENDITE (per liberare liquidità prima degli acquisti)
            for sig in signals:
                if sig.action == "SELL":
                    portfolio.sell(
                        item_id=sig.item_id,
                        quantity=sig.quantity,
                        unit_gross_price=sig.target_price,
                        date=date_str,
                        platform=self.platform,
                        seller_absorbs_shipping=self.seller_absorbs_shipping
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

        # Benchmark: rendimento mensile corrispondente al CAGR annuo target
        bench_monthly_rate = (1.0 + self.benchmark_cagr) ** (1.0 / 12.0) - 1.0
        bench_returns = pd.Series(bench_monthly_rate, index=monthly_returns.index)
        bench_nav = self.initial_cash * (1.0 + bench_returns).cumprod()

        # Metriche quantitative istituzionali
        strat_cagr = cagr(monthly_returns, periods_per_year=12)
        strat_sharpe = sharpe(monthly_returns, rf_annual=0.03, periods_per_year=12)
        strat_max_dd = max_drawdown(monthly_returns)
        strat_calmar = calmar(monthly_returns, periods_per_year=12)
        strat_sortino = sortino_ratio(monthly_returns, rf_annual=0.03, periods_per_year=12)
        strat_ulcer = ulcer_index(monthly_returns)

        # Alpha netto annualizzato rispetto al benchmark
        alpha_annual = strat_cagr - self.benchmark_cagr

        # Statistiche sui singoli trade
        trades_df = portfolio.get_trades_df()
        trade_stats = compute_trade_metrics(trades_df)

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
            alpha_annualized=alpha_annual
        )
