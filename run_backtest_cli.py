"""
run_backtest_cli.py — Esecutore da riga di comando per benchmark quantitativi veloci.
Confronta strategie su collezionabili Pokémon vs Benchmark di mercato tradizionali (S&P 500, Oro).
"""

from __future__ import annotations
import sys
import numpy as np
import pandas as pd

from poke_quant.data.storage import load_price_matrix, load_metadata
from poke_quant.data.price_fetcher import build_and_cache_universe
from poke_quant.engine.backtester import Backtester
from poke_quant.engine.strategies.sealed_accumulator import SealedAccumulatorStrategy
from poke_quant.engine.strategies.chase_dip_buyer import ChaseDipBuyerStrategy
from poke_quant.engine.strategies.optimal_sealed_strategy import OptimalSealedStrategy
from poke_quant.validation.statistical_validation import deflated_sharpe_ratio, pbo_cscv


def print_banner(title: str):
    print("\n" + "=" * 78)
    print(f"  {title.upper()}")
    print("=" * 78)


def format_pct(val: float) -> str:
    if np.isnan(val):
        return "N/A"
    return f"{val * 100:+.2f}%"


def format_ratio(val: float) -> str:
    if np.isnan(val) or np.isinf(val):
        return "N/A"
    return f"{val:.2f}"


def main():
    print_banner("PokeQuant — Audit Quantitativo Istituzionale Collezionabili Pokémon")
    
    # 1. Carica dati storici
    prices_df, metadata = build_and_cache_universe(force_refresh=False)
    print(f"• Dati storici caricati: {len(prices_df)} mesi ({prices_df.index[0].strftime('%Y-%m')} -> {prices_df.index[-1].strftime('%Y-%m')})")
    print(f"• Universo prodotti: {len(metadata)} asset reali (Booster Box, Chase Cards, Vintage)")
    print(f"• Piattaforma di vendita simulata: Cardmarket (fee 5% + imballaggio)")

    initial_capital = 10000.0

    # 2. Configura ed esegue Strategia 1: Optimal Sealed Strategy (Tier S/A, Mesi 4-14)
    strat_opt = OptimalSealedStrategy(
        allowed_tiers=["S", "A"],
        min_buy_age_months=4,
        max_buy_age_months=14,
        max_msrp_multiplier=1.15,
        min_hold_months=30,
        target_roi=1.50,
        max_hold_months=48,
        max_allocation_pct=0.20
    )
    bt_opt = Backtester(
        strategy=strat_opt,
        historical_prices_df=prices_df,
        items_metadata=metadata,
        initial_cash=initial_capital,
        platform="cardmarket",
        benchmark_cagr=0.10
    )
    res_opt = bt_opt.run()

    # 3. Configura ed esegue Strategia 2: Sealed Accumulator Standard
    strat_sealed = SealedAccumulatorStrategy(
        max_allocation_per_set_pct=0.25,
        max_buy_age_months=14,
        min_hold_months=20,
        target_profit_roi=0.80,
        msrp_max_multiplier=1.20
    )
    bt_sealed = Backtester(
        strategy=strat_sealed,
        historical_prices_df=prices_df,
        items_metadata=metadata,
        initial_cash=initial_capital,
        platform="cardmarket",
        benchmark_cagr=0.10
    )
    res_sealed = bt_sealed.run()

    # 4. Configura ed esegue Strategia 3: Chase Dip Buyer (Singole)
    strat_chase = ChaseDipBuyerStrategy(
        min_dip_months=4,
        max_dip_months=10,
        min_drop_from_launch_pct=0.15,
        target_profit_roi=0.50,
        max_allocation_per_card_pct=0.15
    )
    bt_chase = Backtester(
        strategy=strat_chase,
        historical_prices_df=prices_df,
        items_metadata=metadata,
        initial_cash=initial_capital,
        platform="cardmarket",
        benchmark_cagr=0.10
    )
    res_chase = bt_chase.run()

    # 5. Tabella comparativa delle metriche
    print_banner("Confronto Performance Netta vs Benchmark (S&P 500 @ 10% annuo)")
    results = [res_opt, res_sealed, res_chase]

    cols = ["Strategia", "Capitale Finale", "ROI Netto", "CAGR Netto", "Alpha vs S&P", "Sharpe", "MaxDD", "Calmar", "Trade Conclusi", "Win Rate"]
    rows = []
    for r in results:
        rows.append([
            r.strategy_name.replace("Strategy", ""),
            f"{r.final_nav:,.2f} €",
            format_pct(r.total_net_return),
            format_pct(r.cagr),
            format_pct(r.alpha_annualized),
            format_ratio(r.sharpe),
            format_pct(r.max_drawdown),
            format_ratio(r.calmar),
            r.total_trades,
            format_pct(r.win_rate)
        ])

    df_comp = pd.DataFrame(rows, columns=cols)
    print(df_comp.to_string(index=False))

    # 5. Dettagli Frizioni e Costi
    print_banner("Analisi di Attrito e Costi Reali")
    for r in results:
        print(f"\n[{r.strategy_name}]")
        print(f"  • PnL Netto generato: {r.total_net_pnl:,.2f} €")
        print(f"  • Commissioni & Frizioni pagate: {r.total_fees_paid:,.2f} €")
        if not r.trades_df.empty:
            print("  • Trade più redditizio:")
            best_t = r.trades_df.sort_values(by="net_pnl", ascending=False).iloc[0]
            print(f"    - {best_t['item_name']}: Acquisto {best_t['buy_price_unit']:.2f}€ ({best_t['buy_date']}) -> Vendita {best_t['sell_price_unit']:.2f}€ ({best_t['sell_date']}) | ROI Netto: {best_t['net_roi']*100:+.1f}%")

    # 6. Validazione anti-overfitting (DSR e PBO)
    print_banner("Validazione Statistica Anti-Overfitting (Bailey & Lopez de Prado)")
    # Converte i rendimenti delle due strategie in matrice per PBO
    common_idx = res_sealed.monthly_returns.index.intersection(res_chase.monthly_returns.index)
    perf_matrix = np.column_stack([
        res_sealed.monthly_returns.loc[common_idx].values,
        res_chase.monthly_returns.loc[common_idx].values
    ])

    # Calcola DSR assumendo 10 tentativi di selezione parametri
    dsr_sealed = deflated_sharpe_ratio(observed_sr=res_sealed.sharpe, n_trials=10, n_obs=len(res_sealed.monthly_returns))
    dsr_chase = deflated_sharpe_ratio(observed_sr=res_chase.sharpe, n_trials=10, n_obs=len(res_chase.monthly_returns))

    print(f"• Deflated Sharpe Ratio Sealed Accumulator (su 10 varianti): {dsr_sealed:.4f} (Confidenza: {dsr_sealed*100:.1f}%)")
    print(f"• Deflated Sharpe Ratio Chase Dip Buyer (su 10 varianti):   {dsr_chase:.4f} (Confidenza: {dsr_chase*100:.1f}%)")
    
    # PBO calcolato su blocchi di 4 se T è sufficiente
    t_len = len(perf_matrix)
    splits = 4 if t_len >= 16 else 2
    rem = t_len % splits
    if rem > 0:
        perf_matrix = perf_matrix[rem:, :]
    pbo = pbo_cscv(perf_matrix, n_splits=splits)
    print(f"• Probability of Backtest Overfitting (PBO / CSCV):         {pbo:.2f} ({pbo*100:.1f}%)")
    print(f"  (Un PBO inferiore al 30% indica che l'Edge in-sample si trasferisce con elevata probabilità out-of-sample).")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
