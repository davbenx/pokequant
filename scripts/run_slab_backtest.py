#!/usr/bin/env python3
"""
scripts/run_slab_backtest.py — CLI per il Backtest Storico e Suite di Robustezza delle Carte Gradate.
Esegue la simulazione 2021-2026, calcola le metriche istituzionali (CAGR, Sharpe, Sortino, MaxDD),
e lancia la suite di invalidazione popperiana (H1, H2, H3, H4) + DSR + PBO.
"""

import sys
import os
from pathlib import Path

# Assicura importazione di poke_quant
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.slabs.slab_backtester import SlabBacktester
from poke_quant.slabs.slab_falsification import run_popperian_falsification_suite


def main():
    print("\n" + "=" * 80)
    print("  💎 POKEQUANT — BACKTEST STORICO E SUITE DI ROBUSTEZZA SLABS (2021-2026)")
    print("=" * 80)

    print("\n[1/3] Esecuzione simulazione cronologica mensile (Frizioni reali Cardmarket + Slippage)...")
    bt = SlabBacktester(initial_capital=5000.0, max_card_allocation_pct=0.30)
    res = bt.run()

    print("\n" + "-" * 80)
    print("  RISULTATI BACKTEST STRATEGIA SLABS (PSA · BGS · CGC)")
    print("-" * 80)
    print(f"  Capitale Iniziale:           {res.initial_capital:,.2f} €")
    print(f"  Valore Finale NAV (2026):    {res.final_nav:,.2f} €")
    print(f"  Rendimento Totale Cumulato:  +{res.total_return_pct:.2f}%")
    print(f"  CAGR (Tasso Annuo Composto): +{res.cagr_pct:.2f}%")
    print(f"  Sharpe Ratio Annualizzato:   {res.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio (Downside):    {res.sortino_ratio:.2f}")
    print(f"  Max Drawdown:                {res.max_drawdown_pct:.2f}%")
    print(f"  Durata Max Drawdown:         {res.max_drawdown_duration_months} mesi")
    print(f"  Operazioni Eseguite:         {res.total_trades} (Vinte: {res.winning_trades})")
    print(f"  Win Rate:                    {res.win_rate_pct:.1f}%")
    print(f"  Profit Factor:               {res.profit_factor:.2f}")
    print(f"  Commissioni e Spedizioni:    {res.total_fees_paid_eur:.2f} €")
    print("-" * 80)

    if res.trades_history:
        print("\n[2/3] Registro Operazioni Eseguite (Ultime 5 o più rilevanti):")
        for idx, t in enumerate(res.trades_history[:8], 1):
            g_str = t.grade.value if hasattr(t.grade, "value") else str(t.grade)
            print(f"  {idx}. {t.card_name} ({g_str}) x{t.quantity}")
            print(f"     Acquisto: {t.buy_date} a {t.buy_price_eur:.1f} € | Vendita: {t.sell_date} a {t.sell_price_eur:.1f} €")
            print(f"     Net PnL: {t.net_pnl_eur:+.1f} € (ROI: {t.net_roi_pct:+.1f}%) dopo {t.holding_months} mesi")
            print(f"     Trigger: {t.exit_reason}")
            print()

    print("\n[3/3] Esecuzione Suite di Invalidazione Popperiana & Test Bias (López de Prado)...")
    fals_report = run_popperian_falsification_suite(base_result=res, num_mc_simulations=100)

    print("\n" + "=" * 80)
    print("  SUITE DI INVALIDAZIONE POPPERIANA (STRESS TEST DI ROTTURA)")
    print("=" * 80)
    for t in fals_report["popperian_stress_tests"]:
        status = "✅ PASSATO" if t["passed"] else "❌ FALLITO"
        print(f"  {status} — {t['test_name']}")
        print(f"     Dettaglio: {t['conclusion']}\n")

    print("-" * 80)
    print("  METRICHE DI ROBUSTEZZA ISTITUZIONALI E CONTROLLO DEI BIAS")
    print("-" * 80)
    dsr = fals_report["dsr"]
    dsr_status = "✅ PASSATO" if dsr["passed"] else "❌ FALLITO"
    print(f"  {dsr_status} — Deflated Sharpe Ratio (DSR): {dsr['score']:.4f} (Soglia >= {dsr['threshold']})")
    print(f"     {dsr['interpretation']}\n")

    pbo = fals_report["pbo"]
    pbo_status = "✅ PASSATO" if pbo["passed"] else "❌ FALLITO"
    print(f"  {pbo_status} — Probability of Backtest Overfitting (PBO): {pbo['score']:.4f} (Soglia < {pbo['threshold']})")
    print(f"     {pbo['interpretation']}\n")

    surv = fals_report["anti_survivorship_bias"]
    surv_status = "✅ PASSATO" if surv["passed"] else "❌ FALLITO"
    print(f"  {surv_status} — Controllo Anti-Survivorship Bias:")
    print(f"     {surv['conclusion']}\n")

    overall = "✅ STRATEGIA SCIENTIFICAMENTE ROBUSTA" if fals_report["all_popperian_tests_passed"] else "⚠️ REVISIONE RICHIESTA"
    print("=" * 80)
    print(f"  VERDETTO FINALE: {overall}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
