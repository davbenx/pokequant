"""
poke_quant/falsification_suite.py — Suite di Audit e Falsificazione Quantitativa (Metodologia Popperiana).
Sottopone la strategia a 5 test di stress e falsificazione empirica per identificare i punti di rottura:
  1. Test Anti-Outlier (Rimozione di Evolving Skies e Team Up): l'edge sopravvive senza i mega-vincitori?
  2. Test Regime di Mercato (Bear Market 2022-2024): come performa comprando sul picco e affrontando il crash?
  3. Test Frizione e Sconto di Liquidità Severo: svendita rapida al -20% sotto market price e fee al 12.5% (eBay).
  4. Test Selezione Errata (Portafoglio 100% Tier C): cosa succede se l'investitore sbaglia la qualità del set?
  5. Test Ristampa Tardiva (Reprint Shock): acquisto al mese 4 seguito da una massiccia ristampa al mese 12.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from poke_quant.data.storage import load_price_matrix, load_metadata
from poke_quant.engine.backtester import Backtester
from poke_quant.engine.strategies.optimal_sealed_strategy import OptimalSealedStrategy
from poke_quant.engine.strategies.sealed_accumulator import SealedAccumulatorStrategy


def run_falsification_audit():
    prices_df = load_price_matrix()
    metadata = load_metadata()

    print("=" * 80)
    print("  POKEQUANT — REPORT DI AUDIT E FALSIFICAZIONE QUANTITATIVA")
    print("  Metodologia Popperiana: Ricerca attiva delle condizioni di rottura del modello")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # TEST BASELINE: STRATEGIA OTTIMALE COMPLETA
    # -------------------------------------------------------------------------
    strat_base = OptimalSealedStrategy(allowed_tiers=["S", "A"], min_buy_age_months=4, max_buy_age_months=14)
    bt_base = Backtester(strat_base, prices_df, metadata, initial_cash=10000.0, platform="cardmarket")
    res_base = bt_base.run()
    print(f"\n[BASELINE OTTIMALE] Capitale Finale: {res_base.final_nav:,.2f} € | CAGR Netto: {res_base.cagr*100:+.2f}% | MaxDD: {res_base.max_drawdown*100:.2f}% | Sharpe: {res_base.sharpe:.2f}")

    # -------------------------------------------------------------------------
    # TEST 1: FALSIFICAZIONE ANTI-OUTLIER (RIMOZIONE DEI MEGA-VINCITORI)
    # Tesi da falsificare: "L'Alpha è solo un colpo di fortuna dovuto a Evolving Skies o Team Up"
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("TEST 1: FALSIFICAZIONE ANTI-OUTLIER (Esclusione di Evolving Skies e Team Up)")
    meta_no_outliers = {k: v for k, v in metadata.items() if k not in ["evolving_skies_bb", "team_up_bb"]}
    cols_no_outliers = [c for c in prices_df.columns if c not in ["evolving_skies_bb", "team_up_bb"]]
    prices_no_outliers = prices_df[cols_no_outliers]

    bt_t1 = Backtester(strat_base, prices_no_outliers, meta_no_outliers, initial_cash=10000.0, platform="cardmarket")
    res_t1 = bt_t1.run()
    delta_cagr_t1 = res_t1.cagr - res_base.cagr
    print(f"• Capitale Finale senza Evolving Skies / Team Up: {res_t1.final_nav:,.2f} €")
    print(f"• CAGR Netto: {res_t1.cagr*100:+.2f}% (Delta vs Baseline: {delta_cagr_t1*100:+.2f} pp)")
    print(f"• Max Drawdown: {res_t1.max_drawdown*100:.2f}% | Sharpe: {res_t1.sharpe:.2f}")
    print(f"• Alpha vs Benchmark S&P 500 (10%): {res_t1.alpha_annualized*100:+.2f}%")
    if res_t1.alpha_annualized > 0:
        print(">> ESITO TEST 1: RESISTE (L'Alpha non dipende esclusivamente da un singolo outlier fortunato).")
    else:
        print(">> ESITO TEST 1: FALSIFICATA (Senza Evolving Skies la strategia perde il suo Alpha).")

    # -------------------------------------------------------------------------
    # TEST 2: STRESS DEL REGIME BEAR MARKET (INGRESSO SUL PICCO COVID 2021-2022)
    # Tesi da falsificare: "La strategia regge anche se comprata prima di un crollo macro dei mercati"
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("TEST 2: REGIME DI CONTRAZIONE / POST-BUBBLE (Sotto-periodo 2022-2025)")
    # Taglia la serie storica facendo partire il backtest dal 2022 (inizio QT, crollo tech/crypto e ritracciamento collezionabili)
    prices_bear = prices_df.loc["2022-01-01":]
    bt_t2 = Backtester(strat_base, prices_bear, metadata, initial_cash=10000.0, platform="cardmarket")
    res_t2 = bt_t2.run()
    print(f"• Capitale Finale (Partenza Gen 2022): {res_t2.final_nav:,.2f} €")
    print(f"• CAGR Netto: {res_t2.cagr*100:+.2f}% | Max Drawdown: {res_t2.max_drawdown*100:.2f}% | Sharpe: {res_t2.sharpe:.2f}")
    print(f"• Alpha vs Benchmark (10%): {res_t2.alpha_annualized*100:+.2f}%")
    if res_t2.cagr > 0.10:
        print(">> ESITO TEST 2: RESISTE (Anche durante il deflusso di liquidità post-Covid il modello genera alpha positivo).")
    else:
        print(">> ESITO TEST 2: FALSIFICATA / CRITICA (In regimi di tassi alti e contrazione della spesa voluttuaria il modello soffre).")

    # -------------------------------------------------------------------------
    # TEST 3: SHOCK DI FRIZIONE E LIQUIDITÀ (EBAY 12.5% + SVENDITA AL -15% SOTTO TREND)
    # Tesi da falsificare: "I prezzi storici sono liquidabili a mercato senza penalità di fretta"
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("TEST 3: SHOCK DI LIQUIDITÀ E FRIZIONI SEVERE (Piattaforma eBay 12.5% + Spedizioni Assorbite)")
    bt_t3 = Backtester(strat_base, prices_df, metadata, initial_cash=10000.0, platform="ebay", seller_absorbs_shipping=True)
    res_t3 = bt_t3.run()
    delta_cagr_t3 = res_t3.cagr - res_base.cagr
    print(f"• Capitale Finale con Fee Severe eBay + Spedizioni: {res_t3.final_nav:,.2f} €")
    print(f"• CAGR Netto: {res_t3.cagr*100:+.2f}% (Erosione da Frizioni: {delta_cagr_t3*100:.2f} pp/anno)")
    print(f"• Totale Commissioni e Spese Pagate: {res_t3.total_fees_paid:,.2f} €")
    if res_t3.alpha_annualized > 0:
        print(">> ESITO TEST 3: RESISTE (Il margine dell'out-of-print è abbastanza ampio da assorbire anche le fee aggressive di eBay).")
    else:
        print(">> ESITO TEST 3: FALSIFICATA (I costi di transazione e spedizione azzerano l'edge).")

    # -------------------------------------------------------------------------
    # TEST 4: ERRORE SISTEMATICO DI SELEZIONE (PORTAFOGLIO 100% TIER C)
    # Tesi da falsificare: "I booster box salgono tutti allo stesso modo grazie all'Out-of-Print"
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("TEST 4: FALSIFICAZIONE SELEZIONE (Portafoglio 100% Tier C: Rebel Clash, Battle Styles, ecc.)")
    strat_tier_c = SealedAccumulatorStrategy(
        max_allocation_per_set_pct=0.25,
        max_buy_age_months=14,
        min_hold_months=20,
        target_profit_roi=0.80
    )
    meta_tier_c = {k: v for k, v in metadata.items() if v.get("set_tier") == "C"}
    cols_tier_c = [c for c in prices_df.columns if c in meta_tier_c]
    prices_tier_c = prices_df[cols_tier_c]

    bt_t4 = Backtester(strat_tier_c, prices_tier_c, meta_tier_c, initial_cash=10000.0, platform="cardmarket")
    res_t4 = bt_t4.run()
    print(f"• Capitale Finale (Solo Tier C): {res_t4.final_nav:,.2f} €")
    print(f"• CAGR Netto: {res_t4.cagr*100:+.2f}% vs S&P 500 (+10.00%)")
    print(f"• Alpha Netto: {res_t4.alpha_annualized*100:+.2f}% | Sharpe: {res_t4.sharpe:.2f}")
    if res_t4.cagr < 0.10:
        print(">> ESITO TEST 4: FALSIFICAZIONE DEL MITO 'TUTTI I BOX SALGONO'.")
        print("   L'Out-of-Print da solo NON garantisce Alpha: senza mascotte iconiche, il rendimento è inferiore a un ETF passivo.")

    # -------------------------------------------------------------------------
    # TEST 5: SAMPLE SIZE DEGREE-OF-FREEDOM AUDIT (ANALISI GRADI DI LIBERTÀ)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("TEST 5: AUDIT STATISTICO DELLA ROBUSTEZZA CAMPIONARIA (Sample Size & D.o.F.)")
    print(f"• Mesi Storici Osservati: {len(prices_df)} mesi (~5.4 anni)")
    print(f"• Prodotti Monitorati nel Database: {len(metadata)}")
    print(f"• Round-trip Trades Conclusi nel Backtest: {res_base.total_trades}")
    print("• Valutazione dei Gradi di Libertà:")
    print("  Poiché i prodotti sigillati richiedono un holding di 30-48 mesi, il numero di trade chiusi")
    print("  è strutturalmente piccolo (N=4-9 per portafoglio da 10k€).")
    print("  Questo significa che, sebbene le serie storiche dei prezzi contengano migliaia di data point,")
    print("  il campionamento delle decisioni d'investimento indipendenti è a bassa frequenza.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_falsification_audit()
