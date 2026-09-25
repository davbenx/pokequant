#!/usr/bin/env python3
"""
scripts/rarity_premium_clean_control_retest.py — Ritesta il fattore "rarità premium"
(PREMIUM_RARITIES, RarityTierFactorStrategy default) dopo aver ampliato il campione di
controllo casuale (scripts/discover_random_control_singles.py --per-set 30, invece del
default 5) per ridurre la contaminazione già documentata in rarity_tier_factor.py e
scripts/retest_promo_premium_illustrator_current_universe.py: su 653 carte, 241
eleggibili per rarità premium, il 96% veniva dal campione chase (selezionato sul prezzo
corrente, survivorship bias) e solo il 4% dal controllo casuale - troppo pochi per
isolare un effetto indipendente dal survivorship bias di discover_chase_cards.py.

Questo NON è un nuovo fattore da validare - è la RIPETIZIONE dello stesso test con un
campione di controllo più grande, per vedere se la conclusione (probabile riscoperta del
survivorship bias, non un premio di rarità indipendente) cambia una volta che il
controllo casuale pesa di più nella composizione.

Riporta la composizione (chase vs controllo) SIA prima che dopo l'ampliamento, poi il
verdetto completo (Sharpe/DSR/PBO/bootstrap/walk-forward) su:
  (a) l'universo intero (chase + controllo, come lo vede la produzione)
  (b) SOLO il campione di controllo (isolando l'effetto dal survivorship bias per
      costruzione - se il fattore "rarità premium" è un premio strutturale reale, deve
      mostrarsi anche qui, non solo dove è mescolato al campione chase)

ESITO: NON VALIDATO, CHIUSO IN MODO CONCLUSIVO. Composizione dopo l'ampliamento: 280
carte eleggibili (era 241), 69% da chase / 31% da controllo (era 96%/4%) - la
contaminazione è ridotta ma non eliminata, ma ora il campione di controllo puro (87
carte premium, zero survivorship bias) è grande abbastanza per un verdetto
indipendente:

  (a) Universo intero: Sharpe 0.69 | CAGR +15.84% | MaxDD -31.14% | DSR(72) 0.215 |
      walk-forward H1 -0.56 -> H2 +2.07
  (b) SOLO controllo puro (87 carte): Sharpe 0.81 | CAGR +16.49% | MaxDD -34.44% |
      DSR(72) 0.303 | walk-forward H1 -0.99 -> H2 +2.73
  Bootstrap (a): CAGR mediana +14.5% [90% CI -3.3%,+40.0%], P(CAGR>0)=89%;
      Sharpe mediana 0.81, P(Sharpe>0)=92%

Il dubbio "campione troppo piccolo per essere conclusivo" lasciato aperto nel
docstring di rarity_tier_factor.py è risolto: NON è (solo) una riscoperta del
survivorship bias di discover_chase_cards.py - anche isolato dalla contaminazione, il
fattore mostra lo stesso identico schema boom/bust di ogni altro fattore categoriale
già respinto in questa ricerca (illustratore, promo, mascotte, popolarità specie),
con un DSR ben sotto qualsiasi soglia usata altrove (0.90-0.95) sia mescolato che
isolato. I bootstrap P(>0) spettacolari (89-92%) sono lo stesso schema ingannevole già
visto sul fattore promo - non bastano quando il walk-forward mostra un'inversione di
segno netta tra le due metà del campione.

NOTA METODOLOGICA IMPORTANTE (trovata in corso, non specifica a questo fattore):
l'ampliamento del campione di controllo casuale ha inizialmente introdotto migliaia di
salti mensili di prezzo estremi e irrealistici (fino a +1726% in un mese, quasi tutti
su carte comuni economiche del campione di controllo) perché scripts/flag_unreliable_
assets.py (il filtro anti-salto-estremo) non era mai stato rieseguito dopo l'aggiunta
delle nuove carte - un passo mancante, non un nuovo tipo di bug. Rieseguito prima di
questo test (543 carte flaggate, 508 singole) - i numeri sopra sono POST-fix. Lo
stesso problema ha gonfiato in modo molto più serio la strategia di produzione
(scarcity/value) proprio perché quella compra in quantità multiple per trade - vedi
scripts/max_quantity_retest_expanded_universe.py.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.data.liquidity_filter import liquid_singles_ids
from poke_quant.engine.backtester import Backtester
from poke_quant.engine.strategies.rarity_tier_factor import RarityTierFactorStrategy, PREMIUM_RARITIES
from poke_quant.validation.statistical_validation import deflated_sharpe_ratio, pbo_cscv
from poke_quant.validation.bootstrap import block_bootstrap_metrics, summarize_bootstrap


def run_bt(strat, prices_df, metadata):
    bt = Backtester(strat, prices_df, metadata, initial_cash=10000.0, platform="cardmarket",
                     apply_liquidity_slippage=True, apply_holding_cost=True, apply_buy_side_shipping=True)
    return bt.run()


def report(label, res, n_trials):
    dsr = deflated_sharpe_ratio(observed_sr=res.sharpe / np.sqrt(12), n_trials=n_trials, n_obs=len(res.monthly_returns))
    n_obs = len(res.monthly_returns)
    h = n_obs // 2
    r1, r2 = res.monthly_returns[:h], res.monthly_returns[h:]
    h1 = (r1.mean() / r1.std()) * np.sqrt(12) if r1.std() > 0 else 0.0
    h2 = (r2.mean() / r2.std()) * np.sqrt(12) if r2.std() > 0 else 0.0
    print(f"  {label:38s} | Sharpe {res.sharpe:6.2f} | CAGR {res.cagr*100:+7.2f}% | "
          f"MaxDD {res.max_drawdown*100:7.2f}% | Trade {res.total_trades:3d} | DSR({n_trials}) {dsr:.3f} | H1 {h1:5.2f} H2 {h2:5.2f}")
    return dsr


def main():
    metadata = load_metadata()
    grade9_prices = load_price_matrix("historical_prices_graded_singles_grade9.csv")
    all_ids = liquid_singles_ids(metadata, grade9_prices)
    print(f"Universo liquid_singles_ids totale: {len(all_ids)}")

    n_chase = sum(1 for k in all_ids if metadata[k].get("selection_method") == "chase_price_filter_survivorship_biased")
    n_control = sum(1 for k in all_ids if metadata[k].get("selection_method") == "random_control")
    print(f"  di cui chase (survivorship-biased): {n_chase} | controllo casuale: {n_control}")

    premium_ids = [k for k in all_ids if metadata[k].get("rarity") in PREMIUM_RARITIES]
    n_premium_chase = sum(1 for k in premium_ids if metadata[k].get("selection_method") == "chase_price_filter_survivorship_biased")
    n_premium_control = sum(1 for k in premium_ids if metadata[k].get("selection_method") == "random_control")
    print(f"\nCarte eleggibili rarità premium: {len(premium_ids)} | da chase: {n_premium_chase} "
          f"({n_premium_chase/max(1,len(premium_ids))*100:.0f}%) | da controllo: {n_premium_control} "
          f"({n_premium_control/max(1,len(premium_ids))*100:.0f}%)")
    print("(confronto: nella run precedente, 241 eleggibili, 96% da chase, 4% da controllo)")

    control_only_ids = [k for k in all_ids if metadata[k].get("selection_method") == "random_control"]

    print("\n=== (a) Universo intero (chase + controllo, come in produzione) ===")
    meta_sub = {k: metadata[k] for k in all_ids}
    prices_sub = grade9_prices[all_ids]
    strat = RarityTierFactorStrategy(rebalance_every_months=6, min_age_months=6)
    res_a = run_bt(strat, prices_sub, meta_sub)
    dsr_a = report("Rarita' premium, universo intero", res_a, 72)

    print("\n=== (b) SOLO controllo casuale (isola l'effetto dal survivorship bias) ===")
    meta_ctrl = {k: metadata[k] for k in control_only_ids}
    prices_ctrl = grade9_prices[control_only_ids]
    n_premium_only_ctrl = sum(1 for k in control_only_ids if metadata[k].get("rarity") in PREMIUM_RARITIES)
    print(f"Carte nel campione di controllo puro: {len(control_only_ids)} | di cui rarità premium: {n_premium_only_ctrl}")
    if n_premium_only_ctrl < 15:
        print("TROPPO POCHE per un test conclusivo anche dopo l'ampliamento - non procedo con questo sotto-test.")
    else:
        strat_ctrl = RarityTierFactorStrategy(rebalance_every_months=6, min_age_months=6, max_positions=999)
        res_b = run_bt(strat_ctrl, prices_ctrl, meta_ctrl)
        dsr_b = report("Rarita' premium, SOLO controllo puro", res_b, 72)

    sims = block_bootstrap_metrics(res_a.monthly_returns, n_sims=500, block_size=6)
    print("\n" + summarize_bootstrap(sims))


if __name__ == "__main__":
    main()
