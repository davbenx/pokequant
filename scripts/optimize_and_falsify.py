#!/usr/bin/env python3
"""
scripts/optimize_and_falsify.py — Round di ottimizzazione/falsificazione rigorosa
per i due candidati validati:
  1. TS Momentum su box sigillati era moderna (2019+)
  2. Carry/Scarsità su singole gradate Grade 9

REGOLA METODOLOGICA (per non ricadere nell'errore che abbiamo già trovato e
corretto più volte in questa sessione): non si cerca "il parametro migliore" su
questi 69 mesi — si verifica la STABILITÀ su una griglia di parametri vicini
(un picco netto su un solo valore è un segnale di overfitting, un plateau è
rassicurante), si calcola il PBO su TUTTA la griglia (non solo tra strategie
diverse, l'uso classico di Bailey&LopezDePrado è esattamente "ho provato N
varianti, quanto è probabile che la migliore in-sample non lo sia out-of-sample"),
e si stima un intervallo di confidenza via block bootstrap invece di un singolo
numero puntuale.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.engine.backtester import Backtester
from poke_quant.engine.strategies.time_series_momentum import TimeSeriesMomentumStrategy
from poke_quant.engine.strategies.carry_scarcity_factor import CarryScarcityFactorStrategy
from poke_quant.engine.strategies.cross_sectional_momentum import CrossSectionalMomentumStrategy
from poke_quant.engine.strategies.dip_mean_reversion import DipMeanReversionStrategy
from poke_quant.engine.strategies.rarity_tier_factor import RarityTierFactorStrategy
from poke_quant.validation.statistical_validation import deflated_sharpe_ratio, pbo_cscv
from poke_quant.validation.bootstrap import block_bootstrap_metrics, summarize_bootstrap

MODERN_ERA_CUTOFF = "2019-01-01"


def run_bt(strategy, prices_df, metadata):
    bt = Backtester(strategy, prices_df, metadata, initial_cash=10000.0, platform="cardmarket",
                     apply_liquidity_slippage=True, apply_holding_cost=True)
    return bt.run()


def section_tsmom_sealed():
    print("=" * 100)
    print("  1) TS MOMENTUM — BOX SIGILLATI ERA MODERNA (2019+)")
    print("=" * 100)
    metadata = load_metadata()
    prices_full = load_price_matrix()
    sealed_ids = [
        k for k, v in metadata.items()
        if v.get("type") == "sealed" and v.get("data_quality") != "thin_unreliable"
        and k in prices_full.columns and v.get("release_date") and v["release_date"] >= MODERN_ERA_CUTOFF
    ]
    meta_sub = {k: v for k, v in metadata.items() if k in sealed_ids}
    prices_sub = prices_full[sealed_ids]

    lookbacks = [6, 9, 12, 15, 18]
    results = {}
    print(f"\nGriglia lookback (mesi) — stabilità, non 'il migliore':")
    for lb in lookbacks:
        strat = TimeSeriesMomentumStrategy(prices_sub, lookback_months=lb)
        res = run_bt(strat, prices_sub, meta_sub)
        results[lb] = res
        print(f"  lookback={lb:2d}m | CAGR {res.cagr*100:+6.2f}% | Sharpe {res.sharpe:5.2f} | "
              f"MaxDD {res.max_drawdown*100:6.2f}% | Trade {res.total_trades:3d}")

    common_idx = None
    for res in results.values():
        common_idx = res.monthly_returns.index if common_idx is None else common_idx.intersection(res.monthly_returns.index)
    perf_matrix = np.column_stack([results[lb].monthly_returns.loc[common_idx].values for lb in lookbacks])
    t_len = len(perf_matrix)
    splits = 8 if t_len >= 32 else 4
    rem = t_len % splits
    pbo = pbo_cscv(perf_matrix[rem:, :] if rem else perf_matrix, n_splits=splits)
    print(f"\nPBO sulla griglia lookback ({splits} split): {pbo:.2f} ({pbo*100:.1f}%)")

    base = results[12]
    dsr = deflated_sharpe_ratio(observed_sr=base.sharpe / np.sqrt(12), n_trials=len(lookbacks), n_obs=len(base.monthly_returns))
    print(f"DSR lookback=12m (n_trials={len(lookbacks)}): {dsr:.3f}")

    sims = block_bootstrap_metrics(base.monthly_returns, n_sims=500, block_size=6)
    print(f"\nBlock bootstrap (500 sim, blocchi 6m) su lookback=12m:")
    print(summarize_bootstrap(sims))


def section_carry_singles():
    print("\n\n" + "=" * 100)
    print("  2) CARRY/SCARSITÀ — SINGOLE GRADATE GRADE 9")
    print("=" * 100)
    metadata = load_metadata()
    prices_full = load_price_matrix("historical_prices_graded_singles_grade9.csv")
    singles_ids = [
        k for k, v in metadata.items()
        if v.get("type") == "single" and v.get("data_quality") != "thin_unreliable" and k in prices_full.columns
    ]
    meta_sub = {k: v for k, v in metadata.items() if k in singles_ids}
    prices_sub = prices_full[singles_ids]

    quantiles = [0.20, 0.25, 0.30, 0.40, 0.50]
    results = {}
    print(f"\nGriglia top_quantile — stabilità:")
    for q in quantiles:
        strat = CarryScarcityFactorStrategy(top_quantile=q, item_type_filter="single")
        res = run_bt(strat, prices_sub, meta_sub)
        results[q] = res
        print(f"  quantile={q:.2f} | CAGR {res.cagr*100:+6.2f}% | Sharpe {res.sharpe:5.2f} | "
              f"MaxDD {res.max_drawdown*100:6.2f}% | Trade {res.total_trades:3d}")

    common_idx = None
    for res in results.values():
        common_idx = res.monthly_returns.index if common_idx is None else common_idx.intersection(res.monthly_returns.index)
    perf_matrix = np.column_stack([results[q].monthly_returns.loc[common_idx].values for q in quantiles])
    t_len = len(perf_matrix)
    splits = 8 if t_len >= 32 else 4
    rem = t_len % splits
    pbo = pbo_cscv(perf_matrix[rem:, :] if rem else perf_matrix, n_splits=splits)
    print(f"\nPBO sulla griglia top_quantile ({splits} split): {pbo:.2f} ({pbo*100:.1f}%)")

    base = results[0.30]
    dsr = deflated_sharpe_ratio(observed_sr=base.sharpe / np.sqrt(12), n_trials=len(quantiles), n_obs=len(base.monthly_returns))
    print(f"DSR quantile=0.30 (n_trials={len(quantiles)}): {dsr:.3f}")

    sims = block_bootstrap_metrics(base.monthly_returns, n_sims=500, block_size=6)
    print(f"\nBlock bootstrap (500 sim, blocchi 6m) su quantile=0.30:")
    print(summarize_bootstrap(sims))


def section_survivorship_bias_check():
    """
    Quantifica l'inflazione da survivorship bias sulla Carry/Scarsita' singles:
    discover_chase_cards.py seleziona sul prezzo Cardmarket CORRENTE (oggi), quindi
    ogni carta in quell'universo e', per costruzione, una carta che sappiamo con
    informazione 2026 essersi rivelata valida. discover_random_control_singles.py
    aggiunge un campione casuale (nessun filtro di prezzo/rarita) sugli stessi 98 set.
    Confrontiamo la stessa strategia, stessi parametri, sui due universi.
    """
    print("\n\n" + "=" * 100)
    print("  3) QUANTO COSTA IL SURVIVORSHIP BIAS? Chase-only vs Chase+Controllo casuale")
    print("=" * 100)
    metadata = load_metadata()
    prices_full = load_price_matrix("historical_prices_graded_singles_grade9.csv")

    def build_universe(selection_methods):
        ids = [
            k for k, v in metadata.items()
            if v.get("type") == "single" and v.get("data_quality") != "thin_unreliable"
            and k in prices_full.columns and v.get("selection_method") in selection_methods
        ]
        return {k: v for k, v in metadata.items() if k in ids}, prices_full[ids]

    chase_meta, chase_prices = build_universe({"chase_price_filter_survivorship_biased"})
    combined_meta, combined_prices = build_universe(
        {"chase_price_filter_survivorship_biased", "random_control"}
    )
    control_meta, control_prices = build_universe({"random_control"})

    print(f"\nUniverso chase-only: {len(chase_meta)} carte")
    print(f"Universo solo controllo casuale: {len(control_meta)} carte")
    print(f"Universo combinato: {len(combined_meta)} carte\n")

    for label, meta_sub, prices_sub in [
        ("CHASE-ONLY (biased)", chase_meta, chase_prices),
        ("SOLO CONTROLLO (no bias di prezzo)", control_meta, control_prices),
        ("CHASE+CONTROLLO (combinato)", combined_meta, combined_prices),
    ]:
        if prices_sub.empty or len(meta_sub) < 3:
            print(f"  {label:36s} | universo troppo piccolo ({len(meta_sub)} carte), salto")
            continue
        strat = CarryScarcityFactorStrategy(top_quantile=0.30, item_type_filter="single")
        res = run_bt(strat, prices_sub, meta_sub)
        print(f"  {label:36s} | CAGR {res.cagr*100:+6.2f}% | Sharpe {res.sharpe:5.2f} | "
              f"MaxDD {res.max_drawdown*100:6.2f}% | Trade {res.total_trades:3d}")

    print("\nLettura: se CHASE-ONLY batte nettamente CHASE+CONTROLLO, la differenza e' la stima "
          "dell'inflazione da survivorship bias. Se il factor Carry/Scarsita' regge anche sul "
          "campione combinato (Sharpe comparabile), l'eta' come proxy di scarsita' ha un effetto "
          "reale al netto della selezione sull'esito.")


def section_singles_factor_search():
    """
    Carry/Scarsita' (fattore eta') NON supera la validazione su nessun sotto-universo
    (vedi section_survivorship_bias_check): Sharpe -0.14/-0.04, DSR 0.10, bootstrap
    P(Sharpe>0) 54-61% - indistinguibile dal rumore. Cerchiamo un segnale alternativo
    invece di rinunciare alle singole, sullo stesso universo combinato (meno biased)
    e con la stessa griglia di rigore.

    ATTENZIONE su RarityTierFactorStrategy: la sua whitelist di rarita' premium
    coincide con CHASE_RARITIES di discover_chase_cards.py. Nell'universo attuale,
    270 delle 288 carte eleggibili per questo fattore vengono dal campione chase
    (selezionato sul prezzo corrente), solo 18 dal campione di controllo casuale -
    rapporto 15:1. Un risultato POSITIVO qui non sarebbe evidenza pulita (la
    popolazione e' ancora in gran parte quella biased); solo un risultato NEGATIVO
    (nessun edge nemmeno su questa popolazione favorevole) e' informativo cosi' com'e'.

    DSR del vincitore usa n_trials = numero di candidati provati in QUESTA ricerca,
    non solo la sua griglia interna - altrimenti si ricade nello stesso errore di
    data-snooping che l'intera sessione ha cercato di correggere.
    """
    print("\n\n" + "=" * 100)
    print("  4) RICERCA SISTEMATICA DI UN FATTORE ALTERNATIVO SULLE SINGOLE")
    print("=" * 100)
    metadata = load_metadata()
    prices_full = load_price_matrix("historical_prices_graded_singles_grade9.csv")
    singles_ids = [
        k for k, v in metadata.items()
        if v.get("type") == "single" and v.get("data_quality") != "thin_unreliable" and k in prices_full.columns
    ]
    meta_sub = {k: v for k, v in metadata.items() if k in singles_ids}
    prices_sub = prices_full[singles_ids]
    print(f"\nUniverso: {len(singles_ids)} singole (chase+controllo combinato)\n")

    # Factory (non istanze dirette): serve per poter ricostruire ogni strategia su
    # sottoinsiemi temporali H1/H2 per il walk-forward sotto, senza condividere stato.
    candidate_factories = {
        "TSMOM lb=6m": lambda p: TimeSeriesMomentumStrategy(p, lookback_months=6, item_type_filter="single"),
        "TSMOM lb=9m": lambda p: TimeSeriesMomentumStrategy(p, lookback_months=9, item_type_filter="single"),
        "TSMOM lb=12m": lambda p: TimeSeriesMomentumStrategy(p, lookback_months=12, item_type_filter="single"),
        "XSMOM lb=6m q=0.30": lambda p: CrossSectionalMomentumStrategy(p, lookback_months=6, top_quantile=0.30, item_type_filter="single"),
        "XSMOM lb=12m q=0.30": lambda p: CrossSectionalMomentumStrategy(p, lookback_months=12, top_quantile=0.30, item_type_filter="single"),
        "Dip lb=12m q=0.20": lambda p: DipMeanReversionStrategy(p, lookback_months=12, bottom_quantile=0.20),
        "Dip lb=15m q=0.20": lambda p: DipMeanReversionStrategy(p, lookback_months=15, bottom_quantile=0.20),
        "Dip lb=18m q=0.15": lambda p: DipMeanReversionStrategy(p, lookback_months=18, bottom_quantile=0.15),
        "RarityTier (94% biased)": lambda p: RarityTierFactorStrategy(),
    }

    results = {}
    for name, factory in candidate_factories.items():
        res = run_bt(factory(prices_sub), prices_sub, meta_sub)
        results[name] = res
        print(f"  {name:26s} | CAGR {res.cagr*100:+6.2f}% | Sharpe {res.sharpe:5.2f} | "
              f"MaxDD {res.max_drawdown*100:6.2f}% | Trade {res.total_trades:3d}")

    best_name = max(results, key=lambda k: results[k].sharpe)
    best = results[best_name]
    print(f"\nMiglior candidato per Sharpe (campione intero): '{best_name}' (Sharpe {best.sharpe:.2f})")

    if len(best.monthly_returns) < 2:
        print("Serie troppo corta per DSR/bootstrap/walk-forward sul vincitore.")
        return

    dsr = deflated_sharpe_ratio(
        observed_sr=best.sharpe / np.sqrt(12), n_trials=len(candidate_factories), n_obs=len(best.monthly_returns)
    )
    print(f"DSR corretto per TUTTI e {len(candidate_factories)} i candidati di questa ricerca "
          f"(non solo la griglia interna del vincitore): {dsr:.3f}")

    sims = block_bootstrap_metrics(best.monthly_returns, n_sims=500, block_size=6)
    print(f"\nBlock bootstrap (500 sim, blocchi 6m) sul vincitore (SOLO campione intero, "
          f"non corretto per walk-forward - vedi sotto):")
    print(summarize_bootstrap(sims))

    # TEST DECISIVO: il bootstrap sopra ricampiona a blocchi la STESSA serie storica del
    # vincitore - non dice se l'edge e' stabile nel tempo, solo quanto e' incerta la sua
    # media. Lo split H1/H2 (prima vs seconda meta' del campione) e' l'unico modo per
    # vedere se il vincitore regge in un regime di mercato diverso da quello su cui e'
    # stato scelto - lo stesso principio che porto' a correggere la conclusione errata
    # "Carry/Scarsita' batte TSMOM" in una sessione precedente.
    print(f"\nWalk-forward H1/H2 sul vincitore '{best_name}' (split a meta' campione, "
          "stessa configurazione, nessun nuovo fitting):")
    mid = len(prices_sub) // 2
    h1_dates, h2_dates = prices_sub.index[:mid], prices_sub.index[mid:]
    factory = candidate_factories[best_name]
    wf_results = {}
    for label, dates in [("H1", h1_dates), ("H2", h2_dates)]:
        sub_prices = prices_sub.loc[dates]
        res = run_bt(factory(sub_prices), sub_prices, meta_sub)
        wf_results[label] = res
        print(f"  {label} ({dates[0].strftime('%Y-%m')} -> {dates[-1].strftime('%Y-%m')}) | "
              f"CAGR {res.cagr*100:+6.2f}% | Sharpe {res.sharpe:5.2f} | MaxDD {res.max_drawdown*100:6.2f}%")
    sharpe_flip = (wf_results["H1"].sharpe < 0) != (wf_results["H2"].sharpe < 0)
    print(("\nATTENZIONE: lo Sharpe cambia segno tra H1 e H2 - l'edge del campione intero e' "
           "guidato da un solo regime di mercato, non e' un fattore stabile nel tempo. NON "
           "considerare questo candidato validato, anche con DSR/bootstrap favorevoli sul "
           "campione intero." if sharpe_flip else
           "\nLo Sharpe mantiene lo stesso segno in entrambe le meta' - non un flip completo, "
           "ma verificare comunque l'ampiezza del divario prima di trattarlo come stabile."))


def section_sealed_exit_logic_search():
    """
    La regola di uscita in produzione (lookback=12m, exit_threshold=0.0,
    trailing_stop=None) non era mai stata testata contro alternative - era
    semplicemente "lo stesso lookback usato per entrare, simmetrico". Qui si
    testano due assi economicamente motivati, tenendo fisso l'ingresso
    (lookback=12m, il punto validato):
      A) exit_lookback_months piu' corto dell'ingresso (uscita piu' reattiva)
      B) trailing_stop_pct come rete di sicurezza indipendente dal momentum

    Stesso standard di rigore usato per le singole: griglia per stabilita',
    PBO, DSR corretto per TUTTI i candidati provati, bootstrap sul vincitore,
    E walk-forward H1/H2 - un miglioramento che non regge lo split non conta.
    """
    print("\n\n" + "=" * 100)
    print("  5) OTTIMIZZAZIONE LOGICA DI USCITA — TS MOMENTUM SEALED")
    print("=" * 100)
    metadata = load_metadata()
    prices_full = load_price_matrix()
    sealed_ids = [
        k for k, v in metadata.items()
        if v.get("type") == "sealed" and v.get("data_quality") != "thin_unreliable"
        and k in prices_full.columns and v.get("release_date") and v["release_date"] >= MODERN_ERA_CUTOFF
    ]
    meta_sub = {k: v for k, v in metadata.items() if k in sealed_ids}
    prices_sub = prices_full[sealed_ids]

    baseline_factory = lambda p: TimeSeriesMomentumStrategy(p, lookback_months=12)
    candidate_factories = {"BASELINE (lookback=12m simmetrico, in produzione)": baseline_factory}
    for exit_lb in [3, 6, 9]:
        candidate_factories[f"exit_lookback={exit_lb}m"] = (
            lambda p, lb=exit_lb: TimeSeriesMomentumStrategy(p, lookback_months=12, exit_lookback_months=lb)
        )
    for stop_pct in [0.10, 0.15, 0.20, 0.25, 0.30]:
        candidate_factories[f"trailing_stop={stop_pct*100:.0f}%"] = (
            lambda p, sp=stop_pct: TimeSeriesMomentumStrategy(p, lookback_months=12, trailing_stop_pct=sp)
        )

    results = {}
    print(f"\nEntrata fissa a lookback=12m (validata) - solo la regola di uscita varia:")
    for name, factory in candidate_factories.items():
        res = run_bt(factory(prices_sub), prices_sub, meta_sub)
        results[name] = res
        print(f"  {name:45s} | CAGR {res.cagr*100:+6.2f}% | Sharpe {res.sharpe:5.2f} | "
              f"MaxDD {res.max_drawdown*100:6.2f}% | Trade {res.total_trades:3d}")

    common_idx = None
    for res in results.values():
        common_idx = res.monthly_returns.index if common_idx is None else common_idx.intersection(res.monthly_returns.index)
    perf_matrix = np.column_stack([results[k].monthly_returns.loc[common_idx].values for k in candidate_factories])
    t_len = len(perf_matrix)
    splits = 8 if t_len >= 32 else 4
    rem = t_len % splits
    pbo = pbo_cscv(perf_matrix[rem:, :] if rem else perf_matrix, n_splits=splits)
    print(f"\nPBO sulla griglia completa ({len(candidate_factories)} candidati, {splits} split): {pbo:.3f} ({pbo*100:.1f}%)")

    baseline_sharpe = results["BASELINE (lookback=12m simmetrico, in produzione)"].sharpe
    best_name = max(results, key=lambda k: results[k].sharpe)
    best = results[best_name]
    print(f"\nBaseline Sharpe: {baseline_sharpe:.2f} | Migliore della griglia: '{best_name}' (Sharpe {best.sharpe:.2f})")

    if best_name == "BASELINE (lookback=12m simmetrico, in produzione)":
        print("La baseline resta la migliore - nessuna modifica alla logica di uscita in produzione.")
        return

    dsr = deflated_sharpe_ratio(
        observed_sr=best.sharpe / np.sqrt(12), n_trials=len(candidate_factories), n_obs=len(best.monthly_returns)
    )
    print(f"DSR del vincitore corretto per TUTTI e {len(candidate_factories)} i candidati: {dsr:.3f} "
          f"(baseline: {deflated_sharpe_ratio(observed_sr=baseline_sharpe/np.sqrt(12), n_trials=len(candidate_factories), n_obs=len(results[best_name].monthly_returns)):.3f})")

    sims = block_bootstrap_metrics(best.monthly_returns, n_sims=500, block_size=6)
    print(f"\nBlock bootstrap (500 sim, blocchi 6m) sul vincitore:")
    print(summarize_bootstrap(sims))

    print(f"\nWalk-forward H1/H2 sul vincitore '{best_name}' vs baseline:")
    mid = len(prices_sub) // 2
    h1_dates, h2_dates = prices_sub.index[:mid], prices_sub.index[mid:]
    for label_strat, factory in [("BASELINE", baseline_factory), (best_name, candidate_factories[best_name])]:
        for label, dates in [("H1", h1_dates), ("H2", h2_dates)]:
            sub_prices = prices_sub.loc[dates]
            res = run_bt(factory(sub_prices), sub_prices, meta_sub)
            print(f"  {label_strat:45s} {label} ({dates[0].strftime('%Y-%m')}->{dates[-1].strftime('%Y-%m')}) | "
                  f"Sharpe {res.sharpe:5.2f} | MaxDD {res.max_drawdown*100:6.2f}%")

    print("\nAdottare il vincitore in produzione SOLO se: DSR >= baseline, nessun sign-flip H1/H2 "
          "peggiore della baseline, e il miglioramento non e' concentrato in 1-2 trade isolati.")


if __name__ == "__main__":
    section_tsmom_sealed()
    section_carry_singles()
    section_survivorship_bias_check()
    section_singles_factor_search()
    section_sealed_exit_logic_search()
