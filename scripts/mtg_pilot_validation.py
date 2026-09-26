#!/usr/bin/env python3
"""
scripts/mtg_pilot_validation.py — PROGETTO PILOTA Magic: The Gathering,
validazione completa (stesso rigore usato per Pokemon/One Piece in questa
ricerca: DSR/PBO/bootstrap/walk-forward, verifica su campione di controllo
puro per isolare il survivorship bias).

Testa, in ordine:
  1. Box (TS Momentum) SOLO su MTG - l'ipotesi tiene sulla nuova franchise presa
     da sola?
  2. Box (TS Momentum) su Pokemon+One Piece+MTG UNIFICATI - una sola strategia,
     un solo portafoglio (richiesto esplicitamente: "il tutto integrato in una
     sola strategia").
  3. Singole (Fattore Scarsita', con is_mtg aggiunto) SOLO su MTG, sul pannello
     RAW (non Grade 9 - scelta confermata dall'utente).
  4. Singole UNIFICATE Pokemon+One Piece+MTG (pannello unificato, vedi
     poke_quant/data/unified_singles_panel.py), con verifica separata sul solo
     campione di controllo MTG (isola il survivorship bias, stesso principio
     usato ovunque in questa ricerca).

RISCHIO STRUTTURALE DICHIARATO (vedi anche discover_mtg_sealed_universe.py):
Magic ha una politica di ristampa molto piu' sistematica di Pokemon (Masters,
"The List", edizioni "Remastered") - l'assunzione economica dietro il momentum
sui box ("l'offerta si riduce solo, mai si rinnova") e' meno solida qui.
Questo NON e' un motivo per scartare a priori il risultato, ma va tenuto
presente interpretandolo - un buon backtest su un campione 2015-2024 non
dimostra che l'assunzione tenga sui prossimi 10 anni quanto lo dimostra per
Pokemon.

ESITO (2026-09-26): PILOTA NON VALIDATO, sia per i box che per le singole.
NON integrato in produzione (nessuna modifica alla dashboard/segnale live).

  Box SOLO MTG:      Sharpe -0.09 | DSR(49) 0.006 | MaxDD -40.22% | 15 box liquidi
                      Walk-forward: H1 -1.13 -> H2 +1.05 (segno invertito, instabile)
  Box UNIFICATO:      Sharpe  1.14 | DSR(49) 0.646 (attribuibile a diluizione nel
                      portafoglio Pokemon+OP gia' forte, NON a un contributo
                      positivo di MTG preso da solo - vedi riga sopra)
  -> Coerente con il rischio di ristampa dichiarato: campione piccolo (15 box) e
     performance negativa; nessuna prova che MTG aggiunga edge ai box.

  Singole SOLO MTG:   Sharpe  1.74 | DSR(69) 0.934 | qty media/trade 445.2 copie
  Singole controllo:  Sharpe  0.94 | DSR(69) 0.418 | qty media/trade 3359.9 copie
  Singole UNIFICATE:  Sharpe  2.49 | DSR(70) 0.998 | qty media/trade 217.2 copie
  -> Numeri "spettacolari" ma smontati dallo stress test sul tetto di quantita'
     per trade (stesso artefatto scoperto in questa sessione sul campione di
     controllo Pokemon ampliato, qui MOLTO piu' estremo per via dei prezzi RAW
     bassissimi di molte comuni MTG - centesimi/pochi euro):

       Config                         tetto=1   tetto=2   tetto=3   tetto=None
       Singole SOLO MTG (Sharpe)        -3.33     -3.09     -2.83       1.74
       Singole controllo MTG (Sharpe)   -6.94        -         -        0.94
       Singole UNIFICATE (Sharpe)       -0.38      0.50      0.86       2.49
       DSR a QUALSIASI tetto realistico (1-3 copie): sempre 0.000 in ogni config.

     A qualunque tetto realistico di quantita' per trade, l'edge scompare o
     diventa negativo. L'apparente performance e' un artefatto del backtest
     (budget fisso in EUR che compra quantita' illimitate di carte da pochi
     centesimi), non alfa reale. Rigettato con la stessa logica applicata a
     ogni altro fattore candidato invalidato in questa ricerca.

  CONCLUSIONE: nessuna delle due gambe del pilota (box, singole) supera la
  bar decisionale di questa ricerca (DSR alto E stabile a parametri realistici
  E walk-forward stabile). MTG resta fuori dalla strategia di produzione.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.data.liquidity_filter import liquid_sealed_ids, liquid_singles_ids
from poke_quant.data.unified_singles_panel import build_unified_singles_price_panel
from poke_quant.engine.backtester import Backtester
from poke_quant.engine.strategies.time_series_momentum import TimeSeriesMomentumStrategy
from poke_quant.engine.strategies.scarcity_value_factor import ScarcityValueFactorStrategy
from poke_quant.validation.statistical_validation import deflated_sharpe_ratio, pbo_cscv
from poke_quant.validation.bootstrap import block_bootstrap_metrics, summarize_bootstrap
from scripts.generate_singles_signal import PRODUCTION_PARAMS

N_TRIALS_BOX_FULL_SESSION = 49   # invariato finche' non si decide di adottare MTG in produzione
N_TRIALS_SINGLES_FULL_SESSION = 69


def run_box_bt(prices_df, metadata):
    strat = TimeSeriesMomentumStrategy(prices_df, lookback_months=12)
    bt = Backtester(strat, prices_df, metadata, initial_cash=10000.0, platform="cardmarket",
                     apply_liquidity_slippage=True, apply_holding_cost=True, apply_buy_side_shipping=True)
    return bt.run()


def run_singles_bt(strat, prices_df, metadata):
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
    print(f"  {label:42s} | Sharpe {res.sharpe:6.2f} | CAGR {res.cagr*100:+7.2f}% | "
          f"MaxDD {res.max_drawdown*100:7.2f}% | Trade {res.total_trades:3d} | DSR({n_trials}) {dsr:.3f} | "
          f"H1 {h1:5.2f} H2 {h2:5.2f}")
    return dsr


def main():
    metadata = load_metadata()
    sealed_prices = load_price_matrix("historical_prices.csv")
    grade9_prices = load_price_matrix("historical_prices_graded_singles_grade9.csv")
    raw_prices = load_price_matrix("historical_prices_graded_singles_raw.csv")

    print("=" * 100)
    print("1-2. BOX (TS Momentum)")
    print("=" * 100)
    sealed_ids = liquid_sealed_ids(metadata, sealed_prices)
    mtg_box_ids = [k for k in sealed_ids if metadata[k].get("franchise") == "magic"]
    print(f"Box MTG liquidi: {len(mtg_box_ids)} | Box totali (Pokemon+OP+MTG): {len(sealed_ids)}")

    if len(mtg_box_ids) >= 5:
        meta_mtg = {k: metadata[k] for k in mtg_box_ids}
        res_mtg_box = run_box_bt(sealed_prices[mtg_box_ids], meta_mtg)
        report("Box SOLO MTG", res_mtg_box, N_TRIALS_BOX_FULL_SESSION)
    else:
        print(f"  Troppo pochi box MTG liquidi ({len(mtg_box_ids)}) per un backtest a se' stante.")

    meta_all = {k: metadata[k] for k in sealed_ids}
    res_unified_box = run_box_bt(sealed_prices[sealed_ids], meta_all)
    report("Box UNIFICATO (Pokemon+OP+MTG)", res_unified_box, N_TRIALS_BOX_FULL_SESSION)

    print("\n" + "=" * 100)
    print("3-4. SINGOLE (Fattore Scarsita')")
    print("=" * 100)
    unified_panel = build_unified_singles_price_panel(grade9_prices, raw_prices)
    singles_ids = liquid_singles_ids(metadata, unified_panel)
    mtg_singles_ids = [k for k in singles_ids if metadata[k].get("franchise") == "magic"]
    mtg_control_ids = [k for k in mtg_singles_ids if metadata[k].get("selection_method") == "random_control"]
    print(f"Singole MTG liquide: {len(mtg_singles_ids)} (di cui controllo puro: {len(mtg_control_ids)}) | "
          f"Singole totali (Pokemon+OP+MTG): {len(singles_ids)}")

    if len(mtg_singles_ids) >= PRODUCTION_PARAMS["min_cross_section"]:
        meta_mtg_s = {k: metadata[k] for k in mtg_singles_ids}
        strat_mtg = ScarcityValueFactorStrategy(**PRODUCTION_PARAMS)
        res_mtg_singles = run_singles_bt(strat_mtg, unified_panel[mtg_singles_ids], meta_mtg_s)
        report("Singole SOLO MTG", res_mtg_singles, N_TRIALS_SINGLES_FULL_SESSION)
    else:
        print(f"  Troppo poche singole MTG liquide ({len(mtg_singles_ids)}) per un backtest a se' stante.")

    if len(mtg_control_ids) >= 10:
        meta_mtg_c = {k: metadata[k] for k in mtg_control_ids}
        strat_mtg_c = ScarcityValueFactorStrategy(**dict(PRODUCTION_PARAMS, min_cross_section=10))
        res_mtg_control = run_singles_bt(strat_mtg_c, unified_panel[mtg_control_ids], meta_mtg_c)
        report("Singole SOLO controllo puro MTG (isola survivorship bias)", res_mtg_control, N_TRIALS_SINGLES_FULL_SESSION)

    meta_all_s = {k: metadata[k] for k in singles_ids}
    strat_unified = ScarcityValueFactorStrategy(**PRODUCTION_PARAMS)
    res_unified_singles = run_singles_bt(strat_unified, unified_panel[singles_ids], meta_all_s)
    dsr_unified = report("Singole UNIFICATE (Pokemon+OP+MTG)", res_unified_singles, N_TRIALS_SINGLES_FULL_SESSION + 1)

    print("\nBootstrap a blocchi (singole unificate):")
    sims = block_bootstrap_metrics(res_unified_singles.monthly_returns, n_sims=500, block_size=6)
    print(summarize_bootstrap(sims))

    print("\n" + "=" * 100)
    print("STRESS TEST: tetto di quantita' per trade (scripts/max_quantity_per_trade_test.py,"
          " qui riapplicato alle singole MTG - vedi ESITO nel docstring)")
    print("=" * 100)
    for cap_label, cap in (("tetto=1", 1), ("tetto=2", 2), ("tetto=3", 3), ("tetto=None", None)):
        if len(mtg_singles_ids) >= PRODUCTION_PARAMS["min_cross_section"]:
            strat = ScarcityValueFactorStrategy(max_quantity_per_trade=cap, **PRODUCTION_PARAMS)
            res = run_singles_bt(strat, unified_panel[mtg_singles_ids], meta_mtg_s)
            avg_qty = res.trades_df["quantity"].abs().mean() if len(res.trades_df) else 0.0
            dsr = deflated_sharpe_ratio(observed_sr=res.sharpe / np.sqrt(12), n_trials=N_TRIALS_SINGLES_FULL_SESSION,
                                         n_obs=len(res.monthly_returns))
            print(f"  MTG-only    {cap_label:12s} | Sharpe {res.sharpe:6.2f} | qty media {avg_qty:8.1f} | DSR {dsr:.3f}")
        if len(mtg_control_ids) >= 10:
            strat_c = ScarcityValueFactorStrategy(max_quantity_per_trade=cap,
                                                   **dict(PRODUCTION_PARAMS, min_cross_section=10))
            res_c = run_singles_bt(strat_c, unified_panel[mtg_control_ids], meta_mtg_c)
            avg_qty_c = res_c.trades_df["quantity"].abs().mean() if len(res_c.trades_df) else 0.0
            dsr_c = deflated_sharpe_ratio(observed_sr=res_c.sharpe / np.sqrt(12), n_trials=N_TRIALS_SINGLES_FULL_SESSION,
                                           n_obs=len(res_c.monthly_returns))
            print(f"  MTG-control {cap_label:12s} | Sharpe {res_c.sharpe:6.2f} | qty media {avg_qty_c:8.1f} | DSR {dsr_c:.3f}")
        strat_u = ScarcityValueFactorStrategy(max_quantity_per_trade=cap, **PRODUCTION_PARAMS)
        res_u = run_singles_bt(strat_u, unified_panel[singles_ids], meta_all_s)
        avg_qty_u = res_u.trades_df["quantity"].abs().mean() if len(res_u.trades_df) else 0.0
        dsr_u = deflated_sharpe_ratio(observed_sr=res_u.sharpe / np.sqrt(12), n_trials=N_TRIALS_SINGLES_FULL_SESSION + 1,
                                       n_obs=len(res_u.monthly_returns))
        print(f"  Unificato   {cap_label:12s} | Sharpe {res_u.sharpe:6.2f} | qty media {avg_qty_u:8.1f} | DSR {dsr_u:.3f}")


if __name__ == "__main__":
    main()
