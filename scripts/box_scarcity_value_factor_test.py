#!/usr/bin/env python3
"""
scripts/box_scarcity_value_factor_test.py — Validazione del fattore scarsita'
lato BOX SIGILLATI (poke_quant/engine/strategies/scarcity_value_factor_sealed.py),
richiesto dall'utente dopo la discussione sul "prezzo massimo per un sealed"
(il tetto attuale, MSRP x 21.6, e' un backstop di coda - mai scattato - non un
vero fair-value ceiling come quello delle singole).

Vedi il docstring del modulo strategia per il proxy di scarsita' adottato
(log(MSRP) al posto della rarita' - nessun dato diretto di scarsita' del pull
chase esiste per i box) e il suo limite dichiarato.

LIMITE CAMPIONARIO (dichiarato PRIMA di guardare l'esito): cross-section
mediana storica 20 box (universo liquido+MSRP cresciuto nel tempo, min 0 max
40) - molto piu' piccolo delle 932 singole. Qualunque esito qui ha potenza
statistica debole, interpretare con cautela proporzionale.

METODO (stessa disciplina di ogni altro fattore in questa sessione):
  1. Diagnostica pura sulla cross-section piu' recente (R^2, residui top/bottom)
     PRIMA di costruire qualunque regola di trading.
  2. Griglia (top_quantile x rebalance_every_months), PBO sulla griglia intera.
  3. DSR corretto per la sola griglia di QUESTO test (non la sessione intera -
     un audit di sessione separato, aggiornato, esula da questo task).
  4. Walk-forward H1/H2 sulla configurazione scelta.
  5. Confronto separato (2 trial) min_age_months 0 vs 4 (esclude box appena
     uscito, finestra ristampe) sulla configurazione scelta.
  6. Confronto diretto con la baseline validata in produzione (TS Momentum,
     Sharpe 1.27) sullo stesso universo/periodo.

ESITO: vedi output.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.data.liquidity_filter import liquid_sealed_ids, MAX_PRICE_TO_MSRP_RATIO
from poke_quant.engine.backtester import Backtester
from poke_quant.engine.strategies.time_series_momentum import TimeSeriesMomentumStrategy
from poke_quant.engine.strategies.scarcity_value_factor_sealed import SealedScarcityValueFactorStrategy
from poke_quant.validation.statistical_validation import deflated_sharpe_ratio, pbo_cscv


def run_bt(strat, prices_df, metadata):
    bt = Backtester(strat, prices_df, metadata, initial_cash=10000.0, platform="cardmarket",
                     apply_liquidity_slippage=True, apply_holding_cost=True, apply_buy_side_shipping=True)
    return bt.run()


def diagnostic(metadata, prices, sealed_ids):
    latest = prices.index[-1]
    rows, ids = [], []
    for iid in sealed_ids:
        info = metadata[iid]
        msrp = info.get("msrp")
        px = prices.loc[latest, iid] if iid in prices.columns else None
        rel = info.get("release_date")
        if not msrp or px is None or pd.isna(px) or px <= 0 or not rel:
            continue
        rel_dt = pd.to_datetime(rel)
        age_m = (latest.year - rel_dt.year) * 12 + (latest.month - rel_dt.month)
        is_op = 1.0 if info.get("franchise") == "one_piece" else 0.0
        rows.append([1.0, np.log(msrp), np.log(age_m + 1.0), is_op, np.log(px)])
        ids.append(iid)

    arr = np.array(rows)
    X, y = arr[:, :-1], arr[:, -1]
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    ss_res, ss_tot = np.sum(resid**2), np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / ss_tot

    print(f"1. DIAGNOSTICA PURA (cross-section {latest.strftime('%Y-%m')}, n={len(ids)}):")
    print(f"   R^2 (log_msrp + log_eta' + is_op) = {r2:.3f}\n")
    ranked = sorted(zip(ids, resid), key=lambda x: x[1])
    print("   Piu' sottovalutate vs pari (residuo piu' negativo):")
    for iid, r in ranked[:5]:
        print(f"     {metadata[iid]['name']:40s} residuo={r:+.3f}")
    print("   Piu' sopravvalutate vs pari (residuo piu' positivo):")
    for iid, r in ranked[-5:]:
        print(f"     {metadata[iid]['name']:40s} residuo={r:+.3f}")
    print()


def main():
    metadata = load_metadata()
    prices_full = load_price_matrix()
    sealed_ids = liquid_sealed_ids(metadata, prices_full)
    meta_sub = {k: metadata[k] for k in sealed_ids}
    prices_sub = prices_full[sealed_ids]

    diagnostic(metadata, prices_full, sealed_ids)

    print("2. GRIGLIA (top_quantile x rebalance_every_months), min_age_months=0:")
    grid = []
    for tq in [0.15, 0.20, 0.25, 0.30]:
        for rb in [3, 6]:
            strat = SealedScarcityValueFactorStrategy(rebalance_every_months=rb, top_quantile=tq, min_age_months=0)
            res = run_bt(strat, prices_sub, meta_sub)
            grid.append((f"q={tq:.2f} rebal={rb}m", res))
            print(f"   {grid[-1][0]:18s} | Sharpe {res.sharpe:5.2f} | CAGR {res.cagr*100:+7.2f}% | "
                  f"MaxDD {res.max_drawdown*100:7.2f}% | Trade {res.total_trades:3d}")

    common_idx = grid[0][1].monthly_returns.index
    for _, res in grid[1:]:
        common_idx = common_idx.intersection(res.monthly_returns.index)
    perf_matrix = np.column_stack([res.monthly_returns.loc[common_idx].values for _, res in grid])
    t_len = len(perf_matrix)
    splits = 8 if t_len >= 16 else 4
    rem = t_len % splits
    pbo = pbo_cscv(perf_matrix[rem:, :] if rem else perf_matrix, n_splits=splits)
    print(f"\n   PBO ({len(grid)} candidati, {splits} split, {t_len} mesi comuni): {pbo:.3f}")

    best_label, best_res = max(grid, key=lambda g: g[1].sharpe)
    dsr_own_grid = deflated_sharpe_ratio(observed_sr=best_res.sharpe / np.sqrt(12), n_trials=len(grid),
                                          n_obs=len(best_res.monthly_returns))
    print(f"   Migliore: {best_label} (Sharpe {best_res.sharpe:.2f}) | "
          f"DSR corretto per questa griglia (n_trials={len(grid)}): {dsr_own_grid:.3f}")

    best_tq = float(best_label.split("q=")[1].split(" ")[0])
    best_rb = int(best_label.split("rebal=")[1].rstrip("m"))

    # BUG METODOLOGICO TROVATO (e corretto PRIMA di riportare l'esito): uno
    # split H1/H2 ingenuo sull'intera storia (70 mesi) da' H1=0 trade/Sharpe
    # 0.00 - non perche' la strategia fallisca in quel periodo, ma perche' la
    # cross-section liquida+MSRP non raggiunge min_cross_section=20 fino a
    # ottobre 2023 (verificato: 0 mesi operativi prima di allora). Uno split
    # cosi' misurerebbe "nessun segnale" vs "tutto il segnale", non due regimi
    # storici comparabili - esattamente il tipo di artefatto che questa
    # sessione ha imparato a riconoscere altrove. Il walk-forward onesto va
    # fatto SOLO sulla finestra in cui la strategia e' davvero mai stata
    # operativa (36 mesi, ott-2023 -> oggi), divisa a meta'.
    print(f"\n3. WALK-FORWARD onesto sulla finestra REALMENTE operativa ({best_label}):")
    op_start = None
    for d in prices_sub.index:
        n = sum(1 for iid in sealed_ids if meta_sub[iid].get("msrp") and iid in prices_sub.columns
                and not pd.isna(prices_sub.loc[d, iid]) and prices_sub.loc[d, iid] > 0
                and meta_sub[iid].get("release_date") and pd.to_datetime(meta_sub[iid]["release_date"]) <= d)
        if n >= 20:
            op_start = d
            break
    if op_start is None:
        print("   [ATTENZIONE] la cross-section non raggiunge mai min_cross_section=20 - nessun walk-forward possibile.")
    else:
        op_window = prices_sub.loc[op_start:]
        print(f"   Finestra operativa (cross-section >= 20 per la prima volta): {op_start.strftime('%Y-%m')} -> "
              f"{op_window.index[-1].strftime('%Y-%m')} ({len(op_window)} mesi, su {len(prices_sub)} mesi totali di storico)")
        mid = len(op_window) // 2
        for wf_label, dates in [("OP-H1", op_window.index[:mid]), ("OP-H2", op_window.index[mid:])]:
            sub = prices_sub.loc[dates[0]:dates[-1]]
            strat = SealedScarcityValueFactorStrategy(rebalance_every_months=best_rb, top_quantile=best_tq, min_age_months=0)
            res = run_bt(strat, sub, meta_sub)
            print(f"   {wf_label} ({dates[0].strftime('%Y-%m')}->{dates[-1].strftime('%Y-%m')}, {len(dates)}m) | "
                  f"Sharpe {res.sharpe:5.2f} | CAGR {res.cagr*100:+6.2f}% | Trade {res.total_trades:3d}")
        print(f"\n   Per trasparenza, lo split NAIVE sull'intera storia (70 mesi, da NON usare come walk-forward):")
        mid_naive = len(prices_sub) // 2
        for wf_label, dates in [("H1-naive", prices_sub.index[:mid_naive]), ("H2-naive", prices_sub.index[mid_naive:])]:
            sub = prices_sub.loc[dates]
            strat = SealedScarcityValueFactorStrategy(rebalance_every_months=best_rb, top_quantile=best_tq, min_age_months=0)
            res = run_bt(strat, sub, meta_sub)
            print(f"   {wf_label} ({dates[0].strftime('%Y-%m')}->{dates[-1].strftime('%Y-%m')}) | "
                  f"Sharpe {res.sharpe:5.2f} | CAGR {res.cagr*100:+6.2f}% | Trade {res.total_trades:3d}  "
                  f"{'<- artefatto: cross-section sotto soglia, non un vero fallimento' if res.total_trades == 0 else ''}")

    print(f"\n4. Confronto separato: min_age_months 0 vs 4 (esclude finestra ristampe) a {best_label}:")
    age_variants = []
    for ma in [0, 4]:
        strat = SealedScarcityValueFactorStrategy(rebalance_every_months=best_rb, top_quantile=best_tq, min_age_months=ma)
        res = run_bt(strat, prices_sub, meta_sub)
        age_variants.append((f"min_age={ma}m", res))
        print(f"   min_age={ma:2d}m | Sharpe {res.sharpe:5.2f} | CAGR {res.cagr*100:+6.2f}% | Trade {res.total_trades:3d}")
    common_idx2 = age_variants[0][1].monthly_returns.index.intersection(age_variants[1][1].monthly_returns.index)
    perf2 = np.column_stack([r.monthly_returns.loc[common_idx2].values for _, r in age_variants])
    t2 = len(perf2)
    pbo2 = pbo_cscv(perf2[t2 % 4:, :] if t2 % 4 else perf2, n_splits=4)
    print(f"   PBO (2 candidati, 4 split): {pbo2:.3f}")

    print(f"\n5. Confronto con la baseline validata in produzione (TS Momentum, stesso universo):")
    strat_tsmom = TimeSeriesMomentumStrategy(prices_sub, lookback_months=12, max_price_msrp_ratio=MAX_PRICE_TO_MSRP_RATIO)
    res_tsmom = run_bt(strat_tsmom, prices_sub, meta_sub)
    print(f"   TS Momentum (prod. attuale)       | Sharpe {res_tsmom.sharpe:5.2f} | CAGR {res_tsmom.cagr*100:+7.2f}% | "
          f"MaxDD {res_tsmom.max_drawdown*100:7.2f}% | Trade {res_tsmom.total_trades:3d}")
    print(f"   Fattore Scarsita' Sealed ({best_label}) | Sharpe {best_res.sharpe:5.2f} | CAGR {best_res.cagr*100:+7.2f}% | "
          f"MaxDD {best_res.max_drawdown*100:7.2f}% | Trade {best_res.total_trades:3d}")

    common_idx3 = res_tsmom.monthly_returns.index.intersection(best_res.monthly_returns.index)
    if len(common_idx3) > 2:
        corr = res_tsmom.monthly_returns.loc[common_idx3].corr(best_res.monthly_returns.loc[common_idx3])
        print(f"   Correlazione rendimenti mensili (periodo comune, n={len(common_idx3)}): {corr:.2f}")

    print(f"\n{'='*90}\nVERDETTO\n{'='*90}")
    threshold_ok = dsr_own_grid >= 0.90
    print(f"DSR (griglia propria, n_trials={len(grid)}): {dsr_own_grid:.3f} - "
          f"{'SOPRA' if threshold_ok else 'SOTTO'} la soglia di comfort 0.90-0.95 usata in questa ricerca.")
    print("NOTA: questo e' il DSR corretto solo per la griglia di QUESTO test, non per l'intera sessione -")
    print("coerente con lo standard di ogni altro script di calibrazione in questo repo, ma un audit di")
    print("sessione aggiornato (come scripts/dsr_session_audit.py, oggi non aggiornato dai molti script")
    print("box lanciati dopo la sua scrittura) applicherebbe una correzione piu' severa.")
    print(f"Campione: mediana storica 20 box per mese (molto piu' debole delle 932 singole) - interpretare")
    print(f"con cautela proporzionale anche se il DSR passa.")
    print(f"TRACK RECORD EFFETTIVO: la strategia e' davvero operativa (cross-section >= 20) solo da")
    print(f"{op_start.strftime('%Y-%m') if op_start is not None else 'N/D'} - {len(op_window) if op_start is not None else 0} mesi su {len(prices_sub)} di storico totale. Il DSR/Sharpe di produzione")
    print(f"qui sopra sono calcolati sull'intera storia (includono ~34 mesi senza alcun segnale), quindi")
    print(f"NON sono direttamente comparabili al DSR della baseline TS Momentum (operativa da sempre) o")
    print(f"delle singole (operative da sempre, 932 nomi) - un fattore onesto da leggere come 'promettente")
    print(f"su un track record reale di soli 36 mesi', non come 'validato allo stesso livello di confidenza'.")


if __name__ == "__main__":
    main()
