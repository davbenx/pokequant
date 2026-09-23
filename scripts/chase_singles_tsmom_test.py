#!/usr/bin/env python3
"""
scripts/chase_singles_tsmom_test.py — Applica la STESSA TimeSeriesMomentumStrategy
usata sui box (momentum assoluto per asset, indipendente, non un ranking
cross-sezionale) alle singole "chase" - le carte selezionate da
discover_chase_cards.py perche' il loro prezzo CORRENTE supera una soglia
(selection_method="chase_price_filter_survivorship_biased", 639 carte con
storico) - contro il campione di controllo casuale (225 carte, nessun filtro
di prezzo) e contro l'universo pieno.

Motivazione: ogni test precedente sulle singole usava CrossSectionalFactorStrategy
(ranking relativo, compra il top 30% per momentum) - mai la stessa logica
assoluta/indipendente per asset usata sui box. Se l'edge dei box "viene dalle
sue carte" nel senso specifico delle chase card (non dalla media di tutte le
carte, gia' testata in scripts/box_vs_singles_basket_test.py), dovrebbe
comparire qui.

ATTENZIONE METODOLOGICA: "chase" qui e' selezionato sul prezzo CORRENTE
(survivorship bias) - lo stesso identico problema documentato in
poke_quant/engine/strategies/rarity_tier_factor.py. Il confronto con
random_control e' la diagnostica per separarlo da un vero effetto.

ESITO: NON VALIDATO, e il confronto chase-vs-random-control lo dimostra invece
di limitarsi a sospettarlo. Chase (n=639): Sharpe 0.18/CAGR +4.6%/MaxDD -47%.
Random control (n=225, stesso identico metodo, nessun filtro di prezzo):
Sharpe -0.81/CAGR -11.4%/MaxDD -60% - la stessa strategia sulle stesse carte
di controllo PERDE, non solo "non vince". PBO sui 3 candidati 11.4%, ma DSR
delle chase corretto per 3 trial e' 0.331 - molto sotto la soglia di comfort
usata in questa ricerca (0.90-0.95). Il walk-forward chiude la questione:
CHASE H1 Sharpe -1.51 -> H2 +1.16 (il solito schema boom/bust); RANDOM CONTROL
H1 -0.97 -> H2 -0.27 - negativo IN ENTRAMBE le metà, nessun recupero. Le chase
card sono selezionate perche' il loro prezzo CORRENTE e' alto (selezione
sull'esito) - e' quasi tautologico che una strategia long-biased su un
campione scelto cosi' mostri un recupero nella seconda meta' del periodo,
mentre lo stesso identico metodo su un campione onesto (random control) non
lo mostra per niente. Questo e' il caso piu' diretto di survivorship bias
osservato in questa ricerca, non solo il piu' sospetto (vedi anche
poke_quant/engine/strategies/rarity_tier_factor.py per lo stesso pattern su
un universo di eleggibilita' diverso ma altrettanto contaminato).
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.engine.strategies.time_series_momentum import TimeSeriesMomentumStrategy
from poke_quant.validation.statistical_validation import deflated_sharpe_ratio, pbo_cscv
from poke_quant.validation.bootstrap import block_bootstrap_metrics, summarize_bootstrap
from scripts.optimize_and_falsify import run_bt


def main():
    metadata = load_metadata()
    prices_full = load_price_matrix("historical_prices_graded_singles_grade9.csv")

    universes = {}
    for method in ["chase_price_filter_survivorship_biased", "random_control"]:
        ids = [
            k for k, v in metadata.items()
            if v.get("type") == "single" and v.get("selection_method") == method
            and v.get("data_quality") != "thin_unreliable" and k in prices_full.columns
        ]
        universes[method] = ids
    universes["FULL (chase+control)"] = universes["chase_price_filter_survivorship_biased"] + universes["random_control"]

    results = {}
    print("TS Momentum (lookback=12m, stessa strategia dei box) per universo:\n")
    for name, ids in universes.items():
        meta_sub = {k: metadata[k] for k in ids}
        prices_sub = prices_full[ids]
        strat = TimeSeriesMomentumStrategy(prices_sub, lookback_months=12, item_type_filter="single")
        res = run_bt(strat, prices_sub, meta_sub)
        results[name] = (res, prices_sub, meta_sub)
        print(f"  {name:42s} (n={len(ids):4d}) | CAGR {res.cagr*100:+6.2f}% | Sharpe {res.sharpe:5.2f} | "
              f"MaxDD {res.max_drawdown*100:6.2f}% | Trade {res.total_trades:4d}")

    common_idx = None
    for res, _, _ in results.values():
        common_idx = res.monthly_returns.index if common_idx is None else common_idx.intersection(res.monthly_returns.index)
    perf_matrix = np.column_stack([results[k][0].monthly_returns.loc[common_idx].values for k in universes])
    t_len = len(perf_matrix)
    splits = 8 if t_len >= 32 else 4
    rem = t_len % splits
    pbo = pbo_cscv(perf_matrix[rem:, :] if rem else perf_matrix, n_splits=splits)
    print(f"\nPBO (3 candidati, {splits} split): {pbo:.3f}")

    chase_res = results["chase_price_filter_survivorship_biased"][0]
    dsr = deflated_sharpe_ratio(observed_sr=chase_res.sharpe / np.sqrt(12), n_trials=len(universes), n_obs=len(chase_res.monthly_returns))
    print(f"DSR carte 'chase' (n_trials={len(universes)}): {dsr:.3f}")

    sims = block_bootstrap_metrics(chase_res.monthly_returns, n_sims=500, block_size=6)
    print(f"\nBlock bootstrap (500 sim, blocchi 6m) sulle carte 'chase':")
    print(summarize_bootstrap(sims))

    print(f"\nWalk-forward H1/H2:")
    for name, (res, prices_sub, meta_sub) in results.items():
        mid = len(prices_sub) // 2
        h1_dates, h2_dates = prices_sub.index[:mid], prices_sub.index[mid:]
        for label, dates in [("H1", h1_dates), ("H2", h2_dates)]:
            sub = prices_sub.loc[dates]
            strat = TimeSeriesMomentumStrategy(sub, lookback_months=12, item_type_filter="single")
            r = run_bt(strat, sub, meta_sub)
            print(f"  {name:42s} {label} ({dates[0].strftime('%Y-%m')}->{dates[-1].strftime('%Y-%m')}) | "
                  f"Sharpe {r.sharpe:5.2f} | CAGR {r.cagr*100:+6.2f}%")


if __name__ == "__main__":
    main()
