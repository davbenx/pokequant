#!/usr/bin/env python3
"""
scripts/graded_raw_ratio_reliability_test.py — Trovato verificando un prezzo reale
(l'utente ha comprato/controllato Raichu #14 [Fossil 1999] a 230EUR + 20EUR
spedizione = 250EUR, contro 107.60EUR mostrati in dashboard - uno scarto del +132%,
troppo grande per essere solo "sold vs ask" o gap di liquidità EU): il filtro di
attendibilita' esistente (poke_quant/data/liquidity_filter.py) controlla solo
SALTI/VOLATILITA' della serie storica, non il LIVELLO assoluto. La serie grade9
di raichu_14 e' liscia (nessun salto), ma persistentemente troppo bassa - il che
il filtro attuale non puo' vedere per costruzione.

Test: ogni singola in metadata ha gia' un campo "cardmarket_ref_price_eur" (prezzo
raw di riferimento, MAI usato finora in nessuna pipeline) - un'ancora INDIPENDENTE
dal pannello grade9 di PriceCharting. Calcolando il rapporto grade9/raw per ogni
carta e confrontandolo con la coorte di carte della stessa era (+-3 anni di uscita,
richiede >=20 carte in coorte), raichu_14 e' al 24* percentile della sua coorte
(mediana coorte 4.98x, raichu_14 solo 1.92x) - se si applicasse il moltiplicatore
mediano della coorte al suo prezzo raw, il prezzo implicito sarebbe 278EUR, molto
piu' vicino ai 250EUR reali trovati dall'utente che ai 107.60EUR mostrati.

Verificato che NON e' un caso isolato: 9 delle 15 carte nel quantile BUY di oggi
(quelle piu' costose, non le carte bulk/comuni) sono sotto il 40* percentile della
loro coorte sullo stesso rapporto - la stessa direzione, sistematica, non rumore.

ESITO: vedi output.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix, save_metadata
from poke_quant.engine.backtester import Backtester
from poke_quant.engine.strategies.scarcity_value_factor import ScarcityValueFactorStrategy
from poke_quant.validation.statistical_validation import deflated_sharpe_ratio, pbo_cscv

# Soglia scelta PRIMA di guardare l'effetto sul backtest (stesso principio di
# MAX_PRICE_TO_MSRP_RATIO): sotto il 10* percentile della propria coorte d'eta'
# e' un outlier statistico vero, non solo "un po' sotto la mediana" - coorte
# richiede >=20 carte per essere affidabile, altrimenti nessun giudizio (dato
# insufficiente, non si inventa una soglia).
RATIO_PERCENTILE_CUTOFF = 0.10
MIN_COHORT_SIZE = 20
COHORT_WINDOW_YEARS = 3


def compute_ratio_flags(metadata: dict, prices_df: pd.DataFrame) -> dict:
    rows = []
    for item_id, info in metadata.items():
        if info.get("type") != "single":
            continue
        ref = info.get("cardmarket_ref_price_eur")
        rel = info.get("release_date")
        if not ref or ref <= 0 or not rel or item_id not in prices_df.columns:
            continue
        s = prices_df[item_id].dropna()
        s = s[s > 0]
        if s.empty:
            continue
        ratio = float(s.iloc[-1]) / float(ref)
        if not (0.3 < ratio < 30):  # rimuove artefatti grossolani (raw~0 o dato corrotto), non l'oggetto del test
            continue
        rows.append({"item_id": item_id, "year": pd.to_datetime(rel).year, "ratio": ratio})
    df = pd.DataFrame(rows)

    flags = {}
    for _, row in df.iterrows():
        cohort = df[(df["year"] >= row["year"] - COHORT_WINDOW_YEARS) & (df["year"] <= row["year"] + COHORT_WINDOW_YEARS)]
        if len(cohort) < MIN_COHORT_SIZE:
            continue
        cutoff = cohort["ratio"].quantile(RATIO_PERCENTILE_CUTOFF)
        if row["ratio"] < cutoff:
            flags[row["item_id"]] = (row["ratio"], cutoff)
    return flags


def run_bt(strat, prices_df, metadata):
    bt = Backtester(strat, prices_df, metadata, initial_cash=10000.0, platform="cardmarket",
                     apply_liquidity_slippage=True, apply_holding_cost=True, apply_buy_side_shipping=True)
    return bt.run()


def main():
    metadata = load_metadata()
    prices_full = load_price_matrix("historical_prices_graded_singles_grade9.csv")

    flags = compute_ratio_flags(metadata, prices_full)
    print(f"Carte flaggate (grade9/raw sotto il {RATIO_PERCENTILE_CUTOFF*100:.0f}° percentile della loro coorte d'eta'): {len(flags)}")
    for item_id, (ratio, cutoff) in sorted(flags.items(), key=lambda x: x[1][0])[:15]:
        print(f"  {metadata[item_id].get('name', item_id):35s} ratio={ratio:5.2f} (soglia coorte {cutoff:5.2f})")

    singles_ids_base = [
        k for k, v in metadata.items()
        if v.get("type") == "single" and v.get("data_quality") != "thin_unreliable" and k in prices_full.columns
    ]
    singles_ids_strict = [k for k in singles_ids_base if k not in flags]
    print(f"\nUniverso attuale: {len(singles_ids_base)} carte | con il nuovo filtro: {len(singles_ids_strict)} "
          f"({len(singles_ids_base) - len(singles_ids_strict)} escluse in più)")

    from scripts.generate_singles_signal import PRODUCTION_PARAMS

    for label, ids in [("SENZA filtro ratio grade9/raw (attuale)", singles_ids_base),
                        ("CON filtro ratio grade9/raw", singles_ids_strict)]:
        meta_sub = {k: metadata[k] for k in ids}
        prices_sub = prices_full[ids]
        strat = ScarcityValueFactorStrategy(**PRODUCTION_PARAMS)
        res = run_bt(strat, prices_sub, meta_sub)
        dsr = deflated_sharpe_ratio(observed_sr=res.sharpe / np.sqrt(12), n_trials=65, n_obs=len(res.monthly_returns))
        mid = len(prices_sub) // 2
        h1, h2 = prices_sub.index[:mid], prices_sub.index[mid:]
        wf = []
        for dates in (h1, h2):
            sub = prices_sub.loc[dates]
            s2 = ScarcityValueFactorStrategy(**PRODUCTION_PARAMS)
            r2 = run_bt(s2, sub, meta_sub)
            wf.append(r2.sharpe)
        print(f"\n  {label}")
        print(f"    Sharpe {res.sharpe:5.2f} | CAGR {res.cagr*100:+7.2f}% | MaxDD {res.max_drawdown*100:7.2f}% | "
              f"Trade {res.total_trades:3d} | DSR(n=65) {dsr:.3f} | WF H1/H2 {wf[0]:.2f}/{wf[1]:.2f}")


if __name__ == "__main__":
    main()
