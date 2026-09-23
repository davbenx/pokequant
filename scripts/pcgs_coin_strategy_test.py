#!/usr/bin/env python3
"""
scripts/pcgs_coin_strategy_test.py — Prima verifica di due strategie sul nuovo
universo di monete PCGS (18 monete, storico reale 2010-2026 per la maggior
parte - vedi scripts/fetch_pcgs_coins.py):

  1. TS MOMENTUM ASSOLUTO — stessa identica classe validata sui box sigillati
     (TimeSeriesMomentumStrategy). Prima verifica se il trasferimento diretto
     regge su un asset con struttura di liquidita' diversa dalle singole TCG.
  2. PREMIO NUMISMATICO EX-ORO (solo monete d'oro, n=8) — scorpora il prezzo
     nel valore di fusione (once d'oro pure x oro spot reale, da FMP) piu' il
     premio da collezione, e applica TS Momentum al SOLO premio, non al
     prezzo totale - idea specifica alle monete, mai possibile sui TCG.

Rigore applicato FIN DALL'INIZIO (non dopo, come nel caso delle singole TCG):
ogni candidato di griglia conta per il PBO/DSR, e il walk-forward usa lo split
REGIME reale (pre-2020 vs 2020+), non solo meta' campione - il vero motivo
per cui si e' scelto questo asset.

ESITO: ENTRAMBE NON VALIDATE, in modo piu' netto di qualunque fattore testato
sui TCG - non "Sharpe insufficiente", Sharpe NEGATIVO con alta confidenza.

Prima versione (media mensile, 18 monete): MaxDD -83/-88%, rendimenti mensili
con deviazione standard fino all'81% e picchi +360%/-76% per le monete meno
liquide - artefatto di aggregare 1-2 vendite d'asta isolate al mese, non un
vero movimento di mercato. Corretto con mediana + filtro di liquidita' minima
(>=75/200 mesi con vendite, sopravvivono 5 monete su 18) - il risultato NON
cambia qualitativamente, solo si attenua: Sharpe -0.96/-1.01/-1.00 (3
candidati), PBO 75.7%, DSR 0.000, walk-forward per REGIME (non meta' campione)
negativo in ENTRAMBI - PRE-2020 (2010-2019) Sharpe -1.14, 2020+ Sharpe -0.26.
Il premio numismatico ex-oro (4 monete d'oro liquide): stessa storia, Sharpe
-1.01/-1.20/-1.24, PBO 70.0%, DSR 0.000.

Perche' ha senso, non solo "un altro fallimento": le monete gold-backed sono
state scelte apposta per avere un prezzo ancorato a una commodity macro molto
liquida ed efficientemente prezzata (oro spot), proprio per uscire dal singolo
super-ciclo hype-pandemico che ha contaminato ogni fattore TCG. Ma questo e'
in tensione diretta con l'obiettivo di trovare momentum sfruttabile: l'oro,
seguito da capitale istituzionale enorme, non ha lo stesso tipo di inefficienza
da hype retail che rendeva profittevole il trend-following sui box sigillati
Pokemon (2020-2021 boom, 2023+ recupero - mosse sostenute, non un mercato
laterale). Il periodo 2011-2015 (mercato ribassista dell'oro, ben documentato,
da $1900 a $1050/oz) e' probabilmente la causa del pessimo Sharpe PRE-2020:
esattamente il tipo di mercato "senza trend, choppy" in cui il trend-following
perde per attrito su falsi segnali, il classico kryptonite del momentum.
Conclusione: la stessa caratteristica che rende le monete un buon campione
multi-regime (prezzo ancorato a una commodity efficiente) le rende anche un
pessimo candidato per il momentum - le due proprieta' sono in tensione, non
indipendenti.
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.engine.strategies.time_series_momentum import TimeSeriesMomentumStrategy
from poke_quant.engine.backtester import Backtester
from poke_quant.validation.statistical_validation import deflated_sharpe_ratio, pbo_cscv
from poke_quant.validation.bootstrap import block_bootstrap_metrics, summarize_bootstrap

COIN_FILE = Path(__file__).resolve().parent.parent / "data_cache" / "pcgs_coin_prices.json"
GOLD_FILE = Path(__file__).resolve().parent.parent / "data_cache" / "gold_spot_usd_monthly.json"
GOLD_OZ_PER_20DOLLAR = 0.96750  # once di oro puro in un $20 Liberty/St. Gaudens (.900 fine, 33.436g)


def run_bt(strategy, prices_df, metadata):
    bt = Backtester(strategy, prices_df, metadata, initial_cash=10000.0, platform="cardmarket",
                     apply_liquidity_slippage=True, apply_holding_cost=True)
    return bt.run()


MIN_VALID_MONTHS = 75  # su 200 mesi totali - esclude le monete troppo sottili (poche vendite/anno,
                        # dove una singola asta anomala genera oscillazioni mensili del 300%+ che non
                        # sono un vero movimento di mercato, solo rumore da lotto isolato)


def build_monthly_panel(coin_data: dict) -> tuple[pd.DataFrame, dict]:
    series = {}
    meta = {}
    for item_id, info in coin_data.items():
        auctions = info["auctions"]
        if not auctions:
            continue
        rows = []
        for a in auctions:
            try:
                dt = pd.to_datetime(a["Date"], format="%m-%Y")
                rows.append((dt, float(a["Price"])))
            except Exception:
                continue
        if not rows:
            continue
        # Mediana, non media: robusta al singolo lotto d'asta anomalo (provenienza, difetti,
        # sticker CAC) che altrimenti domina il prezzo del mese quando le vendite sono poche.
        s = pd.DataFrame(rows, columns=["date", "price"]).groupby("date")["price"].median()
        s.index = s.index.to_period("M").to_timestamp()
        series[item_id] = s
        meta[item_id] = {
            "name": info["name"], "type": "single", "segment": info["segment"],
            "release_date": "1850-01-01",  # eta' irrilevante per monete storiche - min_age_months=0
        }
    full_index = pd.date_range(
        min(s.index.min() for s in series.values()), max(s.index.max() for s in series.values()), freq="MS"
    )
    df = pd.DataFrame({k: v.reindex(full_index) for k, v in series.items()})

    valid_counts = df.notna().sum()
    thin = valid_counts[valid_counts < MIN_VALID_MONTHS].index.tolist()
    if thin:
        print(f"Escluse per liquidita' insufficiente (<{MIN_VALID_MONTHS}/{len(full_index)} mesi con vendite): "
              f"{[meta[c]['name'] for c in thin]}")
    df = df.drop(columns=thin)
    meta = {k: v for k, v in meta.items() if k not in thin}

    df = df.ffill(limit=2)  # buchi brevi (nessuna vendita quel mese) - non oltre 2 mesi
    return df, meta


def main():
    coin_data = json.loads(COIN_FILE.read_text())
    prices_df, meta = build_monthly_panel(coin_data)
    print(f"Pannello: {prices_df.shape[1]} monete, {prices_df.shape[0]} mesi "
          f"({prices_df.index[0].strftime('%Y-%m')} -> {prices_df.index[-1].strftime('%Y-%m')})")

    # ================= STRATEGIA 1: TS MOMENTUM ASSOLUTO =================
    print("\n" + "=" * 90)
    print("1) TS MOMENTUM ASSOLUTO SU MONETE (stessa classe validata sui box sigillati)")
    print("=" * 90)
    lookbacks = [6, 9, 12]
    results = {}
    for lb in lookbacks:
        strat = TimeSeriesMomentumStrategy(prices_df, lookback_months=lb, item_type_filter="single", min_age_months=0)
        res = run_bt(strat, prices_df, meta)
        results[f"lb={lb}m"] = res
        print(f"  lb={lb:2d}m | CAGR {res.cagr*100:+6.2f}% | Sharpe {res.sharpe:5.2f} | "
              f"MaxDD {res.max_drawdown*100:6.2f}% | Trade {res.total_trades:4d}")

    common_idx = None
    for res in results.values():
        common_idx = res.monthly_returns.index if common_idx is None else common_idx.intersection(res.monthly_returns.index)
    perf_matrix = np.column_stack([results[k].monthly_returns.loc[common_idx].values for k in results])
    t_len = len(perf_matrix)
    splits = 8 if t_len >= 32 else 4
    rem = t_len % splits
    pbo = pbo_cscv(perf_matrix[rem:, :] if rem else perf_matrix, n_splits=splits)
    print(f"\nPBO ({len(results)} candidati, {splits} split): {pbo:.3f}")

    best_name = max(results, key=lambda k: results[k].sharpe)
    best = results[best_name]
    dsr = deflated_sharpe_ratio(observed_sr=best.sharpe / np.sqrt(12), n_trials=len(results), n_obs=len(best.monthly_returns))
    print(f"Migliore: '{best_name}' Sharpe {best.sharpe:.2f} | Trade {best.total_trades} | DSR: {dsr:.3f}")

    sims = block_bootstrap_metrics(best.monthly_returns, n_sims=500, block_size=6)
    print(summarize_bootstrap(sims))

    lb_best = int(best_name.split("=")[1].rstrip("m"))
    regime_split = pd.Timestamp("2020-01-01")
    pre_dates = prices_df.index[prices_df.index < regime_split]
    post_dates = prices_df.index[prices_df.index >= regime_split]
    print(f"\nWalk-forward per REGIME (non meta' campione): PRE-2020 ({len(pre_dates)} mesi) vs 2020+ ({len(post_dates)} mesi)")
    for label, dates in [("PRE-2020", pre_dates), ("2020+", post_dates)]:
        if len(dates) < 12:
            print(f"  {label}: troppo pochi mesi ({len(dates)}), salto")
            continue
        sub = prices_df.loc[dates]
        strat = TimeSeriesMomentumStrategy(sub, lookback_months=lb_best, item_type_filter="single", min_age_months=0)
        r = run_bt(strat, sub, meta)
        print(f"  {label} ({dates[0].strftime('%Y-%m')}->{dates[-1].strftime('%Y-%m')}) | "
              f"Sharpe {r.sharpe:5.2f} | CAGR {r.cagr*100:+6.2f}% | Trade {r.total_trades}")

    # ================= STRATEGIA 3: PREMIO NUMISMATICO EX-ORO =================
    print("\n" + "=" * 90)
    print("3) PREMIO NUMISMATICO EX-ORO (solo monete d'oro)")
    print("=" * 90)
    gold_spot = pd.Series({pd.to_datetime(k): v for k, v in json.loads(GOLD_FILE.read_text()).items()}).sort_index()
    gold_ids = [k for k, v in meta.items() if v["segment"] == "gold"]
    gold_prices = prices_df[gold_ids]
    common_gold_idx = gold_prices.index.intersection(gold_spot.index)
    gold_prices, gold_spot_aligned = gold_prices.loc[common_gold_idx], gold_spot.loc[common_gold_idx]

    melt_value = gold_spot_aligned * GOLD_OZ_PER_20DOLLAR
    premium_df = gold_prices.sub(melt_value, axis=0)
    premium_pct_df = gold_prices.div(melt_value, axis=0) - 1.0  # premio in % del valore di fusione

    print(f"Universo oro: {len(gold_ids)} monete, {len(common_gold_idx)} mesi")
    print(f"Premio medio attuale (ultimo mese): {premium_pct_df.iloc[-1].mean()*100:+.1f}% sopra il valore di fusione")

    n_negative = (premium_df < 0).sum().sum()
    print(f"Mesi con premio negativo (prezzo sotto il valore di fusione): {n_negative} su {premium_df.size}")

    gold_meta = {k: meta[k] for k in gold_ids}
    results_premium = {}
    for lb in lookbacks:
        # Segnale sul PREMIO IN DOLLARI (prezzo - valore di fusione, resta positivo per queste
        # monete da collezione), esecuzione sul prezzo reale della moneta (gold_prices) - stessa
        # decoupling segnale/esecuzione usata per il rapporto tra gradi nella ricerca sui TCG.
        strat = TimeSeriesMomentumStrategy(premium_df, lookback_months=lb, item_type_filter="single", min_age_months=0)
        res_bt = Backtester(strat, gold_prices, gold_meta, initial_cash=10000.0, platform="cardmarket",
                             apply_liquidity_slippage=True, apply_holding_cost=True)
        res = res_bt.run()
        results_premium[f"premio lb={lb}m"] = res
        print(f"  premio lb={lb:2d}m | CAGR {res.cagr*100:+6.2f}% | Sharpe {res.sharpe:5.2f} | "
              f"MaxDD {res.max_drawdown*100:6.2f}% | Trade {res.total_trades:4d}")

    if results_premium:
        common_idx2 = None
        for res in results_premium.values():
            common_idx2 = res.monthly_returns.index if common_idx2 is None else common_idx2.intersection(res.monthly_returns.index)
        perf_matrix2 = np.column_stack([results_premium[k].monthly_returns.loc[common_idx2].values for k in results_premium])
        t_len2 = len(perf_matrix2)
        splits2 = 8 if t_len2 >= 32 else 4
        rem2 = t_len2 % splits2
        pbo2 = pbo_cscv(perf_matrix2[rem2:, :] if rem2 else perf_matrix2, n_splits=splits2)
        print(f"\nPBO premio ({len(results_premium)} candidati, {splits2} split): {pbo2:.3f}")
        best_name2 = max(results_premium, key=lambda k: results_premium[k].sharpe)
        best2 = results_premium[best_name2]
        dsr2 = deflated_sharpe_ratio(observed_sr=best2.sharpe / np.sqrt(12), n_trials=len(results_premium), n_obs=len(best2.monthly_returns))
        print(f"Migliore: '{best_name2}' Sharpe {best2.sharpe:.2f} | Trade {best2.total_trades} | DSR: {dsr2:.3f}")


if __name__ == "__main__":
    main()
