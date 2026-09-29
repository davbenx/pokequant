#!/usr/bin/env python3
"""
scripts/update_monthly_cache.py — Precomputa e serializza tutti i dati della
dashboard di PokeQuant in file JSON statici salvati in data_cache/:
  - data_cache/precomputed_dashboard_data.json (segnali box, singole, indici, backtest)
  - data_cache/product_image_cache.json (link immagini cover pre-scaricati)

Permette alla dashboard Streamlit di caricarsi istantaneamente (<0.3s) leggendo
direttamente da disco/GitHub, senza eseguire computazioni OLS pesanti, backtest
da 8-16s o chiamate di rete ripetute a PriceCharting ad ogni refresh pagina.

Esecuzione:
  python scripts/update_monthly_cache.py
"""

from __future__ import annotations
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.data.liquidity_filter import (
    liquid_sealed_ids, liquid_singles_ids, MAX_PRICE_TO_MSRP_RATIO
)
from poke_quant.engine.backtester import Backtester
from poke_quant.engine.strategies.time_series_momentum import TimeSeriesMomentumStrategy
from poke_quant.engine.strategies.scarcity_value_factor import ScarcityValueFactorStrategy
from poke_quant.engine.position_sizing import inverse_vol_split
from poke_quant.data.price_fetcher import fetch_pricecharting_cover_image_url
from scripts.generate_monthly_signal import compute_signal_rows, MODERN_ERA_CUTOFF
from scripts.generate_singles_signal import (
    compute_singles_signal_rows, compute_singles_alternative_rows, compute_singles_avoid_rows,
    PRODUCTION_PARAMS as SINGLES_PROD_PARAMS, DAC7_SINGLES_PARAMS
)

DATA_CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"
PRECOMPUTED_FILE = DATA_CACHE_DIR / "precomputed_dashboard_data.json"
IMAGE_CACHE_FILE = DATA_CACHE_DIR / "product_image_cache.json"


def _sanitize_for_json(obj):
    """Converte ricorsivamente float np.nan, Timestamp e numpy types in formati JSON standard."""
    if isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
        return str(obj)
    elif isinstance(obj, np.generic):
        val = obj.item()
        if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
            return None
        return val
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, pd.Series):
        s_dict = {str(k): _sanitize_for_json(v) for k, v in obj.dropna().to_dict().items()}
        return s_dict
    elif isinstance(obj, pd.DataFrame):
        df_clean = obj.copy()
        for c in df_clean.columns:
            if pd.api.types.is_datetime64_any_dtype(df_clean[c]):
                df_clean[c] = df_clean[c].dt.strftime("%Y-%m-%d")
        return df_clean.to_dict(orient="records")
    return obj


def compute_market_indices_data(metadata: dict, prices_full: pd.DataFrame) -> dict:
    sealed_ids = liquid_sealed_ids(metadata, prices_full, modern_era_cutoff=MODERN_ERA_CUTOFF)

    segments: dict[str, list[str]] = {"Pokémon EN": [], "Pokémon JP": [], "One Piece TCG": []}
    for k in sealed_ids:
        v = metadata[k]
        if v.get("franchise") == "one_piece":
            segments["One Piece TCG"].append(k)
        elif v.get("language") == "jp":
            segments["Pokémon JP"].append(k)
        else:
            segments["Pokémon EN"].append(k)

    def build_index(ids: list[str]) -> pd.Series:
        sub = prices_full[ids]
        basket_ret = sub.pct_change().mean(axis=1, skipna=True).fillna(0.0)
        idx = (1.0 + basket_ret).cumprod() * 100.0
        idx.iloc[0] = 100.0
        return idx

    overall_index = build_index(sealed_ids)
    segment_indices = {name: build_index(ids) for name, ids in segments.items() if len(ids) >= 3}
    segment_counts = {name: len(ids) for name, ids in segments.items()}

    lookback = 12
    breadth = {}
    for i in range(lookback, len(prices_full.index)):
        date = str(prices_full.index[i])[:10]
        n_pos = n_tot = 0
        for k in sealed_ids:
            series = prices_full[k].iloc[: i + 1].dropna()
            series = series[series > 0]
            if len(series) < lookback + 1:
                continue
            past, now = float(series.iloc[-(lookback + 1)]), float(series.iloc[-1])
            if past <= 0:
                continue
            n_tot += 1
            if (now - past) / past > 0:
                n_pos += 1
        if n_tot > 0:
            breadth[date] = n_pos / n_tot * 100.0

    return {
        "overall_index": {str(k)[:10]: round(float(v), 2) for k, v in overall_index.items()},
        "segment_indices": {
            seg: {str(k)[:10]: round(float(v), 2) for k, v in s.items()}
            for seg, s in segment_indices.items()
        },
        "segment_counts": segment_counts,
        "breadth_series": {k: round(float(v), 1) for k, v in breadth.items()},
    }


def serialize_backtest_res(res, n_universe: int) -> dict:
    m_ret = {str(k)[:10]: float(v) for k, v in res.monthly_returns.items()}
    nav_hist = []
    for dt, row in res.nav_history.iterrows():
        nav_hist.append({
            "date": str(dt)[:10],
            "nav": round(float(row["nav"]), 2),
            "cash": round(float(row.get("cash", 0.0)), 2),
            "portfolio_value": round(float(row.get("portfolio_value", 0.0)), 2),
        })

    trades_list = []
    if not res.trades_df.empty:
        t_df = res.trades_df.copy()
        for _, tr in t_df.iterrows():
            trades_list.append({
                "item_id": tr.get("item_id"),
                "item_name": tr.get("item_name"),
                "buy_date": str(tr.get("buy_date"))[:10],
                "sell_date": str(tr.get("sell_date"))[:10],
                "holding_months": float(tr.get("holding_months", 0)),
                "buy_price_unit": round(float(tr.get("buy_price_unit", 0)), 2),
                "sell_price_unit": round(float(tr.get("sell_price_unit", 0)), 2),
                "quantity": int(tr.get("quantity", 1)),
                "net_roi": round(float(tr.get("net_roi", 0)), 4),
                "net_pnl": round(float(tr.get("net_pnl", 0)), 2),
            })

    years = max(1e-6, (pd.to_datetime(res.trades_df["sell_date"]).max() - pd.to_datetime(res.trades_df["sell_date"]).min()).days / 365.25) if not res.trades_df.empty else 1.0
    eur_per_year = (res.trades_df["sell_price_unit"] * res.trades_df["quantity"]).sum() / years if not res.trades_df.empty else 0.0
    trades_per_year = len(res.trades_df) / years if not res.trades_df.empty else 0.0

    return {
        "strategy_name": res.strategy_name,
        "n_universe": n_universe,
        "total_trades": res.total_trades,
        "win_rate": round(float(res.win_rate), 4),
        "profit_factor": round(float(res.profit_factor), 2),
        "sharpe": round(float(res.sharpe), 2),
        "cagr": round(float(res.cagr), 4),
        "max_drawdown": round(float(res.max_drawdown), 4),
        "monthly_returns": m_ret,
        "nav_history": nav_hist,
        "trades": trades_list,
        "trades_per_year": round(trades_per_year, 1),
        "eur_per_year": round(eur_per_year, 2),
    }


def prefetch_and_update_image_cache(all_pairs: set[tuple[str, str]]) -> dict:
    """Carica l'image cache esistente, scarica in parallelo le immagini mancanti e salva."""
    cache = {}
    if IMAGE_CACHE_FILE.exists():
        try:
            with open(IMAGE_CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception as e:
            print(f"Warning reading image cache: {e}")

    missing = []
    for gs, isl in all_pairs:
        key = f"{gs}:{isl}"
        if key not in cache or cache[key] is None:
            missing.append((gs, isl))

    if missing:
        print(f"Prefetching {len(missing)} missing product images politely (max_workers=2)...")
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_pair = {}
            for gs, isl in missing:
                future_to_pair[executor.submit(fetch_pricecharting_cover_image_url, gs, isl)] = (gs, isl)
                time.sleep(0.15)
            done_count = 0
            for future in as_completed(future_to_pair):
                gs, isl = future_to_pair[future]
                key = f"{gs}:{isl}"
                try:
                    url = future.result()
                    if url:
                        cache[key] = url
                except Exception:
                    pass
                done_count += 1
                if done_count % 25 == 0 or done_count == len(missing):
                    print(f"  Images fetched: {done_count}/{len(missing)}")

        with open(IMAGE_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
        found_cnt = sum(1 for v in cache.values() if v)
        print(f"Image cache written to {IMAGE_CACHE_FILE} ({found_cnt}/{len(cache)} valid images).")
    else:
        print(f"All {len(all_pairs)} product images are already cached in {IMAGE_CACHE_FILE}.")

    return cache


def main():
    t_start = time.time()
    print("=" * 65)
    print("🚀 POKEQUANT: AVVIO PRECOMPUTAZIONE DASHBOARD E CACHE STATICA")
    print("=" * 65)

    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("\n1. Caricamento metadati e prezzi storici...")
    metadata = load_metadata()
    prices_box = load_price_matrix()
    prices_singles = load_price_matrix("historical_prices_graded_singles_grade9.csv")
    latest_date_box = str(prices_box.index[-1])[:10]
    latest_date_singles = str(prices_singles.index[-1])[:10]
    print(f"   Prezzi box al: {latest_date_box} ({len(prices_box)} mesi, {prices_box.shape[1]} serie)")
    print(f"   Prezzi singole al: {latest_date_singles} ({len(prices_singles)} mesi, {prices_singles.shape[1]} serie)")

    print("\n2. Calcolo segnali Box sigillati (TS Momentum)...")
    box_rows, _ = compute_signal_rows()
    print(f"   {len(box_rows)} segnali box calcolati.")

    print("\n3. Calcolo indici di mercato ed ampiezza...")
    market_indices = compute_market_indices_data(metadata, prices_box)
    print(f"   Indici calcolati: {list(market_indices['segment_indices'].keys())}")

    print("\n4. Calcolo segnali Singole Grade 9 (Produzione & DAC7)...")
    print("   -> Singole BUY Produzione...")
    singles_prod_buy, _ = compute_singles_signal_rows(SINGLES_PROD_PARAMS)
    print("   -> Singole Alternative Produzione...")
    # shown_rows=singles_prod_buy evita che questa funzione ricalcoli
    # internamente compute_singles_signal_rows(SINGLES_PROD_PARAMS) (trovato
    # in audit performance 2026-09-29: veniva rifatto identico due righe sopra).
    singles_prod_alt, _ = compute_singles_alternative_rows(SINGLES_PROD_PARAMS, shown_rows=singles_prod_buy)
    print("   -> Singole Uscite/Avoid Produzione...")
    singles_prod_avoid, _ = compute_singles_avoid_rows(SINGLES_PROD_PARAMS)

    # DAC7_SINGLES_PARAMS e' un alias letterale di PRODUCTION_PARAMS (vedi
    # generate_singles_signal.py) - stesso identico output della Produzione,
    # nessun bisogno di ricalcolare (fit OLS + universo liquido inclusi) 3
    # volte in piu' per parametri che coincidono esattamente. Se in futuro
    # DAC7_SINGLES_PARAMS smette di essere un vero alias, il check sotto se
    # ne accorge da solo e torna al calcolo separato.
    if DAC7_SINGLES_PARAMS == SINGLES_PROD_PARAMS:
        print("   -> DAC7 identico a Produzione (stessi parametri) - riuso risultato, nessun ricalcolo.")
        singles_dac7_buy, singles_dac7_alt, singles_dac7_avoid = singles_prod_buy, singles_prod_alt, singles_prod_avoid
    else:
        print("   -> Singole BUY DAC7...")
        singles_dac7_buy, _ = compute_singles_signal_rows(DAC7_SINGLES_PARAMS)
        print("   -> Singole Alternative DAC7...")
        singles_dac7_alt, _ = compute_singles_alternative_rows(DAC7_SINGLES_PARAMS, shown_rows=singles_dac7_buy)
        print("   -> Singole Uscite/Avoid DAC7...")
        singles_dac7_avoid, _ = compute_singles_avoid_rows(DAC7_SINGLES_PARAMS)

    print(f"   Produzione: {len(singles_prod_buy)} BUY, {len(singles_prod_alt)} Alt, {len(singles_prod_avoid)} Avoid")
    print(f"   DAC7:       {len(singles_dac7_buy)} BUY, {len(singles_dac7_alt)} Alt, {len(singles_dac7_avoid)} Avoid")

    print("\n5. Esecuzione backtest completi (una tantum per serializzazione)...")
    # Backtest Box
    sealed_ids = liquid_sealed_ids(metadata, prices_box, modern_era_cutoff=MODERN_ERA_CUTOFF)
    meta_box_sub = {k: metadata[k] for k in sealed_ids}
    prices_box_sub = prices_box[sealed_ids]
    strat_box = TimeSeriesMomentumStrategy(prices_box_sub, lookback_months=12, max_price_msrp_ratio=MAX_PRICE_TO_MSRP_RATIO)
    bt_box = Backtester(strat_box, prices_box_sub, meta_box_sub, initial_cash=10000.0, platform="cardmarket",
                         apply_liquidity_slippage=True, apply_holding_cost=True, apply_buy_side_shipping=True)
    res_box = bt_box.run()
    bt_box_dict = serialize_backtest_res(res_box, len(sealed_ids))
    print(f"   Backtest Box completato: Sharpe {res_box.sharpe:.2f}, CAGR +{res_box.cagr*100:.1f}%, Trades {res_box.total_trades}")

    # Backtest Singole Produzione
    singles_ids = liquid_singles_ids(metadata, prices_singles)
    meta_singles_sub = {k: metadata[k] for k in singles_ids}
    prices_singles_sub = prices_singles[singles_ids]
    strat_singles_prod = ScarcityValueFactorStrategy(**SINGLES_PROD_PARAMS)
    bt_singles_prod = Backtester(strat_singles_prod, prices_singles_sub, meta_singles_sub, initial_cash=10000.0, platform="cardmarket",
                                 apply_liquidity_slippage=True, apply_holding_cost=True, apply_buy_side_shipping=True)
    res_singles_prod = bt_singles_prod.run()
    bt_singles_prod_dict = serialize_backtest_res(res_singles_prod, len(singles_ids))
    print(f"   Backtest Singole PROD completato: Sharpe {res_singles_prod.sharpe:.2f}, CAGR +{res_singles_prod.cagr*100:.1f}%, Trades {res_singles_prod.total_trades}")

    # Backtest Singole DAC7
    strat_singles_dac7 = ScarcityValueFactorStrategy(**DAC7_SINGLES_PARAMS)
    bt_singles_dac7 = Backtester(strat_singles_dac7, prices_singles_sub, meta_singles_sub, initial_cash=10000.0, platform="cardmarket",
                                 apply_liquidity_slippage=True, apply_holding_cost=True, apply_buy_side_shipping=True)
    res_singles_dac7 = bt_singles_dac7.run()
    bt_singles_dac7_dict = serialize_backtest_res(res_singles_dac7, len(singles_ids))
    print(f"   Backtest Singole DAC7 completato: Sharpe {res_singles_dac7.sharpe:.2f}, CAGR +{res_singles_dac7.cagr*100:.1f}%, Trades {res_singles_dac7.total_trades}")

    # Risk parity split - formula centralizzata in
    # poke_quant.engine.position_sizing.inverse_vol_split (trovato in audit
    # generale, 2026-09-29: questo script e app.py::get_box_singles_split()
    # reimplementavano la stessa formula in due punti indipendenti, a rischio
    # di divergere silenziosamente).
    common_idx = res_box.monthly_returns.index.intersection(res_singles_prod.monthly_returns.index)
    vol_box = float(res_box.monthly_returns.loc[common_idx].std())
    vol_singles = float(res_singles_prod.monthly_returns.loc[common_idx].std())
    w_box, w_singles = inverse_vol_split(vol_box, vol_singles, round_to=2)
    print(f"   Risk-parity split calibrato: {w_box*100:.0f}% Box / {w_singles*100:.0f}% Singole")

    print("\n6. Identificazione prodotti e prefetch copertine...")
    all_items = set()
    for r in box_rows:
        all_items.add(r["item_id"])
    for r in singles_prod_buy + singles_prod_alt + singles_prod_avoid + singles_dac7_buy + singles_dac7_alt:
        all_items.add(r["item_id"])

    all_pairs = set()
    for item_id in all_items:
        info = metadata.get(item_id, {})
        gs, isl = info.get("game_slug"), info.get("item_slug")
        if gs and isl:
            all_pairs.add((gs, isl))

    prefetch_and_update_image_cache(all_pairs)

    print("\n7. Serializzazione dati completi in JSON...")
    precomputed_data = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "latest_date_box": latest_date_box,
        "latest_date_singles": latest_date_singles,
        "risk_parity": {
            "w_box": w_box,
            "w_singles": w_singles,
            "vol_box": round(vol_box, 4),
            "vol_singles": round(vol_singles, 4),
        },
        "box_signals": _sanitize_for_json(box_rows),
        "market_indices": _sanitize_for_json(market_indices),
        "singles_signals": {
            "production": {
                "buy_rows": _sanitize_for_json(singles_prod_buy),
                "alt_rows": _sanitize_for_json(singles_prod_alt),
                "avoid_rows": _sanitize_for_json(singles_prod_avoid),
            },
            "dac7": {
                "buy_rows": _sanitize_for_json(singles_dac7_buy),
                "alt_rows": _sanitize_for_json(singles_dac7_alt),
                "avoid_rows": _sanitize_for_json(singles_dac7_avoid),
            },
        },
        "backtest_results": {
            "box": bt_box_dict,
            "singles_production": bt_singles_prod_dict,
            "singles_dac7": bt_singles_dac7_dict,
        },
    }

    with open(PRECOMPUTED_FILE, "w", encoding="utf-8") as f:
        json.dump(precomputed_data, f, indent=2)

    fsize_mb = PRECOMPUTED_FILE.stat().st_size / (1024 * 1024)
    elapsed = time.time() - t_start
    print(f"\n✅ File salvato con successo: {PRECOMPUTED_FILE} ({fsize_mb:.2f} MB)")
    print(f"⏱️ Tempo totale impiegato: {elapsed:.2f} secondi.")
    print("=" * 65)


if __name__ == "__main__":
    main()
