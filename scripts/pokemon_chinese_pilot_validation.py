#!/usr/bin/env python3
"""
scripts/pokemon_chinese_pilot_validation.py — Progetto pilota Pokemon Cinese,
validazione box (stesso rigore di scripts/mtg_pilot_validation.py).

Universo scoperto da scripts/discover_pokemon_chinese_sealed_universe.py (15
prodotti sealed reali, verificati dal vivo). Prima di qualunque Sharpe/DSR,
verifica quanti superano il requisito minimo di TimeSeriesMomentumStrategy
(lookback_months=12, cioe' >=13 mesi di storico reale per calcolare un solo
rendimento trailing) - un franchise con mercato secondario troppo giovane
non e' "debole", e' semplicemente non ancora testabile.

Uso: python scripts/pokemon_chinese_pilot_validation.py
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.data.liquidity_filter import liquid_sealed_ids
from poke_quant.engine.backtester import Backtester
from poke_quant.engine.strategies.time_series_momentum import TimeSeriesMomentumStrategy
from poke_quant.validation.statistical_validation import deflated_sharpe_ratio
from poke_quant.validation.bootstrap import block_bootstrap_metrics, summarize_bootstrap

N_TRIALS_BOX = 52  # +1 su jp_pilot_validation.py


def run_box_bt(prices_df, metadata):
    strat = TimeSeriesMomentumStrategy(prices_df, lookback_months=12)
    bt = Backtester(strat, prices_df, metadata, initial_cash=10000.0, platform="cardmarket",
                     apply_liquidity_slippage=True, apply_holding_cost=True, apply_buy_side_shipping=True)
    return bt.run()


def main():
    metadata = load_metadata()
    prices = load_price_matrix()

    zh_ids_all = [k for k, v in metadata.items() if v.get("franchise") == "pokemon_chinese"]
    print(f"Prodotti sealed Pokemon Cinese scoperti: {len(zh_ids_all)}")
    for item_id in zh_ids_all:
        if item_id in prices.columns:
            s = prices[item_id].dropna()
            s = s[s > 0]
            print(f"  {item_id:28s} {len(s):3d} mesi di storico reale (dal {s.index.min().strftime('%Y-%m') if len(s) else 'N/A'})")

    print("\n" + "=" * 100)
    print("BOX Pokemon Cinese (TS Momentum) - richiede >=13 mesi di storico per un trailing return")
    print("=" * 100)
    liquid_all = liquid_sealed_ids(metadata, prices, exclude_franchises=frozenset())
    zh_liquid = [k for k in liquid_all if metadata[k].get("franchise") == "pokemon_chinese"]
    print(f"Prodotti Cinesi con >=13 mesi di storico (requisito lookback_months=12): {len(zh_liquid)}")

    if len(zh_liquid) < 5:
        print(f"\n  ESITO: solo {len(zh_liquid)} prodotto/i con storico sufficiente per calcolare anche un solo "
              "rendimento trailing a 12 mesi. NON e' possibile un backtest statisticamente significativo -\n"
              "  il mercato secondario Pokemon Cinese su PriceCharting e' troppo giovane (la maggior parte dei\n"
              "  set scoperti ha 1-8 mesi di storico, alcuni un solo mese). Non e' un rigetto per debolezza\n"
              "  della strategia - e' un dato mancante strutturale: la storia dei prezzi non esiste ancora.\n"
              "  DECISIONE: NON adottare in produzione ora. Rivalutare tra 6-12 mesi quando i set piu' vecchi\n"
              "  (151 Collection, Gem Pack 2/3) avranno accumulato storico sufficiente per un test vero.")
        return

    meta_zh = {k: metadata[k] for k in zh_liquid}
    res_zh = run_box_bt(prices[zh_liquid], meta_zh)
    dsr = deflated_sharpe_ratio(observed_sr=res_zh.sharpe / np.sqrt(12), n_trials=N_TRIALS_BOX, n_obs=len(res_zh.monthly_returns))
    print(f"\n  Box SOLO Pokemon Cinese | Sharpe {res_zh.sharpe:.2f} | CAGR {res_zh.cagr*100:+.2f}% | "
          f"MaxDD {res_zh.max_drawdown*100:.2f}% | Trade {res_zh.total_trades} | DSR({N_TRIALS_BOX}) {dsr:.3f}")
    sims = block_bootstrap_metrics(res_zh.monthly_returns, n_sims=500, block_size=3)
    print(f"  Bootstrap: {summarize_bootstrap(sims)}")


if __name__ == "__main__":
    main()
