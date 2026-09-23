#!/usr/bin/env python3
"""
scripts/pcgs_coin_strategy_test2.py — Secondo giro di test sulle monete PCGS,
motivato dal fallimento netto del momentum (Sharpe negativo in entrambi i
regimi, vedi pcgs_coin_strategy_test.py): se il mercato e' troppo efficiente
per il trend-following, la mean-reversion e' l'ipotesi successiva da
falsificare, non un'altra variante dello stesso meccanismo.

1. MEAN-REVERSION SUL PREMIO NUMISMATICO (monete d'oro, n=4 dopo filtro
   liquidita'): z-score del premio (prezzo - valore di fusione) sulla propria
   storia, compra quando il premio e' anormalmente BASSO (z molto negativo),
   aspettando la normalizzazione - esecuzione sul prezzo reale della moneta,
   segnale sul premio (stessa decoupling della ricerca precedente).

2. RAPPORTO ORO/ARGENTO COME FILTRO DI REGIME sul paniere oro: non un vero
   rotation a due gambe (le monete d'argento non hanno superato il filtro di
   liquidita' - dati troppo rumorosi per essere l'esecuzione), ma un timing
   sul paniere oro basato sullo z-score del rapporto spot oro/argento (dati
   puliti, FMP). Testa entrambe le direzioni (compra quando l'oro e' a buon
   mercato relativo, o quando e' caro relativo), nessuna assunta a priori.

ESITO: NON VALIDATE, ma con un profilo diverso dal momentum - non un
fallimento catastrofico, un'assenza di segnale.

1) Mean-reversion sul premio: Sharpe 0,03-0,04 - praticamente piatto, non
negativo come il momentum ma indistinguibile dal rumore. DSR 0,246. Walk-
forward mostra ancora PRE-2020 negativo (-0,45) e 2020+ positivo (+0,33) -
stesso schema visto ovunque sui TCG, solo molto piu' attenuato in ampiezza.
Il mercato non compensa chi compra il premio basso ne' chi lo insegue.

2) Rapporto oro/argento come filtro di regime: migliore Sharpe -0,19 (ancora
negativo), DSR 0,013. La direzione "compra quando l'oro e' caro relativo
all'argento" e' sistematicamente peggiore (-0,53 a -0,78) di "compra quando
e' a buon mercato" (-0,19 a -0,30) - un'indicazione debole ma consistente che
comprare quando l'oro NON e' relativamente costoso e' meno dannoso, non che
sia profittevole.

Conclusione del ciclo di ricerca sulle monete (4 famiglie di strategia
testate: momentum assoluto, momentum sul premio ex-oro, mean-reversion sul
premio, timing su rapporto oro/argento): nessuna produce un edge. Il quadro
che emerge e' coerente - le monete gold/silver-backed sono un mercato troppo
maturo/arbitraggiato per qualsiasi fattore tecnico semplice testato finora,
sia trend-following che mean-reversion. Non si esclude un edge more
sofisticato (es. cross-sezionale con popolazione reale, idea 2 della lista
proposta ma non ancora testata - richiede un universo piu' ampio), ma le
quattro ipotesi piu' semplici ed economicamente motivate sono tutte
falsificate.
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.engine.strategies import Signal
from poke_quant.engine.portfolio import Portfolio
from poke_quant.engine.strategies.cross_sectional_factor import CrossSectionalFactorStrategy, zscore_factor
from poke_quant.engine.backtester import Backtester
from poke_quant.validation.statistical_validation import deflated_sharpe_ratio, pbo_cscv
from poke_quant.validation.bootstrap import block_bootstrap_metrics, summarize_bootstrap
from scripts.pcgs_coin_strategy_test import build_monthly_panel, GOLD_OZ_PER_20DOLLAR, COIN_FILE, GOLD_FILE

SILVER_FILE = Path(__file__).resolve().parent.parent / "data_cache" / "silver_spot_usd_monthly.json"


def run_bt(strategy, prices_df, metadata):
    bt = Backtester(strategy, prices_df, metadata, initial_cash=10000.0, platform="cardmarket",
                     apply_liquidity_slippage=True, apply_holding_cost=True)
    return bt.run()


class RatioRegimeTimingStrategy:
    """Compra equal-weight tutto il paniere quando lo z-score del rapporto
    esterno (oro/argento) e' oltre la soglia nella direzione scelta, altrimenti
    liquida tutto e resta in cassa. Nessun ranking per asset - un singolo
    segnale macro applicato all'intero paniere."""
    def __init__(self, ratio_zscore: pd.Series, lookback_months: int, z_threshold: float,
                 buy_when: str = "low", max_allocation_pct: float = 0.30, item_type_filter: str = "single"):
        self.ratio_zscore = ratio_zscore
        self.lookback_months = lookback_months
        self.z_threshold = z_threshold
        self.buy_when = buy_when
        self.max_allocation_pct = max_allocation_pct
        self.item_type_filter = item_type_filter

    def reset(self):
        pass

    def generate_signals(self, current_date, portfolio, market_snapshot):
        signals = []
        cur_dt = pd.to_datetime(current_date)
        if cur_dt not in self.ratio_zscore.index:
            return signals
        z = self.ratio_zscore.loc[cur_dt]
        if pd.isna(z):
            return signals
        in_regime = (z <= -self.z_threshold) if self.buy_when == "low" else (z >= self.z_threshold)

        eligible_ids = [k for k, v in market_snapshot.items()
                         if v.get("type") == self.item_type_filter and v.get("current_price", 0) > 0]
        if not in_regime:
            for item_id, pos in list(portfolio.positions.items()):
                if item_id in market_snapshot:
                    signals.append(Signal(action="SELL", item_id=item_id, item_name=pos.item_name,
                                           item_type=pos.item_type, quantity=pos.quantity,
                                           target_price=market_snapshot[item_id]["current_price"], reason="fuori regime"))
            return signals

        total_nav = portfolio.get_total_nav({k: v["current_price"] for k, v in market_snapshot.items()})
        target = total_nav * min(self.max_allocation_pct, 1.0 / max(1, len(eligible_ids)))
        for item_id in eligible_ids:
            if item_id in portfolio.positions:
                continue
            info = market_snapshot[item_id]
            px = info["current_price"]
            budget = min(portfolio.cash, target)
            qty = int(budget // px)
            if qty < 1 and portfolio.cash >= px and px <= total_nav * 0.5:
                qty = 1
            if qty >= 1:
                signals.append(Signal(action="BUY", item_id=item_id, item_name=info.get("name", item_id),
                                       item_type=self.item_type_filter, quantity=qty, target_price=px, reason="in regime"))
        return signals


def main():
    coin_data = json.loads(COIN_FILE.read_text())
    prices_df, meta = build_monthly_panel(coin_data)
    gold_ids = [k for k, v in meta.items() if v["segment"] == "gold"]
    gold_prices = prices_df[gold_ids]
    gold_meta = {k: meta[k] for k in gold_ids}

    gold_spot = pd.Series({pd.to_datetime(k): v for k, v in json.loads(GOLD_FILE.read_text()).items()}).sort_index()
    silver_spot = pd.Series({pd.to_datetime(k): v for k, v in json.loads(SILVER_FILE.read_text()).items()}).sort_index()
    common_idx = gold_prices.index.intersection(gold_spot.index)
    gold_prices, gold_spot_a, silver_spot_a = gold_prices.loc[common_idx], gold_spot.loc[common_idx], silver_spot.loc[common_idx]
    melt_value = gold_spot_a * GOLD_OZ_PER_20DOLLAR
    premium_df = gold_prices.sub(melt_value, axis=0)

    regime_split = pd.Timestamp("2020-01-01")

    # ================= 1) MEAN-REVERSION SUL PREMIO =================
    print("=" * 90)
    print("1) MEAN-REVERSION SUL PREMIO NUMISMATICO (z-score, compra il premio anormalmente basso)")
    print("=" * 90)
    results = {}
    for lb in [6, 9, 12]:
        strat = CrossSectionalFactorStrategy(premium_df, zscore_factor, lookback_months=lb, top_quantile=0.5,
                                              ascending=True, rebalance_every_months=3, item_type_filter="single", min_age_months=0)
        res = run_bt(strat, gold_prices, gold_meta)
        results[f"meanrev lb={lb}m"] = res
        print(f"  lb={lb:2d}m | CAGR {res.cagr*100:+6.2f}% | Sharpe {res.sharpe:5.2f} | "
              f"MaxDD {res.max_drawdown*100:6.2f}% | Trade {res.total_trades:4d}")

    common_idx_r = None
    for res in results.values():
        common_idx_r = res.monthly_returns.index if common_idx_r is None else common_idx_r.intersection(res.monthly_returns.index)
    perf_matrix = np.column_stack([results[k].monthly_returns.loc[common_idx_r].values for k in results])
    t_len = len(perf_matrix)
    splits = 8 if t_len >= 32 else 4
    rem = t_len % splits
    pbo = pbo_cscv(perf_matrix[rem:, :] if rem else perf_matrix, n_splits=splits)
    best_name = max(results, key=lambda k: results[k].sharpe)
    best = results[best_name]
    dsr = deflated_sharpe_ratio(observed_sr=best.sharpe / np.sqrt(12), n_trials=len(results), n_obs=len(best.monthly_returns))
    print(f"\nPBO ({len(results)} candidati, {splits} split): {pbo:.3f}")
    print(f"Migliore: '{best_name}' Sharpe {best.sharpe:.2f} | Trade {best.total_trades} | DSR: {dsr:.3f}")
    sims = block_bootstrap_metrics(best.monthly_returns, n_sims=500, block_size=6)
    print(summarize_bootstrap(sims))
    lb_best = int(best_name.split("=")[1].rstrip("m"))
    pre_dates = gold_prices.index[gold_prices.index < regime_split]
    post_dates = gold_prices.index[gold_prices.index >= regime_split]
    print("\nWalk-forward per regime:")
    for label, dates in [("PRE-2020", pre_dates), ("2020+", post_dates)]:
        if len(dates) < 12:
            continue
        sub_premium, sub_gold = premium_df.loc[dates], gold_prices.loc[dates]
        strat = CrossSectionalFactorStrategy(sub_premium, zscore_factor, lookback_months=lb_best, top_quantile=0.5,
                                              ascending=True, rebalance_every_months=3, item_type_filter="single")
        r = run_bt(strat, sub_gold, gold_meta)
        print(f"  {label} ({dates[0].strftime('%Y-%m')}->{dates[-1].strftime('%Y-%m')}) | Sharpe {r.sharpe:5.2f} | CAGR {r.cagr*100:+6.2f}% | Trade {r.total_trades}")

    # ================= 2) RAPPORTO ORO/ARGENTO COME FILTRO DI REGIME =================
    print("\n" + "=" * 90)
    print("2) RAPPORTO ORO/ARGENTO COME FILTRO DI REGIME sul paniere oro")
    print("=" * 90)
    ratio = gold_spot_a / silver_spot_a
    results2 = {}
    for lb in [12, 24]:
        z = (ratio - ratio.rolling(lb).mean()) / ratio.rolling(lb).std()
        for thresh in [0.5, 1.0]:
            for direction in ["low", "high"]:
                strat = RatioRegimeTimingStrategy(z, lookback_months=lb, z_threshold=thresh, buy_when=direction)
                res = run_bt(strat, gold_prices, gold_meta)
                name = f"lb={lb}m z={thresh} {direction}"
                results2[name] = res
                print(f"  {name:22s} | CAGR {res.cagr*100:+6.2f}% | Sharpe {res.sharpe:5.2f} | "
                      f"MaxDD {res.max_drawdown*100:6.2f}% | Trade {res.total_trades:4d}")

    common_idx2 = None
    for res in results2.values():
        common_idx2 = res.monthly_returns.index if common_idx2 is None else common_idx2.intersection(res.monthly_returns.index)
    perf_matrix2 = np.column_stack([results2[k].monthly_returns.loc[common_idx2].values for k in results2])
    t_len2 = len(perf_matrix2)
    splits2 = 8 if t_len2 >= 32 else 4
    rem2 = t_len2 % splits2
    pbo2 = pbo_cscv(perf_matrix2[rem2:, :] if rem2 else perf_matrix2, n_splits=splits2)
    best_name2 = max(results2, key=lambda k: results2[k].sharpe)
    best2 = results2[best_name2]
    dsr2 = deflated_sharpe_ratio(observed_sr=best2.sharpe / np.sqrt(12), n_trials=len(results2), n_obs=len(best2.monthly_returns))
    print(f"\nPBO ({len(results2)} candidati, {splits2} split): {pbo2:.3f}")
    print(f"Migliore: '{best_name2}' Sharpe {best2.sharpe:.2f} | Trade {best2.total_trades} | DSR: {dsr2:.3f}")
    sims2 = block_bootstrap_metrics(best2.monthly_returns, n_sims=500, block_size=6)
    print(summarize_bootstrap(sims2))


if __name__ == "__main__":
    main()
