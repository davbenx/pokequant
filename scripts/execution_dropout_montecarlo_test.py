#!/usr/bin/env python3
"""
scripts/execution_dropout_montecarlo_test.py — L'utente ha chiesto: "se non
trovo carte o box nel prezzo previsto, il portafoglio perde alcune posizioni.
Cosa comporta?" - domanda distinta da tutto il resto testato oggi (costo
sdoganato, livello prezzi EU): qui l'ipotesi e' la piu' semplice possibile,
segnali BUY che falliscono a caso all'esecuzione (prezzo reale sopra al
"massimo", nessuna disponibilita' trovata), non per un motivo sistematico
legato alla carta - per isolare l'effetto puro di "meno posizioni con lo
stesso capitale" dalla domanda (gia' testata altrove) se le carte perse
sarebbero state migliori o peggiori della media.

Metodo: wrapper attorno alla strategia che scarta ogni segnale BUY con
probabilita' fissa p (mai i segnali SELL - quelli chiudono posizioni gia'
aperte, non serve "trovare" nulla per venderle). N simulazioni Monte Carlo
per ogni tasso di dropout, per isolare l'effetto della sola CASUALITA' di
esecuzione da un singolo run fortunato/sfortunato.

ESITO (40 simulazioni box, 40+20 singole - vedi anche lo script gemello per
i tassi 40/60/73% sulle singole, spezzato per limiti di tempo):

  BOX: quasi insensibile al dropout casuale, anche al 73%. Sharpe 1,18->1,22
  (leggermente MEGLIO), MaxDD medio -11,12%->-10,65% (leggermente meglio).
  Motivo: solo 32 trade in tutto il backtest, raramente piu' di una manciata
  di posizioni aperte insieme - poca diversificazione simultanea da perdere
  in origine.

  SINGOLE: il CAGR scende in modo chiaro e monotono col dropout (26,7%->16,7%
  al 73%, il tasso osservato oggi) - meno capitale viene mai investito, quindi
  meno crescita assoluta. MA lo Sharpe (rischio aggiustato) resta relativamente
  stabile (1,57->1,39, -11% relativo) - le posizioni perse sono un
  sottoinsieme CASUALE, non selettivamente le migliori, quindi il rischio per
  euro investito non peggiora in modo sistematico. Il MaxDD MEDIO addirittura
  migliora (-12,0%->-8,7%) - meno posizioni aperte insieme = meno esposizione
  concentrata durante un crollo di mercato generale. PERO' il range di esiti
  possibili si allarga: gia' al 20% di dropout, il MaxDD PEGGIORE osservato
  tra 40 tentativi casuali e' -15,7% (peggio del -12,0% base, senza alcun
  dropout) - qualunque livello di dropout introduce una possibilita' di
  risultato peggiore della media che a dropout zero semplicemente non esiste.

CONCLUSIONE PRATICA: perdere posizioni per indisponibilita' di prezzo reale
NON rompe l'edge in media (il rischio per euro investito resta simile) - il
costo reale e' un CAGR assoluto piu' basso (meno capitale mai impiegato) E
piu' incertezza sul risultato SPECIFICO che si otterra' (non quella media).
Non e' un motivo per forzare acquisti sopra al "massimo" per "compensare" le
posizioni perse - l'allocazione di capitale (build_allocation/
build_equal_allocation in app.py) gia' ridistribuisce automaticamente il
capitale sulle posizioni che SONO eseguibili, fino al tetto per posizione.
"""

import random
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.data.liquidity_filter import liquid_sealed_ids, MAX_PRICE_TO_MSRP_RATIO
from poke_quant.engine.backtester import Backtester
from poke_quant.engine.strategies.time_series_momentum import TimeSeriesMomentumStrategy
from poke_quant.engine.strategies.scarcity_value_factor import ScarcityValueFactorStrategy
from scripts.generate_singles_signal import PRODUCTION_PARAMS

N_SIMULATIONS = 40
DROP_RATES = [0.0, 0.2, 0.4, 0.6, 0.73]  # 0.73 = tasso reale osservato oggi sulle singole (11/15)


class ExecutionDropoutWrapper:
    """Scarta ogni segnale BUY con probabilita' drop_prob - mai i SELL, che
    chiudono posizioni gia' aperte e non richiedono "trovare" nulla sul
    mercato. Simula "il prezzo reale era sopra al massimo, nessuna
    disponibilita' trovata" in modo puramente casuale, non selettivo."""

    def __init__(self, inner_strategy, drop_prob: float, seed: int):
        self.inner = inner_strategy
        self.drop_prob = drop_prob
        self.rng = random.Random(seed)

    def reset(self):
        if hasattr(self.inner, "reset"):
            self.inner.reset()

    def generate_signals(self, current_date, portfolio, market_snapshot):
        signals = self.inner.generate_signals(current_date, portfolio, market_snapshot)
        if self.drop_prob <= 0:
            return signals
        return [s for s in signals if s.action != "BUY" or self.rng.random() >= self.drop_prob]


def run_bt(strat, prices_df, metadata):
    bt = Backtester(strat, prices_df, metadata, initial_cash=10000.0, platform="cardmarket",
                     apply_liquidity_slippage=True, apply_holding_cost=True, apply_buy_side_shipping=True)
    return bt.run()


def monte_carlo(make_strategy, prices_df, metadata, drop_prob, n_sims):
    sharpes, cagrs, maxdds, trades = [], [], [], []
    for seed in range(n_sims):
        strat = ExecutionDropoutWrapper(make_strategy(), drop_prob, seed) if drop_prob > 0 else make_strategy()
        res = run_bt(strat, prices_df, metadata)
        sharpes.append(res.sharpe)
        cagrs.append(res.cagr)
        maxdds.append(res.max_drawdown)
        trades.append(res.total_trades)
    return {
        "sharpe_mean": np.mean(sharpes), "sharpe_p05": np.percentile(sharpes, 5), "sharpe_p95": np.percentile(sharpes, 95),
        "cagr_mean": np.mean(cagrs),
        "maxdd_mean": np.mean(maxdds), "maxdd_worst": np.min(maxdds),
        "trades_mean": np.mean(trades),
    }


def main():
    metadata = load_metadata()

    print(f"=== BOX (TS Momentum) — {N_SIMULATIONS} simulazioni per tasso di dropout ===")
    sealed_prices = load_price_matrix()
    sealed_ids = liquid_sealed_ids(metadata, sealed_prices)
    meta_box = {k: metadata[k] for k in sealed_ids}
    prices_box = sealed_prices[sealed_ids]
    for drop in DROP_RATES:
        stats = monte_carlo(
            lambda: TimeSeriesMomentumStrategy(prices_box, lookback_months=12, max_price_msrp_ratio=MAX_PRICE_TO_MSRP_RATIO),
            prices_box, meta_box, drop, N_SIMULATIONS)
        print(f"  dropout {drop*100:4.0f}% | Sharpe medio {stats['sharpe_mean']:5.2f} (5-95%: {stats['sharpe_p05']:5.2f} / {stats['sharpe_p95']:5.2f}) | "
              f"CAGR medio {stats['cagr_mean']*100:+6.2f}% | MaxDD medio {stats['maxdd_mean']*100:6.2f}% | MaxDD peggiore {stats['maxdd_worst']*100:6.2f}% | "
              f"Trade medi {stats['trades_mean']:4.1f}")

    print(f"\n=== SINGOLE (Fattore Scarsita') — {N_SIMULATIONS} simulazioni per tasso di dropout ===")
    grade9_prices = load_price_matrix("historical_prices_graded_singles_grade9.csv")
    singles_ids = [k for k, v in metadata.items() if v.get("type") == "single"
                   and v.get("data_quality") != "thin_unreliable" and k in grade9_prices.columns]
    meta_singles = {k: metadata[k] for k in singles_ids}
    prices_singles = grade9_prices[singles_ids]
    for drop in DROP_RATES:
        stats = monte_carlo(lambda: ScarcityValueFactorStrategy(**PRODUCTION_PARAMS),
                             prices_singles, meta_singles, drop, N_SIMULATIONS)
        print(f"  dropout {drop*100:4.0f}% | Sharpe medio {stats['sharpe_mean']:5.2f} (5-95%: {stats['sharpe_p05']:5.2f} / {stats['sharpe_p95']:5.2f}) | "
              f"CAGR medio {stats['cagr_mean']*100:+6.2f}% | MaxDD medio {stats['maxdd_mean']*100:6.2f}% | MaxDD peggiore {stats['maxdd_worst']*100:6.2f}% | "
              f"Trade medi {stats['trades_mean']:4.1f}")


if __name__ == "__main__":
    main()
