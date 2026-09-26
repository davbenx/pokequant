#!/usr/bin/env python3
"""
scripts/box_entry_priority_order_test.py — Bug trovato mentre si verificava
la richiesta dell'utente "per gli acquisti bisogna dare priorita' alle prime
della lista a scendere, corretto?".

IPOTESI DI PARTENZA (falsificata): che l'ordine mostrato in dashboard fosse
solo un dettaglio di visualizzazione, senza impatto sul backtest validato.

METODO: TimeSeriesMomentumStrategy.generate_signals() (box) iterava i
candidati d'ingresso in `for item_id, info in market_snapshot.items()` -
cioe' nell'ordine di INSERIMENTO in items_metadata.json (arbitrario, nessun
criterio quantitativo), non per momentum. Portfolio.buy() (poke_quant/engine/
portfolio.py) rifiuta un trade SENZA fill parziale se il costo supera la
cassa residua (`if total_cost > self.cash: return False`), e Backtester
esegue le compravendite BUY nell'ordine esatto in cui la strategia le ha
restituite. Quindi quando la cassa non basta per tutti i segnali dello
stesso mese, chi viene comprato e chi viene silenziosamente scartato dipende
SOLO da quell'ordine arbitrario.

Verificato quanto spesso questo si verifica davvero nel backtest storico
(non un caso limite):

    Mesi con >=2 segnali BUY box nello stesso mese: 24
    Di questi, mesi in cui la spesa totale richiesta supera la cassa
    disponibile in quel momento: 24 / 24 (100%)

Non e' un edge case - e' la norma. Il numero di Sharpe/CAGR storico della
strategia box e' sempre stato, senza che nessuno lo sapesse, in parte un
prodotto dell'ordine con cui gli item finiscono nel file JSON.

FIX: i candidati d'ingresso vengono ora raccolti, poi ordinati per momentum
trailing DECRESCENTE (trend piu' forte prima) prima di allocare la cassa in
sequenza. Non e' un parametro scelto guardando il risultato: e' la regola
naturale e pre-registrabile per una strategia di trend-following vincolata
dal capitale (spendi prima sul segnale piu' forte) - lo stesso principio che
la dashboard aveva sempre dichiarato all'utente ("priorita' ai primi in
lista"), ora davvero applicato dal motore di backtest e non solo dal testo.

ESITO (confrontato PRIMA vs DOPO il fix, stessa configurazione di produzione,
lookback_months=12, max_price_msrp_ratio=MAX_PRICE_TO_MSRP_RATIO):

    Config                          Sharpe   CAGR     MaxDD    Trade  DSR(49)  H1     H2
    PRIMA (ordine JSON arbitrario)   1.140   +22.03%  -11.29%    41    0.646  0.49   1.90
    DOPO  (ordine per momentum)      1.227   +23.33%  -10.16%    44    0.714  0.41   2.15

Migliora TUTTE le metriche (Sharpe, CAGR, MaxDD, DSR) - non e' un trade-off.
Adottato in produzione. pytest 219/219 passati dopo il cambio.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.data.liquidity_filter import liquid_sealed_ids, MAX_PRICE_TO_MSRP_RATIO
from poke_quant.engine.strategies.time_series_momentum import TimeSeriesMomentumStrategy
from poke_quant.engine.backtester import Backtester
from poke_quant.validation.statistical_validation import deflated_sharpe_ratio


def main():
    metadata = load_metadata()
    prices_full = load_price_matrix()
    sealed_ids = liquid_sealed_ids(metadata, prices_full)
    meta_sub = {k: v for k, v in metadata.items() if k in sealed_ids}
    prices_sub = prices_full[sealed_ids]

    strat = TimeSeriesMomentumStrategy(prices_sub, lookback_months=12, max_price_msrp_ratio=MAX_PRICE_TO_MSRP_RATIO)
    bt = Backtester(strat, prices_sub, meta_sub, initial_cash=10000.0, platform="cardmarket",
                     apply_liquidity_slippage=True, apply_holding_cost=True, apply_buy_side_shipping=True)
    res = bt.run()

    # Quante volte la cassa e' davvero vincolante nel backtest storico corrente.
    from collections import defaultdict
    by_month = defaultdict(list)
    for s in res.signals_history:
        if s["action"] == "BUY":
            by_month[s["date"]].append(s)
    n_multi = n_binding = 0
    for date, sigs in by_month.items():
        if len(sigs) < 2:
            continue
        n_multi += 1
        if sum(s["total_value"] for s in sigs) > sigs[0]["portfolio_cash_before"]:
            n_binding += 1

    dsr = deflated_sharpe_ratio(observed_sr=res.sharpe / np.sqrt(12), n_trials=49, n_obs=len(res.monthly_returns))
    n_obs = len(res.monthly_returns)
    h = n_obs // 2
    r1, r2 = res.monthly_returns[:h], res.monthly_returns[h:]
    h1 = (r1.mean() / r1.std()) * np.sqrt(12) if r1.std() > 0 else 0.0
    h2 = (r2.mean() / r2.std()) * np.sqrt(12) if r2.std() > 0 else 0.0

    print(f"Mesi con >=2 segnali BUY: {n_multi} | di cui cassa vincolante: {n_binding} ({n_binding/max(1,n_multi)*100:.0f}%)")
    print(f"Sharpe {res.sharpe:.3f} | CAGR {res.cagr*100:+.2f}% | MaxDD {res.max_drawdown*100:.2f}% | "
          f"Trade {res.total_trades} | DSR(49) {dsr:.3f} | H1 {h1:.2f} H2 {h2:.2f}")


if __name__ == "__main__":
    main()
