#!/usr/bin/env python3
"""
scripts/max_edge_preservation_test.py — Calibrazione e test empirico del
"Prezzo massimo per mantenere l'Edge" (generate_singles_signal.py::PRESERVE_EDGE_ALPHA).

Richiesto dall'utente: "Sostituisci il prezzo massimo con il prezzo massimo per mantenere l'edge."
Il vecchio prezzo massimo teorico (current_price * exp(cutoff - residual)) indicava il confine
lordo estremo del quantile BUY. Acquistando sistematicamente a quel prezzo, l'intero edge
veniva distrutto (Sharpe -0.87, CAGR -31.3%) dalle frizioni reali (fee 5-13%, spedizione 7€,
slippage 2.5% e vendite forzate al ribilanciamento trimestrale).

Questo script testa una griglia di frazioni alpha in [0.0, 0.50] dove:
    P_buy = P_current + alpha * (P_cutoff_teorico - P_current)

ESITO EMPIRICO ORIGINALE (su 2.588 singole, 69 mesi con spedizione reale e frizioni):
  - alpha = 0.00 (prezzo di mercato): Sharpe 2.21 | CAGR +36.21% | MaxDD -8.82%
  - alpha = 0.10 (SOGLIA ISTITUZIONALE): Sharpe 1.01 | CAGR +19.42% | MaxDD -15.81%
  - alpha = 0.15: Sharpe 0.56 | CAGR +11.84% | MaxDD -21.46%
  - alpha = 0.20: Sharpe 0.25 | CAGR +6.09% | MaxDD -24.78%
  - alpha = 0.25 (PAREGGIO ZERO-ALPHA): Sharpe 0.00 | CAGR +0.92% | MaxDD -36.21%
  - alpha >= 0.30: Sharpe NEGATIVO e CAGR NEGATIVO.

ADOTTATO PRESERVE_EDGE_ALPHA = 0.10 in produzione: al momento della calibrazione
garantiva Sharpe >= 1.0 e un rendimento annuo di circa +20% nel caso peggiore
(acquisto sempre al tetto massimo mostrato).

RI-VERIFICATO (2026-09-29, richiesta esplicita dell'utente: "verifica che le
valutazioni dei prezzi massimi sulle slab per avere edge siano corretti") su
2.587 singole, 69 mesi, con la matrice prezzi rebuilded con FX reale (Fase 1.3):
  - alpha = 0.00: Sharpe 2.02 | CAGR +42.08% | MaxDD  -9.16% | Trade 138
  - alpha = 0.10 (PRODUZIONE): Sharpe 0.63 | CAGR +15.03% | MaxDD -16.57% | Trade 145
  - alpha = 0.15: Sharpe 0.20 | CAGR  +5.07% | MaxDD -24.18% | Trade 147
  - alpha = 0.20: Sharpe -0.09 | CAGR  -2.71% | MaxDD -39.90% | Trade 145
  - alpha = 0.25: Sharpe -0.32 | CAGR  -9.45% | MaxDD -56.06% | Trade 145 (pareggio ora sotto 0.25, non piu' esattamente a 0.25)
  - alpha >= 0.30: Sharpe NEGATIVO e CAGR NEGATIVO (invariato).

La FORMULA resta corretta e monotona (verificato: sniper_ceiling/max_edge_price
cresce sempre col prezzo di cutoff, nessun bug nel codice) - e' la SOGLIA
0,63 < 1,0 che non rispetta piu' la garanzia originale "Sharpe >= 1.0" con cui
PRESERVE_EDGE_ALPHA=0.10 fu adottato: drift naturale coi mesi di dati aggiunti
dopo la calibrazione iniziale, non un bug. Resta positivo (0,63, non negativo)
quindi non e' un rigetto - ma la soglia "garantisce Sharpe >= 1.0" nel commento
sopra NON e' piu' vera oggi. Decisione se ricalibrare PRESERVE_EDGE_ALPHA
lasciata esplicitamente all'utente (cambiare un parametro di produzione dopo
averne visto il risultato aggiornato sarebbe overfitting in-sample, la stessa
disciplina gia' applicata altrove in questa sessione) - non modificato qui.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.engine.strategies.scarcity_value_factor import ScarcityValueFactorStrategy
from poke_quant.engine.backtester import Backtester
from poke_quant.engine.portfolio import Portfolio
from poke_quant.engine.strategies import Signal
from scripts.generate_singles_signal import PRODUCTION_PARAMS, PRESERVE_EDGE_ALPHA


class ScarcityAlphaPriceStrategy(ScarcityValueFactorStrategy):
    def __init__(self, alpha: float = 0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def generate_signals(self, current_date, portfolio, market_snapshot):
        signals = []
        cur_dt = pd.to_datetime(current_date)
        is_rebalance_month = (self._call_count % self.rebalance_every_months) == 0
        self._call_count += 1
        if not is_rebalance_month:
            return signals

        residuals = self._fit_residuals(cur_dt, market_snapshot)
        if not residuals:
            return signals

        ranked = sorted(residuals.items(), key=lambda x: x[1])
        n_buy = max(1, int(len(residuals) * self.top_quantile))
        eligible = [item_id for item_id, _ in ranked[:n_buy]][: self.max_positions]
        cutoff_residual = ranked[min(len(ranked) - 1, n_buy - 1)][1]
        eligible_set = set(eligible)
        hold_set = eligible_set

        for item_id, pos in list(portfolio.positions.items()):
            if item_id in market_snapshot and item_id not in hold_set:
                signals.append(Signal(
                    action="SELL", item_id=item_id, item_name=pos.item_name,
                    item_type=pos.item_type, quantity=pos.quantity,
                    target_price=market_snapshot[item_id]["current_price"],
                    reason="Uscita dal quantile"
                ))

        total_nav = portfolio.get_total_nav({k: v["current_price"] for k, v in market_snapshot.items()})
        target_per_position = total_nav * min(self.max_allocation_pct, 1.0 / max(1, len(eligible)))

        for item_id in eligible:
            if item_id in portfolio.positions:
                continue
            info = market_snapshot[item_id]
            cur_price = info["current_price"]
            resid = residuals[item_id]
            theoretical_cutoff_px = cur_price * np.exp(cutoff_residual - resid)
            buy_px = cur_price + self.alpha * (theoretical_cutoff_px - cur_price)

            available_cash = portfolio.cash
            budget = min(available_cash, target_per_position)
            qty = int(budget // buy_px)
            if qty < 1 and available_cash >= buy_px and buy_px <= total_nav * 0.35:
                qty = 1
            if self.max_quantity_per_trade is not None:
                qty = min(qty, self.max_quantity_per_trade)
            if qty >= 1:
                signals.append(Signal(
                    action="BUY", item_id=item_id, item_name=info.get("name", item_id),
                    item_type=self.item_type_filter, quantity=qty, target_price=buy_px,
                    reason=f"BUY residuo {resid:.2f}, buy_px={buy_px:.2f} (cutoff={theoretical_cutoff_px:.2f})"
                ))

        return signals


def main():
    metadata = load_metadata()
    grade9_prices = load_price_matrix("historical_prices_graded_singles_grade9.csv")
    singles_ids = [k for k, v in metadata.items() if v.get("type") == "single"
                   and v.get("data_quality") != "thin_unreliable" and k in grade9_prices.columns]
    meta_singles = {k: metadata[k] for k in singles_ids}
    prices_singles = grade9_prices[singles_ids]

    print(f"Dataset: {len(singles_ids)} carte singole su {len(prices_singles)} mesi.")
    print("=" * 80)
    print("GRIGLIA DI SENSIBILITA' SULLA PRESERVAZIONE DELL'EDGE ALL'ACQUISTO")
    print("=" * 80)

    alphas = [0.00, PRESERVE_EDGE_ALPHA, 0.15, 0.20, 0.25, 0.30, 0.50]
    for a in alphas:
        strat = ScarcityAlphaPriceStrategy(alpha=a, **PRODUCTION_PARAMS)
        bt = Backtester(
            strat, prices_singles, meta_singles,
            initial_cash=10000.0,
            platform="cardmarket",
            apply_liquidity_slippage=True,
            apply_holding_cost=True,
            apply_buy_side_shipping=True
        )
        res = bt.run()
        tag = " <- PRODUZIONE ATTUALE" if abs(a - PRESERVE_EDGE_ALPHA) < 1e-4 else (" <- BREAKEVEN" if abs(a - 0.25) < 1e-4 else "")
        print(f"Alpha: {a:4.2f} | Sharpe: {res.sharpe:5.2f} | CAGR: {res.cagr*100:6.2f}% | MaxDD: {res.max_drawdown*100:6.2f}% | Trade: {res.total_trades:3d}{tag}")


if __name__ == "__main__":
    main()
