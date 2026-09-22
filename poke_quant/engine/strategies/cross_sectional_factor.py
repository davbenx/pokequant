"""
poke_quant/engine/strategies/cross_sectional_factor.py — Fattore Cross-Sezionale
Generalizzato: stessa meccanica di ranking/rotazione di CrossSectionalMomentumStrategy,
ma con la funzione di scoring intercambiabile. Un solo motore testato una volta (incluso
il fix anti-nondeterminismo: liste ordinate per rank, mai set) per tre ipotesi tecniche
distinte, invece di tre classi quasi-duplicate:

  - momentum_factor: rendimento trailing "skip-month" (12-1) - salta l'ultimo mese per
    evitare la contaminazione da reversione di breve periodo (Jegadeesh 1990), un
    raffinamento della Time-Series/Cross-Sectional Momentum già testate.
  - proximity_to_high_factor: prezzo attuale / massimo osservato nella finestra - un
    fattore concettualmente diverso dal momentum puro (George & Hwang, "The 52-Week
    High and Momentum Investing", JF 2004): due carte con lo stesso rendimento a 12m
    possono avere posizione molto diversa rispetto al proprio massimo.
  - low_volatility_factor: deviazione standard dei rendimenti mensili nella finestra
    (ascending=True compra la meno volatile) - testa l'anomalia low-vol, ipotesi opposta
    a "più rischio premia di più".
  - zscore_factor: deviazione standard del valore attuale vs media della finestra -
    generico, usato anche sul RAPPORTO singola/box dello stesso set (vedi
    scripts/build_ratio_matrix.py), non solo sul prezzo puro.

ESITO VALIDAZIONE (griglia 9 candidati momentum/52w-high/low-vol + griglia 6 candidati
sul rapporto singola/box, entrambe su scripts/optimize_and_falsify.py-style rigor):
NESSUNO VALIDATO. Pattern identico e ripetuto 3 volte su costruzioni indipendenti:
  - LOW-VOL lb=12 q=0.30: Sharpe 1.10 full-sample, PBO 5.7% (ottimo!), DSR 0.846 - ma
    walk-forward H1 Sharpe -1.09 / H2 Sharpe +1.33: inversione di segno netta.
  - RATIO-MOM (rapporto singola/box) lb=12 q=0.30: Sharpe 0.75, ma PBO 65.7% (pessimo)
    E H1 Sharpe -1.35 / H2 Sharpe +2.63.
  - momentum skip-month e 52w-high: troppo debole ovunque (Sharpe 0.08-0.34).
Nota metodologica: nel caso LOW-VOL il PBO da solo (5.7%, eccellente) NON ha segnalato
il problema - solo lo split H1/H2 lo rivela. Il PBO su CSCV a blocchi contigui non
sostituisce un vero test walk-forward quando l'intero campione e' dominato da UN solo
ciclo di mercato (boom 2021 -> bust 2022-23 -> recupero 2023+): qualunque fattore
long-biased/rotazionale eredita quel ciclo, mascherando l'assenza di alpha idiosincratico
reale. Documentato per non richiedere questo esperimento una quarta volta.
"""

from __future__ import annotations
from typing import Callable, Dict, List, Any, Optional
import numpy as np
import pandas as pd
from poke_quant.engine.strategies import Signal
from poke_quant.engine.portfolio import Portfolio


def momentum_factor(series: pd.Series, current_date: pd.Timestamp, lookback: int, skip: int = 0) -> Optional[float]:
    s = series[series.index <= current_date].dropna()
    s = s[s > 0]
    if len(s) < lookback + skip + 1:
        return None
    now = float(s.iloc[-(skip + 1)])
    past = float(s.iloc[-(skip + lookback + 1)])
    if past <= 0:
        return None
    return (now - past) / past


def proximity_to_high_factor(series: pd.Series, current_date: pd.Timestamp, lookback: int, skip: int = 0) -> Optional[float]:
    s = series[series.index <= current_date].dropna()
    s = s[s > 0]
    if len(s) < lookback + skip + 1:
        return None
    window = s.iloc[-(lookback + skip + 1):len(s) - skip] if skip > 0 else s.iloc[-(lookback + 1):]
    if window.empty:
        return None
    now = float(s.iloc[-(skip + 1)])
    peak = float(window.max())
    if peak <= 0:
        return None
    return now / peak


def low_volatility_factor(series: pd.Series, current_date: pd.Timestamp, lookback: int, skip: int = 0) -> Optional[float]:
    s = series[series.index <= current_date].dropna()
    s = s[s > 0]
    if len(s) < lookback + skip + 2:
        return None
    window = s.iloc[-(lookback + skip + 1):len(s) - skip] if skip > 0 else s.iloc[-(lookback + 1):]
    rets = window.pct_change().dropna()
    if len(rets) < 3:
        return None
    return float(rets.std(ddof=1))


def zscore_factor(series: pd.Series, current_date: pd.Timestamp, lookback: int, skip: int = 0) -> Optional[float]:
    """Deviazione standard del valore attuale rispetto a media/std della finestra
    precedente. Generico - usato sia su prezzo puro (dip mean-reversion) sia su un
    rapporto costruito (es. singola/box dello stesso set, vedi build_ratio_matrix)."""
    s = series[series.index <= current_date].dropna()
    if len(s) < lookback + skip + 1:
        return None
    now = float(s.iloc[-(skip + 1)])
    window = s.iloc[-(lookback + skip + 1):len(s) - skip] if skip > 0 else s.iloc[-(lookback + 1):-1]
    mean_p, std_p = float(window.mean()), float(window.std())
    if std_p <= 1e-9:
        return None
    return (now - mean_p) / std_p


class CrossSectionalFactorStrategy:
    def __init__(
        self,
        prices_df: pd.DataFrame,
        factor_fn: Callable[[pd.Series, pd.Timestamp, int, int], Optional[float]],
        lookback_months: int = 12,
        skip_months: int = 0,
        top_quantile: float = 0.30,
        ascending: bool = False,
        rebalance_every_months: int = 3,
        max_allocation_pct: float = 0.15,
        item_type_filter: str = "single",
        min_age_months: int = 0,
    ):
        self.prices_df = prices_df.sort_index()
        self.factor_fn = factor_fn
        self.lookback_months = lookback_months
        self.skip_months = skip_months
        self.top_quantile = top_quantile
        self.ascending = ascending
        self.rebalance_every_months = rebalance_every_months
        self.max_allocation_pct = max_allocation_pct
        self.item_type_filter = item_type_filter
        self.min_age_months = min_age_months
        self._call_count = 0

    def reset(self):
        self._call_count = 0

    def _score(self, item_id: str, current_date: pd.Timestamp) -> Optional[float]:
        if item_id not in self.prices_df.columns:
            return None
        return self.factor_fn(self.prices_df[item_id], current_date, self.lookback_months, self.skip_months)

    def generate_signals(
        self,
        current_date: str,
        portfolio: Portfolio,
        market_snapshot: Dict[str, Dict[str, Any]],
    ) -> List[Signal]:
        signals: List[Signal] = []
        cur_dt = pd.to_datetime(current_date)
        is_rebalance_month = (self._call_count % self.rebalance_every_months) == 0
        self._call_count += 1
        if not is_rebalance_month:
            return signals

        ranked = []
        for item_id, info in market_snapshot.items():
            if info.get("type") != self.item_type_filter or info.get("current_price", 0) <= 0:
                continue
            if self.min_age_months > 0 and info.get("release_date"):
                rel_dt = pd.to_datetime(info["release_date"])
                age_m = (cur_dt.year - rel_dt.year) * 12 + (cur_dt.month - rel_dt.month)
                if age_m < self.min_age_months:
                    continue
            score = self._score(item_id, cur_dt)
            if score is not None and np.isfinite(score):
                ranked.append((item_id, score))
        if not ranked:
            return signals

        # Ordine deterministico: per score (direzione scelta), pareggi rotti per item_id -
        # lista, non set, per un'esecuzione riproducibile a corto di cassa (vedi bug fix
        # in carry_scarcity_factor.py/cross_sectional_momentum.py in questa sessione).
        ranked.sort(key=lambda x: (x[1], x[0]), reverse=not self.ascending)
        n_top = max(1, int(round(len(ranked) * self.top_quantile)))
        top_ranked = ranked[:n_top]
        top_ids = {item_id for item_id, _ in top_ranked}

        for item_id, pos in list(portfolio.positions.items()):
            if item_id in market_snapshot and item_id not in top_ids:
                signals.append(Signal(
                    action="SELL", item_id=item_id, item_name=pos.item_name,
                    item_type=pos.item_type, quantity=pos.quantity,
                    target_price=market_snapshot[item_id]["current_price"],
                    reason="Fattore cross-sezionale: uscito dal quantile top"
                ))

        total_nav = portfolio.get_total_nav({k: v["current_price"] for k, v in market_snapshot.items()})
        target_per_position = total_nav * min(self.max_allocation_pct, 1.0 / max(1, n_top))

        for item_id, score in top_ranked:
            if item_id in portfolio.positions:
                continue
            info = market_snapshot[item_id]
            cur_price = info["current_price"]
            available_cash = portfolio.cash
            budget = min(available_cash, target_per_position)
            qty = int(budget // cur_price)
            if qty < 1 and available_cash >= cur_price and cur_price <= total_nav * 0.35:
                qty = 1
            if qty >= 1:
                signals.append(Signal(
                    action="BUY", item_id=item_id, item_name=info.get("name", item_id),
                    item_type=self.item_type_filter, quantity=qty, target_price=cur_price,
                    reason=f"Fattore cross-sezionale: score={score:.4f}"
                ))
        return signals
