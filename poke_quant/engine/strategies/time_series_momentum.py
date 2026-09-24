"""
poke_quant/engine/strategies/time_series_momentum.py — Time-Series Momentum
(Moskowitz, Ooi & Pedersen, "Time Series Momentum", JFE 2012).

Regola canonica e NON discrezionale: per ciascun asset sealed, indipendentemente
dagli altri, posizione lunga se il rendimento trailing di N mesi è positivo,
flat altrimenti. Nessun filtro di tier, MSRP o finestra d'acquisto discrezionale.

Serve come benchmark "pre-registrato": una regola presa identica dalla letteratura
accademica (la stessa logica di trend-following con cui ApexEngine tratta SPY/GLD/BTC),
applicata senza alcun parametro scelto guardando i dati Pokémon/One Piece.

min_age_months (opzionale, default 0 = nessun filtro): esclude dall'INGRESSO gli
asset più giovani di N mesi da release. Motivazione empirica, non discrezionale:
in Fase 2 (walk-forward per annata) la coorte di asset più recenti ha mostrato
drawdown molto più ampi con strategie che non li escludono (Equal-Weight: -49%
sulla coorte 2023+) rispetto a chi li esclude (Carry/Scarsità: -9.7%). Qui si
testa se la stessa esclusione, applicata SOLO come filtro di ingresso a monte
del segnale di momentum (non come strategia a sé), migliora il profilo di
rischio senza perdere l'edge di TSMOM.

Parametri di uscita (default = comportamento originale invariato, per non
alterare silenziosamente la strategia già validata e deployata):
  - exit_lookback_months (default None -> usa lookback_months): permette un
    lookback di uscita diverso da quello di ingresso (es. uscita piu' rapida
    su una finestra piu' corta).
  - exit_threshold (default 0.0): soglia di rendimento trailing sotto cui
    vendere. Con 0.0 e' la regola originale (mom<=0 vende). Negativo = più
    tolerante al rumore, positivo = uscita anticipata.
  - trailing_stop_pct (default None = disattivato): se impostato, vende anche
    se il prezzo scende di questa percentuale dal massimo osservato da quando
    la posizione e' stata aperta, indipendentemente dal segnale di momentum -
    un controllo del rischio aggiuntivo, non dalla letteratura originale.
Testati in scripts/optimize_and_falsify.py::section_sealed_exit_logic_search
contro la regola base (DSR 0.913, PBO 28.6%) prima di essere adottati come
default in produzione.

  - max_age_months (opzionale, default None = nessun tetto): esclude
    dall'INGRESSO gli asset più vecchi di N mesi da release, a complemento di
    min_age_months - insieme definiscono una finestra d'eta' [min, max) in cui
    il segnale di momentum può aprire una posizione. Ipotesi testata in
    scripts/sealed_age_window_search.py: comprare solo box "di mezza eta'"
    (né appena uscito né molto vecchio) potrebbe migliorare Sharpe/MaxDD
    rispetto alla regola senza tetto. ESITO: vedi docstring di quel file.

  - max_holding_months (opzionale, default None = nessun time stop): forza la
    vendita se la posizione e' aperta da almeno N mesi, indipendentemente dal
    segnale di momentum - un limite temporale puro, non un filtro sul
    rendimento. Ipotesi testata in scripts/sealed_time_stop_search.py: la
    distribuzione dei tempi di possesso in produzione e' molto asimmetrica
    (mediana 4 mesi, ma alcuni trade durano 29-45 mesi) - un time stop
    tronca la coda destra, cioe' proprio i trade che nel trend-following
    tipicamente generano la maggior parte del rendimento. ESITO: vedi
    docstring di quel file.

  - min_confirm_months (opzionale, default 1 = nessuna conferma, comportamento
    originale invariato): richiede che il momentum sia positivo per almeno N
    mesi CONSECUTIVI (non solo il mese corrente) prima di aprire una nuova
    posizione - l'opposto del filtro di freschezza usato sulle singole
    (scarcity_value_factor.py). Motivazione EMPIRICA, non simmetrica per
    principio: un residuo di valore che persiste senza correggersi e' un
    segnale di allarme (value trap), ma un momentum che persiste da mesi e'
    un trend confermato - vedi scripts/sealed_momentum_confirmation_search.py
    per il test diretto (rendimento forward per box in funzione di quanti
    mesi consecutivi il momentum e' stato positivo) che ha motivato questo
    parametro, e per l'esito della ricerca della soglia che massimizza il
    rendimento per trade.

  - max_price_msrp_ratio (opzionale, default None = nessun tetto, comportamento
    originale invariato): blocca l'INGRESSO su un box il cui prezzo corrente
    supera max_price_msrp_ratio * MSRP - stesso numero e stessa motivazione di
    poke_quant.data.liquidity_filter.MAX_PRICE_TO_MSRP_RATIO (21.6x, calibrato
    UNA VOLTA sull'universo moderno prima di guardare l'effetto, gia' usato per
    decidere quali box vintage entrano nell'universo). Qui e' lo stesso confine
    applicato non solo all'ammissione nell'universo ma ad OGNI ingresso live,
    per rispondere alla richiesta di un "prezzo massimo che non rompa l'edge"
    anche per i box - vedi scripts/box_max_price_ratio_test.py per l'esito
    empirico prima di adottarlo in produzione. Box senza MSRP noto in metadata
    non sono soggetti a questo tetto (nessun dato fabbricato).
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
import pandas as pd
from poke_quant.engine.strategies import Signal
from poke_quant.engine.portfolio import Portfolio


class TimeSeriesMomentumStrategy:
    def __init__(
        self,
        prices_df: pd.DataFrame,
        lookback_months: int = 12,
        max_allocation_pct: float = 0.12,
        item_type_filter: str = "sealed",
        min_age_months: int = 0,
        max_age_months: Optional[int] = None,
        exit_lookback_months: Optional[int] = None,
        exit_threshold: float = 0.0,
        trailing_stop_pct: Optional[float] = None,
        max_holding_months: Optional[int] = None,
        min_confirm_months: int = 1,
        max_price_msrp_ratio: Optional[float] = None,
    ):
        self.prices_df = prices_df.sort_index()
        self.lookback_months = lookback_months
        self.max_allocation_pct = max_allocation_pct
        self.item_type_filter = item_type_filter
        self.min_age_months = min_age_months
        self.max_age_months = max_age_months
        self.exit_lookback_months = exit_lookback_months or lookback_months
        self.exit_threshold = exit_threshold
        self.trailing_stop_pct = trailing_stop_pct
        self.max_holding_months = max_holding_months
        self.min_confirm_months = max(1, min_confirm_months)
        self.max_price_msrp_ratio = max_price_msrp_ratio

    def reset(self):
        pass

    def _trailing_return(self, item_id: str, current_date: pd.Timestamp, lookback: Optional[int] = None) -> Optional[float]:
        lb = lookback or self.lookback_months
        if item_id not in self.prices_df.columns:
            return None
        series = self.prices_df[item_id]
        series = series[series.index <= current_date].dropna()
        series = series[series > 0]
        if len(series) < lb + 1:
            return None
        past = float(series.iloc[-(lb + 1)])
        now = float(series.iloc[-1])
        if past <= 0:
            return None
        return (now - past) / past

    def _confirmed_positive(self, item_id: str, current_date: pd.Timestamp, months: int) -> bool:
        """True se il momentum e' stato positivo negli ultimi `months` mesi di
        valutazione CONSECUTIVI (incluso current_date) - non solo oggi."""
        idx = self.prices_df.index
        pos = idx.searchsorted(current_date, side="right") - 1
        if pos < 0:
            return False
        for k in range(months):
            j = pos - k
            if j < 0:
                return False
            mom = self._trailing_return(item_id, idx[j])
            if mom is None or mom <= 0:
                return False
        return True

    def _peak_since(self, item_id: str, buy_date: str, current_date: pd.Timestamp) -> Optional[float]:
        if item_id not in self.prices_df.columns:
            return None
        series = self.prices_df[item_id]
        window = series[(series.index >= pd.to_datetime(buy_date)) & (series.index <= current_date)].dropna()
        window = window[window > 0]
        if window.empty:
            return None
        return float(window.max())

    def generate_signals(
        self,
        current_date: str,
        portfolio: Portfolio,
        market_snapshot: Dict[str, Dict[str, Any]],
    ) -> List[Signal]:
        signals: List[Signal] = []
        cur_dt = pd.to_datetime(current_date)
        total_nav = portfolio.get_total_nav({k: v["current_price"] for k, v in market_snapshot.items()})
        max_item_budget = total_nav * self.max_allocation_pct

        # Uscita: il segnale di momentum (sulla finestra di uscita) scende sotto la
        # soglia -> liquida tutto. Con exit_lookback_months=lookback_months e
        # exit_threshold=0.0 (default) e' esattamente la regola originale.
        for item_id, pos in list(portfolio.positions.items()):
            if item_id not in market_snapshot:
                continue
            cur_price = market_snapshot[item_id]["current_price"]

            if self.trailing_stop_pct is not None:
                peak = self._peak_since(item_id, pos.buy_date, cur_dt)
                if peak is not None and cur_price <= peak * (1.0 - self.trailing_stop_pct):
                    signals.append(Signal(
                        action="SELL", item_id=item_id, item_name=pos.item_name,
                        item_type=pos.item_type, quantity=pos.quantity, target_price=cur_price,
                        reason=f"TSMOM: trailing stop {self.trailing_stop_pct*100:.0f}% dal massimo ({peak:.2f}€)"
                    ))
                    continue

            if self.max_holding_months is not None:
                d_buy = pd.to_datetime(pos.buy_date)
                held_m = (cur_dt.year - d_buy.year) * 12 + (cur_dt.month - d_buy.month)
                if held_m >= self.max_holding_months:
                    signals.append(Signal(
                        action="SELL", item_id=item_id, item_name=pos.item_name,
                        item_type=pos.item_type, quantity=pos.quantity, target_price=cur_price,
                        reason=f"TSMOM: time stop a {self.max_holding_months} mesi di possesso"
                    ))
                    continue

            mom = self._trailing_return(item_id, cur_dt, lookback=self.exit_lookback_months)
            if mom is not None and mom <= self.exit_threshold:
                signals.append(Signal(
                    action="SELL", item_id=item_id, item_name=pos.item_name,
                    item_type=pos.item_type, quantity=pos.quantity, target_price=cur_price,
                    reason=f"TSMOM: rendimento trailing {self.exit_lookback_months}m sotto soglia ({mom*100:+.1f}% <= {self.exit_threshold*100:.1f}%)"
                ))

        # Ingresso: segnale positivo e posizione non già aperta.
        for item_id, info in market_snapshot.items():
            if info.get("type") != self.item_type_filter or item_id in portfolio.positions:
                continue
            cur_price = info["current_price"]
            if cur_price <= 0:
                continue
            if (self.min_age_months > 0 or self.max_age_months is not None) and info.get("release_date"):
                rel_dt = pd.to_datetime(info["release_date"])
                age_m = (cur_dt.year - rel_dt.year) * 12 + (cur_dt.month - rel_dt.month)
                if age_m < self.min_age_months:
                    continue
                if self.max_age_months is not None and age_m > self.max_age_months:
                    continue
            mom = self._trailing_return(item_id, cur_dt)
            if mom is None or mom <= 0:
                continue
            if self.min_confirm_months > 1 and not self._confirmed_positive(item_id, cur_dt, self.min_confirm_months):
                continue
            if self.max_price_msrp_ratio is not None:
                msrp = info.get("msrp")
                if msrp and cur_price / msrp > self.max_price_msrp_ratio:
                    continue

            available_cash = portfolio.cash
            budget = min(available_cash, max_item_budget)
            qty = int(budget // cur_price)
            if qty < 1 and available_cash >= cur_price and cur_price <= total_nav * 0.35:
                qty = 1  # lotto minimo indivisibile per conti piccoli, come nelle altre strategie
            if qty >= 1:
                signals.append(Signal(
                    action="BUY", item_id=item_id, item_name=info.get("name", item_id),
                    item_type=self.item_type_filter, quantity=qty, target_price=cur_price,
                    reason=f"TSMOM: rendimento trailing {self.lookback_months}m positivo (+{mom*100:.1f}%)"
                ))
        return signals
