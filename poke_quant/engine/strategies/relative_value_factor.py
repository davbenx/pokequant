"""
poke_quant/engine/strategies/relative_value_factor.py — Fattore di Valore Relativo
(regressione cross-sezionale, non time-series).

Diverso da ogni fattore testato finora in questa ricerca: non guarda il proprio
trend/z-score storico di una carta (quello e' zscore_factor/momentum_factor in
cross_sectional_factor.py, entrambi TIME-SERIES sulla propria storia). Qui, ad
ogni ribilanciamento, si stima una regressione OLS cross-sezionale:

    log(prezzo) ~ 1 + rarita_premium + one_piece + lingua_jp + eta_mesi + is_chase

su TUTTE le carte idonee quel mese (chase + controllo casuale insieme), e si va
long sulle carte col residuo piu' negativo - quelle che, dato il loro
grado/rarita'/eta'/franchise, il modello dice "dovrebbero costare piu' di
quanto costano" rispetto ai loro pari. E' l'idea "sottovalutata rispetto al
mercato nonostante grado/rarita' direbbero il contrario" chiesta esplicitamente
dall'utente - non richiede dati di popolazione PSA (bloccati, vedi ricerca
precedente), solo le caratteristiche gia' in metadata.

is_chase (selection_method == "chase_price_filter_survivorship_biased") e'
un regressore OBBLIGATORIO, non opzionale: senza di esso una prima versione di
questo test comprava per il 71% carte "random_control" (26% dell'universo) -
non un segnale di valore, solo la riscoperta che le chase costano di piu'
perche' SONO STATE selezionate per costare di piu'. Con is_chase nella
regressione, il residuo confronta chase-con-chase e control-con-control,
isolando il valore relativo vero da quell'artefatto di campionamento.

ESITO VALIDAZIONE: il candidato piu' promettente di tutta la ricerca sulle
singole - primo a passare il walk-forward SENZA inversione di segno in nessuna
meta', ma non ancora allo stesso livello di certezza della strategia sealed in
produzione. Su 864 singole (chase+controllo, is_chase come regressore
obbligatorio): 'VALUE rebal=3 q=0.20' rende Sharpe 1.06, CAGR +33.0%, MaxDD
-14.9%, PBO 1.4% (5 candidati, 8 split - molto stabile), DSR 0.895 (appena
sotto la soglia di comfort 0.90-0.95 usata in questa ricerca). Walk-forward:
H1 Sharpe +0.38, H2 +1.60 - ENTRAMBI positivi, nessuna inversione.

Verifica decisiva: l'effetto tiene INDIPENDENTEMENTE dentro ciascun
sottocampione preso da solo, non solo nel pool misto - la prova che serviva
per escludere che fosse ancora l'artefatto chase-vs-controllo:
  - Solo chase (n=639): Sharpe 1.33, H1 +0.26, H2 +2.90
  - Solo controllo casuale (n=225, MAI selezionato sul prezzo): Sharpe 1.09,
    H1 +1.79, H2 +1.14
Nel campione di controllo, genuinamente non contaminato da survivorship bias,
l'effetto e' forte in ENTRAMBE le meta' - la prova piu' pulita vista in questa
ricerca. Riserva onesta: e' circa la nona/decima ipotesi indipendente provata
su questa serie di singole in questa sessione - una correzione per multiple
comparisons su TUTTA la sequenza (non solo sui 5 candidati di questa griglia)
abbasserebbe ulteriormente la DSR effettiva, e non e' stata calcolata.
AGGIORNAMENTO - correzione per l'intera sessione di ricerca (richiesta
esplicitamente dall'utente, "procedi" dopo aver elencato il piano):
contando TUTTI i trial indipendenti tentati su questa serie di singole in
questa sessione (non solo i 5 di questa griglia) - section_singles_factor_search
(9), section_singles_technical_fundamental_search (4), griglia illustratore (4),
griglia promo (3), chase-TSMOM (3), box-vs-paniere (2), questa griglia (5),
grade-spread (8), grade-lead-lag (8) = 46 trial totali - il DSR del vincitore
crolla da 0,895 a **0,581**. Ben sotto la soglia di comfort (0,90-0,95) usata
per OGNI altro candidato in questa ricerca, inclusa la strategia sealed in
produzione (DSR 0,913).

Controlli aggiuntivi fatti per completezza:
- Stabilita' per segmento: NON CONCLUSIVA per un problema di dati, non del
  fattore - tutte le singole One Piece hanno release_date=None in metadata
  (escluse dalla regressione per costruzione, zero trade) e non esistono
  singole giapponesi nel pannello grade9. Il segmento "Pokemon EN" testato
  e' di fatto quasi l'intero universo (784/864), non un check indipendente.
- Robustezza della specifica: qualitativamente solida (Sharpe 1,05-1,78 su 4
  varianti di regressione - eta lineare/log, con/senza controlli promo e
  illustratore - H1 resta sempre positivo, nessun collasso). Rassicurante sul
  fatto che non sia fragile a scelte arbitrarie, ma queste 4 varianti sono
  ESSE STESSE altri trial nello stesso spazio di ricerca - non le uso per
  "aggiornare" al numero migliore (sarebbe lo stesso errore di data-snooping
  che questa correzione cerca di evitare).

VERDETTO FINALE: per coerenza con lo standard applicato a ogni altro candidato
di questa ricerca, con DSR 0,581 questo fattore NON PASSA la soglia usata in
questa sessione. Resta il candidato qualitativamente piu' interessante (unico
a non invertire segno nel walk-forward), ma non e' "il fattore che ha
funzionato" - e' un'ipotesi che regge meglio delle altre sotto un occhio meno
rigoroso e non regge sotto lo stesso rigore applicato al resto. Da riprendere
solo con un'ipotesi pre-registrata su dati futuri (non derivata da questa
stessa ricerca esplorativa), non con piu' tentativi su questo stesso campione.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from poke_quant.engine.strategies import Signal
from poke_quant.engine.portfolio import Portfolio
from poke_quant.engine.strategies.rarity_tier_factor import PREMIUM_RARITIES


class RelativeValueFactorStrategy:
    def __init__(
        self,
        rebalance_every_months: int = 6,
        top_quantile: float = 0.20,
        max_positions: int = 60,
        max_allocation_pct: float = 0.06,
        min_age_months: int = 6,
        item_type_filter: str = "single",
        min_cross_section: int = 20,
    ):
        self.rebalance_every_months = rebalance_every_months
        self.top_quantile = top_quantile
        self.max_positions = max_positions
        self.max_allocation_pct = max_allocation_pct
        self.min_age_months = min_age_months
        self.item_type_filter = item_type_filter
        self.min_cross_section = min_cross_section
        self._call_count = 0

    def reset(self):
        self._call_count = 0

    def _fit_residuals(self, cur_dt: pd.Timestamp, market_snapshot: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        rows, ids = [], []
        for item_id, info in market_snapshot.items():
            if info.get("type") != self.item_type_filter or info.get("current_price", 0) <= 0:
                continue
            rel_dt_raw = info.get("release_date")
            if not rel_dt_raw:
                continue
            rel_dt = pd.to_datetime(rel_dt_raw)
            age_m = (cur_dt.year - rel_dt.year) * 12 + (cur_dt.month - rel_dt.month)
            if age_m < self.min_age_months:
                continue
            is_premium = 1.0 if info.get("rarity") in PREMIUM_RARITIES else 0.0
            is_op = 1.0 if info.get("franchise") == "one_piece" else 0.0
            is_jp = 1.0 if info.get("language") == "jp" else 0.0
            is_chase = 1.0 if info.get("selection_method") == "chase_price_filter_survivorship_biased" else 0.0
            log_price = np.log(info["current_price"])
            rows.append([1.0, is_premium, is_op, is_jp, float(age_m), is_chase, log_price])
            ids.append(item_id)

        if len(rows) < self.min_cross_section:
            return {}

        arr = np.array(rows)
        X, y = arr[:, :6], arr[:, 6]
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        fitted = X @ coef
        residual = y - fitted
        return dict(zip(ids, residual))

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

        residuals = self._fit_residuals(cur_dt, market_snapshot)
        if not residuals:
            return signals

        n_buy = max(1, int(len(residuals) * self.top_quantile))
        eligible = [item_id for item_id, _ in sorted(residuals.items(), key=lambda x: x[1])[:n_buy]]
        eligible = eligible[: self.max_positions]
        eligible_set = set(eligible)

        for item_id, pos in list(portfolio.positions.items()):
            if item_id in market_snapshot and item_id not in eligible_set:
                signals.append(Signal(
                    action="SELL", item_id=item_id, item_name=pos.item_name,
                    item_type=pos.item_type, quantity=pos.quantity,
                    target_price=market_snapshot[item_id]["current_price"],
                    reason="Valore Relativo: uscito dal quantile piu' sottovalutato"
                ))

        total_nav = portfolio.get_total_nav({k: v["current_price"] for k, v in market_snapshot.items()})
        target_per_position = total_nav * min(self.max_allocation_pct, 1.0 / max(1, len(eligible)))

        for item_id in eligible:
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
                    reason=f"Valore Relativo: residuo {residuals[item_id]:+.2f} (sottovalutata vs pari)"
                ))
        return signals
