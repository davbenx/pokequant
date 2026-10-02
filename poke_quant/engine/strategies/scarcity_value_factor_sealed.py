"""
poke_quant/engine/strategies/scarcity_value_factor_sealed.py — Fattore di
Valore per Scarsita' Continua, lato BOX SIGILLATI.

RICHIESTA DELL'UTENTE (dopo aver chiesto "qual e' il prezzo massimo al quale
acquistare un sealed" e aver discusso che il tetto attuale, MSRP x 21,6, e'
un backstop di coda mai scattato - non un vero fair-value ceiling come quello
delle singole): costruire l'equivalente box del fattore scarsita' delle
singole (poke_quant/engine/strategies/scarcity_value_factor.py), che usa

    log(prezzo) ~ 1 + log(scarsita' per rarita') + eta' + franchise + ...

Il problema concreto per i box: NON esiste un equivalente diretto di "rarita'"
per un box sigillato - la tabella EXPECTED_COPIES_PER_BOX misura la scarsita'
di una SINGOLA carta dentro il box, non del box stesso. Niente in metadata
descrive in modo affidabile e non fabbricato "quanto raro e' il pull chase di
QUESTO set" per l'intero universo (set_tier esiste ma e' assegnato a mano solo
per i 47 box curati inizialmente, "UNASSIGNED" per 123/170 box scoperti in
automatico - usarlo escluderebbe la maggioranza dell'universo o userebbe un
giudizio soggettivo non riproducibile, la stessa categoria di problema che le
singole hanno risolto abbandonando la dummy binaria "premium si/no").

PROXY ADOTTATO (il piu' onesto disponibile senza fabbricare un numero):
    log(prezzo) ~ 1 + log(MSRP) + log(eta' mesi + 1) + is_one_piece

log(MSRP) sostituisce la scarsita' di rarita': un MSRP piu' alto implica
tipicamente un prodotto piu' grande/premium (piu' slot di pull, piu' carte
rare per box) - e' un proxy MECCANICO del "quanto e' probabile trovare la
carta chase dentro", non una misura diretta, dichiarato esplicitamente come
limite. Il residuo misura quindi "quanto si allontana il prezzo da cio' che
MSRP+eta'+franchise implicherebbero per un box comparabile" - stesso principio
logico delle singole (residuo = sottovalutato/sopravvalutato vs pari), con un
regressore di scarsita' piu' debole perche' non abbiamo un dato diretto.

is_jp/is_zh/is_mtg inclusi per simmetria col file singole (stesso pattern:
colonna a zero riceve peso zero via lstsq quando quei franchise non sono
presenti nel cross-section attuale, vedi commento originale) ma SENZA dati
reali oggi nell'universo liquido+MSRP (MTG escluso di default, JP/Cinese con
zero righe nell'intersezione liquido+MSRP al momento della scrittura - vedi
scripts/box_scarcity_value_factor_test.py per la verifica empirica).

product_type (booster_box/specialty_etb/specialty_bundle) deliberatamente
ESCLUSO come regressore: solo 4/40 righe nell'universo liquido+MSRP non sono
booster_box (3 ETB + 1 bundle) - troppo poche per stimare un coefficiente
affidabile, aggiungerlo rischierebbe di adattarsi al rumore di 4 osservazioni.

DIAGNOSTICA PRIMA DI COSTRUIRE QUALUNQUE REGOLA DI TRADING (stessa disciplina
di population_scarcity_factor_test.py): sulla cross-section piu' recente
(n=40), R^2 = 0,328 - log(MSRP)+eta'+franchise spiegano una parte reale ma
modesta della varianza di prezzo. I residui piu' sopravvalutati corrispondono
a set vintage/sold-out notoriamente scarsi (Unified Minds, Unbroken Bonds,
Lost Thunder) - coerente con la conoscenza di dominio, non uno sbilanciamento
palese - PRIMA di guardare l'effetto su un backtest.

LIMITE CAMPIONARIO DICHIARATO: cross-section mediana storica 20 box (min 0,
max 40 - l'universo liquido+MSRP e' cresciuto nel tempo). Molto piu' piccolo
dei 932 delle singole - qualunque esito di validazione qui ha potenza
statistica molto piu' debole e va interpretato con cautela proporzionale.
Vedi scripts/box_scarcity_value_factor_test.py per l'esito completo
(griglia/PBO/walk-forward/DSR) prima di qualunque uso in produzione.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from poke_quant.engine.strategies import Signal
from poke_quant.engine.portfolio import Portfolio


class SealedScarcityValueFactorStrategy:
    def __init__(
        self,
        rebalance_every_months: int = 6,
        top_quantile: float = 0.20,
        max_positions: int = 20,
        max_allocation_pct: float = 0.12,
        min_age_months: int = 0,
        min_cross_section: int = 20,
        use_log_age: bool = True,
        max_quantity_per_trade: Optional[int] = None,
    ):
        self.rebalance_every_months = rebalance_every_months
        self.top_quantile = top_quantile
        self.max_positions = max_positions
        self.max_allocation_pct = max_allocation_pct
        self.min_age_months = min_age_months
        self.min_cross_section = min_cross_section
        self.use_log_age = use_log_age
        self.max_quantity_per_trade = max_quantity_per_trade
        self._call_count = 0

    def reset(self):
        self._call_count = 0

    def _fit_residuals(self, cur_dt: pd.Timestamp, market_snapshot: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        rows, ids = [], []
        for item_id, info in market_snapshot.items():
            if info.get("type") != "sealed" or info.get("current_price", 0) <= 0:
                continue
            msrp = info.get("msrp")
            rel_dt_raw = info.get("release_date")
            if not msrp or not rel_dt_raw:
                continue
            rel_dt = pd.to_datetime(rel_dt_raw)
            age_m = (cur_dt.year - rel_dt.year) * 12 + (cur_dt.month - rel_dt.month)
            if age_m < self.min_age_months:
                continue
            is_op = 1.0 if info.get("franchise") == "one_piece" else 0.0
            is_mtg = 1.0 if info.get("franchise") == "magic" else 0.0
            is_jp_or_zh = 1.0 if info.get("language") in ("jp", "zh") else 0.0
            age_feat = np.log(age_m + 1.0) if self.use_log_age else float(age_m)
            feat = [1.0, np.log(float(msrp)), age_feat, is_op, is_mtg, is_jp_or_zh]
            log_price = np.log(info["current_price"])
            rows.append(feat + [log_price])
            ids.append(item_id)

        if len(rows) < self.min_cross_section:
            return {}

        arr = np.array(rows)
        X, y = arr[:, :-1], arr[:, -1]
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        residual = y - X @ coef
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

        ranked = sorted(residuals.items(), key=lambda x: x[1])
        n_buy = max(1, int(len(residuals) * self.top_quantile))
        eligible = [item_id for item_id, _ in ranked[:n_buy]]
        eligible = eligible[: self.max_positions]
        eligible_set = set(eligible)

        for item_id, pos in list(portfolio.positions.items()):
            if item_id in market_snapshot and item_id not in eligible_set and pos.item_type == "sealed":
                signals.append(Signal(
                    action="SELL", item_id=item_id, item_name=pos.item_name,
                    item_type=pos.item_type, quantity=pos.quantity,
                    target_price=market_snapshot[item_id]["current_price"],
                    reason="Valore Scarsita' Sealed: uscito dal quantile piu' sottovalutato"
                ))

        total_nav = portfolio.get_total_nav({k: v["current_price"] for k, v in market_snapshot.items()})
        max_item_budget = total_nav * self.max_allocation_pct

        for item_id in eligible:
            if item_id in portfolio.positions:
                continue
            info = market_snapshot[item_id]
            cur_price = info["current_price"]
            available_cash = portfolio.cash
            budget = min(available_cash, max_item_budget)
            qty = int(budget // cur_price)
            if qty < 1 and available_cash >= cur_price and cur_price <= total_nav * 0.35:
                qty = 1
            if self.max_quantity_per_trade is not None:
                qty = min(qty, self.max_quantity_per_trade)
            if qty >= 1:
                signals.append(Signal(
                    action="BUY", item_id=item_id, item_name=info.get("name", item_id),
                    item_type="sealed", quantity=qty, target_price=cur_price,
                    reason=f"Valore Scarsita' Sealed: residuo {residuals[item_id]:+.2f} (sottovalutato vs pari per MSRP/eta')"
                ))
        return signals
