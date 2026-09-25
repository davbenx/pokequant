"""
poke_quant/engine/strategies/scarcity_value_factor.py — Fattore di Valore per
Scarsita' Continua (non categoriale).

Evoluzione di relative_value_factor.py: quel fattore usava una dummy binaria
(is_premium: 0/1, "rarita' premium o no") per catturare la rarita' - una
semplificazione grezza. Qui si usa la SCARSITA' CONTINUA (1 / copie attese
per box, dalla stessa tabella generica per fascia di rarita' usata in
scripts/box_ev_theory_test.py per il valore atteso del box) come regressore -
la stessa idea richiesta dall'utente per le singole: il prezzo di una carta
dovrebbe riflettere quanto e' difficile trovarla, non solo se e' "rara si/no".

    log(prezzo) ~ 1 + log(scarsita') + one_piece + lingua_jp + eta_mesi + is_chase

Compra il quantile con residuo piu' negativo (sottovalutata rispetto a quanto
la sua scarsita' - non solo la sua categoria di rarita' - implicherebbe).

ESITO VALIDAZIONE: PRIMO FATTORE SULLE SINGOLE A SUPERARE IL DSR CORRETTO PER
L'INTERA RICERCA. Su 864 singole (chase+controllo), 'SCARCITY rebal=3 q=0.20':
Sharpe 1.74, CAGR +27.8%, MaxDD -9.8%, PBO 1.4% (5 candidati, 8 split). DSR
corretto solo per la sua griglia: 0,996. DSR corretto per TUTTA la ricerca
sulle singole (51 trial totali: 46 precedenti + 5 di questa griglia): 0,943 -
sopra la soglia di comfort 0,90-0,95 usata ovunque in questa sessione, la
prima volta che succede sulle singole.

Walk-forward: H1 Sharpe +0,83, H2 +3,16 - entrambi fortemente positivi,
nessuna inversione. Verifica decisiva (tiene anche senza survivorship bias):
solo campione di controllo casuale (n=225, mai selezionato sul prezzo)
Sharpe 0,92, H1 +1,63, H2 +1,01 - ancora forte in entrambe le meta'.

Robustezza alla tabella di probabilita' (che e' una stima GENERICA della
community, non dati ufficiali): perturbando ogni fascia di +-40% in modo
casuale (5 varianti indipendenti), lo Sharpe resta 1,66-1,75 - il risultato
non dipende dai numeri esatti della tabella, solo dall'ORDINE relativo delle
fasce di rarita' (piu' raro = piu' scarso), un'assunzione molto piu' debole
e defendibile.

Nessuna riserva residua identificata al momento - il candidato piu' solido
di tutta la ricerca, sealed incluso (DSR sealed corretto per l'intera sessione:
0,675 - vedi scripts/dsr_session_audit.py).

AGGIORNAMENTO - rifinitura della specifica (richiesta esplicitamente, "stessa
idea, specifiche alternative" invece di un meccanismo nuovo): testate 6
varianti di regressione sullo stesso fattore - eta' log-trasformata, controlli
extra (promo, illustratore), RANGO ORDINALE della fascia di rarita' invece del
valore continuo (elimina la dipendenza dalla tabella di probabilita' assunta -
usa solo l'ordine: piu' raro = rango piu' alto), winsorizzazione dei prezzi
outlier, e la combinazione delle tre. PBO tra le 6 varianti: 0,000 - nessuna
e' un caso isolato, tutte reggono (Sharpe 1,55-2,02, walk-forward sempre
positivo in entrambe le meta'). La combinazione migliore (eta' log + rango +
controlli extra, ora i default) da' Sharpe 2,02, CAGR +32,6%, MaxDD -8,5%,
DSR 0,999 sulla sola griglia di 6 varianti -> 0,980 corretto per le 62 prove
totali sulle singole in questa sessione (46 + 5 griglia scarsita' + 5 topdown
+ 6 di qui) - ancora sopra soglia, anzi piu' alto dell'originale (0,943).
use_rank=True e' la scelta piu' defendibile epistemicamente: non assume nessun
numero specifico di probabilita' di pull, solo che le fasce siano ordinate
correttamente per rarita' (rango dalla stessa tabella EXPECTED_COPIES_PER_BOX,
ma solo il suo ordine, non i valori). I parametri precedenti (scarsita'
continua, eta' lineare, senza extra controlli) restano disponibili passando
use_rank=False, use_log_age=False, extra_controls=False.

Non ancora in produzione a piena scala: un fattore che passa ogni controllo
disponibile oggi merita comunque cautela pratica (allocazione iniziale
piccola, monitoraggio, non un salto immediato a pari livello del sealed)
prima di qualsiasi capitale reale.

ISTERESI (exit_quantile) - TESTATA, NON ADOTTATA: vedi scripts/singles_hysteresis_search.py.
Tenere una posizione finche' non esce da un quantile piu' largo di quello di
ingresso (invece del quantile stretto singolo) NON migliora la strategia -
PBO 0,0% sulla griglia, la baseline (exit_quantile=None) vince su ogni
candidato per Sharpe. Il parametro resta disponibile (default None = nessun
cambiamento) perche' testato e documentato, non perche' consigliato.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from poke_quant.engine.strategies import Signal
from poke_quant.engine.portfolio import Portfolio

# Stessa tabella generica per fascia di rarita' usata in box_ev_theory_test.py -
# qui non serve il numero di copie per box in se', solo il RAPPORTO relativo tra
# fasce (la scarsita' e' l'inverso delle copie attese, in scala logaritmica).
EXPECTED_COPIES_PER_BOX = {
    "Rare Secret": 36 / 30, "Special Illustration Rare": 36 / 30, "Hyper Rare": 36 / 30,
    "Rare Shining": 36 / 30, "Rare BREAK": 36 / 30, "Rare ACE": 36 / 30,
    "Rare Rainbow": 36 / 9, "Rare Ultra": 36 / 9, "Ultra Rare": 36 / 9, "Double Rare": 36 / 9,
    "Rare Holo VMAX": 36 / 9, "Rare Holo VSTAR": 36 / 9, "Rare Holo V": 36 / 9,
    "Rare Holo GX": 36 / 9, "Rare Holo EX": 36 / 9, "Rare Holo LV.X": 36 / 9, "Rare Holo Star": 36 / 9,
    "Illustration Rare": 36 / 4,
    "Rare Holo": 36 * 0.5, "Rare": 36 * 0.5, "Common": 36 * 1.0, "Uncommon": 36 * 1.0,
}
EXCLUDED_RARITIES = {"Promo", None}

# Rango ordinale per fascia (0 = meno scarsa, N-1 = piu' scarsa) - stesso ordine
# di EXPECTED_COPIES_PER_BOX, ma senza usarne i valori numerici assoluti. Usato
# quando use_rank=True: la scelta piu' defendibile epistemicamente, dato che la
# tabella di probabilita' e' una stima generica della community, non dati
# ufficiali (vedi robustezza alla perturbazione nel docstring del modulo).
_RANK_ORDER = sorted(EXPECTED_COPIES_PER_BOX.items(), key=lambda x: x[1])
RARITY_RANK = {rarity: i for i, (rarity, _) in enumerate(_RANK_ORDER)}
_N_TIERS = len(RARITY_RANK)


class ScarcityValueFactorStrategy:
    def __init__(
        self,
        rebalance_every_months: int = 6,
        top_quantile: float = 0.20,
        max_positions: int = 60,
        max_allocation_pct: float = 0.06,
        min_age_months: int = 6,
        item_type_filter: str = "single",
        min_cross_section: int = 20,
        use_log_age: bool = True,
        use_rank: bool = True,
        extra_controls: bool = True,
        exit_quantile: Optional[float] = None,
        max_quantity_per_trade: Optional[int] = None,
    ):
        self.rebalance_every_months = rebalance_every_months
        self.top_quantile = top_quantile
        self.max_positions = max_positions
        self.max_allocation_pct = max_allocation_pct
        self.min_age_months = min_age_months
        self.item_type_filter = item_type_filter
        self.min_cross_section = min_cross_section
        self.use_log_age = use_log_age
        self.use_rank = use_rank
        self.extra_controls = extra_controls
        # Banda di isteresi (default None = nessuna, comportamento originale
        # invariato: entrata e uscita sullo STESSO quantile). Se impostato
        # (deve essere >= top_quantile), una posizione aperta resta in
        # portafoglio finche' non esce da questo quantile PIU' LARGO, non
        # appena esce dal quantile stretto di ingresso - riduce il turnover
        # da residui che oscillano attorno al taglio esatto senza un vero
        # cambio di tesi. Vedi scripts/singles_hysteresis_search.py per l'esito.
        self.exit_quantile = exit_quantile
        # Trovato indagando "e se compro piu' copie?": senza tetto, qty =
        # budget_posizione // prezzo compra decine di copie IDENTICHE (stessa
        # carta, stesso grado) su una carta economica - nel backtest validato
        # arriva a 95 copie in un solo mese (Aegislash Grade9 a 2,20€). Nessun
        # mercato reale ha mai 95 slab identici disponibili insieme allo
        # stesso prezzo tracciato - il backtest assume liquidita' infinita per
        # costruzione. Default None = nessun tetto, comportamento storico
        # invariato (i numeri validati finora NON hanno questo limite) - vedi
        # scripts/max_quantity_per_trade_test.py per l'impatto misurato di
        # aggiungerne uno realistico.
        self.max_quantity_per_trade = max_quantity_per_trade
        self._call_count = 0

    def reset(self):
        self._call_count = 0

    def _fit_residuals(self, cur_dt: pd.Timestamp, market_snapshot: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        rows, ids = [], []
        for item_id, info in market_snapshot.items():
            if info.get("type") != self.item_type_filter or info.get("current_price", 0) <= 0:
                continue
            rarity = info.get("rarity")
            if rarity in EXCLUDED_RARITIES or rarity not in EXPECTED_COPIES_PER_BOX:
                continue
            rel_dt_raw = info.get("release_date")
            if not rel_dt_raw:
                continue
            rel_dt = pd.to_datetime(rel_dt_raw)
            age_m = (cur_dt.year - rel_dt.year) * 12 + (cur_dt.month - rel_dt.month)
            if age_m < self.min_age_months:
                continue
            if self.use_rank:
                scarcity_feat = float(RARITY_RANK[rarity]) / _N_TIERS
            else:
                scarcity_feat = np.log(1.0 / EXPECTED_COPIES_PER_BOX[rarity])
            is_op = 1.0 if info.get("franchise") == "one_piece" else 0.0
            is_jp = 1.0 if info.get("language") == "jp" else 0.0
            is_chase = 1.0 if info.get("selection_method") == "chase_price_filter_survivorship_biased" else 0.0
            age_feat = np.log(age_m + 1.0) if self.use_log_age else float(age_m)
            feat = [1.0, scarcity_feat, is_op, is_jp, age_feat, is_chase]
            if self.extra_controls:
                is_promo = 1.0 if info.get("is_promo") else 0.0
                has_artist = 1.0 if info.get("artist") else 0.0
                feat += [is_promo, has_artist]
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

        if self.exit_quantile is not None:
            n_hold = max(n_buy, int(len(residuals) * self.exit_quantile))
            hold_set = {item_id for item_id, _ in ranked[:n_hold]}
        else:
            hold_set = eligible_set

        for item_id, pos in list(portfolio.positions.items()):
            if item_id in market_snapshot and item_id not in hold_set:
                signals.append(Signal(
                    action="SELL", item_id=item_id, item_name=pos.item_name,
                    item_type=pos.item_type, quantity=pos.quantity,
                    target_price=market_snapshot[item_id]["current_price"],
                    reason="Valore Scarsita': uscito dal quantile piu' sottovalutato"
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
            if self.max_quantity_per_trade is not None:
                qty = min(qty, self.max_quantity_per_trade)
            if qty >= 1:
                signals.append(Signal(
                    action="BUY", item_id=item_id, item_name=info.get("name", item_id),
                    item_type=self.item_type_filter, quantity=qty, target_price=cur_price,
                    reason=f"Valore Scarsita': residuo {residuals[item_id]:+.2f} (sottovalutata vs pari per rarita' pesata)"
                ))
        return signals
