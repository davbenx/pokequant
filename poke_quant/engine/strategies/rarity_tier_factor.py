"""
poke_quant/engine/strategies/rarity_tier_factor.py — Fattore Rarita' Ex-Ante.

A differenza del filtro di prezzo di discover_chase_cards.py (selezione sull'ESITO,
survivorship bias), qui l'eleggibilita' e' decisa dalla RARITA' TIPOGRAFICA assegnata
da Nintendo/TPCi al momento della stampa - un'informazione disponibile ex-ante,
identica per chi comprava il giorno dell'uscita e per chi guarda lo storico oggi.
Ipotesi: le rarita' con stampa limitata per definizione (1 per booster box o meno)
portano un premio di scarsita' strutturale indipendente dall'esito di prezzo.
A ogni ribilanciamento, equal-weight su tutte le carte eleggibili di rarita' premium
tra quelle correntemente prezzate (non un ranking - la rarita' e' binaria, nota,
fissa nel tempo), fino a un tetto di posizioni per restare diversificati.

ESITO VALIDAZIONE: NON VALIDATO, per un motivo diverso dagli altri candidati - non e'
un problema statistico ma di composizione del campione. PREMIUM_RARITIES coincide con
CHASE_RARITIES di discover_chase_cards.py: nell'universo attuale 270 delle 288 carte
eleggibili (94%) vengono dal campione selezionato sul prezzo corrente, solo 18 (6%) dal
campione di controllo casuale. Il risultato full-sample (Sharpe 1.11, DSR 0.864) e'
quasi certamente una riscoperta del survivorship bias di discover_chase_cards.py sotto
un'altra etichetta, non un fattore indipendente. Da ritestare solo dopo aver ampliato
molto il campione di controllo casuale per queste rarita' specifiche (oggi troppo
piccolo, n=18, per essere conclusivo da solo).

ESITO VALIDAZIONE (field_name="artist", fattore illustratore): NON VALIDATO, per un
motivo statistico stavolta, non di composizione campionaria. Su 783 singole (universo
completo, illustratore noto per 820 carte via pokemontcg.io), il fattore "carta di un
illustratore-star" (rebal=6, minage=6) rende Sharpe 0.11-0.12 full-sample con MaxDD
-60% - già debole di per se'. La griglia (4 candidati) da' PBO 24.3% e DSR 0.221
(sotto la soglia 0.90-0.95 usata in questa serie di test). Il walk-forward e' la prova
decisiva: H1 (2021-01->2023-10) Sharpe -1.92, H2 (2023-11->2026-09) Sharpe +2.51 - lo
stesso schema boom/bust visto in ogni altro fattore rotazionale testato su questo
universo, non un premio strutturale legato all'illustratore.

ESITO VALIDAZIONE (field_name="is_promo_str", fattore promo/SVP): NON VALIDATO, e
questo è il caso più insidioso di tutta la ricerca perché i numeri aggregati erano tra i
MIGLIORI visti in assoluto - migliori anche della strategia sealed in produzione.
Ritestato dopo aver corretto il buco nei pannelli prezzo (864 singole, 80 promo tra
queste, vedi scripts/rebuild_prices_with_real_fx.py): 'PROMO rebal=12 minage=6' rende
Sharpe 1.06, CAGR +23.5%, PBO 0.029 (3 candidati, 8 split), DSR 0.945, bootstrap
P(Sharpe>0)=96% (RIVERIFICATO su richiesta esplicita - "se non le hai fatte tu,
ritestale": l'etichetta era scambiata, rebal=12/minage=6 non rebal=6/minage=12, e i
numeri Sharpe 1.27/DSR 0.980 scritti qui in precedenza non sono piu' riproducibili
identici oggi, probabilmente per la serie prezzi allungata nel frattempo - vedi
scripts/retest_promo_premium_illustrator_current_universe.py per il dettaglio completo,
incluso il numero sull'universo attuale col pavimento di costo di gradazione: Sharpe
1.12, DSR 0.570). Su qualsiasi soglia usata altrove in questa ricerca, questo
passerebbe a pieni voti. MA il walk-forward - il test che in questa stessa ricerca ha
già smontato LOW-VOL (PBO 5.7% eppure fallito) e ILLUSTRATOR - mostra lo stesso
identico schema: H1 (2021-01->2023-10) Sharpe -0.64, CAGR -5.10%; H2 (2023-11->2026-09) Sharpe +2.36,
CAGR +68.30%. Le carte promo (SVP in particolare) sono un sottomercato piccolo e meno
liquido che ha vissuto una mania di prezzo particolarmente estrema nel 2024-2025 (nuovi
set Illustration Rare/SVP) - lo stesso super-ciclo boom/bust/recupero visto ovunque in
questo universo, solo più amplificato qui per la ridotta liquidità. Un PBO=0.029 e un
DSR=0.945 spettacolari NON bastano quando il segno si inverte tra le due metà del
campione: per coerenza con lo standard usato su ogni altro candidato di questa ricerca,
questo fattore resta NON VALIDATO. Non testabile in modo conclusivo con solo ~5.7 anni
di storico e un unico ciclo macro - andrebbe riprovato quando il campione includerà
più di un ciclo boom/bust indipendente.
"""

from __future__ import annotations
from typing import Dict, List, Any, FrozenSet
import pandas as pd
from poke_quant.engine.strategies import Signal
from poke_quant.engine.portfolio import Portfolio

PREMIUM_RARITIES: FrozenSet[str] = frozenset({
    "Rare Secret", "Rare Rainbow", "Rare Ultra", "Special Illustration Rare",
    "Illustration Rare", "Hyper Rare", "Rare Holo VMAX", "Rare Holo VSTAR",
})


class RarityTierFactorStrategy:
    """Nonostante il nome (storico), l'eleggibilita' e' generica su qualsiasi campo
    categorico dei metadata - vedi field_name. Usata anche per il fattore illustratore
    (field_name="artist") con la stessa identica logica di esecuzione."""
    def __init__(
        self,
        rebalance_every_months: int = 6,
        max_positions: int = 40,
        max_allocation_pct: float = 0.08,
        min_age_months: int = 6,
        item_type_filter: str = "single",
        premium_rarities: FrozenSet[str] = PREMIUM_RARITIES,
        field_name: str = "rarity",
    ):
        self.rebalance_every_months = rebalance_every_months
        self.max_positions = max_positions
        self.max_allocation_pct = max_allocation_pct
        self.min_age_months = min_age_months
        self.item_type_filter = item_type_filter
        self.premium_rarities = premium_rarities
        self.field_name = field_name
        self._call_count = 0

    def reset(self):
        self._call_count = 0

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

        # Lista ordinata per item_id (non set) - eleggibilita' deterministica,
        # esecuzione a corto di cassa sempre nello stesso ordine riproducibile.
        eligible: List[str] = []
        for item_id, info in sorted(market_snapshot.items()):
            if info.get("type") != self.item_type_filter or info.get("current_price", 0) <= 0:
                continue
            if info.get(self.field_name) not in self.premium_rarities:
                continue
            if self.min_age_months > 0 and info.get("release_date"):
                rel_dt = pd.to_datetime(info["release_date"])
                age_m = (cur_dt.year - rel_dt.year) * 12 + (cur_dt.month - rel_dt.month)
                if age_m < self.min_age_months:
                    continue
            eligible.append(item_id)
        if not eligible:
            return signals

        eligible = eligible[: self.max_positions]
        eligible_set = set(eligible)  # solo per test di appartenenza, mai iterato

        for item_id, pos in list(portfolio.positions.items()):
            if item_id in market_snapshot and item_id not in eligible_set:
                signals.append(Signal(
                    action="SELL", item_id=item_id, item_name=pos.item_name,
                    item_type=pos.item_type, quantity=pos.quantity,
                    target_price=market_snapshot[item_id]["current_price"],
                    reason="Rarita' Ex-Ante: uscita dal set di rarita' premium eleggibili"
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
                    reason=f"{self.field_name}={info.get(self.field_name)} (categoria eleggibile)"
                ))
        return signals
