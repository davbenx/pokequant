"""
app.py — PokeQuant: Dashboard della strategia validata (TS Momentum, box sigillati).

Mostra SOLO cio' che ha superato la validazione istituzionale di questa sessione
(bootstrap, walk-forward H1/H2 senza inversione di segno - vedi scripts/optimize_and_falsify.py).
Tutto il resto (Slabs Radar, desk discrezionale a tier, OptimalSealedStrategy/
SealedAccumulatorStrategy/ChaseDipBuyerStrategy, l'audit a slider) e' stato rimosso
per direttiva esplicita - "tieni solo quello che abbiamo validato" - non e' sparito:
resta nel codice/git history per ricerca futura, solo non piu' mostrato come se
fosse pronto per capitale reale.

DSR: il numero originale (0,913) era corretto solo per la griglia di lookback con cui
la strategia fu scelta (n_trials=5). Un audit successivo (richiesto esplicitamente
dopo aver scoperto lo stesso problema sul fattore di valore relativo delle singole)
ha ricontato TUTTI i trial tentati sul lato sealed in questa sessione - lookback (5),
logica di uscita (9), finestra d'eta' (7), time stop (7), teoria EV del box (4) = 32,
poi + conferma momentum (7, respinta) + isteresi (7, respinta) + espansione universo
via rapporto prezzo/MSRP (1, adottata) = 47 - il DSR corretto e' 0,778 sull'universo
allargato (era 0,675 sui 32 trial originali e sull'universo a 36 box). Sopra la soglia
0,90-0,95 usata ovunque in questa ricerca ancora non ci arriva, ma l'espansione
dell'universo (36->40 box, vedi scripts/sealed_universe_expansion_test.py) e' un
miglioramento reale, non solo una correzione al ribasso come le altre volte. Mostrati
ENTRAMBI i numeri in dashboard, non solo il piu' favorevole - vedi
scripts/dsr_session_audit.py.

Azionabilita' per l'Italia: link diretti a Cardmarket (mercato primario per chi opera
dall'Italia, vedi OPERATIONS_ITALIA.md) su ogni posizione BUY/HOLD.
"""

from __future__ import annotations
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))

from poke_quant.data.storage import load_price_matrix, load_metadata
from poke_quant.data.cardmarket_bridge import get_cardmarket_deep_link
from poke_quant.engine.backtester import Backtester
from poke_quant.engine.strategies.time_series_momentum import TimeSeriesMomentumStrategy
from poke_quant.engine.strategies.scarcity_value_factor import ScarcityValueFactorStrategy
from poke_quant.engine.position_sizing import age_weight, inverse_vol_split
from scripts.generate_monthly_signal import compute_signal_rows
from poke_quant.data.liquidity_filter import liquid_sealed_ids, MAX_PRICE_TO_MSRP_RATIO, liquid_singles_ids
from poke_quant.data.price_fetcher import (
    fetch_pricecharting_cover_image_url,
    fetch_pricecharting_variant_grade9,
    fetch_pricecharting_variant_grade_tier,
    fetch_pricecharting_grade_tier_price,
    parse_pricecharting_url_or_slug,
    search_metadata_card_by_query,
)
from poke_quant.config import estimate_usa_import_landed_cost, IMPORT_FROM_USA
from scripts.generate_singles_signal import (
    compute_singles_signal_rows, compute_singles_avoid_rows, compute_singles_alternative_rows,
    filter_singles_rows, _set_label,
    PRODUCTION_PARAMS as SINGLES_PARAMS, DAC7_SINGLES_PARAMS,
)
from poke_quant.slabs.grading_multipliers import (
    Era,
    normalize_era,
    adjust_price_for_grading,
    get_variant_multiplier,
    variant_to_pricecharting_key,
    ERA_PSA10_TO_PSA9_RATIO,
    ERA_BGS95_TO_PSA9_RATIO,
    get_recommended_grade_for_card,
    get_recommended_grade_targets,
    get_card_pop_pressure,
    get_card_strategy_and_pop_details,
)

# Soglie DAC7 (direttiva UE 2021/514): sopra queste soglie annue le piattaforme
# (Cardmarket, eBay, ecc.) segnalano il venditore alle autorità fiscali come
# probabile attività commerciale, non occasionale. Non sono soglie di
# performance - sono un vincolo operativo/di conformità richiesto dall'utente.
DAC7_MAX_ANNUAL_EUR = 2000.0
DAC7_MAX_ANNUAL_TRADES = 30

# =============================================================================
# NUMERI VALIDATI. Fissi, non ricalcolati a ogni caricamento pagina - una
# strategia si rivalida ogni 6 mesi (vedi OPERATIONS_ITALIA.md), non ogni
# refresh del browser.
# =============================================================================
VALIDATED_BOX = {
    # Universo allargato a 40 box (era 36): oltre al cutoff 2019, include box
    # piu' vecchi con rapporto prezzo/MSRP reale dentro il range gia' osservato
    # nell'universo moderno - vedi scripts/sealed_universe_expansion_test.py e
    # poke_quant/data/liquidity_filter.py::liquid_sealed_ids.
    # AGGIORNAMENTO: include ora la spedizione REALE a carico del compratore
    # all'acquisto (10€/box, poke_quant.config.SHIPPING_COSTS) - Portfolio.buy()
    # non l'aveva mai applicata (vedi scripts/buy_side_shipping_test.py). Sharpe
    # scende 1,31->1,18, onesto: nessun acquisto e' mai stato gratis nella realta'.
    # AGGIORNAMENTO 2: aggiunto un tetto prezzo/MSRP all'INGRESSO (non solo
    # all'ammissione nell'universo) - stesso 21,6x gia' calibrato in
    # liquidity_filter.py, non una nuova griglia - vedi
    # scripts/box_max_price_ratio_test.py. Impatto storico ZERO (nessun trade
    # passato lo violava: Sharpe/CAGR/MaxDD/Trade identici), ma protegge da
    # oggi in avanti - 1 box su 34 con MSRP noto e' oggi sopra soglia e viene
    # escluso da un nuovo acquisto. Un trial in piu' nel conteggio onesto
    # (48->49) anche se non ha cambiato nessun numero.
    "dsr_own_grid": 0.913, "dsr_full_session": 0.681, "n_trials_full_session": 49,
    "pbo": 0.286, "sharpe": 1.18, "cagr": 22.80, "max_dd": -11.12,
    "bootstrap_cagr_p_pos": 100, "bootstrap_sharpe_p_pos": 100,
    "h1_sharpe": 0.36, "h2_sharpe": 1.82,
}
VALIDATED_SINGLES = {
    # Universo corretto a 935 carte (era 864): il filtro di attendibilita' su
    # historical_prices.csv valutava una serie DIVERSA da quella che il
    # fattore usa davvero (historical_prices_graded_singles_grade9.csv) - vedi
    # scripts/flag_unreliable_assets.py.
    # AGGIORNAMENTO: include ora la spedizione REALE a carico del compratore
    # all'acquisto (7€/carta) - vedi scripts/buy_side_shipping_test.py. Impatto
    # molto piu' grande che sui box: un trade tipico da 50-300€ regge molto
    # meno bene una spedizione fissa di un box da centinaia di euro. DSR scende
    # 0,954->0,836 - SOTTO la soglia di comfort 0,90-0,95 per la prima volta da
    # quando questo fattore l'ha superata. Non e' piu' "il primo candidato a
    # superarla" con questo conto piu' onesto.
    # AGGIORNAMENTO 2 (trovato verificando un prezzo reale - un Raichu #14 a
    # 250€ tutto compreso contro 107,60€ mostrati): il filtro di attendibilita'
    # vedeva solo salti/volatilita', non un prezzo grade9 persistentemente
    # troppo basso su carte vintage poco liquide. Aggiunto un check indipendente
    # (rapporto grade9/raw vs coorte d'eta', vedi liquidity_filter.py::
    # compute_grade_raw_ratio_flags e scripts/graded_raw_ratio_reliability_test.py):
    # universo 935->869 carte, e il risultato MIGLIORA (non solo protegge) -
    # coerente con l'ipotesi che fossero falsi positivi da dato sottile, non
    # alfa reale. DSR risale 0,836->0,876, ancora sotto la soglia di comfort
    # ma piu' vicino. Nota onesta: raichu_14 stesso resta nell'universo (24°
    # percentile della sua coorte, sopra la soglia conservativa del 10°) - il
    # filtro riduce il problema, non lo elimina caso per caso.
    # AGGIORNAMENTO 3 (l'utente ha verificato un prezzo reale - Mantine #64 a
    # 11,48EUR Grade9, quando la sola gradazione PSA/CGC costa piu' di cosi'
    # anche nella fascia bulk): trovato un BUG, non solo un filtro mancante -
    # get_singles_backtest_results() qui sotto filtrava correttamente
    # data_quality="thin_unreliable", ma scripts/generate_singles_signal.py
    # (che genera le liste BUY/alternative/avoid REALMENTE mostrate) non
    # filtrava MAI nulla: 2/15 carte BUY e 11/107 alternative erano gia'
    # flaggate come dato inattendibile, ma proposte come acquisto. Corretto
    # (liquidity_filter.py::liquid_singles_ids, ora usato ovunque). Aggiunto
    # anche un pavimento di costo di gradazione (MIN_SINGLES_MEDIAN_PRICE_EUR
    # = 20EUR, stima ragionata non un listino verificato): 216/869 carte
    # (25%, di cui 208 "random_control" - campione casuale per testare
    # survivorship bias, non scelte come investimento) avevano un prezzo
    # Grade9 mediano storico sotto il costo minimo reale di farle gradare -
    # l'intera "sottovalutazione" era un artefatto della regressione
    # log-lineare compressa vicino allo zero. Testato
    # (scripts/grading_cost_floor_test.py): escluderle NON peggiora l'edge
    # (Sharpe 1,57->1,58, DSR migliora leggermente nonostante +3 trial nel
    # conteggio onesto) - coerente con l'ipotesi che non fossero alfa reale.
    # AGGIORNAMENTO 4 (l'utente: "Vedo ancora Sandslash #42, che e' sotto il
    # prezzo da gradazione"): BUG nel pavimento stesso - usava la mediana su
    # TUTTA la storia della carta, che una carta scesa e rimasta bassa per
    # mesi puo' superare grazie a prezzi vecchi piu' alti (Sandslash: mediana
    # storica 20,32EUR, ma sceso a ~14EUR da 7 mesi consecutivi - passava il
    # filtro nonostante il prezzo reale di oggi sia sotto soglia). Non isolato:
    # altre 9 carte con lo stesso problema. Corretto usando la mediana sui
    # SOLI ultimi 3 mesi (stessa finestra di freschezza del ribilanciamento
    # in produzione) invece di tutta la storia - universo 653->675 (rimuove
    # le carte scese e rimaste basse, riammette quelle genuinamente risalite
    # sopra soglia di recente, es. Lombre #34). Sharpe 1,58->1,52, DSR
    # 0,878->0,850 - piccolo calo onesto, il costo naturale di una carta reale
    # in meno erroneamente inclusa, non un peggioramento del fattore stesso.
    # AGGIORNAMENTO 5 (richiesto dall'utente: "ricostruisci i dati... testaci sopra
    # strategie... includi i miglioramenti in produzione"): campione di controllo
    # casuale ampliato 6x (discover_random_control_singles.py --per-set 30, per
    # risolvere la contaminazione 94%/6% trovata testando la rarita' premium - vedi
    # rarity_tier_factor.py) e pannelli prezzo ricostruiti per l'universo intero
    # (3.207 item). Universo singole 675->1.085. Headline Sharpe 1,52->2,05, DSR
    # 0,850->0,981 - MA verificato (scripts/max_quantity_retest_expanded_universe.py)
    # che il salto NON e' un miglioramento reale: viene quasi interamente dal poter
    # comprare in blocco molte piu' carte comuni economiche (il campione ampliato
    # pesca uniformemente su ogni carta di un set, la maggioranza comuni) - lo stesso
    # limite gia' noto (scripts/max_quantity_per_trade_test.py) ma AMPLIFICATO. A
    # tetto di quantita' realistico (1-2 copie per acquisto, il caso normale per
    # slab gradati) il DSR resta 0,028-0,346 - INVARIATO O PEGGIORE di prima
    # (0,044-0,371), non migliorato. L'headline resta la metodologia standard usata
    # ovunque in questo progetto (nessun tetto), ma il divario tra teorico e
    # realizzabile si e' allargato, non ridotto, con piu' dati - vedi il caveat
    # STRATEGIA UNIFICATA PRODUZIONE:
    # Optimum vincolato verificato empiricamente in scripts/dac7_turnover_search.py:
    # rebalance_every_months=3, max_positions=20.
    # Unifica la modalita' standard e la conformita' DAC7 (< 30 vendite annue):
    # Sharpe 2.39 (DSR 0.999), CAGR +67.68%, MaxDD -9.70%, 28.6 vendite/anno reali.
    # BUG TROVATO (2026-09-29, richiesta esplicita dell'utente "verifica che le
    # variabili della strategia siano ottimizzate/corrette"): dac7_turnover_search.py
    # calcolava il DSR con n_trials = sola griglia locale (24), non il conteggio
    # CUMULATIVO di questa stessa linea di ricerca (62 trial gia' spesi prima -
    # vedi poke_quant/engine/strategies/scarcity_value_factor.py). Corretto: 62+24=86
    # trial totali. Impatto pratico minimo (Sharpe 2,37 su 68 mesi e' quasi
    # insensibile al conteggio: DSR 0,999->0,995) - resta sopra la soglia di
    # comfort 0,90-0,95, la conclusione non cambia.
    "dsr_own_grid": 0.999, "dsr_full_session": 0.995, "n_trials_full_session": 86,
    "pbo": 0.014, "sharpe": 2.39, "cagr": 67.68, "max_dd": -9.70,
    "h1_sharpe": 1.15, "h2_sharpe": 4.10,
}
VALIDATED_BLEND = {"sharpe": 2.94, "cagr": 41.64, "max_dd": -5.42}

st.set_page_config(page_title="PokeQuant — TS Momentum", page_icon="⚡", layout="wide")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: #cbd5e1; }
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #080c14; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 4px; }
.nav-header { background: linear-gradient(180deg, rgba(15,23,42,0.85) 0%, rgba(8,12,20,0.95) 100%); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 12px 18px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; }
.nav-title { font-size: 19px; font-weight: 800; letter-spacing: -0.4px; color: #f8fafc; }
.pill-tag { display: inline-flex; align-items: center; gap: 4px; padding: 3px 9px; border-radius: 9999px; font-size: 11px; font-weight: 600; }
.pill-emerald { background: rgba(16,185,129,0.12); color: #10b981; border: 1px solid rgba(16,185,129,0.28); }
.pill-blue { background: rgba(56,189,248,0.12); color: #38bdf8; border: 1px solid rgba(56,189,248,0.28); }
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 16px; }
.kpi-card { background: rgba(15,23,42,0.55); border: 1px solid rgba(255,255,255,0.07); border-radius: 12px; padding: 12px 15px; }
.kpi-label { font-size: 10.5px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: #94a3b8; margin-bottom: 2px; }
.kpi-value { font-family: 'JetBrains Mono', monospace; font-size: 21px; font-weight: 700; color: #f8fafc; letter-spacing: -0.4px; }
.kpi-sub { font-size: 11px; font-weight: 500; margin-top: 2px; }
.kpi-sub-emerald { color: #10b981; }
.kpi-sub-amber { color: #fbbf24; }
.signal-card { background: rgba(15,23,42,0.65); border-radius: 10px; padding: 12px 14px; margin-bottom: 8px; border-left: 3px solid; border-top: 1px solid rgba(255,255,255,0.05); border-right: 1px solid rgba(255,255,255,0.05); border-bottom: 1px solid rgba(255,255,255,0.05); display: flex; align-items: center; gap: 12px; }
.signal-card-buy { border-left-color: #10b981; }
.signal-card-sell { border-left-color: #f43f5e; }
.signal-card-thumb { width: 56px; height: 56px; object-fit: contain; border-radius: 6px; background: rgba(255,255,255,0.04); flex-shrink: 0; }
.signal-card-body { flex: 1; min-width: 0; }
.section-title { font-size: 16px; font-weight: 700; letter-spacing: -0.3px; color: #f1f5f9; margin-top: 10px; margin-bottom: 2px; }
.section-desc { font-size: 12px; color: #94a3b8; margin-bottom: 10px; }
.cm-btn { display: inline-flex; align-items: center; gap: 5px; background: rgba(16,185,129,0.12); border: 1px solid rgba(16,185,129,0.35); color: #10b981 !important; padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 600; text-decoration: none !important; }
.cm-btn-sell { background: rgba(245,158,11,0.12); border: 1px solid rgba(245,158,11,0.35); color: #fbbf24 !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


DATA_CACHE_DIR = Path(__file__).resolve().parent / "data_cache"
PRECOMPUTED_FILE = DATA_CACHE_DIR / "precomputed_dashboard_data.json"
IMAGE_CACHE_FILE = DATA_CACHE_DIR / "product_image_cache.json"


class PrecomputedBacktestResult:
    """Wrapper leggero che emula l'interfaccia di BacktestResult a partire dai dati
    precomputati in data_cache/precomputed_dashboard_data.json, evitando di rieseguire
    la simulazione su migliaia di serie ad ogni apertura/refresh della dashboard."""
    def __init__(self, data: dict):
        self.strategy_name = data.get("strategy_name", "")
        self.total_trades = data.get("total_trades", 0)
        self.win_rate = float(data.get("win_rate", 0.0))
        self.profit_factor = float(data.get("profit_factor", 0.0))
        self.sharpe = float(data.get("sharpe", 0.0))
        self.cagr = float(data.get("cagr", 0.0))
        self.max_drawdown = float(data.get("max_drawdown", 0.0))

        m_ret = data.get("monthly_returns", {})
        self.monthly_returns = pd.Series(
            {pd.to_datetime(k): float(v) for k, v in m_ret.items()}
        ).sort_index()

        nav_list = data.get("nav_history", [])
        if nav_list:
            df_nav = pd.DataFrame(nav_list)
            df_nav["date"] = pd.to_datetime(df_nav["date"])
            df_nav.set_index("date", inplace=True)
            self.nav_history = df_nav
        else:
            self.nav_history = pd.DataFrame(columns=["nav", "cash", "portfolio_value"])

        trades_list = data.get("trades", [])
        if trades_list:
            df_trades = pd.DataFrame(trades_list)
            df_trades["buy_date"] = pd.to_datetime(df_trades["buy_date"])
            df_trades["sell_date"] = pd.to_datetime(df_trades["sell_date"])
            self.trades_df = df_trades
        else:
            self.trades_df = pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def load_precomputed_dashboard_data() -> Optional[dict]:
    """Carica i dati precomputati mensilmente/trimestralmente da disco/GitHub.
    Ritorna None se il file non esiste o è corrotto, attivando il fallback al calcolo live."""
    if not PRECOMPUTED_FILE.exists():
        return None
    try:
        with open(PRECOMPUTED_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=3600)
def load_persistent_image_cache() -> dict:
    """Carica l'indice URL copertine persistito da data_cache/product_image_cache.json."""
    if not IMAGE_CACHE_FILE.exists():
        return {}
    try:
        with open(IMAGE_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


@st.cache_data(show_spinner=False, ttl=3600)
def get_metadata():
    return load_metadata()


@st.cache_data(show_spinner=False, ttl=3600)
def get_signal():
    cached = load_precomputed_dashboard_data()
    if cached and "box_signals" in cached:
        return cached["box_signals"], cached.get("latest_date_box", "N/A")
    rows, latest_date = compute_signal_rows()
    return rows, latest_date.strftime("%Y-%m-%d")


@st.cache_data(show_spinner=False, ttl=3600)
def get_prices_full():
    return load_price_matrix()


@st.cache_data(show_spinner=False, ttl=3600)
def get_market_indices():
    """Indici equal-weight buy&hold (nessuna strategia, nessun timing) sull'universo
    sealed validato - la 'beta' del mercato, da confrontare con l'alfa della
    strategia (curva NAV più sotto). Utili come contesto, non come segnale
    d'ingresso: mostrano quale segmento è caldo/freddo, non quando comprare."""
    cached = load_precomputed_dashboard_data()
    if cached and "market_indices" in cached:
        idx_data = cached["market_indices"]
        overall = pd.Series({pd.to_datetime(k): float(v) for k, v in idx_data["overall_index"].items()}).sort_index()
        segments = {
            name: pd.Series({pd.to_datetime(k): float(v) for k, v in s.items()}).sort_index()
            for name, s in idx_data.get("segment_indices", {}).items()
        }
        counts = idx_data.get("segment_counts", {})
        breadth = pd.Series({pd.to_datetime(k): float(v) for k, v in idx_data.get("breadth_series", {}).items()}).sort_index()
        return overall, segments, counts, breadth

    metadata = get_metadata()
    prices_full = load_price_matrix()
    sealed_ids = liquid_sealed_ids(metadata, prices_full)

    segments: dict[str, list[str]] = {"Pokémon EN": [], "Pokémon JP": [], "One Piece TCG": []}
    for k in sealed_ids:
        v = metadata[k]
        if v.get("franchise") == "one_piece":
            segments["One Piece TCG"].append(k)
        elif v.get("language") == "jp":
            segments["Pokémon JP"].append(k)
        else:
            segments["Pokémon EN"].append(k)

    def build_index(ids: list[str]) -> pd.Series:
        sub = prices_full[ids]
        basket_ret = sub.pct_change().mean(axis=1, skipna=True).fillna(0.0)
        idx = (1.0 + basket_ret).cumprod() * 100.0
        idx.iloc[0] = 100.0
        return idx

    overall_index = build_index(sealed_ids)
    segment_indices = {name: build_index(ids) for name, ids in segments.items() if len(ids) >= 3}
    segment_counts = {name: len(ids) for name, ids in segments.items()}

    # Ampiezza di mercato: % dell'universo con momentum trailing 12m positivo, mese per mese.
    # Stessa regola della strategia in produzione, solo aggregata invece che tradata.
    lookback = 12
    breadth = {}
    for i in range(lookback, len(prices_full.index)):
        date = prices_full.index[i]
        n_pos = n_tot = 0
        for k in sealed_ids:
            series = prices_full[k][prices_full.index <= date].dropna()
            series = series[series > 0]
            if len(series) < lookback + 1:
                continue
            past, now = float(series.iloc[-(lookback + 1)]), float(series.iloc[-1])
            if past <= 0:
                continue
            n_tot += 1
            if (now - past) / past > 0:
                n_pos += 1
        if n_tot > 0:
            breadth[date] = n_pos / n_tot * 100.0
    breadth_series = pd.Series(breadth)

    return overall_index, segment_indices, segment_counts, breadth_series


@st.cache_data(show_spinner=False, ttl=3600)
def get_backtest_results():
    cached = load_precomputed_dashboard_data()
    if cached and "backtest_results" in cached and "box" in cached["backtest_results"]:
        data = cached["backtest_results"]["box"]
        return PrecomputedBacktestResult(data), data.get("n_universe", 52)

    metadata = get_metadata()
    prices_full = load_price_matrix()
    sealed_ids = liquid_sealed_ids(metadata, prices_full)
    meta_sub = {k: v for k, v in metadata.items() if k in sealed_ids}
    prices_sub = prices_full[sealed_ids]
    strat = TimeSeriesMomentumStrategy(prices_sub, lookback_months=12, max_price_msrp_ratio=MAX_PRICE_TO_MSRP_RATIO)
    bt = Backtester(strat, prices_sub, meta_sub, initial_cash=10000.0, platform="cardmarket",
                     apply_liquidity_slippage=True, apply_holding_cost=True, apply_buy_side_shipping=True)
    res = bt.run()
    return res, len(sealed_ids)


@st.cache_data(show_spinner=False, ttl=3600)
def get_singles_signal(mode: str = "production"):
    cached = load_precomputed_dashboard_data()
    key = "production" if mode == "production" else "dac7"
    if cached and "singles_signals" in cached and key in cached["singles_signals"]:
        buy_rows = cached["singles_signals"][key].get("buy_rows", [])
        return buy_rows, cached.get("latest_date_singles", "N/A")
    params = SINGLES_PARAMS if mode == "production" else DAC7_SINGLES_PARAMS
    rows, latest_date = compute_singles_signal_rows(params)
    return rows, latest_date.strftime("%Y-%m-%d")


@st.cache_data(show_spinner=False, ttl=3600)
def get_singles_avoid_signal(mode: str = "production"):
    cached = load_precomputed_dashboard_data()
    key = "production" if mode == "production" else "dac7"
    if cached and "singles_signals" in cached and key in cached["singles_signals"]:
        avoid_rows = cached["singles_signals"][key].get("avoid_rows", [])
        return avoid_rows, cached.get("latest_date_singles", "N/A")
    params = SINGLES_PARAMS if mode == "production" else DAC7_SINGLES_PARAMS
    rows, latest_date = compute_singles_avoid_rows(params)
    return rows, latest_date.strftime("%Y-%m-%d")


@st.cache_data(show_spinner=False, ttl=3600)
def get_singles_alternatives(mode: str = "production"):
    cached = load_precomputed_dashboard_data()
    key = "production" if mode == "production" else "dac7"
    if cached and "singles_signals" in cached and key in cached["singles_signals"]:
        alt_rows = cached["singles_signals"][key].get("alt_rows", [])
        return alt_rows, cached.get("latest_date_singles", "N/A")
    params = SINGLES_PARAMS if mode == "production" else DAC7_SINGLES_PARAMS
    rows, latest_date = compute_singles_alternative_rows(params)
    return rows, latest_date.strftime("%Y-%m-%d")


def annualized_turnover(trades_df: pd.DataFrame) -> tuple:
    """(vendite/anno, EUR/anno) da un trades_df del backtest - il conteggio non
    scala col capitale (e' strutturale, dipende da quante posizioni/quanto
    spesso ruota), il volume EUR sì (proporzionale al capitale usato nel
    backtest, qui sempre 10.000€ di partenza)."""
    if trades_df.empty:
        return 0.0, 0.0
    sell_dates = pd.to_datetime(trades_df["sell_date"])
    years = max(1e-6, (sell_dates.max() - sell_dates.min()).days / 365.25)
    eur_per_year = (trades_df["sell_price_unit"] * trades_df["quantity"]).sum() / years
    trades_per_year = len(trades_df) / years
    return trades_per_year, eur_per_year


@st.cache_data(show_spinner=False, ttl=3600)
def get_singles_prices_full():
    return load_price_matrix("historical_prices_graded_singles_grade9.csv")


@st.cache_data(show_spinner=False, ttl=3600)
def get_all_database_card_options():
    metadata = get_metadata()
    prices_full = get_singles_prices_full()
    if prices_full.empty:
        return [], {}
    latest = prices_full.iloc[-1]
    options = []
    opt_map = {}
    for item_id, info in metadata.items():
        if info.get("type") == "single" and item_id in latest and latest[item_id] > 0 and not pd.isna(latest[item_id]):
            set_lbl = _set_label(info) or "?"
            lbl = f"{info.get('name', item_id)} [{set_lbl}] — {latest[item_id]:.2f}€"
            options.append(lbl)
            opt_map[lbl] = {
                "item_id": item_id,
                "name": info.get("name", item_id),
                "set_name": set_lbl,
                "current_price_eur": float(latest[item_id]),
                "max_edge_price_eur": float(latest[item_id]) * 1.05,
            }
    options.sort()
    return options, opt_map


@st.cache_data(show_spinner=False, ttl=3600)
def get_singles_backtest_results(mode: str = "production"):
    cached = load_precomputed_dashboard_data()
    key = "singles_production" if mode == "production" else "singles_dac7"
    if cached and "backtest_results" in cached and key in cached["backtest_results"]:
        data = cached["backtest_results"][key]
        return PrecomputedBacktestResult(data), data.get("n_universe", 3105)

    params = SINGLES_PARAMS if mode == "production" else DAC7_SINGLES_PARAMS
    metadata = get_metadata()
    prices_full = get_singles_prices_full()
    singles_ids = liquid_singles_ids(metadata, prices_full)
    meta_sub = {k: v for k, v in metadata.items() if k in singles_ids}
    prices_sub = prices_full[singles_ids]
    strat = ScarcityValueFactorStrategy(**params)
    bt = Backtester(strat, prices_sub, meta_sub, initial_cash=10000.0, platform="cardmarket",
                     apply_liquidity_slippage=True, apply_holding_cost=True, apply_buy_side_shipping=True)
    res = bt.run()
    return res, len(singles_ids)


@st.cache_data(show_spinner=False, ttl=3600)
def get_cached_pc_variant_grade9(game_slug: str, item_slug: str, variant_key: str):
    return fetch_pricecharting_variant_grade9(game_slug, item_slug, variant_key)


@st.cache_data(show_spinner=False, ttl=3600)
def get_cached_pc_variant_grade_tier(game_slug: str, item_slug: str, variant_key: str, tier: str):
    return fetch_pricecharting_variant_grade_tier(game_slug, item_slug, variant_key, tier=tier)


@st.cache_data(show_spinner=False, ttl=3600)
def get_box_singles_split():
    """Split di capitale box/singole per inverse-vol (risk parity).
    Legge dai dati precomputati se disponibili, altrimenti calcola live.

    Formula centralizzata in poke_quant.engine.position_sizing.inverse_vol_split
    (trovato in audit generale, 2026-09-29: questa funzione e' scripts/
    update_monthly_cache.py reimplementavano la stessa formula in due punti
    indipendenti, a rischio di divergere silenziosamente)."""
    cached = load_precomputed_dashboard_data()
    if cached and "risk_parity" in cached:
        rp = cached["risk_parity"]
        return float(rp.get("w_box", 0.52)), float(rp.get("w_singles", 0.48))

    res_box, _ = get_backtest_results()
    res_singles, _ = get_singles_backtest_results("production")
    common_idx = res_box.monthly_returns.index.intersection(res_singles.monthly_returns.index)
    vol_box = res_box.monthly_returns.loc[common_idx].std()
    vol_singles = res_singles.monthly_returns.loc[common_idx].std()
    return inverse_vol_split(vol_box, vol_singles)


# Cache in memoria con TTL
_IMAGE_CACHE: dict = {}
_IMAGE_SUCCESS_TTL = 7 * 24 * 3600
_IMAGE_FAILURE_TTL = 3600


def get_product_image(game_slug: str, item_slug: str) -> Optional[str]:
    if not game_slug or not item_slug:
        return None
    key = (game_slug, item_slug)
    now = time.time()
    cached = _IMAGE_CACHE.get(key)
    if cached is not None:
        url, cached_at, ttl = cached
        if now - cached_at < ttl:
            return url

    # Verifica cache persistita su disco (data_cache/product_image_cache.json)
    p_cache = load_persistent_image_cache()
    p_key = f"{game_slug}:{item_slug}"
    if p_key in p_cache and p_cache[p_key]:
        url = p_cache[p_key]
        _IMAGE_CACHE[key] = (url, now, _IMAGE_SUCCESS_TTL)
        return url

    try:
        url = fetch_pricecharting_cover_image_url(game_slug, item_slug)
    except Exception:
        url = None
    _IMAGE_CACHE[key] = (url, now, _IMAGE_SUCCESS_TTL if url else _IMAGE_FAILURE_TTL)
    return url


def prefetch_product_images(pairs: list) -> None:
    """Carica da cache persistita o scarica in PARALLELO le immagini non ancora note."""
    to_fetch = []
    now = time.time()
    p_cache = load_persistent_image_cache()
    for game_slug, item_slug in pairs:
        if not game_slug or not item_slug:
            continue
        key = (game_slug, item_slug)
        cached = _IMAGE_CACHE.get(key)
        if cached is not None and now - cached[1] < cached[2]:
            continue
        p_key = f"{game_slug}:{item_slug}"
        if p_key in p_cache and p_cache[p_key]:
            _IMAGE_CACHE[key] = (p_cache[p_key], now, _IMAGE_SUCCESS_TTL)
            continue
        to_fetch.append(key)
    if not to_fetch:
        return
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_pricecharting_cover_image_url, gs, isl): (gs, isl) for gs, isl in to_fetch}
        for future in as_completed(futures):
            key = futures[future]
            try:
                url = future.result()
            except Exception:
                url = None
            _IMAGE_CACHE[key] = (url, time.time(), _IMAGE_SUCCESS_TTL if url else _IMAGE_FAILURE_TTL)


def render_thumb_html(img_url: Optional[str]) -> str:
    """Tag HTML per la miniatura prodotto (box o carta) - fattorizzato in audit
    generale (2026-09-29): la stessa identica riga era copiata in 4 punti del
    file, a rischio di divergere se una veniva aggiornata e le altre no."""
    return f'<img class="signal-card-thumb" src="{img_url}" />' if img_url else '<div class="signal-card-thumb"></div>'


def render_usa_import_html(price_eur: float, item_type: str) -> str:
    """Frammento HTML per il costo sdoganato stimato da USA - fattorizzato in
    audit generale (2026-09-29): esisteva copiato due volte (box e singole)
    con formattazione GIA' divergente (.0f vs .2f, "sdoganato da USA" vs
    "sdoganato USA") - lo stesso tipo di drift che questa sessione ha sempre
    trattato come un problema, non un dettaglio estetico."""
    landed = estimate_usa_import_landed_cost(price_eur, item_type=item_type)
    return f' &nbsp;·&nbsp; <span style="color:#fbbf24;">sdoganato da USA ~{landed:.2f}€</span>'


def build_price_chart(item_id: str, name: str, prices_full: pd.DataFrame, months: int = 36):
    """Costruisce un grafico del prezzo fisso e non zoomabile (36 mesi), calibrato per
    evidenziare a colpo d'occhio il trend strutturale ed eventuali anomalie (dump, spike)."""
    if item_id not in prices_full.columns:
        return None
    series = prices_full[item_id].dropna()
    series = series[series > 0].tail(months)
    if len(series) < 2:
        return None

    y_min, y_max = float(series.min()), float(series.max())
    pad = (y_max - y_min) * 0.08 if y_max > y_min else y_max * 0.1
    y_range = [max(0.0, y_min - pad), y_max + pad]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        mode="lines+markers",
        line=dict(color="#38bdf8", width=2.0),
        marker=dict(size=3.5, color="#38bdf8"),
        fill="tozeroy",
        fillcolor="rgba(56, 189, 248, 0.06)",
        hovertemplate="%{x|%b %Y}: <b>%{y:,.0f}€</b><extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=175,
        margin=dict(l=4, r=8, t=6, b=4),
        showlegend=False,
        dragmode=False,
        xaxis=dict(
            fixedrange=True,
            showgrid=False,
            tickfont=dict(size=10, color="#94a3b8"),
            tickformat="%b %y",
        ),
        yaxis=dict(
            fixedrange=True,
            range=y_range,
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            tickfont=dict(size=10, color="#94a3b8"),
            tickformat=",.0f€",
        ),
    )
    return fig


def build_allocation(buy_rows: list, capital: float, metadata: dict, latest_date: str,
                      max_allocation_pct: float = 0.12):
    """Ripartisce il capitale per peso-età, poi applica il tetto per posizione
    dichiarato in sidebar (12% del capitale) con un waterfall: chi sfora il tetto
    viene fissato al tetto e l'eccedenza si ridistribuisce sui restanti, finche'
    nessuno sfora piu' - prima questa funzione calcolava solo la proporzione per
    peso senza applicare alcun tetto, contraddicendo il testo in sidebar.

    L'ordine di RITORNO segue quello di buy_rows in ingresso (per momentum
    decrescente, vedi generate_monthly_signal.py) - non viene piu' riordinato
    per peso-eta'. Bug trovato verificando "priorita' ai primi in lista": il
    riordino per peso faceva mostrare in cima i box piu' anziani (piu' peso di
    sizing), non quelli col miglior segnale, contraddicendo sia la didascalia
    sia l'ordine di acquisto REALE usato dal motore di backtest dopo il fix in
    time_series_momentum.py (vedi scripts/box_entry_priority_order_test.py) -
    peso-eta' resta il criterio di TAGLIA della posizione, non piu' di ordine."""
    latest_dt = pd.to_datetime(latest_date)
    weighted = []
    for r in buy_rows:
        rel_dt = metadata.get(r["item_id"], {}).get("release_date")
        age_m = None
        if rel_dt:
            rd = pd.to_datetime(rel_dt)
            age_m = (latest_dt.year - rd.year) * 12 + (latest_dt.month - rd.month)
        weighted.append((r, age_weight(age_m)))

    cap = capital * max_allocation_pct
    alloc_by_idx = {}
    remaining_capital = capital
    free_idx = set(range(len(weighted)))
    while free_idx:
        total_w = sum(weighted[i][1] for i in free_idx) or 1.0
        over_cap = []
        for i in free_idx:
            share = remaining_capital * (weighted[i][1] / total_w)
            if share > cap:
                over_cap.append(i)
        if not over_cap:
            for i in free_idx:
                alloc_by_idx[i] = remaining_capital * (weighted[i][1] / total_w)
            break
        for i in over_cap:
            alloc_by_idx[i] = cap
            remaining_capital -= cap
            free_idx.remove(i)

    return [(weighted[i][0], alloc_by_idx[i], weighted[i][1]) for i in range(len(weighted))]


def build_equal_allocation(buy_rows: list, capital: float, max_allocation_pct: float = 0.12):
    """Come build_allocation, ma a peso uguale (nessun peso-età per le singole) -
    stesso waterfall del tetto per posizione.

    NOTA (audit generale, 2026-09-29): implementazione indipendente, non
    parametrizzazione di build_allocation - le due funzioni duplicano lo
    stesso algoritmo di waterfall invece di condividerlo. Non unificate qui
    (rischio di introdurre una regressione in un meccanismo di allocazione
    capitale gia' testato, per un guadagno di manutenibilita' non urgente) -
    se in futuro build_allocation viene corretta di nuovo, controllare se la
    stessa correzione serve anche qui."""
    n = len(buy_rows)
    if n == 0:
        return []
    cap = capital * max_allocation_pct
    alloc_by_idx = {}
    remaining_capital = capital
    free_idx = set(range(n))
    while free_idx:
        share = remaining_capital / len(free_idx)
        over_cap = [i for i in free_idx if share > cap]
        if not over_cap:
            for i in free_idx:
                alloc_by_idx[i] = share
            break
        for i in over_cap:
            alloc_by_idx[i] = cap
            remaining_capital -= cap
            free_idx.remove(i)
    return [(buy_rows[i], alloc_by_idx[i]) for i in range(n)]


def get_box_franchise_label(r: dict, metadata: dict | None = None) -> str:
    """Restituisce l'etichetta del franchise per i box (Pokémon EN, Pokémon JP, One Piece TCG, Magic (MTG))."""
    f = r.get("franchise")
    lang = r.get("language")
    if f in ["Pokémon EN", "Pokémon JP", "One Piece TCG", "Magic (MTG)"]:
        return f
    if not f and metadata:
        meta = metadata.get(r.get("item_id"), {})
        f = meta.get("franchise", "pokemon")
        lang = meta.get("language", "en")
    f = f or "pokemon"
    lang = lang or "en"
    if f == "pokemon":
        return "Pokémon JP" if lang == "jp" else "Pokémon EN"
    elif f == "one_piece":
        return "One Piece TCG"
    elif f == "magic":
        return "Magic (MTG)"
    return str(f).title()


def render_box_card(r: dict, alloc: float | None, w: float | None, capital_box: float,
                    metadata: dict, prices_full: pd.DataFrame, show_usa_import: bool,
                    tier_type: str = "core") -> None:
    """Renderizza la card per un box sigillato in base al tier (core, bench, vault)."""
    meta = metadata.get(r["item_id"], {})
    link = get_cardmarket_deep_link(r["name"], franchise=meta.get("franchise", "pokemon"),
                                     language=meta.get("language", "en"), game_slug=meta.get("game_slug"))
    img_url = get_product_image(meta.get("game_slug"), meta.get("item_slug"))
    img_tag = render_thumb_html(img_url)
    max_price_html = (f' &nbsp;·&nbsp; <span style="color:#94a3b8;">massimo (tot.) '
                       f'{r["max_price_eur"]:.0f}€</span>') if r.get("max_price_eur") is not None else ""
    usa_import_html = render_usa_import_html(r["current_price_eur"], "sealed") if show_usa_import else ""
    box_price = r["current_price_eur"]

    if tier_type == "core" and alloc is not None and w is not None:
        if box_price <= 0:
            qty_est, spend_est, skip_reason = 1, alloc, None
        elif alloc >= box_price:
            qty_est, spend_est, skip_reason = int(alloc // box_price), alloc, None
        elif box_price <= capital_box * 0.35:
            qty_est, spend_est, skip_reason = 1, box_price, "budget"
        else:
            qty_est, spend_est, skip_reason = 0, 0.0, "troppo_grande"

        if skip_reason == "troppo_grande":
            qty_html = (f' &nbsp; <span style="color:#f43f5e;">⚠️ non acquistabile a questo capitale — costa {box_price:,.0f}€, '
                        f'sopra il 35% dei {capital_box:,.0f}€ dedicati ai box</span>')
        elif skip_reason == "budget":
            qty_html = (f' &nbsp; <span style="color:#fbbf24;">→ 1 pz. ⚠️ richiede {spend_est:,.0f}€, più dei {alloc:,.0f}€ '
                        f'ideali — comprato per intero (lotto indivisibile) se hai capitale libero</span>')
        else:
            qty_warn = ' ⚠️ <span style="color:#fbbf24;">assume più copie identiche disponibili insieme</span>' if qty_est > 3 else ""
            qty_html = f' &nbsp; <span style="color:#94a3b8;">→ {qty_est} pz.{qty_warn}</span>'
        alloc_display = spend_est if skip_reason == "budget" else alloc
        alloc_line = f'<br><span style="font-family:\'JetBrains Mono\',monospace; font-size:15px; color:#f8fafc;">{alloc_display:,.0f}€</span>{qty_html}'
        weight_html = f" &nbsp;·&nbsp; peso età {w:.2f}"
    elif tier_type == "vault":
        alloc_line = f'<br><span style="font-family:\'JetBrains Mono\',monospace; font-size:14px; color:#fbbf24;">🏛️ Asset Vault / Grail</span> &nbsp; <span style="color:#94a3b8;">(Richiede liquidità dedicata o capitale elevato)</span>'
        weight_html = ""
    else:  # bench
        alloc_line = f'<br><span style="font-family:\'JetBrains Mono\',monospace; font-size:14px; color:#38bdf8;">🛡️ Riserva / Panchina</span> &nbsp; <span style="color:#94a3b8;">(Subentra in sequenza se un box Core è irreperibile a prezzo equo)</span>'
        weight_html = ""

    st.markdown(f"""
    <div class="signal-card signal-card-buy">
        {img_tag}
        <div class="signal-card-body">
        <strong>{r['name']}</strong> &nbsp; <span style="color:#10b981;">+{r['trailing_12m_return_pct']:.0f}% (12m)</span>
        &nbsp;·&nbsp; {r['current_price_eur']:.0f}€ (PriceCharting){weight_html}{max_price_html}{usa_import_html}
        {alloc_line}
        &nbsp; <a class="cm-btn" href="{link}" target="_blank">🛒 Verifica su Cardmarket</a>
        </div>
    </div>
    """, unsafe_allow_html=True)
    chart = build_price_chart(r["item_id"], r["name"], prices_full)
    if chart is not None:
        st.plotly_chart(chart, use_container_width=True,
                         config={"displayModeBar": False, "scrollZoom": False, "staticPlot": False, "doubleClick": False},
                         key=f"chart_buy_{tier_type}_{r['item_id']}")


def render_single_card(r: dict, alloc: float, metadata: dict, singles_prices_full: pd.DataFrame,
                       show_usa_import: bool, key_prefix: str = "single") -> None:
    """Renderizza la card per una singola gradata: Target, Prezzo, Alternative e Pressione Popolazione."""
    meta = {"franchise": r.get("franchise", "pokemon"), "language": r.get("language", "en")}
    full_meta = metadata.get(r["item_id"], {})
    link = get_cardmarket_deep_link(r["name"], franchise=meta["franchise"], language=meta["language"],
                                     item_type="single", game_slug=full_meta.get("game_slug"))
    start = r.get("signal_start_date")
    start_str = start.strftime("%Y-%m") if hasattr(start, "strftime") else str(start)
    img_url = get_product_image(full_meta.get("game_slug"), full_meta.get("item_slug"))
    img_tag = render_thumb_html(img_url)
    card_era = r.get("era") or full_meta.get("era")
    rel_year = int(str(full_meta.get("release_date", "2020"))[:4]) if full_meta.get("release_date") else 2020
    rec = get_recommended_grade_for_card(rel_year=rel_year, era=card_era)

    # Risoluzione Target Grade, Alternative Minori e Pop Pressure
    rec_targets = get_recommended_grade_targets(
        base_psa9_eur=float(r["current_price_eur"]),
        era=rec["era"],
        item_id=r.get("item_id"),
        game_slug=full_meta.get("game_slug"),
        item_slug=full_meta.get("item_slug"),
    )
    ladder = rec_targets["ladder"]
    pop_pressure = rec_targets.get("pop_pressure", {})
    pop_badge = pop_pressure.get("badge_html", "")

    is_target_psa10 = (rec_targets["target_grade"] == "PSA 10")
    if is_target_psa10:
        target_p = float(r.get("target_price_eur") or rec_targets["target_price_eur"])
        era_ratio = ERA_PSA10_TO_PSA9_RATIO.get(normalize_era(rec["era"]), 2.80)
        target_max_edge = float(r.get("target_max_edge_price_eur") or (round(r["max_edge_price_eur"] * era_ratio, 2) if r.get("max_edge_price_eur") else round(target_p * 1.05, 2)))
        target_grade_label = "PSA 10"
        target_badge_text = "🎯 Target: PSA 10"
        target_badge_color = "#f59e0b"
        base_g9_html = f' &nbsp;·&nbsp; <span style="color:#64748b; font-size:12px;">(Base G9: {r["current_price_eur"]:.2f}€)</span>'
    else:
        target_p = float(r["current_price_eur"])
        target_max_edge = float(r["max_edge_price_eur"]) if r.get("max_edge_price_eur") is not None else None
        target_grade_label = "PSA 9"
        target_badge_text = "🎯 Target: PSA 9"
        target_badge_color = "#10b981"
        base_g9_html = ""

    max_price_html = (f' &nbsp;·&nbsp; <span style="color:#94a3b8;">max per edge '
                       f'<strong style="color:#f8fafc;">{target_max_edge:.2f}€</strong></span>') if target_max_edge is not None else ""

    usa_import_html = render_usa_import_html(target_p, "single") if show_usa_import else ""

    # Allocazione e quantità
    if alloc > 0:
        qty_est = max(1, int(alloc // target_p)) if target_p > 0 else 1
        qty_warn = ' ⚠️ <span style="color:#fbbf24;">assume più slab reperibili</span>' if qty_est > 1 else ""
        alloc_display = (f'<span style="font-family:\'JetBrains Mono\',monospace; font-size:15px; color:#f8fafc;">{alloc:,.0f}€</span>'
                         f' &nbsp; <span style="color:#94a3b8;">→ <strong>{qty_est} pz.</strong> {target_grade_label}{qty_warn}</span>')
    else:
        alloc_display = f'<span style="font-family:\'JetBrains Mono\',monospace; font-size:13px; color:#94a3b8;">Alternativa &nbsp;→ 1 pz. {target_grade_label}</span>'

    # Badge fonte dati reale vs stima per la scala delle alternative minori
    has_real_lower = ladder["8.0"]["is_real"] or ladder["7.0"]["is_real"]
    data_badge = (
        '<span style="color:#10b981; font-size:11px; font-weight:600;" title="Prezzi dei gradi minori (8 e 7) basati su vendite reali PriceCharting">✨ Prezzi Alt: Reali PC</span>'
        if has_real_lower else
        '<span style="color:#64748b; font-size:11px;" title="Prezzi dei gradi minori stimati con moltiplicatori quantitativi">📊 Prezzi Alt: Stima</span>'
    )

    # Lean strip: alternative minori + pop pressure + data badge (costruito senza spazi bianchi multipli per evitare bug markdown <div>)
    strip_html = (
        f'<div style="margin:5px 0 4px 0; font-size:12px; line-height:1.4; color:#cbd5e1; background:rgba(30,41,59,0.6); border-radius:4px; padding:5px 8px; border:1px solid rgba(56,189,248,0.2);">'
        f'<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:6px;">'
        f'<span>🥈 <strong>Alternative Minori:</strong> <span style="color:#f8fafc; font-family:\'JetBrains Mono\',monospace;">{rec_targets["minor_alternatives_str"]}</span></span>'
        f'<span>{pop_badge} &nbsp;{data_badge}</span>'
        f'</div>'
        f'</div>'
    )

    card_html = (
        f'<div class="signal-card signal-card-buy">'
        f'{img_tag}'
        f'<div class="signal-card-body">'
        f'<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">'
        f'<div>'
        f'<strong>{r["name"]}</strong> &nbsp; <span style="color:#38bdf8; font-weight:600;">[{r.get("set_name") or "?"}]</span>'
        f' &nbsp; <span style="color:#94a3b8;">{r["rarity"]}</span>'
        f'</div>'
        f'<div>'
        f'<span style="background:{target_badge_color}22; color:{target_badge_color}; border:1px solid {target_badge_color}; border-radius:4px; padding:2px 8px; font-size:12px; font-weight:700;">{target_badge_text}</span>'
        f'</div>'
        f'</div>'
        f'<div style="margin-top:2px;">'
        f'<span style="color:#10b981; font-weight:700; font-size:14px;">~{target_p:.2f}€</span>'
        f' &nbsp;·&nbsp; sconto vs. pari <strong style="color:#38bdf8;">{r["discount_pct"]:+.0f}%</strong>{base_g9_html}'
        f'{max_price_html}'
        f' &nbsp;·&nbsp; <span style="color:#94a3b8;">segnale da {start_str} ({r.get("months_in_signal", 0)}m)</span>{usa_import_html}'
        f'</div>'
        f'{strip_html}'
        f'<div style="margin-top:4px;">'
        f'{alloc_display} &nbsp; <a class="cm-btn" href="{link}" target="_blank">🛒 Verifica su Cardmarket</a>'
        f'</div>'
        f'</div>'
        f'</div>'
    )

    st.markdown(card_html, unsafe_allow_html=True)
    chart = build_price_chart(r["item_id"], r["name"], singles_prices_full)
    if chart is not None:
        st.plotly_chart(chart, use_container_width=True,
                         config={"displayModeBar": False, "scrollZoom": False, "staticPlot": False, "doubleClick": False},
                         key=f"chart_{key_prefix}_{r['item_id']}")


def main():
    metadata = get_metadata()
    prices_full = get_prices_full()
    sig_rows, latest_date = get_signal()
    n_buy = sum(1 for r in sig_rows if r["signal"] == "BUY/HOLD")
    n_sell = sum(1 for r in sig_rows if r["signal"] == "AVOID/SELL")
    n_excessive = sum(1 for r in sig_rows if "PREZZO ECCESSIVO" in r["signal"])
    n_verify = len(sig_rows) - n_buy - n_sell - n_excessive
    # Split box/singole per inverse-vol (risk parity), non piu' 50/50 hardcoded -
    # vedi docstring di get_box_singles_split() e scripts/box_singles_split_optimization.py.
    w_box, w_singles = get_box_singles_split()

    # Trovato in audit generale (2026-09-29, richiesto dall'utente "trova bug...
    # invalida"): precomputed_dashboard_data.json ha un campo generated_at
    # (scripts/update_monthly_cache.py) ma la dashboard non lo mostrava mai -
    # i numeri potevano essere vecchi di settimane senza che l'utente lo sapesse.
    # Nessuna GitHub Action rigenera questa cache automaticamente: serve un
    # umano che ricordi di rilanciare update_monthly_cache.py.
    _cache_data = load_precomputed_dashboard_data()
    if _cache_data and _cache_data.get("generated_at"):
        _gen_dt = pd.to_datetime(_cache_data["generated_at"])
        _age_days = (pd.Timestamp.now() - _gen_dt).days
        if _age_days <= 40:
            _staleness_pill = f'<span class="pill-tag pill-blue">Cache: {_age_days}g fa</span>'
        else:
            _staleness_pill = (f'<span class="pill-tag" style="background:rgba(244,63,94,0.15); '
                                f'color:#f43f5e; border:1px solid rgba(244,63,94,0.35);">⚠️ Cache: {_age_days}g fa — '
                                f'rilancia scripts/update_monthly_cache.py</span>')
    else:
        _staleness_pill = '<span class="pill-tag pill-blue">Calcolo live (no cache)</span>'

    st.markdown(f"""
    <div class="nav-header">
        <div>
            <span class="nav-title">⚡ PokeQuant</span>
            <span style="color:#64748b; font-size:12px; margin-left:8px;">Blend {w_box*100:.0f}/{w_singles*100:.0f} · Box Sigillati (TS Momentum) + Singole (Fattore Scarsità)</span>
        </div>
        <div>
            <span class="pill-tag pill-blue">Blend Sharpe {VALIDATED_BLEND['sharpe']:.2f}</span>
            <span class="pill-tag pill-blue">Segnale {latest_date[:7]}</span>
            {_staleness_pill}
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📋 Come si usa, in pratica", expanded=False):
        st.markdown(f"""
1. **Ogni mese**, consulta le liste 🟢 verdi (Box, poi Singole) già filtrate e pronte all'uso.
2. **Cardmarket**: clicca "Verifica su Cardmarket", verifica la corrispondenza esatta del set **[Set]** e resta **sotto il prezzo massimo per edge**.
3. **Pezzi indicati**: rispetta "→ N pz." — per singole alternative o budget rimanente, consulta la panchina.
4. **Target Grado & Pop**: punta al grado indicato nel badge (PSA 10 moderno, PSA 9 vintage/mid-era); verifica il badge di pressione demografica per le alternative minori.
5. **Capitale**: ripartito {w_box*100:.0f}/{w_singles*100:.0f} box/singole in risk parity. Registra sempre gli acquisti/esiti per calibrare il modello.
        """)

    # --- SIDEBAR: capitale + conformità DAC7 ---
    with st.sidebar:
        st.markdown("### 💰 Capitale")
        capital = st.number_input("Capitale dedicato (€)", min_value=100.0, max_value=1_000_000.0,
                                   value=10000.0, step=500.0)
        st.caption(f"{w_box*100:.0f}% box sigillati, {w_singles*100:.0f}% singole (fattore scarsità, risk "
                   "parity — non più un 50/50 fisso) — le due strategie hanno "
                   "correlazione bassa (0,37): il blend porta Sharpe 1,38→1,90 e MaxDD -11,1%→-7,01% "
                   "rispetto al solo box (stesso periodo comune, frizioni incluse). Cap 12% del capitale per "
                   "singola posizione dentro ciascuna metà, box pesato per età (0,4x sotto i 18 mesi, 1,0x dopo).")

        # --- COMPLIANCE (Fase 3: raggruppati - Import USA e DAC7 sono entrambi
        # "costi/limiti esterni al modello", prima separati da due blocchi di
        # filtri nel mezzo senza un motivo di ordinamento) ---
        st.markdown("---")
        st.markdown("### 🛃🇪🇺 Compliance & Costi Esterni")
        show_usa_import = st.checkbox(
            "Mostra costo sdoganato stimato (TCGplayer/eBay.com)", value=False,
            help="Prezzo oggetto + spedizione + IVA 22% + dazio UE, stimato — comprare SEMPRE a questo costo "
                 "pieno distrugge l'edge (testato). Vedi 'ℹ️ Dettagli' qui sotto per i numeri.")
        with st.expander("ℹ️ Dettagli — import USA"):
            st.caption("Il prezzo PriceCharting è quello USA — per comprarlo davvero a quel livello serve un venditore "
                       "USA, non Cardmarket EU. Dal 1° luglio 2026 (Reg. UE 382/2026) è stata abolita la soglia di "
                       "franchigia doganale a 150€: OGNI spedizione extra-UE paga dazio, qualsiasi valore. Stima: "
                       "oggetto + spedizione internazionale + IVA 22% + dazio forfettario UE 3€ + commissione di "
                       "sdoganamento del corriere (~15€, indicativa — varia per corriere). Alta confidenza su IVA/dazio "
                       "(normativa verificata), bassa sulla commissione corriere — non è un preventivo vincolante. "
                       "⚠️ TESTATO (scripts/usa_landed_cost_edge_test.py): comprare SEMPRE a questo costo pieno "
                       "distrugge l'edge — singole Sharpe 1,57→-0,33 (perdita netta), box Sharpe 1,18→0,58 con MaxDD "
                       "triplicato. Usa questo numero solo come soglia informativa (EU è comunque meglio o peggio "
                       "di importare), non come canale di acquisto regolare.")
        limit_capital_dac7 = st.checkbox("Limita capitale al volume sicuro DAC7 (2.000€/anno)", value=False,
                                         help="Direttiva UE DAC7: sopra 2.000€/anno o 30 vendite, le piattaforme "
                                              "segnalano il profilo al fisco. Limita il capitale per restare sotto "
                                              "soglia. Vedi 'ℹ️ Dettagli' qui sotto.")
        with st.expander("ℹ️ Dettagli — DAC7"):
            st.caption("Direttiva UE DAC7: sopra 2.000€ di incasso lordo o 30 vendite annue, "
                       "le piattaforme come Cardmarket/eBay segnalano il profilo alle autorità fiscali. "
                       "La strategia PokeQuant sulle singole opera già nativamente all'optimum vincolato "
                       "di 20 posizioni (~28 vendite/anno, Sharpe 2.39), rimanendo strutturalmente "
                       "sotto le 30 vendite annue. Questo toggle limita il capitale allocato per restare "
                       "anche sotto il tetto monetario dei 2.000€/anno di vendite stimate.")
        # Verificato in audit generale (2026-09-29): NON un residuo morto - le
        # config "produzione" e "DAC7" per le singole sono state unificate a
        # monte (scripts/generate_singles_signal.py::DAC7_SINGLES_PARAMS =
        # PRODUCTION_PARAMS, vedi commento li') perche' l'optimum vincolato
        # verificato (rebalance=3, max_positions=20, Sharpe 2,39) e' ANCHE il
        # miglior Sharpe assoluto - non serve piu' una seconda modalita' che
        # sacrifichi Sharpe per la conformita', quindi singles_mode e' sempre
        # "production" e la sola leva DAC7 rimasta (il tetto sul VOLUME in EUR,
        # che dipende dal capitale scelto dall'utente, non dai parametri della
        # strategia) e' il checkbox sopra.
        singles_mode = "production"

        res_box_dac7check, _ = get_backtest_results()
        res_singles_dac7check, _ = get_singles_backtest_results(singles_mode)
        box_trades_yr, box_eur_yr_per_10k = annualized_turnover(res_box_dac7check.trades_df)
        singles_trades_yr, singles_eur_yr_per_10k = annualized_turnover(res_singles_dac7check.trades_df)

        # Il conteggio vendite/anno e' strutturale (non scala col capitale) - solo
        # il volume EUR/anno scala linearmente col capitale allocato a ciascuna meta',
        # ORA pesata per w_box/w_singles (non piu' un 50/50 implicito).
        total_trades_yr = box_trades_yr + singles_trades_yr
        eur_yr_at_capital = (box_eur_yr_per_10k * w_box + singles_eur_yr_per_10k * w_singles) * (capital / 10000.0)

        combined_rate_per_eur = (box_eur_yr_per_10k * w_box + singles_eur_yr_per_10k * w_singles) / 10000.0
        safe_max_capital = (DAC7_MAX_ANNUAL_EUR / combined_rate_per_eur) if combined_rate_per_eur > 0 else capital

        effective_capital = min(capital, safe_max_capital) if limit_capital_dac7 else capital

        over_count = total_trades_yr > DAC7_MAX_ANNUAL_TRADES
        over_volume = capital > safe_max_capital

        if limit_capital_dac7:
            if over_count:
                st.error(f"⚠️ Questa configurazione genera ~{total_trades_yr:.0f} vendite/anno — "
                         f"sopra le 30 indipendentemente dal capitale (il conteggio non scala col capitale, solo il volume €).")
            if over_volume:
                st.warning(f"Capitale limitato a **{effective_capital:,.0f}€** (da {capital:,.0f}€ richiesti) per restare "
                           f"sotto {DAC7_MAX_ANNUAL_EUR:,.0f}€/anno di vendite stimate.")
            else:
                st.success(f"✅ ~{total_trades_yr:.0f} vendite/anno, ~{eur_yr_at_capital:,.0f}€/anno stimati — sotto soglia.")
            st.caption(f"Capitale massimo sicuro stimato: **{safe_max_capital:,.0f}€** totali "
                       f"({w_box*100:.0f}%/{w_singles*100:.0f}% box/singole, risk parity). "
                       "Strategia singole unificata: max 20 posizioni (optimum vincolato, Sharpe 2.39).")
        else:
            if over_volume:
                st.info(f"ℹ️ Limite volume disattivato: a {capital:,.0f}€ si stimano ~{eur_yr_at_capital:,.0f}€/anno di incassi "
                        f"(possibile segnalazione se superi 2.000€/anno senza partita IVA).")
            else:
                st.success(f"✅ ~{total_trades_yr:.0f} vendite/anno, ~{eur_yr_at_capital:,.0f}€/anno stimati — sotto soglia.")

        # --- FILTRI BOX (Fase 3: min+max simmetrico alle singole - prima
        # esisteva solo il massimo) ---
        st.markdown("---")
        st.markdown("### 📦 Filtri Box")
        min_box_price = st.number_input(
            "Prezzo minimo per box (€)", min_value=0.0, max_value=100_000.0,
            value=0.0, step=50.0,
            help="Filtra i box in acquisto sotto questa soglia (0 = nessun limite)."
        )
        max_box_price = st.number_input(
            "Prezzo massimo per box (€, 0 = nessun limite)", min_value=0.0, max_value=100_000.0,
            value=0.0, step=50.0,
            help="Filtra i box in acquisto sopra questa soglia (0 = nessun limite)."
        )

        # --- FILTRI SINGOLE ---
        st.markdown("---")
        st.markdown("### 🃏 Filtri Singole (Fattore Scarsità)")
        min_card_price = st.number_input(
            "Prezzo minimo per carta (€)", min_value=0.0, max_value=10_000.0,
            value=40.0, step=5.0,
            help="Filtra le carte troppo economiche (sotto 40-50€ l'incidenza di spedizione e la non-convenienza di gradazione distruggono l'edge operativo)."
        )
        max_card_price = st.number_input(
            "Prezzo massimo per singola carta (€, 0 = nessun limite)", min_value=0.0, max_value=100_000.0,
            value=0.0, step=50.0,
            help="Filtra le carte in acquisto sopra questa soglia - indipendentemente da quanto il modello le "
                 "ritenga sottovalutate. Utile per restare su acquisti pratici/gestibili, non è un giudizio di "
                 "convenienza: una carta esclusa qui può comunque essere un'ottima occasione, solo fuori budget."
        )
        only_holo_specials = st.checkbox(
            "Solo Holo & Rarità Speciali", value=True,
            help="Esclude carte Common, Uncommon e Non-Holo ordinarie, focalizzandosi su Rare Holo vintage, Secret, Ultra Rare, Rainbow, Illustration Rare (SIR/SAR)."
        )
        # Fase 3: stesso pattern pills del selettore franchise box (prima un
        # checkbox binario "Solo carte Pokémon") - le opzioni riflettono
        # esattamente i franchise con singole nell'universo dopo la Fase 1
        # (One Piece non ha mai avuto singole valide - rarity assente per
        # tutte le 119 carte, vedi scripts/one_piece_pilot_validation.py;
        # Pokémon JP non ha singole nell'universo; Pokémon Cinese non ancora
        # adottato). Magic resta selezionabile con lo stesso caveat del box:
        # e' stato rigettato, la vista sara' sempre vuota.
        selected_singles_franchise = st.pills(
            "Segmento / Franchise (Singole):",
            options=["Pokémon EN", "Magic (MTG)", "Tutti i Segmenti"],
            default="Pokémon EN",
            help="Filtra le singole per franchise. 'Pokémon EN' è il segmento principale validato istituzionalmente."
        )
        if selected_singles_franchise == "Magic (MTG)":
            st.warning("⚠️ Magic (MTG) è stato testato come pilota separato ed è stato **rigettato** (per le "
                       "singole l'edge apparente crolla a qualsiasi tetto di quantità realistico — vedi "
                       "scripts/mtg_pilot_validation.py). Questa vista sarà sempre vuota.")

        st.markdown("---")
        st.markdown("### 🇮🇹 Esecuzione dall'Italia")
        st.caption("1. Cardmarket — priorità assoluta (fee 5%, no dogana intra-UE)\n\n"
                   "2. eBay.it / eBay.de — box USA/JP con meno offerta su Cardmarket\n\n"
                   "3. TCGplayer — solo se il differenziale supera nettamente dogana+spedizione")
        st.markdown("---")
        st.caption("⚠️ Nessuna verifica di liquidità reale integrata — se non trovi nulla sotto il \"massimo\" "
                   "mostrato, NON è un errore, è il gate di liquidità che funziona: registralo comunque con "
                   "`log_execution_price.py` (acquisto) o `log_sell_outcome.py` (vendita), non ignorarlo.")

    capital = effective_capital

    # --- AZIONE: BUY/HOLD con allocazione e link Cardmarket (quota box, risk parity) ---
    st.markdown(f'<div class="section-title">📦 Box da comprare/mantenere — {w_box*100:.0f}% del capitale ({capital*w_box:,.0f}€)</div>', unsafe_allow_html=True)
    st.caption("Prezzo da PriceCharting (USA), NON una quota Cardmarket — verifica sempre col prezzo reale dietro "
               "il bottone. Resta sotto il \"massimo\" (oggetto + spedizione). L'€ mostrato è l'allocazione reale: "
               "i box sono ripartiti gerarchicamente in Tier 1 (Core prioritari), Tier 2 (Panchina) e Tier 3 (Vault >500€).")

    # Segmento / Franchise selector (Pills)
    selected_box_franchise = st.pills(
        "Segmento / Franchise (Box Sigillati):",
        options=["Pokémon EN", "Pokémon JP", "One Piece TCG", "Magic (MTG)", "Tutti i Segmenti"],
        default="Pokémon EN",
        help="Filtra i box per franchise e lingua. 'Pokémon EN' è il segmento principale validato istituzionalmente."
    )
    if selected_box_franchise == "Magic (MTG)":
        # Trovato in audit generale (richiesto dall'utente: "trova bug...
        # invalida"): il pilota MTG (scripts/mtg_pilot_validation.py) e' stato
        # RIGETTATO in modo decisivo - box Sharpe -0,09/DSR 0,006, singole
        # Sharpe negativo a qualunque tetto di quantita' realistico - e per
        # questo escluso di DEFAULT dalle statistiche di produzione
        # (poke_quant/data/liquidity_filter.py::DEFAULT_EXCLUDED_FRANCHISES).
        # Questa opzione resta selezionabile (scelta di prodotto di un'altra
        # sessione/dell'utente) ma non mostrera' mai righe: nessuna
        # raccomandazione MTG e' generata in produzione.
        st.warning("⚠️ Magic (MTG) è stato testato come pilota separato ed è stato **rigettato** (Sharpe box "
                   "-0,09, DSR 0,006; per le singole l'edge apparente crolla a qualsiasi tetto di quantità "
                   "realistico — vedi scripts/mtg_pilot_validation.py). Per questo motivo non viene generata "
                   "nessuna raccomandazione MTG in produzione: questa vista sarà sempre vuota.")
    elif selected_box_franchise == "Pokémon JP":
        # Stesso pattern del caveat MTG sopra: scripts/jp_pilot_validation.py
        # (2026-09-29) ha trovato DSR(51) 0,015 e i 4 UNICI trade storici tutti
        # in perdita - eliminato dalla produzione (DEFAULT_EXCLUDED_LANGUAGES
        # in poke_quant/data/liquidity_filter.py). Vista sempre vuota.
        st.warning("⚠️ Pokémon JP è stato testato in isolamento ed è stato **eliminato dalla produzione** "
                   "(DSR 0,015, i 4 unici trade storici tutti in perdita da -18% a -34% — vedi "
                   "scripts/jp_pilot_validation.py). Questa vista sarà sempre vuota.")

    with st.expander("ℹ️ Dettagli — gerarchia a 3 Tier, lotto indivisibile, capitale sequenziale"):
        st.caption("\"Massimo\" è la spesa TOTALE oltre la quale il modello considera il box fuori dal range "
                   "prezzo/MSRP validato (21,6x) — confronta oggetto + spedizione reali dell'inserzione contro questo valore.\n\n"
                   "**Architettura a 3 Tier**:\n"
                   "- **Tier 1 (Core Conviction)**: I top 5-8 box a più alto momentum (<500€) che assorbono la cassa mensile "
                   "rispettando i vincoli di portafoglio.\n"
                   "- **Tier 2 (Panchina & Alternative)**: Riserve liquide ad alto momentum. Subentrano solo se un box Core è "
                   "già in tuo possesso o non reperibile a prezzo equo.\n"
                   "- **Tier 3 (Vault & Grails >500€)**: Pezzi storici/museali ad alto capitale (>10-35% del budget). Acquistabili "
                   "solo con cassa dedicata di livello Vault.")

    all_buy_rows = [r for r in sig_rows if r["signal"] == "BUY/HOLD"]
    if min_box_price > 0:
        all_buy_rows = [r for r in all_buy_rows if r["current_price_eur"] >= min_box_price]
    if max_box_price > 0:
        all_buy_rows = [r for r in all_buy_rows if r["current_price_eur"] <= max_box_price]

    if selected_box_franchise != "Tutti i Segmenti":
        buy_rows = [r for r in all_buy_rows if get_box_franchise_label(r, metadata) == selected_box_franchise]
    else:
        buy_rows = all_buy_rows

    core_rows = [r for r in buy_rows if r.get("tier") == "core"]
    bench_rows = [r for r in buy_rows if r.get("tier") == "bench"]
    vault_rows = [r for r in buy_rows if r.get("tier") == "vault"]

    # Fallback dinamico se i tiers non sono presenti nel dizionario
    if not core_rows and not bench_rows and not vault_rows and buy_rows:
        for r in buy_rows:
            set_tier = metadata.get(r["item_id"], {}).get("set_tier")
            if r["current_price_eur"] >= 500.0 or set_tier == "Grail":
                vault_rows.append(r)
            elif len(core_rows) < 8:
                core_rows.append(r)
            else:
                bench_rows.append(r)

    box_capital_half = capital * w_box

    tab_core, tab_bench, tab_vault = st.tabs([
        f"💎 Tier 1: Core Conviction ({len(core_rows)})",
        f"🛡️ Tier 2: Panchina & Riserve ({len(bench_rows)})",
        f"🏛️ Tier 3: Vault / Grails >500€ ({len(vault_rows)})",
    ])

    with tab_core:
        if not core_rows:
            st.info("Nessun box Core per questo segmento con i filtri attuali.")
        else:
            allocation = build_allocation(core_rows, box_capital_half, metadata, latest_date)
            prefetch_product_images([
                (metadata.get(r["item_id"], {}).get("game_slug"), metadata.get(r["item_id"], {}).get("item_slug"))
                for r, _, _ in allocation
            ])

            _total_real_spend = 0.0
            executable_alloc = []
            skipped_alloc = []
            for r, alloc, w in allocation:
                p = r["current_price_eur"]
                if p <= 0 or alloc >= p:
                    _total_real_spend += alloc
                    executable_alloc.append((r, alloc, w))
                elif p <= box_capital_half * 0.35:
                    _total_real_spend += p
                    executable_alloc.append((r, alloc, w))
                else:
                    skipped_alloc.append((r, alloc, w))

            if not executable_alloc:
                min_core_p = min(r["current_price_eur"] for r in core_rows)
                min_req_budget = min_core_p / 0.35
                st.warning(
                    f"⚠️ **Capitale insufficiente per acquistare box in questo segmento** (Budget box attuale: **{box_capital_half:,.0f}€**).\n\n"
                    f"Per proteggere il portafoglio dal rischio rovina, la regola quantitativa validata vieta di impiegare oltre il **35% del budget** "
                    f"su un singolo pezzo (tetto massimo per singolo box oggi: **{box_capital_half*0.35:,.0f}€**).\n\n"
                    f"Tutti i box Core di questo segmento superano questa soglia (il più economico costa **{min_core_p:,.0f}€**).\n\n"
                    f"💡 **Azioni consigliate**:\n"
                    f"- **Dirotta la liquidità sulle Singole Gradate**: Con {box_capital_half:,.0f}€ puoi comprare 3–5 slab PSA 9 a sconto (40–90€/pezzo) perfettamente diversificati.\n"
                    f"- **Esplora Pokémon JP**: Nel selettore in alto scegli *Pokémon JP* (troverai box Core a 50–80€ compatibili col budget).\n"
                    f"- **Aumenta il budget box**: Per acquistare il primo box rispettando il tetto del 35% serve un budget box di almeno **~{min_req_budget:,.0f}€** (capitale totale consigliato ~{min_req_budget/w_box:,.0f}€)."
                )
                with st.expander(f"🔍 Mostra comunque i {len(skipped_alloc)} box Core (richiedono budget > {box_capital_half:,.0f}€)"):
                    st.caption("Questi box hanno momentum positivo ma a questo livello di capitale rappresenterebbero più del 35% del portafoglio box.")
                    for r, alloc, w in skipped_alloc:
                        render_box_card(r, alloc, w, box_capital_half, metadata, prices_full, show_usa_import, tier_type="core")
            else:
                st.markdown(f"""
                <div class="kpi-grid">
                    <div class="kpi-card"><div class="kpi-label">Budget Box ({w_box*100:.0f}%)</div><div class="kpi-value">{box_capital_half:,.0f} €</div><div class="kpi-sub">Capitale risk-parity</div></div>
                    <div class="kpi-card"><div class="kpi-label">Spesa Reale Stimata</div><div class="kpi-value" style="color:#10b981;">{_total_real_spend:,.0f} €</div><div class="kpi-sub kpi-sub-emerald">{_total_real_spend/box_capital_half*100:.1f}% del budget box</div></div>
                    <div class="kpi-card"><div class="kpi-label">Box Core Acquistabili</div><div class="kpi-value">{len(executable_alloc)} / {len(core_rows)}</div><div class="kpi-sub">Rispettano tetto 35%</div></div>
                    <div class="kpi-card"><div class="kpi-label">Liquidità Residua</div><div class="kpi-value">{max(0.0, box_capital_half - _total_real_spend):,.0f} €</div><div class="kpi-sub">Cassa pronta o per singole</div></div>
                </div>
                """, unsafe_allow_html=True)

                if _total_real_spend > box_capital_half * 1.10:
                    st.warning(
                        f"⚠️ A questo capitale, comprare per intero i box Core selezionati costerebbe ~**{_total_real_spend:,.0f}€**, "
                        f"contro i {box_capital_half:,.0f}€ dedicati (+{(_total_real_spend/box_capital_half-1)*100:.0f}%). "
                        "I box sono lotti indivisibili: il modello reale spende la cassa in sequenza (priorità ai primi in lista)."
                    )
                else:
                    st.caption("✅ **Allocazione equilibrata**: I box in elenco rientrano nel budget dedicato, "
                               "rispettando la regola del lotto indivisibile e la diversificazione.")

                for r, alloc, w in executable_alloc:
                    render_box_card(r, alloc, w, box_capital_half, metadata, prices_full, show_usa_import, tier_type="core")

                if skipped_alloc:
                    with st.expander(f"⚠️ {len(skipped_alloc)} box Core esclusi per tetto di concentrazione (>35% del budget)"):
                        st.caption("Questi box hanno momentum positivo ma a questo livello di capitale rappresenterebbero più del 35% del portafoglio box, violando il limite di concentrazione testato.")
                        for r, alloc, w in skipped_alloc:
                            render_box_card(r, alloc, w, box_capital_half, metadata, prices_full, show_usa_import, tier_type="core")

    with tab_bench:
        st.info("🛡️ **Panchina & Riserve Liquide**: Se possiedi già uno dei box Core o non riesci a trovarlo su Cardmarket "
                "sotto il prezzo massimo per preservare l'edge, acquista in sequenza da questa lista. Condividono tutti "
                "momentum trailing 12m positivo e sono pronti a subentrare senza compromettere la validazione.")
        if not bench_rows:
            st.caption("Nessun box in panchina per questo segmento.")
        else:
            prefetch_product_images([
                (metadata.get(r["item_id"], {}).get("game_slug"), metadata.get(r["item_id"], {}).get("item_slug"))
                for r in bench_rows
            ])
            for r in bench_rows:
                render_box_card(r, None, None, box_capital_half, metadata, prices_full, show_usa_import, tier_type="bench")

    with tab_vault:
        st.info("🏛️ **Tier 3 Vault & Grails**: Box storici e set rari con prezzo unitario > 500€ o catalogati come 'Grail'. "
                "Hanno momentum trailing 12m positivo ma richiederebbero una concentrazione sproporzionata del budget mensile "
                "(>10-35%). Il modello validato li esegue solo se si dispone di liquidità dedicata di livello 'Vault', altrimenti "
                "li salta per proteggere la diversificazione del portafoglio.")
        if not vault_rows:
            st.caption("Nessun box nel Vault per questo segmento.")
        else:
            prefetch_product_images([
                (metadata.get(r["item_id"], {}).get("game_slug"), metadata.get(r["item_id"], {}).get("item_slug"))
                for r in vault_rows
            ])
            for r in vault_rows:
                render_box_card(r, None, None, box_capital_half, metadata, prices_full, show_usa_import, tier_type="vault")

    # --- ROTAZIONE: AVOID/SELL ---
    all_sell_rows = [r for r in sig_rows if r["signal"] == "AVOID/SELL"]
    if selected_box_franchise != "Tutti i Segmenti":
        sell_rows = [r for r in all_sell_rows if get_box_franchise_label(r, metadata) == selected_box_franchise]
    else:
        sell_rows = all_sell_rows

    if sell_rows:
        st.markdown('<div class="section-title">🔴 Uscite (momentum invertito)</div>', unsafe_allow_html=True)
        prefetch_product_images([
            (metadata.get(r["item_id"], {}).get("game_slug"), metadata.get(r["item_id"], {}).get("item_slug"))
            for r in sell_rows
        ])
        for r in sell_rows:
            meta_sell = metadata.get(r["item_id"], {})
            img_url = get_product_image(meta_sell.get("game_slug"), meta_sell.get("item_slug"))
            img_tag = render_thumb_html(img_url)
            st.markdown(f"""
            <div class="signal-card signal-card-sell">
                {img_tag}
                <div class="signal-card-body">
                <strong>{r['name']}</strong> &nbsp; <span style="color:#f43f5e;">{r['trailing_12m_return_pct']:.0f}% (12m)</span>
                &nbsp;·&nbsp; {r['current_price_eur']:.0f}€ (PriceCharting)
                </div>
            </div>
            """, unsafe_allow_html=True)
            chart = build_price_chart(r["item_id"], r["name"], prices_full)
            if chart is not None:
                st.plotly_chart(chart, use_container_width=True,
                                 config={"displayModeBar": False, "scrollZoom": False, "staticPlot": False, "doubleClick": False},
                                 key=f"chart_sell_{r['item_id']}")

    all_excessive_rows = [r for r in sig_rows if "PREZZO ECCESSIVO" in r["signal"]]
    if selected_box_franchise != "Tutti i Segmenti":
        excessive_rows = [r for r in all_excessive_rows if get_box_franchise_label(r, metadata) == selected_box_franchise]
    else:
        excessive_rows = all_excessive_rows

    if excessive_rows:
        with st.expander(f"🚫 Prezzo eccessivo — momentum positivo ma bloccato ({len(excessive_rows)})"):
            st.caption("Il modello direbbe di comprare (momentum 12m positivo), ma il prezzo attuale supera già il "
                       "tetto che preserva l'edge: stesso rapporto prezzo/MSRP già usato per ammettere un box vintage "
                       "nell'universo (21,6x, vedi `poke_quant/data/liquidity_filter.py`), qui applicato anche a un "
                       "nuovo acquisto. Non è impossibile che salga ancora, ma comprare oltre questo confine non è "
                       "ciò che è stato validato — impedisce di comprare a un prezzo che romperebbe l'edge misurato.")
            for r in excessive_rows:
                st.markdown(f"- **{r['name']}** — {r['current_price_eur']:.0f}€ attuale vs **{r['max_price_eur']:.0f}€ massimo "
                            f"(totale)** (+{r['trailing_12m_return_pct']:.0f}% 12m)")

    all_verify_rows = [r for r in sig_rows if "VERIFICARE" in r["signal"]]
    if selected_box_franchise != "Tutti i Segmenti":
        verify_rows = [r for r in all_verify_rows if get_box_franchise_label(r, metadata) == selected_box_franchise]
    else:
        verify_rows = all_verify_rows

    if verify_rows:
        with st.expander(f"⚠️ Da verificare a mano ({len(verify_rows)}) — rendimento implausibile, mercato troppo sottile"):
            for r in verify_rows:
                st.markdown(f"- **{r['name']}** — {r['trailing_12m_return_pct']:+.0f}% (12m), {r['current_price_eur']:.0f}€ (PriceCharting)")
                chart = build_price_chart(r["item_id"], r["name"], prices_full)
                if chart is not None:
                    st.plotly_chart(chart, use_container_width=True,
                                     config={"displayModeBar": False, "scrollZoom": False, "staticPlot": False, "doubleClick": False},
                                     key=f"chart_verify_{r['item_id']}")

    # --- AZIONE: SINGOLE — FATTORE SCARSITÀ (50% del capitale) ---
    st.markdown(f'<div class="section-title">🃏 Singole da comprare — Fattore Scarsità, {w_singles*100:.0f}% del capitale ({capital*w_singles:,.0f}€)</div>', unsafe_allow_html=True)
    st.caption("Verifica sempre set e lingua esatti su Cardmarket (**[Set]** nel badge). Resta sotto il prezzo **massimo per edge**. "
               "Preferisci slab **PSA** (o BGS/CGC). Per alternative minori, controlla il Pop Pressure badge.")
    with st.expander("ℹ️ Strategia, edge, compagnie di gradazione e limiti di modello"):
        st.caption("**Strategia**: optimum validato a 20 posizioni (Sharpe 2.39, CAGR +67.7%, MaxDD -9.7%), "
                   "target di gradazione calibrato per era (PSA 10 su Moderno, PSA 9 su Vintage/Mid-Era).")
        st.caption("• **Massimo per Edge**: spesa totale massima (inclusa spedizione) oltre la quale l'edge netto statistico viene eroso dalle fee.\n\n"
                   "• **Compagnie di Gradazione**: consigliate PSA, poi BGS, CGC e SGC (alta liquidità internazionale). Evita enti regionali senza riconoscimento globale (GRAAD, TAG, ACE) salvo forti sconti.\n\n"
                   "• **Pressione Popolazione (Pop Pressure)**: calcolata sul percentile di diluizione dell'era. Se segnalato sovraffollamento (P > 90), punta esclusivamente al Target primario.")
    singles_rows, singles_latest_date = get_singles_signal(singles_mode)
    alt_rows, _ = get_singles_alternatives(singles_mode)
    singles_prices_full = get_singles_prices_full()
    # Selettore pills -> filtro franchise esatto (None = "Tutti i Segmenti", nessun filtro).
    singles_franchise_filter = {"Pokémon EN": "pokemon", "Magic (MTG)": "magic"}.get(selected_singles_franchise)
    # Applica i filtri qualitativi e di prezzo alle singole (rimuove rumore a basso prezzo / bulk non-holo / altri TCG)
    singles_rows = filter_singles_rows(
        singles_rows,
        min_price=min_card_price,
        max_price=max_card_price,
        only_holo=only_holo_specials,
        franchise=singles_franchise_filter,
        retag_tiers=True,
    )
    alt_rows = filter_singles_rows(
        alt_rows,
        min_price=min_card_price,
        max_price=max_card_price,
        only_holo=only_holo_specials,
        franchise=singles_franchise_filter,
        retag_tiers=False,
    )
    calc_alt_rows = alt_rows[:30]
    singles_allocation = build_equal_allocation(singles_rows, capital * w_singles)


    if not singles_allocation:
        st.info("Nessuna carta nel quantile BUY questo mese.")
    else:
        core_singles = [item for item in singles_allocation if item[0].get("tier") == "core"]
        if not core_singles and singles_allocation:
            core_singles = singles_allocation[:8]
            bench_singles = singles_allocation[8:]
        else:
            bench_singles = [item for item in singles_allocation if item[0].get("tier") != "core"]

        tab_s_core, tab_s_bench, tab_s_alt = st.tabs([
            f"💎 Tier 1: Core Conviction ({len(core_singles)})",
            f"🛡️ Tier 2: Panchina & Riserve ({len(bench_singles)})",
            f"🔄 Alternative Stesso Quantile ({len(alt_rows)})",
        ])

        singles_budget_half = capital * w_singles
        alloc_core_focus = singles_budget_half / max(1, len(core_singles))
        alloc_full_dist = singles_budget_half / max(1, len(singles_allocation))

        with tab_s_core:
            st.markdown(f"""
            <div class="kpi-grid">
                <div class="kpi-card"><div class="kpi-label">Budget Singole ({w_singles*100:.0f}%)</div><div class="kpi-value">{singles_budget_half:,.0f} €</div><div class="kpi-sub">Capitale risk-parity</div></div>
                <div class="kpi-card"><div class="kpi-label">Quota Focus (Top 8)</div><div class="kpi-value" style="color:#10b981;">{alloc_core_focus:,.0f} € / carta</div><div class="kpi-sub kpi-sub-emerald">Concentrazione raccomandata</div></div>
                <div class="kpi-card"><div class="kpi-label">Quota Standard ({len(singles_allocation)} pos.)</div><div class="kpi-value">{alloc_full_dist:,.0f} € / carta</div><div class="kpi-sub">Distribuzione uniforme</div></div>
                <div class="kpi-card"><div class="kpi-label">Carte Core</div><div class="kpi-value">{len(core_singles)}</div><div class="kpi-sub">Top decile residui</div></div>
            </div>
            """, unsafe_allow_html=True)

            prefetch_product_images([
                (metadata.get(r["item_id"], {}).get("game_slug"), metadata.get(r["item_id"], {}).get("item_slug"))
                for r, _ in core_singles
            ])
            for r, alloc in core_singles:
                render_single_card(r, alloc, metadata, singles_prices_full, show_usa_import, key_prefix="single_core")

        with tab_s_bench:
            st.info("🛡️ **Panchina & Posizioni Secondarie**: Carte validate dal modello di scarsità (dalla 9ª in poi). "
                    "Usale se una carta della Top 8 Core non è reperibile su Cardmarket al di sotto del 'Massimo per Edge'.")
            if not bench_singles:
                st.caption("Nessuna carta in panchina (tutte le posizioni rientrano nella Top 8 Core).")
            else:
                prefetch_product_images([
                    (metadata.get(r["item_id"], {}).get("game_slug"), metadata.get(r["item_id"], {}).get("item_slug"))
                    for r, _ in bench_singles[:8]
                ])
                for r, alloc in bench_singles[:8]:
                    render_single_card(r, alloc, metadata, singles_prices_full, show_usa_import, key_prefix="single_bench")

                if len(bench_singles) > 8:
                    with st.expander(f"Altre {len(bench_singles) - 8} carte in panchina"):
                        bench_table_rows = []
                        for r, alloc in bench_singles[8:]:
                            full_meta_b = metadata.get(r["item_id"], {})
                            rec_t = get_recommended_grade_targets(
                                base_psa9_eur=float(r["current_price_eur"]),
                                era=r.get("era") or full_meta_b.get("era", "modern"),
                                item_id=r.get("item_id"),
                                game_slug=full_meta_b.get("game_slug"),
                                item_slug=full_meta_b.get("item_slug"),
                            )
                            is_p10 = (rec_t["target_grade"] == "PSA 10")
                            p_target = float(r.get("target_price_eur") or rec_t["target_price_eur"])
                            era_ratio = ERA_PSA10_TO_PSA9_RATIO.get(normalize_era(rec_t["era"]), 2.80) if is_p10 else 1.0
                            max_edge_target = float(r.get("target_max_edge_price_eur") or (round(r["max_edge_price_eur"] * era_ratio, 2) if r.get("max_edge_price_eur") else round(p_target * 1.05, 2)))
                            target_grade_lbl = rec_t["target_grade"]
                            qty_est = max(1, int(alloc // p_target)) if p_target > 0 and alloc > 0 else 1
                            bench_table_rows.append({
                                "Carta": r["name"],
                                "Set": r.get("set_name") or "?",
                                "Rarità": r["rarity"],
                                "Grado Target": target_grade_lbl,
                                "Alternative Minori Consigliate": rec_t["minor_alternatives_str"],
                                "Prezzo Target (€)": p_target,
                                "Massimo per Edge (€)": max_edge_target,
                                "Base G9 (€)": r["current_price_eur"],
                                "Sconto vs. pari (%)": r["discount_pct"],
                                "Segnale da": r["signal_start_date"].strftime("%Y-%m") if hasattr(r["signal_start_date"], "strftime") else str(r["signal_start_date"]),
                                "Allocazione (€)": alloc,
                                "Quantità Target": qty_est,
                            })
                        rest_df = pd.DataFrame(bench_table_rows)
                        st.dataframe(rest_df, use_container_width=True, hide_index=True,
                                     column_config={
                                         "Prezzo Target (€)": st.column_config.NumberColumn(format="%.2f €"),
                                         "Massimo per Edge (€)": st.column_config.NumberColumn(format="%.2f €"),
                                         "Base G9 (€)": st.column_config.NumberColumn(format="%.2f €"),
                                         "Sconto vs. pari (%)": st.column_config.NumberColumn(format="%+.1f%%"),
                                         "Allocazione (€)": st.column_config.NumberColumn(format="%.0f €"),
                                     })

        with tab_s_alt:
            st.caption("Ripiego, non un secondo BUY: usa il budget non speso qui invece di lasciarlo fermo o "
                       "forzare più copie di una carta — recupera parte dell'edge perso ma non tutto. "
                       "Ancora nel quantile 20% più sottovalutato, solo fuori dalle prime posizioni per rank.")
            if not alt_rows:
                st.caption("Nessuna alternativa disponibile con i filtri attuali.")
            else:
                alt_table_rows = []
                for r in alt_rows[:60]:
                    full_meta_a = metadata.get(r["item_id"], {})
                    rec_t = get_recommended_grade_targets(
                        base_psa9_eur=float(r["current_price_eur"]),
                        era=r.get("era") or full_meta_a.get("era", "modern"),
                        item_id=r.get("item_id"),
                        game_slug=full_meta_a.get("game_slug"),
                        item_slug=full_meta_a.get("item_slug"),
                    )
                    is_p10 = (rec_t["target_grade"] == "PSA 10")
                    p_target = float(r.get("target_price_eur") or rec_t["target_price_eur"])
                    era_ratio = ERA_PSA10_TO_PSA9_RATIO.get(normalize_era(rec_t["era"]), 2.80) if is_p10 else 1.0
                    max_edge_target = float(r.get("target_max_edge_price_eur") or (round(r["max_edge_price_eur"] * era_ratio, 2) if r.get("max_edge_price_eur") else round(p_target * 1.05, 2)))
                    target_grade_lbl = rec_t["target_grade"]
                    alt_table_rows.append({
                        "Carta": r["name"],
                        "Set": r.get("set_name") or "?",
                        "Rarità": r["rarity"],
                        "Grado Target": target_grade_lbl,
                        "Alternative Minori Consigliate": rec_t["minor_alternatives_str"],
                        "Prezzo Target (€)": p_target,
                        "Massimo per Edge (€)": max_edge_target,
                        "Base G9 (€)": r["current_price_eur"],
                        "Sconto vs. pari (%)": r["discount_pct"],
                    })
                alt_df = pd.DataFrame(alt_table_rows)
                st.dataframe(alt_df, use_container_width=True, hide_index=True,
                             column_config={
                                 "Prezzo Target (€)": st.column_config.NumberColumn(format="%.2f €"),
                                 "Massimo per Edge (€)": st.column_config.NumberColumn(format="%.2f €"),
                                 "Base G9 (€)": st.column_config.NumberColumn(format="%.2f €"),
                                 "Sconto vs. pari (%)": st.column_config.NumberColumn(format="%+.1f%%"),
                             })

                with st.expander("🔍 Mostra le Prime 10 Alternative in formato scheda"):
                    st.caption("Visualizza le prime 10 carte alternative come schede singole con prezzi target, pop pressure, grafici e link Cardmarket.")
                    prefetch_product_images([
                        (metadata.get(r["item_id"], {}).get("game_slug"), metadata.get(r["item_id"], {}).get("item_slug"))
                        for r in alt_rows[:10]
                    ])
                    for r in alt_rows[:10]:
                        render_single_card(r, 0.0, metadata, singles_prices_full, show_usa_import, key_prefix="single_alt")

    # --- USCITE/AVOID: SINGOLE SOPRAVVALUTATE (specchio del BUY) ---
    avoid_rows, _ = get_singles_avoid_signal(singles_mode)
    if singles_franchise_filter is not None:
        avoid_rows = [r for r in avoid_rows if r.get("franchise") == singles_franchise_filter]
    if avoid_rows:
        st.markdown('<div class="section-title">🔴 Singole da evitare/vendere — sopravvalutate vs pari</div>', unsafe_allow_html=True)
        st.caption("⚠️ Specchio del quantile BUY (stesso modello, residuo più positivo): la carta costa più di "
                   "quanto la sua rarità/età/set implicherebbero rispetto alle pari. Molte di queste sono chase "
                   "iconiche (Charizard, Lugia, carte ★) — il modello non cattura il premio da fama/desiderabilità, "
                   "solo rarità/età/franchise, quindi un sovrapprezzo enorme spesso riflette un premio reale, non "
                   "un errore di prezzo. A differenza del quantile BUY, qui NON è stato validato un backtest di "
                   "vendita/short — è informativo (come le Uscite dei box), non una strategia a sé testata.")
        prefetch_product_images([
            (metadata.get(r["item_id"], {}).get("game_slug"), metadata.get(r["item_id"], {}).get("item_slug"))
            for r in avoid_rows[:15]
        ])
        for r in avoid_rows[:15]:
            full_meta = metadata.get(r["item_id"], {})
            img_url = get_product_image(full_meta.get("game_slug"), full_meta.get("item_slug"))
            img_tag = render_thumb_html(img_url)
            st.markdown(f"""
            <div class="signal-card signal-card-sell">
                {img_tag}
                <div class="signal-card-body">
                <strong>{r['name']}</strong> &nbsp; <span style="color:#38bdf8; font-weight:600;">[{r.get('set_name') or '?'}]</span>
                &nbsp; <span style="color:#94a3b8;">{r['rarity']}</span>
                &nbsp;·&nbsp; {r['current_price_eur']:.2f}€ <span style="color:#fbbf24;">[Grade 9]</span> (PriceCharting)
                &nbsp;·&nbsp; sovrapprezzo vs. pari {r['discount_pct']:+.0f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        if len(avoid_rows) > 15:
            with st.expander(f"Altre {len(avoid_rows) - 15} carte sopravvalutate"):
                avoid_df = pd.DataFrame([
                    {"Carta": r["name"], "Set": r.get("set_name") or "?", "Rarità": r["rarity"], "Grado": "Grade 9",
                     "Prezzo (€)": r["current_price_eur"], "Sovrapprezzo vs. pari (%)": r["discount_pct"]}
                    for r in avoid_rows[15:]
                ])
                st.dataframe(avoid_df, use_container_width=True, hide_index=True,
                             column_config={
                                 "Prezzo (€)": st.column_config.NumberColumn(format="%.2f €"),
                                 "Sovrapprezzo vs. pari (%)": st.column_config.NumberColumn(format="%+.1f%%"),
                             })

    st.markdown('<div class="section-title">⚖️ Valutatore Slab</div>', unsafe_allow_html=True)
    # --- CALCOLATORE RAPIDO SLAB & CORREZIONI CASE DI GRADAZIONE ---
    with st.expander("⚖️ Calcolatore Inserzioni Slab & Moltiplicatori Case di Gradazione (BGS, CGC, PSA, SGC, TAG, PCA, GRAAD, CCC, AiGrading, ACE)", expanded=True):
        st.markdown("**Valutatore Rapido Inserzioni**: Seleziona una carta dai segnali BUY, cercala nell'intero database (3.100+ carte) o inseriscine una personalizzata. Indica la casa di gradazione, il voto e l'eventuale variante speciale (1st Edition, No Symbol, Shadowless). Il modello recupera il benchmark reale ed applica i correttivi quantitativi per preservare l'edge.")
        
        calc_mode = st.radio(
            "Origine della carta da valutare:",
            options=[
                f"⭐ Segnali Modello ({len(singles_rows)} BUY + {len(calc_alt_rows)} Alternative)",
                "🔍 Cerca tra tutte le 3.100+ carte del Database PokeQuant",
                "✏️ Carta Personalizzata / Inserimento Libero (o Link PriceCharting)",
            ],
            horizontal=True,
            index=0,
            help="Scegli se valutare una carta tra i segnali attuali, cercare una carta qualsiasi del database completo (3.100+ carte con storico e metadati automatici), oppure inserire manualmente nome, benchmark o link PriceCharting."
        )

        card_options = []
        option_to_row = {}

        if calc_mode.startswith("⭐"):
            for i, r in enumerate(singles_rows):
                lbl = f"🟢 [BUY #{i+1}] {r['name']} [{r.get('set_name') or '?'}] — {r['current_price_eur']:.2f}€"
                card_options.append(lbl)
                option_to_row[lbl] = r

            for r in calc_alt_rows:
                lbl = f"🔄 [ALT] {r['name']} [{r.get('set_name') or '?'}] — {r['current_price_eur']:.2f}€"
                card_options.append(lbl)
                option_to_row[lbl] = r
        elif calc_mode.startswith("🔍"):
            all_opts, all_map = get_all_database_card_options()
            card_options = all_opts
            option_to_row = all_map

        is_custom_mode = calc_mode.startswith("✏️")

        with st.form("slab_calculator_form"):
            calc_c1, calc_c2, calc_c3 = st.columns([2, 1, 1])
            with calc_c1:
                if is_custom_mode:
                    custom_card_name = st.text_input(
                        "Nome Carta Personalizzata",
                        value="",
                        placeholder="Es. Charizard Holo Base Set, Umbreon VMAX, Lugia V...",
                        help="Scrivi il nome della carta. Se è presente nel database PokeQuant o inserisci il link PriceCharting, i dati vengono completati in automatico."
                    )
                    chosen_option = "✏️ Personalizzata"
                else:
                    chosen_option = st.selectbox(
                        f"Carta da valutare ({len(card_options)} disponibili)",
                        options=card_options,
                        index=0,
                        help="Seleziona o digita il nome per cercare istantaneamente tra le carte disponibili."
                    )
                    custom_card_name = ""
            with calc_c2:
                company_input = st.selectbox(
                    "Casa di Gradazione", 
                    options=["CGC", "BGS", "PSA", "SGC", "TAG", "PCA", "GRAAD", "CCC", "AiGrading", "ACE"], 
                    index=0
                )
            with calc_c3:
                grade_input = st.selectbox(
                    "Voto Slab", 
                    options=[
                        "10.0 Gem Mint", 
                        "10.0 Pristine", 
                        "10.0 Black Label (BGS Quad 10)", 
                        "9.5 Gem Mint", 
                        "9.0 Mint",
                        "8.5 NM-Mint+",
                        "8.0 NM-Mint",
                        "7.5 Near Mint+",
                        "7.0 Near Mint",
                    ], 
                    index=4,
                    help="Valuta qualsiasi grado dal 10.0 fino al 7.0 con mezzi voti. Privilegia i dati storici reali di PriceCharting per il grado esatto se disponibili."
                )

            # Riga opzionale per Link / Slug PriceCharting (visibile in modalità personalizzata)
            if is_custom_mode:
                custom_pc_url = st.text_input(
                    "🔗 Link o Slug PriceCharting (Opzionale: scarica prezzi live di ogni voto e variante)",
                    value="",
                    placeholder="Es. https://www.pricecharting.com/game/pokemon-base-set/charizard-4 oppure pokemon-base-set/charizard-4",
                    help="Incolla l'URL o lo slug della pagina PriceCharting per recuperare all'istante il prezzo reale per il grado e la variante selezionati."
                )
            else:
                custom_pc_url = ""

            # Seconda riga: Variante Speciale, Benchmark PSA manuale e Era
            r2_c1, r2_c2, r2_c3 = st.columns([1.5, 1.2, 1.3])
            with r2_c1:
                variant_input = st.selectbox(
                    "Edizione / Variante Speciale",
                    options=[
                        "Standard (Unlimited / Regolare)",
                        "1st Edition (Rileva reale da PriceCharting o ~2.5x WotC)",
                        "1st Edition Base Set (Rileva reale da PriceCharting o ~6.0x)",
                        "No Symbol Error (Rileva reale da PriceCharting o ~1.4x)",
                        "Shadowless Base Set (Rileva reale da PriceCharting o ~3.0x)",
                        "Reverse Holo Legendary Collection (~3.0x vs Regular)",
                    ],
                    index=0,
                    help="I prezzi del database PokeQuant sono su edizioni Unlimited. Se la slab in vendita è una 1st Edition o un errore noto (es. No Symbol Jungle), il tool interroga in automatico PriceCharting per recuperare l'esatto benchmark di mercato della variante (se disponibile) o scala con moltiplicatore set-aware."
                )
            with r2_c2:
                benchmark_label = "Benchmark PSA (€) [Opzionale se c'è Link o DB]" if is_custom_mode else "Benchmark PSA (€) [0 = auto da DB]"
                benchmark_help = "Prezzo indicativo di mercato della carta in PSA 9 (o PSA 10). Se inserisci il link PriceCharting o la carta è nel database, puoi lasciare 0.0 per scaricarlo in automatico." if is_custom_mode else "Lascia 0.0 per usare il prezzo di mercato della carta selezionata sopra. Inserisci un valore > 0 per forzare un benchmark personalizzato."
                manual_psa_override = st.number_input(
                    benchmark_label, 
                    min_value=0.0, 
                    value=0.0, 
                    step=5.0,
                    help=benchmark_help,
                )
            with r2_c3:
                era_input = st.selectbox(
                    "Era Collezionistica",
                    options=["Auto (rileva dalla carta/variante)", "vintage", "mid_era", "modern"],
                    index=0,
                    format_func=lambda x: "Auto (rileva dalla carta)" if x.startswith("Auto") else ("Vintage (1999–2003)" if x == "vintage" else ("Mid-Era (2004–2016)" if x == "mid_era" else "Moderno (2017+)"))
                )

            # Terza riga: Prezzo offerta, spedizione e toggle USA
            inp_c1, inp_c2, inp_c3 = st.columns([1.5, 1.2, 1.5])
            with inp_c1:
                offer_price_eur = st.number_input("Prezzo Annuncio / Offerta (€)", min_value=1.0, value=75.0, step=5.0)
            with inp_c2:
                shipping_eur = st.number_input("Spese Sped. (€)", min_value=0.0, value=6.0, step=1.0)
            with inp_c3:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                is_usa_import = st.checkbox(
                    "🌍 Inserzione USA / Extra-UE", 
                    value=False, 
                    help="Applica IVA 22% su oggetto+spedizione, dazio forfettario 3€ e oneri corriere 15€ per determinare il costo reale sdoganato in Italia."
                )

            submit_calc = st.form_submit_button("🔍 Calcola Valutazione Slab", use_container_width=True)

        # Gestione del calcolo al submit (e salvataggio in session_state)
        if submit_calc:
            is_black_label = "black label" in grade_input.lower()
            is_pristine = "pristine" in grade_input.lower()
            g_in = grade_input.lower()
            if "10" in g_in:
                grade_val = "10.0"
                grade_tier = "psa10"
                tier_label = "PSA 10"
            elif "9.5" in g_in:
                grade_val = "9.5"
                grade_tier = "grade9_5"
                tier_label = "Grado 9.5"
            elif "8.5" in g_in:
                grade_val = "8.5"
                grade_tier = "grade8_5"
                tier_label = "Grado 8.5"
            elif "8.0" in g_in or "8" in g_in:
                grade_val = "8.0"
                grade_tier = "grade8"
                tier_label = "Grado 8.0"
            elif "7.5" in g_in:
                grade_val = "7.5"
                grade_tier = "grade7_5"
                tier_label = "Grado 7.5"
            elif "7.0" in g_in or "7" in g_in:
                grade_val = "7.0"
                grade_tier = "grade7"
                tier_label = "Grado 7.0"
            else:
                grade_val = "9.0"
                grade_tier = "grade9"
                tier_label = "Grado 9.0"

            # Risoluzione metadati, slug, era e prezzo base
            sel_meta = {}
            sel_row = None
            matched_db_info = None
            is_custom = is_custom_mode or (chosen_option == "✏️ Personalizzata")

            # Se l'utente ha inserito un URL o slug PriceCharting
            pc_slugs = parse_pricecharting_url_or_slug(custom_pc_url) if is_custom and custom_pc_url else None

            if is_custom:
                card_name = custom_card_name.strip() if custom_card_name.strip() else ""
                if pc_slugs:
                    g_slug, i_slug = pc_slugs
                    sel_meta["game_slug"] = g_slug
                    sel_meta["item_slug"] = i_slug
                    if not card_name:
                        card_name = f"{i_slug.replace('-', ' ').title()} [{g_slug.replace('pokemon-', '').replace('-', ' ').title()}]"
                    slug_low = f"{g_slug} {i_slug}".lower()
                    if any(v in slug_low for v in ["base-set", "fossil", "jungle", "rocket", "neo-", "gym-", "legendary-collection", "expedition", "aquapolis", "skyridge"]):
                        era_detected = "vintage"
                    elif any(m in slug_low for m in ["ex-", "diamond", "pearl", "platinum", "heartgold", "black-white", "xy-"]):
                        era_detected = "mid_era"
                    else:
                        era_detected = "modern"
                else:
                    # Tenta riconoscimento automatico nel database delle 3.600+ carte
                    matched = search_metadata_card_by_query(card_name, metadata) if card_name else None
                    if matched:
                        matched_id, matched_info = matched
                        matched_db_info = matched_info
                        sel_meta = matched_info.copy()
                        sel_meta["item_id"] = matched_id
                        g_slug = sel_meta.get("game_slug", "")
                        i_slug = sel_meta.get("item_slug", "")
                        if not card_name or len(card_name) < 4:
                            card_name = matched_info.get("name", card_name)
                        rel_year = int(str(sel_meta.get("release_date", "2020"))[:4]) if sel_meta.get("release_date") else 2020
                        if rel_year <= 2003:
                            era_detected = "vintage"
                        elif rel_year <= 2016:
                            era_detected = "mid_era"
                        else:
                            era_detected = "modern"
                    else:
                        g_slug = ""
                        i_slug = ""
                        era_detected = "modern"
                        if not card_name:
                            card_name = "Carta Personalizzata"
                target_item_id = matched_db_info.get("item_id") if matched_db_info else None
                if not target_item_id and i_slug:
                    target_item_id = i_slug.replace("-", "_")
            elif chosen_option in option_to_row:
                sel_row = option_to_row[chosen_option]
                target_item_id = sel_row.get("item_id")
                card_name = sel_row["name"]
                sel_meta = metadata.get(sel_row["item_id"], {})
                g_slug = sel_meta.get("game_slug", "")
                i_slug = sel_meta.get("item_slug", "")
                rel_year = int(str(sel_meta.get("release_date", "2020"))[:4]) if sel_meta.get("release_date") else 2020
                if rel_year <= 2003:
                    era_detected = "vintage"
                elif rel_year <= 2016:
                    era_detected = "mid_era"
                else:
                    era_detected = "modern"
            else:
                card_name = "Carta Non Selezionata"
                era_detected = "vintage"
                g_slug = ""
                i_slug = ""
                target_item_id = None

            # Determinazione del prezzo base e benchmark
            base_psa_raw = 0.0
            base_psa_final = 0.0
            base_max_edge = 0.0
            effective_max_edge = 0.0
            benchmark_source = ""
            is_pc_grade_resolved = False
            is_variant_grade9_only = False
            pc_live_info = None

            # Priorità 1: Se l'utente ha inserito un override manuale > 0.0
            if manual_psa_override > 0.0:
                base_psa_raw = manual_psa_override
                base_psa_final = manual_psa_override
                base_max_edge = base_psa_raw * 1.05
                effective_max_edge = base_max_edge
                benchmark_source = f"Benchmark Manuale ({manual_psa_override:.2f} €)"
            # Priorità 2: Se abbiamo game_slug e item_slug (da link PriceCharting o da metadati/DB)
            elif g_slug and i_slug:
                is_special_variant = not variant_input.startswith("Standard")
                if is_special_variant:
                    v_key = variant_to_pricecharting_key(variant_input)
                    if v_key:
                        # Richiesta esplicita dell'utente dopo il fix del voto
                        # slab: "deve prendere i dati reali quanto possibile".
                        # Prova PRIMA il dato reale PriceCharting per il grado
                        # ESATTO scelto sulla pagina della variante (non solo
                        # il Grado 9 di get_cached_pc_variant_grade9 sotto) -
                        # solo se quella pagina non ha un dato per QUESTO grado
                        # specifico si cade sul Grado 9 reale + scalatura.
                        pc_tier_variant = get_cached_pc_variant_grade_tier(g_slug, i_slug, v_key, grade_tier)
                        if pc_tier_variant:
                            pc_eur, pc_usd, pc_url = pc_tier_variant
                            pc_live_info = {"eur": pc_eur, "usd": pc_usd, "url": pc_url, "source": "PriceCharting Variante", "tier": tier_label}
                            base_psa_final = pc_eur
                            base_psa_raw = pc_eur
                            effective_max_edge = round(pc_eur * 1.05, 2)
                            benchmark_source = f"PriceCharting Variante Reale {tier_label} (${pc_usd:.2f} USD)"
                            is_pc_grade_resolved = True
                        else:
                            pc_data = get_cached_pc_variant_grade9(g_slug, i_slug, v_key)
                            if pc_data:
                                pc_eur, pc_usd, pc_url = pc_data
                                pc_live_info = {"eur": pc_eur, "usd": pc_usd, "url": pc_url, "source": "PriceCharting Variante", "tier": "Grado 9"}
                                base_psa_final = pc_eur
                                base_psa_raw = pc_eur
                                effective_max_edge = round(pc_eur * 1.05, 2)
                                benchmark_source = f"PriceCharting Variante Reale (${pc_usd:.2f} USD)"
                                is_pc_grade_resolved = True
                                # BUG TROVATO (l'utente: "modificare il voto slab non
                                # modifica i prezzi consigliati", 2026-09-29):
                                # get_cached_pc_variant_grade9()/fetch_pricecharting_
                                # variant_grade9() restituisce SEMPRE il prezzo reale
                                # al grado 9.0 della variante (chiave "graded" del
                                # JSON PriceCharting), indipendente dal voto scelto -
                                # ma is_pc_grade_resolved=True disabilitava la
                                # scalatura di grado sotto per QUALUNQUE voto,
                                # lasciando il prezzo bloccato al valore Grado 9 pur
                                # etichettato come benchmark del grado scelto.
                                # Verificato: Jolteon #4 No Symbol Error, CGC 9.0 ->
                                # 283,46€, CGC 10.0 -> 134,65€ (invertito, un voto
                                # migliore valeva MENO). is_variant_grade9_only
                                # marca questo caso specifico, usato sotto per capire
                                # quando il prezzo NON e' davvero abbinato al voto.
                                is_variant_grade9_only = True

                if not is_pc_grade_resolved:
                    sel_item_id = sel_meta.get("item_id")
                    pc_tier = fetch_pricecharting_grade_tier_price(g_slug, i_slug, tier=grade_tier, item_id=sel_item_id)
                    if pc_tier:
                        pc_eur, pc_usd, pc_url, pc_source = pc_tier
                        base_psa_final = pc_eur
                        base_psa_raw = pc_eur
                        effective_max_edge = round(pc_eur * 1.05, 2)
                        pc_live_info = {"eur": pc_eur, "usd": pc_usd, "url": pc_url, "source": pc_source, "tier": tier_label}
                        benchmark_source = f"PriceCharting Reale {tier_label} (${pc_usd:.2f} USD)"
                        is_pc_grade_resolved = True

            # Priorità 3: Se non risolto con PriceCharting tier ma abbiamo il prezzo del DB / metadati
            if not is_pc_grade_resolved and manual_psa_override == 0.0:
                if sel_row:
                    base_psa_raw = float(sel_row["current_price_eur"])
                    base_max_edge = float(sel_row.get("max_edge_price_eur") or (base_psa_raw * 1.05))
                    base_psa_final = base_psa_raw
                    effective_max_edge = base_max_edge
                    benchmark_source = f"Database PokeQuant (PSA 9: {base_psa_raw:.2f}€)"
                elif matched_db_info:
                    db_price = matched_db_info.get("last_psa_price") or matched_db_info.get("cardmarket_ref_price_eur") or 0.0
                    if db_price > 0:
                        base_psa_raw = float(db_price)
                        base_max_edge = base_psa_raw * 1.05
                        base_psa_final = base_psa_raw
                        effective_max_edge = base_max_edge
                        benchmark_source = f"Riconosciuta da DB ({matched_db_info.get('name')}: {base_psa_raw:.2f}€)"

            # Gestione errore se benchmark è 0.0
            if base_psa_final <= 0.0:
                if "slab_eval_res" in st.session_state:
                    del st.session_state["slab_eval_res"]
                st.error(
                    "⚠️ **Impossibile calcolare il Fair Value**: Non è stato possibile determinare un prezzo di riferimento per questa carta.\n\n"
                    "👉 **Cosa fare**:\n"
                    "- Inserisci il **Benchmark PSA (€)** nel campo dedicato (es. il valore indicativo di mercato della carta in PSA 9 o PSA 10).\n"
                    "- Oppure incolla il **Link o Slug PriceCharting** (es. `https://www.pricecharting.com/game/...` o `pokemon-base-set/charizard-4`) per scaricare automaticamente i prezzi in tempo reale."
                )
            else:
                # Gestione variante speciale (se non già risolta da PriceCharting)
                is_special_variant = not variant_input.startswith("Standard")
                # is_pc_grade_resolved resta True anche quando il prezzo reale
                # trovato e' quello Grado-9-soltanto della variante
                # (is_variant_grade9_only) - qui invece serve sapere se il
                # prezzo e' DAVVERO abbinato al voto selezionato nel menu, per
                # decidere se scalarlo per grado sotto e se passarlo come
                # benchmark "gia' del grado giusto" a adjust_price_for_grading.
                is_price_grade_matched = is_pc_grade_resolved and not (is_variant_grade9_only and grade_val != "9.0")
                if manual_psa_override > 0.0:
                    v_mult = 1.0
                    v_desc = "Benchmark manuale inserito dall'utente"
                elif is_special_variant and not is_pc_grade_resolved:
                    v_mult, v_desc = get_variant_multiplier(variant_input, sel_meta.get("game_slug"))
                    base_psa_final = round(base_psa_raw * v_mult, 2)
                    effective_max_edge = round(base_max_edge * v_mult, 2)
                elif is_special_variant:
                    # BUG TROVATO (trovato indagando il bug del voto slab, stessa
                    # richiesta dell'utente): quando la variante aveva un prezzo
                    # reale (is_pc_grade_resolved), v_desc cadeva sempre nel ramo
                    # else sotto ("Versione Standard / Unlimited") - etichetta
                    # sbagliata per una carta che e' invece proprio la variante
                    # speciale scelta (es. "No Symbol Error"), solo perche' il
                    # suo prezzo veniva da un dato reale invece che da una stima.
                    v_mult = 1.0
                    v_desc = f"{variant_input.split('(')[0].strip()} — dato reale PriceCharting"
                else:
                    v_mult = 1.0
                    v_desc = "Versione Standard / Unlimited"

                if is_special_variant and ("1st" in variant_input.lower() or "symbol" in variant_input.lower() or "shadowless" in variant_input.lower()):
                    era_detected = "vintage"

                era_final = era_detected if era_input.startswith("Auto") else era_input
                display_title = f"{card_name} [{variant_input.split('(')[0].strip()}]" if is_special_variant else card_name

                # Stima se non risolto da tier reale (o se risolto solo al Grado 9
                # della variante ma il voto scelto e' un altro - vedi
                # is_price_grade_matched sopra)
                if not is_price_grade_matched and manual_psa_override == 0.0:
                    is_grade_10 = "10" in grade_val
                    if is_grade_10:
                        p10_ratio = ERA_PSA10_TO_PSA9_RATIO.get(normalize_era(era_final), 3.00)
                        base_psa_final = round(base_psa_final * p10_ratio, 2)
                        effective_max_edge = round(effective_max_edge * p10_ratio, 2)
                        benchmark_source = f"Stima Algoritmica PSA 10 ({p10_ratio:.2f}x era)"
                    # BUG TROVATO (trovato indagando il bug del voto slab sopra,
                    # stessa richiesta dell'utente): il grado 9.5 applicava QUI
                    # una pre-scalatura (ERA_BGS95_TO_PSA9_RATIO) e POI
                    # adjust_price_for_grading applicava DI NUOVO un
                    # moltiplicatore gia' "vs PSA9" da EMPIRICAL_RATIOS_GRADE9
                    # (get_grading_adjustment assegna benchmark_ref="PSA_9"
                    # anche al grado 9.5, non "PSA_10" - verificato leggendo il
                    # codice) - doppio conteggio che sovrastimava il fair value
                    # di un fattore pari esattamente a ERA_BGS95_TO_PSA9_RATIO
                    # (es. CGC 9.5 Moderno: 114,72€ corretto vs 184,69€ col
                    # doppio conteggio, +61%). Il grado 10 ne ha davvero bisogno
                    # (EMPIRICAL_RATIOS_GRADE10 e' "vs PSA10" - PSA/10.0=1.0x
                    # esatto, "Benchmark base Grado 10"), ma 9.5/8.5/8.0/7.5/7.0
                    # no: EMPIRICAL_RATIOS_GRADE9 e' gia' l'intero rapporto vs
                    # PSA9, nessuna pre-scalatura va applicata qui (stesso
                    # trattamento che 8.5/8.0/7.5/7.0 gia' ricevevano).
                    elif is_variant_grade9_only:
                        benchmark_source = f"PriceCharting Variante Reale, base Grado 9 ({base_psa_final:.2f}€)"
                    elif pc_live_info:
                        benchmark_source = f"PriceCharting Reale Grado 9 (${pc_live_info['usd']:.2f} USD)"
                    elif is_special_variant:
                        benchmark_source = f"Stima Variante ({v_mult:.2f}x)"

                rec_grade = get_recommended_grade_for_card(era=era_final)
                is_modern_sub10 = (normalize_era(era_final) == Era.MODERN and ("10" not in grade_val and "9.5" not in grade_val))
                card_rarity = sel_meta.get("rarity") if 'sel_meta' in locals() and sel_meta else (matched_db_info.get("rarity") if matched_db_info else None)
                card_reldate = sel_meta.get("release_date") if 'sel_meta' in locals() and sel_meta else (matched_db_info.get("release_date") if matched_db_info else None)
                strat_pop_details = get_card_strategy_and_pop_details(
                    item_id=target_item_id,
                    game_slug=g_slug,
                    item_slug=i_slug,
                    card_name=card_name,
                    era=era_final,
                    rarity=card_rarity,
                    release_date=card_reldate,
                    mode=singles_mode if 'singles_mode' in locals() else "production",
                )
                pop_pressure = strat_pop_details.get("pop_pressure") or get_card_pop_pressure(target_item_id, era_final)

                fair_value_calib, sniper_ceiling_calib, adj = adjust_price_for_grading(
                    base_psa_price_eur=base_psa_final,
                    company=company_input,
                    grade=grade_val,
                    era=era_final,
                    subgrades_black_label=is_black_label,
                    is_pristine=is_pristine,
                    is_grade_benchmark_price=is_price_grade_matched,
                )

                if not is_price_grade_matched:
                    sniper_ceiling_calib = round(effective_max_edge * adj.sniper_ceiling_factor, 2)

                if is_price_grade_matched and pc_live_info:
                    bench_tier_label = pc_live_info.get("tier", f"Grado {grade_val}")
                elif is_price_grade_matched:
                    bench_tier_label = f"Grado {grade_val}"
                elif "10" in grade_val:
                    bench_tier_label = "PSA 10"
                else:
                    bench_tier_label = "PSA 9"

                if is_usa_import:
                    landed_cost = estimate_usa_import_landed_cost(offer_price_eur, item_type="single")
                    total_offer = landed_cost
                    fixed_customs = (IMPORT_FROM_USA.intl_shipping_single_eur * (1.0 + IMPORT_FROM_USA.vat_rate)) + IMPORT_FROM_USA.eu_customs_duty_flat_eur + IMPORT_FROM_USA.courier_handling_fee_eur
                    sniper_net = max(0.0, round((sniper_ceiling_calib - fixed_customs) / (1.0 + IMPORT_FROM_USA.vat_rate), 2))
                else:
                    total_offer = offer_price_eur + shipping_eur
                    sniper_net = max(0.0, round(sniper_ceiling_calib - shipping_eur, 2))

                discount_real_pct = (1.0 - (total_offer / fair_value_calib)) * 100.0 if fair_value_calib > 0 else 0.0

                if total_offer <= fair_value_calib * 0.75:
                    v_badge = "🚨 DEEP VALUE / COLPACCIO (-25%+ di sconto)"
                    v_color = "#10b981"
                elif total_offer <= fair_value_calib * 0.95:
                    v_badge = "🟢 BUY CONSIGLIATO (A Sconto)"
                    v_color = "#10b981"
                elif total_offer <= sniper_ceiling_calib:
                    v_badge = "🟡 FAIR VALUE / AL LIMITE DELL'EDGE"
                    v_color = "#fbbf24"
                else:
                    over_pct = ((total_offer / sniper_ceiling_calib) - 1.0) * 100.0
                    v_badge = f"🔴 OVERPRICED (+{over_pct:.1f}% sopra il tetto)"
                    v_color = "#f43f5e"

                st.session_state["slab_eval_res"] = {
                    "display_title": display_title,
                    "target_item_id": target_item_id,
                    "grade_val": grade_val,
                    "bench_tier_label": bench_tier_label,
                    "pop_pressure": pop_pressure,
                    "strat_pop_details": strat_pop_details,
                    "base_psa_raw": base_psa_raw,
                    "v_mult": v_mult,
                    "v_desc": v_desc,
                    "base_psa_final": base_psa_final,
                    "effective_max_edge": effective_max_edge,
                    "pc_live_info": pc_live_info,
                    "benchmark_source": benchmark_source,
                    "rec_grade": rec_grade,
                    "is_modern_sub10": is_modern_sub10,
                    "company_name": adj.company.value,
                    "grade_input": grade_input,
                    "is_black_label": is_black_label,
                    "is_pristine": is_pristine,
                    "is_usa_import": is_usa_import,
                    "era_final": era_final,
                    "fair_value_calib": fair_value_calib,
                    "sniper_ceiling_calib": sniper_ceiling_calib,
                    "sniper_net": sniper_net,
                    "total_offer": total_offer,
                    "offer_price_eur": offer_price_eur,
                    "shipping_eur": shipping_eur,
                    "discount_real_pct": discount_real_pct,
                    "v_badge": v_badge,
                    "v_color": v_color,
                    "adj": adj,
                    "matched_db_info": matched_db_info,
                }

        # Mostra i risultati se calcolati
        if "slab_eval_res" in st.session_state:
            res = st.session_state["slab_eval_res"]
            if res.get("matched_db_info"):
                m_info = res["matched_db_info"]
                st.info(f"💡 Carta riconosciuta automaticamente nel catalogo PokeQuant: **{m_info.get('name')}** [{m_info.get('game_slug')}]. Prezzo storico rilevato: **{res['base_psa_final']:.2f} €**.")
                if m_info.get("data_quality") == "thin_unreliable":
                    st.warning(
                        f"🚩 **Mercato sottile**: questa carta è flaggata come dato inaffidabile "
                        f"(scripts/flag_unreliable_assets.py). {m_info.get('data_quality_reason', '')} "
                        f"Il benchmark sopra può essere gonfiato da poche vendite reali al grado — "
                        f"verificare a mano prima di procedere."
                    )

            pc_info = res.get("pc_live_info")
            if pc_info and "live" in pc_info.get("source", "").lower():
                st.success(f"🌐 Dati PriceCharting recuperati in tempo reale: **${pc_info['usd']:.2f} USD** ({pc_info['eur']:.2f} €) · Fonte: {pc_info.get('source')}")
            if pc_info and pc_info.get("url"):
                sub_benchmark = f"<a href='{pc_info['url']}' target='_blank' style='color: #38bdf8; text-decoration: underline;'>{res.get('benchmark_source', 'PriceCharting ↗')}</a>"
                variant_note = f" · Variante: <strong>{res['v_desc']} (<a href='{pc_info['url']}' target='_blank' style='color:#38bdf8;'>PriceCharting ↗</a>)</strong>"
            elif res['v_mult'] > 1.0:
                sub_benchmark = f"Base: {res['base_psa_raw']:.2f}€ × {res['v_mult']:.2f}x ({res.get('benchmark_source', 'Variante')})"
                variant_note = f" · Variante: <strong>{res['v_desc']} ({res['v_mult']:.2f}x)</strong>"
            else:
                sub_benchmark = res.get("benchmark_source", f"Prezzo mercato PSA (Max Edge: {res['effective_max_edge']:.2f}€)")
                variant_note = ""

            sub_offer_label = "🎯 Max Puntata eBay USA" if res.get("is_usa_import") else "🎯 Max Sniper (Netto)"
            sub_offer_desc = f"Max puntata consentita (Tua offerta sdoganata: {res['total_offer']:.2f}€ all-in)" if res.get("is_usa_import") else f"Max puntata asta (Tua offerta inserita: {res['total_offer']:.2f}€ all-in)"

            rec_grade = res.get("rec_grade")
            rec_badge_html = f"<span style='background: {rec_grade['badge_color']}22; color: {rec_grade['badge_color']}; border: 1px solid {rec_grade['badge_color']}; border-radius: 4px; padding: 2px 7px; font-size: 11px; font-weight: 700; margin-left: 8px;'>{rec_grade['target_badge']}</span>" if rec_grade else ""

            pop_pressure = res.get("pop_pressure", {})
            pop_badge = pop_pressure.get("badge_html", "")
            pop_badge_html = f"<span style='margin-left: 6px;'>{pop_badge}</span>" if pop_badge else ""

            # Dettagli Pop Report Multi-Fonte e Strategia
            strat = res.get("strat_pop_details") or {}
            has_pop = strat.get("has_pop_report", False)
            psa = strat.get("psa_census", {})
            cgc = strat.get("cgc_census", {})
            pc_pop_url = strat.get("pricecharting_pop_url")
            psa_search_url = strat.get("psa_search_url")
            gem_rate = strat.get("gem_rate_psa")
            gem_rate_str = f"{gem_rate:.1f}%" if gem_rate is not None else "N/D"

            r89 = pop_pressure.get("ratio_8_9", 0.0) or 0.0
            pct = pop_pressure.get("percentile", 50.0) or 50.0

            pc_link_btn = f"<a href='{pc_pop_url}' target='_blank' style='color:#38bdf8; text-decoration:none; background:rgba(56,189,248,0.12); border:1px solid rgba(56,189,248,0.3); padding:3px 8px; border-radius:5px; font-weight:600; font-size:11px;'>🌐 PriceCharting Pop ↗</a>" if pc_pop_url else ""
            psa_link_btn = f"<a href='{psa_search_url}' target='_blank' style='color:#fbbf24; text-decoration:none; background:rgba(245,158,11,0.12); border:1px solid rgba(245,158,11,0.3); padding:3px 8px; border-radius:5px; font-weight:600; font-size:11px;'>🏛️ Cerca su PSA Pop ↗</a>" if psa_search_url else ""

            if has_pop:
                p10_val = psa.get('10', 0)
                p9_val = psa.get('9', 0)
                p8_val = psa.get('8', 0)
                p7_val = psa.get('7', 0)
                psa_tot_val = psa.get('total', 0)

                c10_val = cgc.get('10', 0)
                c95_val = cgc.get('9.5', 0)
                c9_val = cgc.get('9', 0)
                c85_val = cgc.get('8.5', 0)
                cgc_tot_val = cgc.get('total', 0)
                mkt_tot_val = strat.get('market_total', 0)

                pop_report_html = (
                    f'<div style="margin-top:12px; background:rgba(30,41,59,0.7); border:1px solid rgba(56,189,248,0.25); border-radius:8px; padding:12px 14px;">'
                    f'<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; margin-bottom:10px; border-bottom:1px solid rgba(255,255,255,0.07); padding-bottom:8px;">'
                    f'<div style="display:flex; align-items:center; gap:8px;">'
                    f'<span style="font-weight:700; font-size:13px; color:#f8fafc;">📊 Population Report Multi-Fonte</span>'
                    f'<span style="font-size:11px; color:#94a3b8;">(Fonti: PSA Census & PriceCharting)</span>'
                    f'</div>'
                    f'<div style="display:flex; gap:6px;">{pc_link_btn}{psa_link_btn}</div>'
                    f'</div>'
                    f'<div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(210px, 1fr)); gap:12px; font-size:12px;">'
                    f'<div>'
                    f'<div style="font-weight:700; color:#38bdf8; margin-bottom:4px; font-size:11px; text-transform:uppercase; letter-spacing:0.5px;">Censimento PSA</div>'
                    f'<div style="color:#cbd5e1; line-height:1.6;">'
                    f'<div>• PSA 10 (Gem Mint): <strong style="color:#f8fafc;">{p10_val:,}</strong> <span style="font-size:10px; color:#94a3b8;">({gem_rate_str})</span></div>'
                    f'<div>• PSA 9 (Mint): <strong style="color:#f8fafc;">{p9_val:,}</strong></div>'
                    f'<div>• PSA 8 (NM-MT): <strong style="color:#f8fafc;">{p8_val:,}</strong></div>'
                    f'<div>• PSA 7 (Near Mint): <strong style="color:#f8fafc;">{p7_val:,}</strong></div>'
                    f'<div style="margin-top:2px; font-weight:600; color:#94a3b8;">Totale PSA: <strong style="color:#f8fafc;">{psa_tot_val:,}</strong> copie</div>'
                    f'</div>'
                    f'</div>'
                    f'<div>'
                    f'<div style="font-weight:700; color:#fbbf24; margin-bottom:4px; font-size:11px; text-transform:uppercase; letter-spacing:0.5px;">CGC & Mercato Totale</div>'
                    f'<div style="color:#cbd5e1; line-height:1.6;">'
                    f'<div>• CGC 10 Pristine: <strong style="color:#f8fafc;">{c10_val:,}</strong></div>'
                    f'<div>• CGC 9.5 / Gem: <strong style="color:#f8fafc;">{c95_val:,}</strong></div>'
                    f'<div>• CGC 9 (Mint): <strong style="color:#f8fafc;">{c9_val:,}</strong></div>'
                    f'<div>• CGC 8.5 (NM-Mt+): <strong style="color:#f8fafc;">{c85_val:,}</strong></div>'
                    f'<div style="margin-top:2px; font-weight:600; color:#94a3b8;">Totale Mercato: <strong style="color:#f8fafc;">{mkt_tot_val:,}</strong> copie</div>'
                    f'</div>'
                    f'</div>'
                    f'<div>'
                    f'<div style="font-weight:700; color:#a78bfa; margin-bottom:4px; font-size:11px; text-transform:uppercase; letter-spacing:0.5px;">Pressione Demografica</div>'
                    f'<div style="color:#cbd5e1; line-height:1.6;">'
                    f'<div>• Ratio Pop(8)/Pop(9): <strong style="color:#38bdf8; font-family:\'JetBrains Mono\',monospace;">{r89:.2f}x</strong></div>'
                    f'<div>• Posizione Era: <strong style="color:#f8fafc;">Top {100-pct:.0f}%</strong> (P{pct:.0f})</div>'
                    f'<div style="margin-top:4px;">{pop_badge}</div>'
                    f'<div style="margin-top:4px; font-size:11px; color:#94a3b8;">{strat.get("census_verdict", "")}</div>'
                    f'</div>'
                    f'</div>'
                    f'</div>'
                    f'</div>'
                )
            else:
                pop_report_html = (
                    f'<div style="margin-top:12px; background:rgba(30,41,59,0.4); border:1px solid rgba(255,255,255,0.06); border-radius:8px; padding:10px 14px;">'
                    f'<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">'
                    f'<span>📊 <strong>Population Report:</strong> Dati di censimento dettagliati non registrati in cache per questa referenza.</span>'
                    f'<div style="display:flex; gap:6px;">{pc_link_btn}{psa_link_btn}</div>'
                    f'</div>'
                    f'</div>'
                )

            # Box Analisi Scarsità & Value
            disc_pct = strat.get("discount_pct", 0.0)
            res_val = strat.get("residual")
            st_tier = strat.get("strategy_tier", "CUSTOM")
            if st_tier in ["CORE_BUY", "BENCH_BUY", "ALT_BUY"]:
                val_color = "#10b981"
                val_disalign_str = f"Sconto Edonico: -{abs(disc_pct):.1f}% vs peers"
                res_str = f" (Residuo ε = {res_val:.2f})" if res_val is not None else ""
            elif st_tier == "AVOID":
                val_color = "#f43f5e"
                val_disalign_str = f"Sovrapprezzo Speculativo: +{abs(disc_pct):.1f}% vs peers"
                res_str = f" (Residuo ε = {res_val:.2f})" if res_val is not None else ""
            elif st_tier == "NEUTRAL":
                val_color = "#94a3b8"
                val_disalign_str = "Prezzo Coerente con il Fair Value"
                res_str = ""
            else:
                val_color = "#64748b"
                val_disalign_str = "Valutazione Parametrica Empirica"
                res_str = ""

            strat_factors_html = (
                f'<div style="margin-top:10px; display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); gap:10px;">'
                f'<div style="background:rgba(15,23,42,0.85); border:1px solid rgba(16,185,129,0.3); border-radius:8px; padding:12px 14px;">'
                f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:6px;">'
                f'<span style="font-weight:700; font-size:13px; color:#10b981;">💎 Fattore Scarsità (Fisica & Temporale)</span>'
                f'<span style="font-size:10px; background:rgba(16,185,129,0.15); color:#10b981; border:1px solid rgba(16,185,129,0.4); border-radius:4px; padding:1px 6px; font-weight:700;">{strat.get("supply_status", "")}</span>'
                f'</div>'
                f'<div style="font-size:12px; color:#cbd5e1; line-height:1.6;">'
                f'<div>• <strong>Rarità & Pull Rate:</strong> <span style="color:#f8fafc;">{strat.get("rarity_tier_label", "Standard")}</span> ({strat.get("pull_rate_desc", "")})</div>'
                f'<div>• <strong>Tiratura & Anzianità:</strong> Anno <span style="color:#f8fafc;">{strat.get("release_year", 2020)}</span> ({strat.get("age_years", 0)} anni fa)</div>'
                f'<div style="margin-top:4px; font-size:11px; color:#94a3b8; font-style:italic;">{strat.get("supply_elasticity", "")}</div>'
                f'</div>'
                f'</div>'
                f'<div style="background:rgba(15,23,42,0.85); border:1px solid rgba(56,189,248,0.3); border-radius:8px; padding:12px 14px;">'
                f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:6px;">'
                f'<span style="font-weight:700; font-size:13px; color:#38bdf8;">⚖️ Fattore Value (Strategia PokeQuant)</span>'
                f'<span style="font-size:10px; background:rgba(56,189,248,0.15); color:{strat.get("strategy_badge_color", "#38bdf8")}; border:1px solid {strat.get("strategy_badge_color", "#38bdf8")}55; border-radius:4px; padding:1px 6px; font-weight:700;">{strat.get("strategy_badge", "Fascia Neutra")}</span>'
                f'</div>'
                f'<div style="font-size:12px; color:#cbd5e1; line-height:1.6;">'
                f'<div>• <strong>Disallineamento Edonico:</strong> <strong style="color:{val_color}; font-size:13px;">{val_disalign_str}</strong>{res_str}</div>'
                f'<div style="margin-top:4px; font-size:11px; color:#94a3b8; line-height:1.4;">{strat.get("value_verdict", "")}</div>'
                f'</div>'
                f'</div>'
                f'</div>'
            )

            if res.get("is_modern_sub10"):
                st.warning(
                    f"⚠️ **Avviso Liquidità Moderno ({res['grade_input']})**: Nelle carte moderne (2017+), i gradi ≤ 9.0 soffrono di scarsa "
                    "liquidità secondaria e scambiano spesso a ridosso o sotto il valore della carta Raw perché i pop report sono dominati da PSA 10 (>70–80%). "
                    "Per il moderno si raccomanda di puntare a **PSA 10** (o BGS 9.5 / Pristine 10) per preservare la rivendibilità."
                )

            # Warning contestuale per sovraffollamento pop su gradi intermedi
            g_val = str(res.get("grade_val", ""))
            is_sub9 = ("8" in g_val) or ("7" in g_val)
            if is_sub9 and pop_pressure.get("is_overcrowded"):
                r_val = pop_pressure.get("ratio_8_9", 0.0)
                pct_val = pop_pressure.get("percentile", 0.0)
                st.warning(
                    f"⚠️ **Avviso Sovraffollamento Pop ({res['company_name']} {res['grade_input']})**: "
                    f"Questa carta registra un rapporto Pop(8)/Pop(9) di **{r_val:.2f}x** (Percentile {pct_val:.0f} nell'era {res['era_final']}). "
                    "L'eccesso demografico sui gradi intermedi diluisce il valore collezionistico e crea una forte concorrenza tra venditori. "
                    "Se decidi di acquistare questo grado, esigi uno sconto ben superiore al fair value o privilegia il target primario raccomandato."
                )

            bench_ref_display = res.get("bench_tier_label") or ("PSA 10" if "10" in str(res.get("grade_val", "")) else "PSA 9")
            eval_card_html = (
                f'<div style="background: rgba(15,23,42,0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 14px 18px; margin-top: 10px;">'
                f'<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 10px;">'
                f'<div>'
                f'<span style="font-weight: 700; font-size: 15px; color: #f8fafc;">Valutazione per <u>{res["display_title"]}</u>: <span style="color: {res["v_color"]};">{res["v_badge"]}</span></span>'
                f'{rec_badge_html}'
                f'{pop_badge_html}'
                f'</div>'
                f'<span style="font-size: 12px; color: #94a3b8;">Slab: <strong>{res["company_name"]} {res["grade_input"]}</strong>{variant_note} · Moltiplicatore: <strong>{res["adj"].multiplier:.3f}x</strong> · Penalità liquidità: <strong>-{res["adj"].liquidity_penalty_pct:.0f}%</strong></span>'
                f'</div>'
                f'<div class="kpi-grid" style="margin-bottom: 0;">'
                f'<div class="kpi-card"><div class="kpi-label">Benchmark PSA ({bench_ref_display})</div><div class="kpi-value">{res["base_psa_final"]:.2f} €</div><div class="kpi-sub">{sub_benchmark}</div></div>'
                f'<div class="kpi-card"><div class="kpi-label">Fair Value {res["company_name"]}</div><div class="kpi-value">{res["fair_value_calib"]:.2f} €</div><div class="kpi-sub kpi-sub-emerald">Valore atteso reale</div></div>'
                f'<div class="kpi-card"><div class="kpi-label">Tetto Max (All-in)</div><div class="kpi-value">{res["sniper_ceiling_calib"]:.2f} €</div><div class="kpi-sub">Soffitto max per edge</div></div>'
                f'<div class="kpi-card"><div class="kpi-label">{sub_offer_label}</div><div class="kpi-value" style="color: #38bdf8;">{res["sniper_net"]:.2f} €</div><div class="kpi-sub">{sub_offer_desc}</div></div>'
                f'</div>'
                f'</div>'
            )
            st.markdown(eval_card_html, unsafe_allow_html=True)
            usa_warn = " · **Nota Dogana**: Inserzione extra-UE attiva (applicata IVA 22% su oggetto+spedizione, dazio forfettario 3€ e oneri corriere)." if res.get("is_usa_import") else ""
            if res['discount_real_pct'] >= 0:
                disc_str = f"Sconto effettivo: **+{res['discount_real_pct']:.1f}%**"
            else:
                disc_str = f"🔴 Sovrapprezzo offerta: **+{abs(res['discount_real_pct']):.1f}%**"
            rec_advice_str = f" · 💡 {rec_grade['short_advice']}" if rec_grade else ""
            st.caption(f"📝 **Logica**: {res['adj'].notes}. {disc_str} rispetto al fair value di una slab {res['company_name']} {res['grade_input']} ({res['era_final'].upper()}){usa_warn}{rec_advice_str}.")
            with st.expander("🔍 Perché questo fair value — popolazione, scarsità e modello"):
                st.markdown(pop_report_html, unsafe_allow_html=True)
                st.markdown(strat_factors_html, unsafe_allow_html=True)
            # Trovato in audit generale (richiesto dall'utente: "trova bug...
            # invalida"): molte celle di grading_multipliers.py sono stime a
            # mano senza riscontro nella ricerca empirica citata, non dati
            # verificati - marcate nel campo notes (vedi _mark_unverified_cells).
            # Qui il warning viene ripetuto in modo prominente, non solo
            # sepolto a meta' della caption sopra, perche' guida una decisione
            # di acquisto reale.
            if "STIMA NON VERIFICATA" in res['adj'].notes:
                st.warning("⚠️ Questo moltiplicatore è una stima non verificata (nessun riscontro in dati reali "
                           "raccolti da questo codice) — trattalo come un'indicazione di massima, non come un "
                           "fair value misurato. Verifica sempre con comparabili reali su PriceCharting/eBay prima di decidere.")
        else:
            st.caption("ℹ️ *Seleziona i parametri sopra e clicca su **'Calcola Valutazione Slab'** per vedere l'analisi istantanea senza ricaricare la pagina.*")

    st.markdown('<div class="section-title">📊 Validazione & Storico</div>', unsafe_allow_html=True)
    st.caption("Metriche di validazione, indici di mercato e giornale trade completo — contesto/audit, non un'azione da compiere ora. Tutto qui sotto è chiuso di default.")
    # --- METRICHE VALIDATE (box, singole, blend) — contesto/audit, non un'azione settimanale: chiuso di default ---
    with st.expander(
        f"📊 Metriche di validazione — Blend Sharpe {VALIDATED_BLEND['sharpe']:.2f} · "
        f"Box Sharpe {VALIDATED_BOX['sharpe']:.2f} (DSR sotto soglia) · Singole Sharpe {VALIDATED_SINGLES['sharpe']:.2f} (DSR nominale sopra soglia — ⚠️ vedi caveat quantità)"
    ):
        st.caption("Numeri fissi da `scripts/optimize_and_falsify.py` e `scripts/dac7_turnover_search.py` — "
                   "non ricalcolati a ogni refresh. Rivalidare ogni 6 mesi.")
        st.warning(
            f"**Box**: DSR {VALIDATED_BOX['dsr_full_session']:.3f} — **sotto** soglia comfort 0,90-0,95. "
            f"**Singole**: DSR {VALIDATED_SINGLES['dsr_full_session']:.3f} è nominale **sopra** soglia, ma gonfiato "
            f"dall'assunzione di molte copie/trade — a un tetto realistico (1-2 copie) scende a 0,028-0,346, "
            f"**sotto** soglia come il box. Non trattare 0,981 come validazione pulita — vedi caveat \"→ N pz.\" "
            f"nella sezione singole BUY."
        )
        st.caption(f"Griglia originale box: {VALIDATED_BOX['dsr_own_grid']:.3f} ({VALIDATED_BOX['n_trials_full_session']} trial). "
                   f"Singole: {VALIDATED_SINGLES['n_trials_full_session']} trial. Storico completo: "
                   "`scripts/dsr_session_audit.py`, `scripts/max_quantity_retest_expanded_universe.py`.")
        st.markdown('<div class="section-desc"><strong>📦 Box sigillati — TS Momentum</strong></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="kpi-grid">
            <div class="kpi-card"><div class="kpi-label">DSR (sessione intera)</div><div class="kpi-value">{VALIDATED_BOX['dsr_full_session']:.3f}</div><div class="kpi-sub kpi-sub-amber">Sotto soglia · griglia propria: {VALIDATED_BOX['dsr_own_grid']:.3f}</div></div>
            <div class="kpi-card"><div class="kpi-label">Sharpe</div><div class="kpi-value">{VALIDATED_BOX['sharpe']:.2f}</div><div class="kpi-sub kpi-sub-emerald">CAGR +{VALIDATED_BOX['cagr']:.1f}%</div></div>
            <div class="kpi-card"><div class="kpi-label">PBO (8 split)</div><div class="kpi-value">{VALIDATED_BOX['pbo']*100:.1f}%</div><div class="kpi-sub kpi-sub-amber">Sopra fascia comfort (&lt;20-25%)</div></div>
            <div class="kpi-card"><div class="kpi-label">Max Drawdown</div><div class="kpi-value">{VALIDATED_BOX['max_dd']:.1f}%</div><div class="kpi-sub kpi-sub-emerald">Bootstrap P(&gt;0)={VALIDATED_BOX['bootstrap_cagr_p_pos']}%</div></div>
            <div class="kpi-card"><div class="kpi-label">Walk-forward H1</div><div class="kpi-value">{VALIDATED_BOX['h1_sharpe']:.2f}</div><div class="kpi-sub kpi-sub-amber">Sharpe 2020-12→2023-10</div></div>
            <div class="kpi-card"><div class="kpi-label">Walk-forward H2</div><div class="kpi-value">{VALIDATED_BOX['h2_sharpe']:.2f}</div><div class="kpi-sub kpi-sub-emerald">Sharpe 2023-11→2026-09</div></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="section-desc"><strong>🃏 Singole — Fattore Scarsità (log-prezzo ~ scarsità continua + controlli)</strong></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="kpi-grid">
            <div class="kpi-card"><div class="kpi-label">DSR (sessione intera)</div><div class="kpi-value">{VALIDATED_SINGLES['dsr_full_session']:.3f}</div><div class="kpi-sub kpi-sub-amber">⚠️ Gonfiato dal tetto di quantità — a 1-2 copie: 0,03-0,35</div></div>
            <div class="kpi-card"><div class="kpi-label">Sharpe</div><div class="kpi-value">{VALIDATED_SINGLES['sharpe']:.2f}</div><div class="kpi-sub kpi-sub-emerald">CAGR +{VALIDATED_SINGLES['cagr']:.1f}%</div></div>
            <div class="kpi-card"><div class="kpi-label">PBO (8 split)</div><div class="kpi-value">{VALIDATED_SINGLES['pbo']*100:.1f}%</div><div class="kpi-sub kpi-sub-emerald">Molto stabile</div></div>
            <div class="kpi-card"><div class="kpi-label">Max Drawdown</div><div class="kpi-value">{VALIDATED_SINGLES['max_dd']:.1f}%</div></div>
            <div class="kpi-card"><div class="kpi-label">Walk-forward H1</div><div class="kpi-value">{VALIDATED_SINGLES['h1_sharpe']:.2f}</div><div class="kpi-sub kpi-sub-emerald">Sharpe 2021-01→2023-10</div></div>
            <div class="kpi-card"><div class="kpi-label">Walk-forward H2</div><div class="kpi-value">{VALIDATED_SINGLES['h2_sharpe']:.2f}</div><div class="kpi-sub kpi-sub-emerald">Sharpe 2023-11→2026-09</div></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="section-desc"><strong>🔗 Blend risk-parity — correlazione 0,37 tra le due strategie</strong></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="kpi-grid">
            <div class="kpi-card"><div class="kpi-label">Sharpe blend</div><div class="kpi-value">{VALIDATED_BLEND['sharpe']:.2f}</div><div class="kpi-sub kpi-sub-emerald">vs 1,38 box da solo (stesso periodo)</div></div>
            <div class="kpi-card"><div class="kpi-label">CAGR blend</div><div class="kpi-value">+{VALIDATED_BLEND['cagr']:.1f}%</div></div>
            <div class="kpi-card"><div class="kpi-label">Max Drawdown blend</div><div class="kpi-value">{VALIDATED_BLEND['max_dd']:.1f}%</div><div class="kpi-sub kpi-sub-emerald">vs -11,1% solo box</div></div>
        </div>
        """, unsafe_allow_html=True)

    # --- INDICI DI MERCATO (contesto, non segnale d'ingresso) — chiuso di default ---
    with st.expander("📉 Indici di mercato (contesto)"):
        st.caption("Indici equal-weight buy&hold (nessun timing, nessuna strategia) sull'universo sealed "
                   "validato — la 'beta' del mercato da confrontare con l'alfa della strategia. Segmenti JP "
                   "e One Piece hanno pochi titoli (5 e 4): direzionali, non statisticamente robusti da soli.")
        overall_index, segment_indices, segment_counts, breadth_series = get_market_indices()

        idx_fig = go.Figure()
        idx_fig.add_trace(go.Scatter(x=overall_index.index, y=overall_index.values, mode="lines",
                                      name=f"Sealed complessivo (n={sum(segment_counts.values())})",
                                      line=dict(color="#f8fafc", width=2.5)))
        seg_colors = {"Pokémon EN": "#38bdf8", "Pokémon JP": "#f43f5e", "One Piece TCG": "#fbbf24"}
        for name, series in segment_indices.items():
            idx_fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines",
                                          name=f"{name} (n={segment_counts[name]})",
                                          line=dict(color=seg_colors.get(name, "#94a3b8"), width=1.5, dash="dot")))
        idx_fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(15,23,42,0.4)", plot_bgcolor="rgba(15,23,42,0.4)",
                               height=320, margin=dict(l=20, r=20, t=30, b=20),
                               legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                               yaxis_title="Indice (base 100)")
        st.plotly_chart(idx_fig, use_container_width=True)

        breadth_fig = go.Figure()
        breadth_fig.add_trace(go.Scatter(x=breadth_series.index, y=breadth_series.values, mode="lines",
                                          fill="tozeroy", line=dict(color="#10b981", width=1.8),
                                          fillcolor="rgba(16,185,129,0.12)", name="Ampiezza"))
        breadth_fig.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.25)")
        breadth_fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(15,23,42,0.4)", plot_bgcolor="rgba(15,23,42,0.4)",
                                   height=180, margin=dict(l=20, r=20, t=10, b=20), showlegend=False,
                                   yaxis=dict(range=[0, 100], title="% con momentum 12m positivo"))
        st.plotly_chart(breadth_fig, use_container_width=True, config={"displayModeBar": False})
        st.caption(f"Ampiezza di mercato: quota dell'universo con momentum trailing 12m positivo — stessa regola "
                   f"della strategia, aggregata. Oggi: {breadth_series.iloc[-1]:.0f}%. Un calo ampio e prolungato "
                   "sotto il 50% è un segnale di regime, non di un singolo box.")

    # --- BACKTEST + GIORNALE TRADE (box, singole, blend) — audit, non un'azione settimanale: chiuso di default ---
    res, n_universe = get_backtest_results()
    res_singles, n_universe_singles = get_singles_backtest_results(singles_mode)
    with st.expander(f"📈 Backtest 2020-2026 e giornale trade — {res.total_trades + res_singles.total_trades} trade chiusi in tutto"):
        common_idx = res.monthly_returns.index.intersection(res_singles.monthly_returns.index)
        # w_box/w_singles = split per inverse-vol (risk parity), non piu' un
        # 50/50 fisso - vedi get_box_singles_split().
        blend_ret = w_box * res.monthly_returns.loc[common_idx] + w_singles * res_singles.monthly_returns.loc[common_idx]
        blend_nav = 10000.0 * (1.0 + blend_ret).cumprod()

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3],
                             subplot_titles=(f"NAV (€, base 10.000€ · {w_box*100:.0f}/{w_singles*100:.0f} box/singole)", "Drawdown (%)"))
        fig.add_trace(go.Scatter(x=res.nav_history.index, y=res.nav_history["nav"], mode="lines", name="Box (TS Momentum)",
                                  line=dict(color="#38bdf8", width=1.5, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=res_singles.nav_history.index, y=res_singles.nav_history["nav"], mode="lines", name="Singole (Scarsità)",
                                  line=dict(color="#fbbf24", width=1.5, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=blend_nav.index, y=blend_nav.values, mode="lines", name=f"Blend {w_box*100:.0f}/{w_singles*100:.0f}",
                                  line=dict(color="#10b981", width=2.5)), row=1, col=1)
        peak = blend_nav.cummax()
        dd = (blend_nav - peak) / peak * 100.0
        fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", fill="tozeroy", name="Drawdown Blend",
                                  line=dict(color="#f43f5e", width=1), fillcolor="rgba(244,63,94,0.15)"), row=2, col=1)
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(15,23,42,0.4)", plot_bgcolor="rgba(15,23,42,0.4)",
                           height=420, margin=dict(l=20, r=20, t=30, b=20),
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Universo: {n_universe} box/ETB era 2019+, {n_universe_singles} singole. Ogni metà simulata con "
                   "10.000€ propri, poi combinata come media dei rendimenti mensili — frizioni reali incluse: vendita "
                   "Cardmarket 5% + imballaggio 0,60€ + slippage + custodia; acquisto con spedizione reale a carico "
                   "del compratore (10€/box, 7€/carta), mai gratis nella realtà.")

        st.markdown('<div class="section-desc"><strong>📜 Trade chiusi — Box</strong></div>', unsafe_allow_html=True)
        trades_df = res.trades_df
        win_rate = res.win_rate * 100.0
        avg_holding = trades_df["holding_months"].mean() if not trades_df.empty else 0.0
        st.markdown(f"""
        <div class="kpi-grid">
            <div class="kpi-card"><div class="kpi-label">Trade chiusi</div><div class="kpi-value">{res.total_trades}</div></div>
            <div class="kpi-card"><div class="kpi-label">Win Rate</div><div class="kpi-value">{win_rate:.0f}%</div><div class="kpi-sub kpi-sub-amber">Pochi vincenti, grandi — tipico trend-following</div></div>
            <div class="kpi-card"><div class="kpi-label">Profit Factor</div><div class="kpi-value">{res.profit_factor:.2f}</div><div class="kpi-sub kpi-sub-emerald">Utile lordo / perdita lorda</div></div>
            <div class="kpi-card"><div class="kpi-label">Holding medio</div><div class="kpi-value">{avg_holding:.1f}m</div><div class="kpi-sub kpi-sub-amber">Mediana {trades_df['holding_months'].median():.0f}m, max {trades_df['holding_months'].max():.0f}m</div></div>
        </div>
        """, unsafe_allow_html=True)
        if trades_df.empty:
            st.info("Nessun trade chiuso nel backtest.")
        else:
            display_df = trades_df.sort_values("sell_date", ascending=False).copy()
            display_df["net_roi_pct"] = display_df["net_roi"] * 100.0
            display_df = display_df[["item_name", "buy_date", "sell_date", "holding_months",
                                      "buy_price_unit", "sell_price_unit", "net_roi_pct", "net_pnl"]]
            display_df.columns = ["Prodotto", "Acquisto", "Vendita", "Holding (m)",
                                   "Prezzo acquisto (€)", "Prezzo vendita (€)", "ROI netto (%)", "P&L netto (€)"]
            st.dataframe(
                display_df, use_container_width=True, hide_index=True,
                column_config={
                    "Prezzo acquisto (€)": st.column_config.NumberColumn(format="%.2f €"),
                    "Prezzo vendita (€)": st.column_config.NumberColumn(format="%.2f €"),
                    "ROI netto (%)": st.column_config.NumberColumn(format="%+.1f%%"),
                    "P&L netto (€)": st.column_config.NumberColumn(format="%+.2f €"),
                },
            )

        st.markdown('<div class="section-desc"><strong>📜 Trade chiusi — Singole</strong></div>', unsafe_allow_html=True)
        trades_df_s = res_singles.trades_df
        win_rate_s = res_singles.win_rate * 100.0
        avg_holding_s = trades_df_s["holding_months"].mean() if not trades_df_s.empty else 0.0
        st.markdown(f"""
        <div class="kpi-grid">
            <div class="kpi-card"><div class="kpi-label">Trade chiusi</div><div class="kpi-value">{res_singles.total_trades}</div></div>
            <div class="kpi-card"><div class="kpi-label">Win Rate</div><div class="kpi-value">{win_rate_s:.0f}%</div></div>
            <div class="kpi-card"><div class="kpi-label">Profit Factor</div><div class="kpi-value">{res_singles.profit_factor:.2f}</div><div class="kpi-sub kpi-sub-emerald">Utile lordo / perdita lorda</div></div>
            <div class="kpi-card"><div class="kpi-label">Holding medio</div><div class="kpi-value">{avg_holding_s:.1f}m</div><div class="kpi-sub kpi-sub-amber">Mediana {trades_df_s['holding_months'].median():.0f}m, max {trades_df_s['holding_months'].max():.0f}m</div></div>
        </div>
        """, unsafe_allow_html=True)
        if trades_df_s.empty:
            st.info("Nessun trade chiuso nel backtest.")
        else:
            display_df_s = trades_df_s.sort_values("sell_date", ascending=False).copy()
            display_df_s["net_roi_pct"] = display_df_s["net_roi"] * 100.0
            display_df_s = display_df_s[["item_name", "buy_date", "sell_date", "holding_months",
                                          "buy_price_unit", "sell_price_unit", "net_roi_pct", "net_pnl"]]
            display_df_s.columns = ["Carta", "Acquisto", "Vendita", "Holding (m)",
                                     "Prezzo acquisto (€)", "Prezzo vendita (€)", "ROI netto (%)", "P&L netto (€)"]
            col_config_s = {
                "Prezzo acquisto (€)": st.column_config.NumberColumn(format="%.2f €"),
                "Prezzo vendita (€)": st.column_config.NumberColumn(format="%.2f €"),
                "ROI netto (%)": st.column_config.NumberColumn(format="%+.1f%%"),
                "P&L netto (€)": st.column_config.NumberColumn(format="%+.2f €"),
            }
            st.dataframe(display_df_s.head(20), use_container_width=True, hide_index=True, column_config=col_config_s)
            if len(display_df_s) > 20:
                with st.expander(f"Altri {len(display_df_s) - 20} trade chiusi"):
                    st.dataframe(display_df_s.iloc[20:], use_container_width=True, hide_index=True, column_config=col_config_s)

        st.caption("P&L e ROI sono netti di commissione Cardmarket (5%), imballaggio (0,60€), costo di custodia e "
                   "spedizione all'acquisto — vedi \"Metriche di validazione\" per CAGR/Sharpe/MaxDD aggregati.")

    st.markdown("---")
    st.caption("PokeQuant · Blend box+singole scelto per correlazione bassa (0,37), non per rendimento massimo · "
                "box sotto soglia istituzionale dopo l'audit sull'intera sessione, singole sopra (vedi avviso in alto) · "
                "[Runbook Italia](https://github.com/davbenx/pokequant/blob/main/OPERATIONS_ITALIA.md) · "
                "Rivalidare con `scripts/optimize_and_falsify.py` e `scripts/scarcity_value_singles_test.py` ogni 6 mesi.")


if __name__ == "__main__":
    main()
