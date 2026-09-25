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
from poke_quant.engine.position_sizing import age_weight
from scripts.generate_monthly_signal import compute_signal_rows, MODERN_ERA_CUTOFF
from poke_quant.data.liquidity_filter import liquid_sealed_ids, MAX_PRICE_TO_MSRP_RATIO, liquid_singles_ids
from poke_quant.data.price_fetcher import fetch_pricecharting_cover_image_url
from poke_quant.config import estimate_usa_import_landed_cost
from scripts.generate_singles_signal import (
    compute_singles_signal_rows, compute_singles_avoid_rows, compute_singles_alternative_rows,
    PRODUCTION_PARAMS as SINGLES_PARAMS, DAC7_SINGLES_PARAMS,
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
    "dsr_own_grid": 0.999, "dsr_full_session": 0.850, "n_trials_full_session": 69,
    "pbo": 0.000, "sharpe": 1.52, "cagr": 27.52, "max_dd": -14.63,
    "h1_sharpe": 0.42, "h2_sharpe": 3.10,
}
VALIDATED_BLEND = {"sharpe": 1.90, "cagr": 24.73, "max_dd": -7.01}

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


@st.cache_data(show_spinner=False, ttl=3600)
def get_signal():
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
    metadata = load_metadata()
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
    metadata = load_metadata()
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
    params = SINGLES_PARAMS if mode == "production" else DAC7_SINGLES_PARAMS
    rows, latest_date = compute_singles_signal_rows(params)
    return rows, latest_date.strftime("%Y-%m-%d")


@st.cache_data(show_spinner=False, ttl=3600)
def get_singles_avoid_signal(mode: str = "production"):
    params = SINGLES_PARAMS if mode == "production" else DAC7_SINGLES_PARAMS
    rows, latest_date = compute_singles_avoid_rows(params)
    return rows, latest_date.strftime("%Y-%m-%d")


@st.cache_data(show_spinner=False, ttl=3600)
def get_singles_alternatives(mode: str = "production"):
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
def get_singles_backtest_results(mode: str = "production"):
    params = SINGLES_PARAMS if mode == "production" else DAC7_SINGLES_PARAMS
    metadata = load_metadata()
    prices_full = get_singles_prices_full()
    singles_ids = liquid_singles_ids(metadata, prices_full)
    meta_sub = {k: v for k, v in metadata.items() if k in singles_ids}
    prices_sub = prices_full[singles_ids]
    strat = ScarcityValueFactorStrategy(**params)
    bt = Backtester(strat, prices_sub, meta_sub, initial_cash=10000.0, platform="cardmarket",
                     apply_liquidity_slippage=True, apply_holding_cost=True, apply_buy_side_shipping=True)
    res = bt.run()
    return res, len(singles_ids)


# Cache manuale invece di @st.cache_data: serve un TTL DIVERSO per successo e
# fallimento. Trovato verificando "molte immagini delle carte singole non le
# vedo" - la dashboard richiede 30-60 immagini per pagina, tutte sincrone: un
# burst che fa scattare il rate limiting 429 di PriceCharting su una parte di
# esse (mitigato con retry in fetch_pricecharting_cover_image_url, ma non
# eliminato). Con un unico TTL lungo (7gg, giusto per un successo - l'immagine
# di un prodotto non cambia), un fallimento residuo diventava "nessuna
# immagine per una settimana intera". Ora un fallimento si ritenta dopo 1h,
# un successo resta cacheato 7gg.
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
    try:
        url = fetch_pricecharting_cover_image_url(game_slug, item_slug)
    except Exception:
        url = None
    _IMAGE_CACHE[key] = (url, now, _IMAGE_SUCCESS_TTL if url else _IMAGE_FAILURE_TTL)
    return url


def prefetch_product_images(pairs: list) -> None:
    """Scarica in PARALLELO (thread pool) tutte le immagini non ancora in
    cache, prima dei loop di rendering che le richiedono una per una.

    Trovato verificando "la dashboard continua a ricaricare": un caricamento
    a cache fredda richiedeva ~66s (misurato) perché fino a ~50-60 immagini
    (box BUY/uscite + singole BUY/evitare) venivano scaricate in SERIE, una
    alla volta, ciascuna una richiesta HTTP sincrona (aggravato dai retry sul
    429 aggiunti di recente, che a volte aggiungono secondi extra). Un
    caricamento cosi' lento e' plausibilmente la causa del reload continuo
    (timeout del browser/proxy -> refresh manuale -> cache di nuovo fredda ->
    di nuovo lento -> ...). Il fetch di un'immagine è I/O-bound (attesa di
    rete, non CPU) - parallelizzabile quasi linearmente con un thread pool,
    a differenza del calcolo del segnale che resta sequenziale."""
    to_fetch = []
    now = time.time()
    for game_slug, item_slug in pairs:
        if not game_slug or not item_slug:
            continue
        key = (game_slug, item_slug)
        cached = _IMAGE_CACHE.get(key)
        if cached is not None and now - cached[1] < cached[2]:
            continue
        to_fetch.append(key)
    if not to_fetch:
        return
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_pricecharting_cover_image_url, gs, isl): (gs, isl) for gs, isl in to_fetch}
        for future in as_completed(futures):
            key = futures[future]
            try:
                url = future.result()
            except Exception:
                url = None
            _IMAGE_CACHE[key] = (url, time.time(), _IMAGE_SUCCESS_TTL if url else _IMAGE_FAILURE_TTL)


def build_price_chart(item_id: str, name: str, prices_full: pd.DataFrame, months: int = 24):
    if item_id not in prices_full.columns:
        return None
    series = prices_full[item_id].dropna()
    series = series[series > 0].tail(months)
    if len(series) < 2:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines+markers",
                              line=dict(color="#38bdf8", width=1.8), marker=dict(size=3),
                              hovertemplate="%{x|%b %Y}: %{y:.0f}€<extra></extra>"))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       height=140, margin=dict(l=0, r=0, t=4, b=0), showlegend=False,
                       xaxis=dict(showgrid=False, tickfont=dict(size=9)),
                       yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", tickfont=dict(size=9)))
    return fig


def build_allocation(buy_rows: list, capital: float, metadata: dict, latest_date: str,
                      max_allocation_pct: float = 0.12):
    """Ripartisce il capitale per peso-età, poi applica il tetto per posizione
    dichiarato in sidebar (12% del capitale) con un waterfall: chi sfora il tetto
    viene fissato al tetto e l'eccedenza si ridistribuisce sui restanti, finche'
    nessuno sfora piu' - prima questa funzione calcolava solo la proporzione per
    peso senza applicare alcun tetto, contraddicendo il testo in sidebar."""
    latest_dt = pd.to_datetime(latest_date)
    weighted = []
    for r in buy_rows:
        rel_dt = metadata.get(r["item_id"], {}).get("release_date")
        age_m = None
        if rel_dt:
            rd = pd.to_datetime(rel_dt)
            age_m = (latest_dt.year - rd.year) * 12 + (latest_dt.month - rd.month)
        weighted.append((r, age_weight(age_m)))
    weighted.sort(key=lambda x: -x[1])

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
    stesso waterfall del tetto per posizione."""
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


def main():
    metadata = load_metadata()
    prices_full = get_prices_full()
    sig_rows, latest_date = get_signal()
    n_buy = sum(1 for r in sig_rows if r["signal"] == "BUY/HOLD")
    n_sell = sum(1 for r in sig_rows if r["signal"] == "AVOID/SELL")
    n_excessive = sum(1 for r in sig_rows if "PREZZO ECCESSIVO" in r["signal"])
    n_verify = len(sig_rows) - n_buy - n_sell - n_excessive

    st.markdown(f"""
    <div class="nav-header">
        <div>
            <span class="nav-title">⚡ PokeQuant</span>
            <span style="color:#64748b; font-size:12px; margin-left:8px;">Blend 50/50 · Box Sigillati (TS Momentum) + Singole (Fattore Scarsità)</span>
        </div>
        <div>
            <span class="pill-tag pill-blue">Blend Sharpe {VALIDATED_BLEND['sharpe']:.2f}</span>
            <span class="pill-tag pill-blue">Segnale {latest_date[:7]}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📋 Come si usa, in pratica", expanded=True):
        st.markdown("""
1. **Ogni mese**, guarda le liste 🟢 verdi qui sotto (box, poi singole) — sono già filtrate, pronte all'uso.
2. **Per ogni riga**: apri "Verifica su Cardmarket", controlla che **set/edizione coincidano esattamente**
   (badge blu **[Set]** sulle carte), e resta **sotto il "massimo"** mostrato (oggetto + spedizione).
3. **Rispetta i pezzi indicati** ("→ N pz.") — se non trovi tutte le copie di una carta, apri
   "🔄 alternative" invece di forzarne di più; per i box senza scorta, salta.
4. **Registra sempre** cosa hai trovato o venduto, anche se non hai comprato — `log_execution_price.py`
   e `log_sell_outcome.py` — calibra il modello nel tempo, non lasciarlo una stima fissa.
5. 🔴 **Rosso** = non comprare (box: momentum invertito · singole: sopravvalutata, solo informativo).
6. **Capitale** diviso 50/50 come mostrato in barra laterale; resta nei limiti DAC7 se vendi in UE.
7. **Rivalida** ogni 6 mesi con `optimize_and_falsify.py` / `scarcity_value_singles_test.py` — i numeri
   di validazione qui sotto sono fissi, non si aggiornano da soli.
        """)

    # --- SIDEBAR: capitale + conformità DAC7 ---
    with st.sidebar:
        st.markdown("### 💰 Capitale")
        capital = st.number_input("Capitale dedicato (€)", min_value=100.0, max_value=1_000_000.0,
                                   value=10000.0, step=500.0)
        st.caption("50% box sigillati, 50% singole (fattore scarsità) — le due strategie hanno "
                   "correlazione bassa (0,37): il blend porta Sharpe 1,38→1,90 e MaxDD -11,1%→-7,01% "
                   "rispetto al solo box (stesso periodo comune, frizioni incluse). Cap 12% del capitale per "
                   "singola posizione dentro ciascuna metà, box pesato per età (0,4x sotto i 18 mesi, 1,0x dopo).")
        st.markdown("---")
        st.markdown("### 💳 Budget massimo per carta")
        max_card_price = st.number_input(
            "Prezzo massimo per singola carta (€, 0 = nessun limite)", min_value=0.0, max_value=100_000.0,
            value=0.0, step=50.0,
            help="Filtra le carte in acquisto sopra questa soglia - indipendentemente da quanto il modello le "
                 "ritenga sottovalutate. Utile per restare su acquisti pratici/gestibili, non è un giudizio di "
                 "convenienza: una carta esclusa qui può comunque essere un'ottima occasione, solo fuori budget.")
        st.markdown("---")
        st.markdown("### 🛃 Import da venditore USA")
        show_usa_import = st.checkbox(
            "Mostra costo sdoganato stimato (TCGplayer/eBay.com)", value=False,
            help="Il prezzo PriceCharting è quello USA — per comprarlo davvero a quel livello serve un venditore "
                 "USA, non Cardmarket EU. Dal 1° luglio 2026 (Reg. UE 382/2026) è stata abolita la soglia di "
                 "franchigia doganale a 150€: OGNI spedizione extra-UE paga dazio, qualsiasi valore. Stima: "
                 "oggetto + spedizione internazionale + IVA 22% + dazio forfettario UE 3€ + commissione di "
                 "sdoganamento del corriere (~15€, indicativa — varia per corriere). Alta confidenza su IVA/dazio "
                 "(normativa verificata), bassa sulla commissione corriere — non è un preventivo vincolante. "
                 "⚠️ TESTATO (scripts/usa_landed_cost_edge_test.py): comprare SEMPRE a questo costo pieno "
                 "distrugge l'edge — singole Sharpe 1,57→-0,33 (perdita netta), box Sharpe 1,18→0,58 con MaxDD "
                 "triplicato. Usa questo numero solo come soglia informativa (EU è comunque meglio o peggio "
                 "di importare), non come canale di acquisto regolare.")
        st.markdown("---")
        st.markdown("### 🇪🇺 Conformità DAC7")
        dac7_mode = st.checkbox("Resta sotto 2.000€ / 30 vendite annue", value=True,
                                 help="DAC7: sopra queste soglie, Cardmarket/eBay segnalano il venditore "
                                      "come commerciale alle autorità fiscali. Con questa modalità attiva, "
                                      "le singole usano un ribilanciamento meno frequente (12m invece di 3m, "
                                      "meno posizioni: 20 invece di 60) e il capitale effettivo viene limitato "
                                      "al massimo che resta sotto soglia.")
        singles_mode = "dac7" if dac7_mode else "production"

        res_box_dac7check, _ = get_backtest_results()
        res_singles_dac7check, _ = get_singles_backtest_results(singles_mode)
        box_trades_yr, box_eur_yr_per_10k = annualized_turnover(res_box_dac7check.trades_df)
        singles_trades_yr, singles_eur_yr_per_10k = annualized_turnover(res_singles_dac7check.trades_df)

        # Il conteggio vendite/anno e' strutturale (non scala col capitale) - solo
        # il volume EUR/anno scala linearmente col capitale allocato a ciascuna meta'.
        total_trades_yr = box_trades_yr + singles_trades_yr
        eur_yr_at_capital = (box_eur_yr_per_10k + singles_eur_yr_per_10k) * (capital * 0.5 / 10000.0)

        combined_rate_per_eur = (box_eur_yr_per_10k + singles_eur_yr_per_10k) / 10000.0 / 2.0  # per EUR di capitale TOTALE (non solo la meta')
        safe_max_capital = (DAC7_MAX_ANNUAL_EUR / combined_rate_per_eur) if combined_rate_per_eur > 0 else capital

        effective_capital = min(capital, safe_max_capital) if dac7_mode else capital

        over_count = total_trades_yr > DAC7_MAX_ANNUAL_TRADES
        over_volume = capital > safe_max_capital

        if dac7_mode:
            if over_count:
                st.error(f"⚠️ Anche al minimo, questa configurazione genera ~{total_trades_yr:.0f} vendite/anno — "
                         f"sopra le 30 indipendentemente dal capitale (il conteggio non scala col capitale, solo il volume €).")
            if over_volume:
                st.warning(f"Capitale limitato a **{effective_capital:,.0f}€** (da {capital:,.0f}€ richiesti) per restare "
                           f"sotto {DAC7_MAX_ANNUAL_EUR:,.0f}€/anno di vendite stimate.")
            else:
                st.success(f"✅ ~{total_trades_yr:.0f} vendite/anno, ~{eur_yr_at_capital:,.0f}€/anno stimati — sotto soglia.")
            st.caption(f"Capitale massimo sicuro stimato: **{safe_max_capital:,.0f}€** totali (50/50 box+singole). "
                       "Stima da turnover storico del backtest, non una garanzia — la liquidità reale (quanti "
                       "acquirenti/venditori ci sono davvero) non è verificata.")
        else:
            st.error(f"⚠️ Modalità DAC7 disattivata: ~{total_trades_yr:.0f} vendite/anno, ~{eur_yr_at_capital:,.0f}€/anno "
                     f"stimati a questo capitale — probabile segnalazione come venditore commerciale se superi 2.000€/30 vendite.")

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

    # --- METRICHE VALIDATE (box, singole, blend) — contesto/audit, non un'azione settimanale: chiuso di default ---
    with st.expander(
        f"📊 Metriche di validazione — Blend Sharpe {VALIDATED_BLEND['sharpe']:.2f} · "
        f"Box Sharpe {VALIDATED_BOX['sharpe']:.2f} (DSR sotto soglia) · Singole Sharpe {VALIDATED_SINGLES['sharpe']:.2f} (DSR sotto soglia)"
    ):
        st.caption("Numeri fissi da `scripts/optimize_and_falsify.py` e `scripts/scarcity_value_singles_test.py` — "
                   "non ricalcolati a ogni refresh. Rivalidare ogni 6 mesi.")
        if singles_mode == "dac7":
            st.info("Modalità DAC7 attiva: i numeri qui sotto sono della configurazione a turnover pieno (per "
                    "confronto) — le performance EFFETTIVE in modalità DAC7 sono più basse, vedi il Backtest sotto.")
        st.warning(
            f"**DSR corretto per l'intera sessione, sotto la soglia di comfort 0,90-0,95 per entrambe le "
            f"strategie**: box {VALIDATED_BOX['dsr_full_session']:.3f} (griglia originale: {VALIDATED_BOX['dsr_own_grid']:.3f}, "
            f"{VALIDATED_BOX['n_trials_full_session']} trial) · singole {VALIDATED_SINGLES['dsr_full_session']:.3f} "
            f"({VALIDATED_SINGLES['n_trials_full_session']} trial) — resta il miglior risultato di tutta la ricerca su "
            f"ciascun lato, walk-forward ancora positivo in entrambe le metà. Storico completo: "
            f"`scripts/dsr_session_audit.py`, `scripts/scarcity_value_singles_test.py`."
        )
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
            <div class="kpi-card"><div class="kpi-label">DSR (sessione intera)</div><div class="kpi-value">{VALIDATED_SINGLES['dsr_full_session']:.3f}</div><div class="kpi-sub kpi-sub-amber">Sotto soglia · griglia propria: {VALIDATED_SINGLES['dsr_own_grid']:.3f}</div></div>
            <div class="kpi-card"><div class="kpi-label">Sharpe</div><div class="kpi-value">{VALIDATED_SINGLES['sharpe']:.2f}</div><div class="kpi-sub kpi-sub-emerald">CAGR +{VALIDATED_SINGLES['cagr']:.1f}%</div></div>
            <div class="kpi-card"><div class="kpi-label">PBO (8 split)</div><div class="kpi-value">{VALIDATED_SINGLES['pbo']*100:.1f}%</div><div class="kpi-sub kpi-sub-emerald">Molto stabile</div></div>
            <div class="kpi-card"><div class="kpi-label">Max Drawdown</div><div class="kpi-value">{VALIDATED_SINGLES['max_dd']:.1f}%</div></div>
            <div class="kpi-card"><div class="kpi-label">Walk-forward H1</div><div class="kpi-value">{VALIDATED_SINGLES['h1_sharpe']:.2f}</div><div class="kpi-sub kpi-sub-emerald">Sharpe 2021-01→2023-10</div></div>
            <div class="kpi-card"><div class="kpi-label">Walk-forward H2</div><div class="kpi-value">{VALIDATED_SINGLES['h2_sharpe']:.2f}</div><div class="kpi-sub kpi-sub-emerald">Sharpe 2023-11→2026-09</div></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="section-desc"><strong>🔗 Blend 50/50 — correlazione 0,37 tra le due strategie</strong></div>', unsafe_allow_html=True)
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

    # --- AZIONE: BUY/HOLD con allocazione e link Cardmarket (50% del capitale) ---
    st.markdown(f'<div class="section-title">📦 Box da comprare/mantenere — 50% del capitale ({capital*0.5:,.0f}€)</div>', unsafe_allow_html=True)
    st.caption("Prezzo da PriceCharting (USA), NON una quota Cardmarket — verifica sempre col prezzo reale dietro "
               "il bottone. Resta sotto il \"massimo\" (oggetto + spedizione). L'€ mostrato è l'allocazione ideale: "
               "se un box costa più, il modello lo compra comunque per intero finché resta sotto il 35% del "
               "capitale box (giallo); sopra quella soglia, salta (rosso).")
    with st.expander("ℹ️ Dettagli — lotto indivisibile, capitale sequenziale"):
        st.caption("\"Massimo\" è la spesa TOTALE oltre la quale il modello considera il box fuori dal range "
                   "prezzo/MSRP validato (21,6x) — non sottraiamo qui una stima di spedizione, il numero è già "
                   "netto: confronta oggetto + spedizione reali dell'inserzione contro questo valore. L'€ mostrato "
                   "è l'allocazione IDEALE proporzionale (tetto 12% del capitale box): il modello TESTATO non salta "
                   "un box solo perché costa più della sua quota ideale — lo compra per intero (lotto indivisibile) "
                   "finché il prezzo resta sotto il 35% del capitale dedicato ai box, segnalato in giallo con la "
                   "spesa reale richiesta; solo sopra quel 35% la posizione viene saltata (rosso), troppo concentrata "
                   "anche per la regola testata. Questo calcolo è una semplificazione statica (divide il capitale "
                   "proporzionalmente su tutti i segnali di oggi); il backtest reale spende la cassa in sequenza, "
                   "quindi con molti segnali insieme non è garantito che tu possa comprarli tutti — priorità ai "
                   "primi in lista.")
    buy_rows = [r for r in sig_rows if r["signal"] == "BUY/HOLD"]
    if max_card_price > 0:
        buy_rows = [r for r in buy_rows if r["current_price_eur"] <= max_card_price]
    allocation = build_allocation(buy_rows, capital * 0.5, metadata, latest_date)
    prefetch_product_images([
        (metadata.get(r["item_id"], {}).get("game_slug"), metadata.get(r["item_id"], {}).get("item_slug"))
        for r, _, _ in allocation
    ])

    if not allocation:
        st.info("Nessun segnale BUY/HOLD questo mese.")
    else:
        box_capital_half = capital * 0.5
        _total_real_spend, _n_skip = 0.0, 0
        for r, alloc, w in allocation:
            p = r["current_price_eur"]
            if p <= 0 or alloc >= p:
                _total_real_spend += alloc
            elif p <= box_capital_half * 0.35:
                _total_real_spend += p
            else:
                _n_skip += 1
        if _total_real_spend > box_capital_half * 1.10:
            st.warning(
                f"⚠️ A questo capitale, comprare per intero **tutti** i {len(allocation)} box in BUY costerebbe "
                f"~**{_total_real_spend:,.0f}€**, contro i {box_capital_half:,.0f}€ dedicati (+{(_total_real_spend/box_capital_half-1)*100:.0f}%). "
                "Non è un errore: i box sono lotti indivisibili, quindi molti superano l'allocazione ideale per "
                "posizione. Il modello reale spende la cassa in sequenza — priorità ai primi in lista, gli ultimi "
                "potrebbero non essere eseguibili questo mese con questo capitale."
                + (f" {_n_skip} box sopra il 35% del capitale vengono comunque saltati indipendentemente dalla cassa." if _n_skip else "")
            )
    for r, alloc, w in allocation:
        meta = metadata.get(r["item_id"], {})
        link = get_cardmarket_deep_link(r["name"], franchise=meta.get("franchise", "pokemon"),
                                         language=meta.get("language", "en"), game_slug=meta.get("game_slug"))
        img_url = get_product_image(meta.get("game_slug"), meta.get("item_slug"))
        img_tag = f'<img class="signal-card-thumb" src="{img_url}" />' if img_url else '<div class="signal-card-thumb"></div>'
        max_price_html = (f' &nbsp;·&nbsp; <span style="color:#94a3b8;">massimo (tot.) '
                           f'{r["max_price_eur"]:.0f}€</span>') if r.get("max_price_eur") is not None else ""
        usa_import_html = ""
        if show_usa_import:
            landed = estimate_usa_import_landed_cost(r["current_price_eur"], item_type="sealed")
            usa_import_html = (f' &nbsp;·&nbsp; <span style="color:#fbbf24;">sdoganato da USA ~{landed:.0f}€</span>')
        box_price = r["current_price_eur"]
        # Stessa regola di TimeSeriesMomentumStrategy.generate_signals (lotto
        # minimo indivisibile): se il budget proporzionale non basta per 1 pezzo,
        # il modello validato lo compra comunque per intero SOLO se il prezzo
        # resta sotto il 35% del capitale dedicato ai box - altrimenti la salta.
        # Qui capital*0.5 approssima il total_nav dello strategy (stesso valore
        # passato a build_allocation) - una carta/box i cui bisogni superano
        # l'allocazione "ideale" NON è un errore di visualizzazione, è la regola
        # testata (vedi scripts/max_quantity_per_trade_test.py e la sidebar).
        box_capital_half = capital * 0.5
        if box_price <= 0:
            qty_est, spend_est, skip_reason = 1, alloc, None
        elif alloc >= box_price:
            qty_est, spend_est, skip_reason = int(alloc // box_price), alloc, None
        elif box_price <= box_capital_half * 0.35:
            qty_est, spend_est, skip_reason = 1, box_price, "budget"
        else:
            qty_est, spend_est, skip_reason = 0, 0.0, "troppo_grande"

        if skip_reason == "troppo_grande":
            qty_html = (f' &nbsp; <span style="color:#f43f5e;">⚠️ salta a questo capitale — costa {box_price:,.0f}€, '
                        f'sopra il 35% dei {box_capital_half:,.0f}€ dedicati ai box (regola testata, non un tetto arbitrario)</span>')
        elif skip_reason == "budget":
            qty_html = (f' &nbsp; <span style="color:#fbbf24;">→ 1 pz. ⚠️ richiede {spend_est:,.0f}€, più dei {alloc:,.0f}€ '
                        f'ideali — il modello lo compra comunque per intero (lotto indivisibile) se hai il capitale libero</span>')
        else:
            qty_warn = ' ⚠️ <span style="color:#fbbf24;">assume più copie identiche disponibili insieme</span>' if qty_est > 3 else ""
            qty_html = f' &nbsp; <span style="color:#94a3b8;">→ {qty_est} pz.{qty_warn}</span>'
        alloc_display = spend_est if skip_reason == "budget" else alloc
        st.markdown(f"""
        <div class="signal-card signal-card-buy">
            {img_tag}
            <div class="signal-card-body">
            <strong>{r['name']}</strong> &nbsp; <span style="color:#10b981;">+{r['trailing_12m_return_pct']:.0f}% (12m)</span>
            &nbsp;·&nbsp; {r['current_price_eur']:.0f}€ (PriceCharting) &nbsp;·&nbsp; peso età {w:.2f}{max_price_html}{usa_import_html}
            <br><span style="font-family:'JetBrains Mono',monospace; font-size:15px; color:#f8fafc;">{alloc_display:,.0f}€</span>{qty_html}
            &nbsp; <a class="cm-btn" href="{link}" target="_blank">🛒 Verifica su Cardmarket</a>
            </div>
        </div>
        """, unsafe_allow_html=True)
        chart = build_price_chart(r["item_id"], r["name"], prices_full)
        if chart is not None:
            st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False},
                             key=f"chart_buy_{r['item_id']}")

    # --- ROTAZIONE: AVOID/SELL ---
    sell_rows = [r for r in sig_rows if r["signal"] == "AVOID/SELL"]
    if sell_rows:
        st.markdown('<div class="section-title">🔴 Uscite (momentum invertito)</div>', unsafe_allow_html=True)
        prefetch_product_images([
            (metadata.get(r["item_id"], {}).get("game_slug"), metadata.get(r["item_id"], {}).get("item_slug"))
            for r in sell_rows
        ])
        for r in sell_rows:
            meta_sell = metadata.get(r["item_id"], {})
            img_url = get_product_image(meta_sell.get("game_slug"), meta_sell.get("item_slug"))
            img_tag = f'<img class="signal-card-thumb" src="{img_url}" />' if img_url else '<div class="signal-card-thumb"></div>'
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
                st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False},
                                 key=f"chart_sell_{r['item_id']}")

    if n_excessive:
        excessive_rows = [r for r in sig_rows if "PREZZO ECCESSIVO" in r["signal"]]
        with st.expander(f"🚫 Prezzo eccessivo — momentum positivo ma bloccato ({n_excessive})"):
            st.caption("Il modello direbbe di comprare (momentum 12m positivo), ma il prezzo attuale supera già il "
                       "tetto che preserva l'edge: stesso rapporto prezzo/MSRP già usato per ammettere un box vintage "
                       "nell'universo (21,6x, vedi `poke_quant/data/liquidity_filter.py`), qui applicato anche a un "
                       "nuovo acquisto. Non è impossibile che salga ancora, ma comprare oltre questo confine non è "
                       "ciò che è stato validato — impedisce di comprare a un prezzo che romperebbe l'edge misurato.")
            for r in excessive_rows:
                st.markdown(f"- **{r['name']}** — {r['current_price_eur']:.0f}€ attuale vs **{r['max_price_eur']:.0f}€ massimo "
                            f"(totale)** (+{r['trailing_12m_return_pct']:.0f}% 12m)")

    if n_verify:
        with st.expander(f"⚠️ Da verificare a mano ({n_verify}) — rendimento implausibile, mercato troppo sottile"):
            for r in sig_rows:
                if "VERIFICARE" in r["signal"]:
                    st.markdown(f"- **{r['name']}** — {r['trailing_12m_return_pct']:+.0f}% (12m), {r['current_price_eur']:.0f}€ (PriceCharting)")
                    chart = build_price_chart(r["item_id"], r["name"], prices_full)
                    if chart is not None:
                        st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False},
                                         key=f"chart_verify_{r['item_id']}")

    # --- AZIONE: SINGOLE — FATTORE SCARSITÀ (50% del capitale) ---
    st.markdown(f'<div class="section-title">🃏 Singole da comprare — Fattore Scarsità, 50% del capitale ({capital*0.5:,.0f}€)</div>', unsafe_allow_html=True)
    if singles_mode == "dac7":
        st.info("ℹ️ Modalità DAC7 attiva: questa NON è la stessa lista di produzione filtrata più stretta — è "
                "una configurazione diversa (ribilanciamento ogni 12 mesi invece di 3, max 20 posizioni invece di "
                "60, validata separatamente). Le due liste possono non avere **nessuna carta in comune**: una "
                "carta nel quantile top-20 da 8 mesi è \"fresca\" per DAC7 (finestra 12 mesi) ma \"scaduta/value "
                "trap\" per la produzione (finestra 3 mesi) — e viceversa, una carta appena entrata nella top-60 "
                "potrebbe non essere nella top-20 più selettiva di DAC7. Disattiva il toggle in sidebar per "
                "vedere la lista di produzione.")
    st.caption("Da comprare: la carta GIÀ GRADATA Grade 9 (uno slab, non raw, non PSA10). Il badge blu **[Set]** "
               "è il set/espansione esatto — verifica sempre di cercare quel set su Cardmarket, il nome della "
               "carta da solo non basta (⚠️ **(Unlimited)** = esiste anche una 1st Edition più cara, prodotto "
               "diverso). Resta sotto il \"massimo\" mostrato. Prime 15 con grafico, le altre in tabella sotto. "
               "⚠️ **La compagnia di gradazione conta**: compra solo **PSA o CGC** — vedi dettagli sotto.")
    with st.expander("ℹ️ Dettagli — sconto, freschezza, copie, limiti del modello"):
        st.caption("\"Sconto vs. pari\" è quanto la carta costa in meno (%) rispetto a quanto la sua rarità/età/set "
                   "implicherebbero rispetto alle sue pari — più negativo, più sottovalutata secondo il modello. "
                   "\"Segnale da\" è da quanti mesi consecutivi è nel quantile BUY: solo le carte entrate negli "
                   "ultimi 3 mesi (cadenza di ribilanciamento validata) sono mostrate qui — oltre, è un possibile "
                   "*value trap* (sconto persistente che il mercato non corregge), non escluso ma spostato nelle "
                   "alternative sotto. \"Massimo\" è la spesa TOTALE (oggetto + spedizione) oltre la quale la carta "
                   "esce dal confine del quantile BUY — già netto, non sottrarre una stima di spedizione tua. Il "
                   "link cerca solo per nome carta (aggiungere set/edizione/grado dava pagine vuote, verificato) — "
                   "usa i filtri di Cardmarket (espansione, lingua), guidati dal set indicato nel badge. Sulle "
                   "carte vintage poco liquide, il pannello Grade 9 può restare sottostimato anche dopo il filtro "
                   "di attendibilità — se non trovi nulla sotto il \"massimo\", registralo con "
                   "`log_execution_price.py` invece di ignorare il segnale. "
                   "\"→ N pz.\" è quante copie IDENTICHE il budget comprerebbe al prezzo mostrato — ⚠️ "
                   "**VERIFICATO** (`scripts/max_quantity_per_trade_test.py`): il backtest (Sharpe 1,52) assume "
                   "che tu trovi TUTTE queste copie insieme, ogni mese; con un tetto realistico di 1 copia lo "
                   "Sharpe scende drasticamente (~0,6, DSR debole), con 2 copie recupera solo in parte — tratta "
                   "1,52 come un tetto teorico se compri quasi sempre un pezzo singolo, non un'aspettativa "
                   "realistica (numeri esatti non riverificati sull'ultimo aggiustamento d'universo, la direzione "
                   "non cambia).")
        st.caption("⚠️ **Compagnia di gradazione — verificato live su PriceCharting**: il prezzo \"Grade 9\" "
                   "che vedi qui è un dato AGGREGATO cross-company (la tabella prezzi di PriceCharting mostra "
                   "\"Ungraded/Grade 7/Grade 8/Grade 9/Grade 9.5/PSA 10\" — solo il grado 10 è diviso per "
                   "compagnia). Dove il dato è diviso, lo spread è enorme: su Charizard Base Set, PSA 10 vale "
                   "**$12.275** contro **$4.707** di CGC 10 (2,6x) e **$7.365** di SGC 10 — non c'è motivo per "
                   "cui questo effetto scompaia al grado 9, semplicemente lì non è misurato per compagnia. "
                   "**Compra solo slab PSA o CGC** (le due compagnie con vera liquidità/riconoscimento sul "
                   "mercato Pokémon): TAG, GRAAD, ACE, EGS, AI-grading e simili NON sono tra le compagnie "
                   "tracciate da PriceCharting (nemmeno tutte quelle tracciate hanno sempre prezzo — ACE mostra "
                   "\"-\" pure al grado 10) — per queste il numero mostrato in dashboard non è il prezzo che "
                   "otterresti in una rivendita reale, il modello non lo sa distinguere.")
    singles_rows, singles_latest_date = get_singles_signal(singles_mode)
    singles_prices_full = get_singles_prices_full()
    if max_card_price > 0:
        singles_rows = [r for r in singles_rows if r["current_price_eur"] <= max_card_price]
    singles_allocation = build_equal_allocation(singles_rows, capital * 0.5)

    if not singles_allocation:
        st.info("Nessuna carta nel quantile BUY questo mese.")
    prefetch_product_images([
        (metadata.get(r["item_id"], {}).get("game_slug"), metadata.get(r["item_id"], {}).get("item_slug"))
        for r, _ in singles_allocation[:15]
    ])
    for r, alloc in singles_allocation[:15]:
        meta = {"franchise": r.get("franchise", "pokemon"), "language": r.get("language", "en")}
        full_meta = metadata.get(r["item_id"], {})
        link = get_cardmarket_deep_link(r["name"], franchise=meta["franchise"], language=meta["language"],
                                         item_type="single", game_slug=full_meta.get("game_slug"))
        start = r["signal_start_date"]
        start_str = start.strftime("%Y-%m") if hasattr(start, "strftime") else str(start)
        img_url = get_product_image(full_meta.get("game_slug"), full_meta.get("item_slug"))
        img_tag = f'<img class="signal-card-thumb" src="{img_url}" />' if img_url else '<div class="signal-card-thumb"></div>'
        max_price_html = (f' &nbsp;·&nbsp; <span style="color:#94a3b8;">massimo (tot.) '
                           f'{r["max_edge_price_eur"]:.2f}€</span>') if r.get("max_edge_price_eur") is not None else ""
        usa_import_html = ""
        if show_usa_import:
            landed = estimate_usa_import_landed_cost(r["current_price_eur"], item_type="single")
            usa_import_html = (f' &nbsp;·&nbsp; <span style="color:#fbbf24;">sdoganato da USA ~{landed:.2f}€</span>')
        qty_est = max(1, int(alloc // r["current_price_eur"])) if r["current_price_eur"] > 0 else 1
        qty_warn = ' ⚠️ <span style="color:#fbbf24;">assume più slab identici disponibili insieme — verifica quante ne trovi davvero</span>' if qty_est > 1 else ""
        qty_html = f' &nbsp; <span style="color:#94a3b8;">→ {qty_est} pz.{qty_warn}</span>'
        st.markdown(f"""
        <div class="signal-card signal-card-buy">
            {img_tag}
            <div class="signal-card-body">
            <strong>{r['name']}</strong> &nbsp; <span style="color:#38bdf8; font-weight:600;">[{r.get('set_name') or '?'}]</span>
            &nbsp; <span style="color:#94a3b8;">{r['rarity']}</span>
            &nbsp;·&nbsp; {r['current_price_eur']:.2f}€ <span style="color:#fbbf24;">[Grade 9]</span> (PriceCharting) &nbsp;·&nbsp; sconto vs. pari {r['discount_pct']:+.0f}%
            &nbsp;·&nbsp; <span style="color:#94a3b8;">segnale da {start_str} ({r['months_in_signal']}m)</span>{max_price_html}{usa_import_html}
            <br><span style="font-family:'JetBrains Mono',monospace; font-size:15px; color:#f8fafc;">{alloc:,.0f}€</span>{qty_html}
            &nbsp; <a class="cm-btn" href="{link}" target="_blank">🛒 Verifica su Cardmarket</a>
            </div>
        </div>
        """, unsafe_allow_html=True)
        chart = build_price_chart(r["item_id"], r["name"], singles_prices_full)
        if chart is not None:
            st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False},
                             key=f"chart_single_{r['item_id']}")

    if len(singles_allocation) > 15:
        with st.expander(f"Altre {len(singles_allocation) - 15} carte nel quantile BUY"):
            rest_df = pd.DataFrame([
                {"Carta": r["name"], "Set": r.get("set_name") or "?", "Rarità": r["rarity"], "Grado": "Grade 9",
                 "Prezzo (€)": r["current_price_eur"], "Sconto vs. pari (%)": r["discount_pct"],
                 "Massimo totale (€)": r.get("max_edge_price_eur"),
                 "Segnale da": r["signal_start_date"].strftime("%Y-%m") if hasattr(r["signal_start_date"], "strftime") else str(r["signal_start_date"]),
                 "Allocazione (€)": alloc,
                 "Quantità": max(1, int(alloc // r["current_price_eur"])) if r["current_price_eur"] > 0 else 1}
                for r, alloc in singles_allocation[15:]
            ])
            st.dataframe(rest_df, use_container_width=True, hide_index=True,
                         column_config={
                             "Prezzo (€)": st.column_config.NumberColumn(format="%.2f €"),
                             "Sconto vs. pari (%)": st.column_config.NumberColumn(format="%+.1f%%"),
                             "Massimo totale (€)": st.column_config.NumberColumn(format="%.2f €"),
                             "Allocazione (€)": st.column_config.NumberColumn(format="%.0f €"),
                         })

    # --- RIPIEGO: alternative se non trovi le copie/carte sopra ---
    alt_rows, _ = get_singles_alternatives(singles_mode)
    if alt_rows:
        with st.expander(f"🔄 Non trovi una carta o le copie sopra? {len(alt_rows)} alternative nello stesso quantile"):
            st.caption("Ripiego, non un secondo BUY: usa il budget non speso qui invece di lasciarlo fermo o "
                       "forzare più copie di una carta — recupera parte dell'edge perso ma non tutto (⚠️ "
                       "**VERIFICATO**, `scripts/singles_diversify_when_capped_test.py`: Sharpe 0,30→0,80 col tetto "
                       "di 1 copia, MaxDD peggiora -13%→-21%, DSR 0,29 sotto soglia). Ancora nel quantile 20% più "
                       "sottovalutato, solo fuori dalle prime 60 per rank — nessun filtro di freschezza qui, "
                       "verifica sempre il grafico prezzo prima di comprare.")
            alt_df = pd.DataFrame([
                {"Carta": r["name"], "Set": r.get("set_name") or "?", "Rarità": r["rarity"], "Grado": "Grade 9",
                 "Prezzo (€)": r["current_price_eur"], "Sconto vs. pari (%)": r["discount_pct"],
                 "Massimo totale (€)": r.get("max_edge_price_eur")}
                for r in alt_rows[:60]
            ])
            st.dataframe(alt_df, use_container_width=True, hide_index=True,
                         column_config={
                             "Prezzo (€)": st.column_config.NumberColumn(format="%.2f €"),
                             "Sconto vs. pari (%)": st.column_config.NumberColumn(format="%+.1f%%"),
                             "Massimo totale (€)": st.column_config.NumberColumn(format="%.2f €"),
                         })

    # --- USCITE/AVOID: SINGOLE SOPRAVVALUTATE (specchio del BUY) ---
    avoid_rows, _ = get_singles_avoid_signal(singles_mode)
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
            img_tag = f'<img class="signal-card-thumb" src="{img_url}" />' if img_url else '<div class="signal-card-thumb"></div>'
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

    # --- BACKTEST + GIORNALE TRADE (box, singole, blend) — audit, non un'azione settimanale: chiuso di default ---
    res, n_universe = get_backtest_results()
    res_singles, n_universe_singles = get_singles_backtest_results(singles_mode)
    with st.expander(f"📈 Backtest 2020-2026 e giornale trade — {res.total_trades + res_singles.total_trades} trade chiusi in tutto"):
        common_idx = res.monthly_returns.index.intersection(res_singles.monthly_returns.index)
        blend_ret = 0.5 * res.monthly_returns.loc[common_idx] + 0.5 * res_singles.monthly_returns.loc[common_idx]
        blend_nav = 10000.0 * (1.0 + blend_ret).cumprod()

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3],
                             subplot_titles=("NAV (€, base 10.000€ per metà)", "Drawdown (%)"))
        fig.add_trace(go.Scatter(x=res.nav_history.index, y=res.nav_history["nav"], mode="lines", name="Box (TS Momentum)",
                                  line=dict(color="#38bdf8", width=1.5, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=res_singles.nav_history.index, y=res_singles.nav_history["nav"], mode="lines", name="Singole (Scarsità)",
                                  line=dict(color="#fbbf24", width=1.5, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=blend_nav.index, y=blend_nav.values, mode="lines", name="Blend 50/50",
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
        if singles_mode == "dac7":
            st.caption(f"⚠️ Riflette la **modalità DAC7** (singole: ribilanciamento 12m, max 20 posizioni) — Sharpe "
                       f"singole {res_singles.sharpe:.2f} (vs {VALIDATED_SINGLES['sharpe']:.2f} a turnover pieno). Le "
                       f"Metriche di validazione sopra mostrano sempre i numeri a turnover pieno.")

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
