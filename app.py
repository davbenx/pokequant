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
from poke_quant.data.liquidity_filter import liquid_sealed_ids
from poke_quant.data.price_fetcher import fetch_pricecharting_cover_image_url
from scripts.generate_singles_signal import compute_singles_signal_rows, compute_singles_avoid_rows, PRODUCTION_PARAMS as SINGLES_PARAMS

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
    "dsr_own_grid": 0.913, "dsr_full_session": 0.778, "n_trials_full_session": 47,
    "pbo": 0.286, "sharpe": 1.31, "cagr": 23.82, "max_dd": -10.62,
    "bootstrap_cagr_p_pos": 100, "bootstrap_sharpe_p_pos": 100,
    "h1_sharpe": 0.68, "h2_sharpe": 1.28,
}
VALIDATED_SINGLES = {
    # Specifica rifinita (eta' log + rango ordinale + controlli extra, ora i
    # default della classe - vedi scarcity_value_factor.py). pbo qui e' quello
    # della griglia di 6 varianti di specifica (8 split), non del grid rebal/quantile:
    # e' il numero corretto per come n_trials_full_session e' stato conteggiato.
    "dsr_own_grid": 0.999, "dsr_full_session": 0.980, "n_trials_full_session": 62,
    "pbo": 0.000, "sharpe": 2.02, "cagr": 32.64, "max_dd": -8.54,
    "h1_sharpe": 0.95, "h2_sharpe": 3.44,
}
VALIDATED_BLEND = {"sharpe": 2.27, "cagr": 28.75, "max_dd": -5.45}

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
    strat = TimeSeriesMomentumStrategy(prices_sub, lookback_months=12)
    bt = Backtester(strat, prices_sub, meta_sub, initial_cash=10000.0, platform="cardmarket",
                     apply_liquidity_slippage=True, apply_holding_cost=True)
    res = bt.run()
    return res, len(sealed_ids)


@st.cache_data(show_spinner=False, ttl=3600)
def get_singles_signal():
    rows, latest_date = compute_singles_signal_rows()
    return rows, latest_date.strftime("%Y-%m-%d")


@st.cache_data(show_spinner=False, ttl=3600)
def get_singles_avoid_signal():
    rows, latest_date = compute_singles_avoid_rows()
    return rows, latest_date.strftime("%Y-%m-%d")


@st.cache_data(show_spinner=False, ttl=3600)
def get_singles_prices_full():
    return load_price_matrix("historical_prices_graded_singles_grade9.csv")


@st.cache_data(show_spinner=False, ttl=3600)
def get_singles_backtest_results():
    metadata = load_metadata()
    prices_full = get_singles_prices_full()
    singles_ids = [
        k for k, v in metadata.items()
        if v.get("type") == "single" and v.get("data_quality") != "thin_unreliable" and k in prices_full.columns
    ]
    meta_sub = {k: v for k, v in metadata.items() if k in singles_ids}
    prices_sub = prices_full[singles_ids]
    strat = ScarcityValueFactorStrategy(rebalance_every_months=3, **SINGLES_PARAMS)
    bt = Backtester(strat, prices_sub, meta_sub, initial_cash=10000.0, platform="cardmarket",
                     apply_liquidity_slippage=True, apply_holding_cost=True)
    res = bt.run()
    return res, len(singles_ids)


@st.cache_data(show_spinner=False, ttl=7 * 24 * 3600)
def get_product_image(game_slug: str, item_slug: str) -> Optional[str]:
    """Immagine di copertina reale da PriceCharting, cache 7gg (l'immagine di un
    prodotto non cambia) - evita di rifare il fetch di rete a ogni refresh pagina."""
    if not game_slug or not item_slug:
        return None
    try:
        return fetch_pricecharting_cover_image_url(game_slug, item_slug)
    except Exception:
        return None


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
    n_verify = len(sig_rows) - n_buy - n_sell

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

    st.markdown("""
    <div style="background: rgba(15,23,42,0.5); border-radius: 8px; padding: 8px 14px; margin-bottom: 14px;
                font-size: 12px; color: #94a3b8; display: flex; gap: 18px; flex-wrap: wrap; align-items: center;">
        <strong style="color:#f1f5f9;">Come leggere i segnali:</strong>
        <span>🟢 <strong style="color:#10b981;">Verde</strong> = comprare/tenere (segnale validato)</span>
        <span>🔴 <strong style="color:#f43f5e;">Rosso</strong> = box: momentum invertito, non comprare/valuta vendita se lo possiedi ·
              singole: sopravvalutata vs pari (informativo, non una strategia di vendita testata a sé)</span>
    </div>
    """, unsafe_allow_html=True)

    # --- SIDEBAR: capitale ---
    with st.sidebar:
        st.markdown("### 💰 Capitale")
        capital = st.number_input("Capitale dedicato (€)", min_value=100.0, max_value=1_000_000.0,
                                   value=10000.0, step=500.0)
        st.caption("50% box sigillati, 50% singole (fattore scarsità) — le due strategie hanno "
                   "correlazione bassa (0,34): il blend porta Sharpe 1,53→2,27 e MaxDD -10,6%→-5,45% "
                   "rispetto al solo box. Cap 12% del capitale per singola posizione dentro ciascuna metà, "
                   "box pesato per età (0,4x sotto i 18 mesi, 1,0x dopo).")
        st.markdown("---")
        st.markdown("### 🇮🇹 Esecuzione dall'Italia")
        st.caption("1. Cardmarket — priorità assoluta (fee 5%, no dogana intra-UE)\n\n"
                   "2. eBay.it / eBay.de — box USA/JP con meno offerta su Cardmarket\n\n"
                   "3. TCGplayer — solo se il differenziale supera nettamente dogana+spedizione")
        st.markdown("---")
        st.caption("⚠️ Nessuna verifica di liquidità reale integrata. Controlla sempre il prezzo "
                   "reale su Cardmarket prima di comprare — il modello non sa se il box è disponibile.")

    # --- METRICHE VALIDATE (box, singole, blend) ---
    st.markdown('<div class="section-title">📊 Metriche di Validazione</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Numeri fissi da scripts/optimize_and_falsify.py e scripts/scarcity_value_singles_test.py — non ricalcolati a ogni refresh. Rivalidare ogni 6 mesi.</div>', unsafe_allow_html=True)
    st.warning(
        f"**DSR corretto per l'intera sessione**: box {VALIDATED_BOX['dsr_full_session']:.3f} (era {VALIDATED_BOX['dsr_own_grid']:.3f} "
        f"sulla sola griglia originale, {VALIDATED_BOX['n_trials_full_session']} trial totali) — sotto soglia 0,90-0,95. "
        f"Singole (fattore scarsità) {VALIDATED_SINGLES['dsr_full_session']:.3f} ({VALIDATED_SINGLES['n_trials_full_session']} trial totali) — "
        f"**sopra** la soglia, il primo candidato di tutta la ricerca a superarla. Vedi `scripts/dsr_session_audit.py` e "
        f"`scripts/scarcity_value_singles_test.py`."
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
        <div class="kpi-card"><div class="kpi-label">DSR (sessione intera)</div><div class="kpi-value">{VALIDATED_SINGLES['dsr_full_session']:.3f}</div><div class="kpi-sub kpi-sub-emerald">Sopra soglia · griglia propria: {VALIDATED_SINGLES['dsr_own_grid']:.3f}</div></div>
        <div class="kpi-card"><div class="kpi-label">Sharpe</div><div class="kpi-value">{VALIDATED_SINGLES['sharpe']:.2f}</div><div class="kpi-sub kpi-sub-emerald">CAGR +{VALIDATED_SINGLES['cagr']:.1f}%</div></div>
        <div class="kpi-card"><div class="kpi-label">PBO (8 split)</div><div class="kpi-value">{VALIDATED_SINGLES['pbo']*100:.1f}%</div><div class="kpi-sub kpi-sub-emerald">Molto stabile</div></div>
        <div class="kpi-card"><div class="kpi-label">Max Drawdown</div><div class="kpi-value">{VALIDATED_SINGLES['max_dd']:.1f}%</div></div>
        <div class="kpi-card"><div class="kpi-label">Walk-forward H1</div><div class="kpi-value">{VALIDATED_SINGLES['h1_sharpe']:.2f}</div><div class="kpi-sub kpi-sub-emerald">Sharpe 2021-01→2023-10</div></div>
        <div class="kpi-card"><div class="kpi-label">Walk-forward H2</div><div class="kpi-value">{VALIDATED_SINGLES['h2_sharpe']:.2f}</div><div class="kpi-sub kpi-sub-emerald">Sharpe 2023-11→2026-09</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="section-desc"><strong>🔗 Blend 50/50 — correlazione 0,34 tra le due strategie</strong></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card"><div class="kpi-label">Sharpe blend</div><div class="kpi-value">{VALIDATED_BLEND['sharpe']:.2f}</div><div class="kpi-sub kpi-sub-emerald">vs 1,53 box da solo (stesso periodo)</div></div>
        <div class="kpi-card"><div class="kpi-label">CAGR blend</div><div class="kpi-value">+{VALIDATED_BLEND['cagr']:.1f}%</div></div>
        <div class="kpi-card"><div class="kpi-label">Max Drawdown blend</div><div class="kpi-value">{VALIDATED_BLEND['max_dd']:.1f}%</div><div class="kpi-sub kpi-sub-emerald">vs -10,6% solo box</div></div>
    </div>
    """, unsafe_allow_html=True)

    # --- INDICI DI MERCATO (contesto, non segnale d'ingresso) ---
    st.markdown('<div class="section-title">📉 Indici di mercato</div>', unsafe_allow_html=True)
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
               "sotto il 50% è un segnale di regime, non di un singolo box — utile come contesto per capire "
               "se le uscite in corso sono isolate o parte di un raffreddamento generale.")

    # --- AZIONE: BUY/HOLD con allocazione e link Cardmarket (50% del capitale) ---
    st.markdown(f'<div class="section-title">📦 Box da comprare/mantenere — 50% del capitale ({capital*0.5:,.0f}€)</div>', unsafe_allow_html=True)
    st.caption("⚠️ Il prezzo mostrato viene da PriceCharting (mercato USA), convertito in EUR al tasso "
               "reale del mese — è il dato su cui il modello calcola il segnale, NON una quota Cardmarket. "
               "Il mercato europeo ha domanda/offerta propria: può differire, anche di molto. Il grafico "
               "mostra lo storico usato dal modello — confronta sempre col prezzo reale dietro al bottone.")
    buy_rows = [r for r in sig_rows if r["signal"] == "BUY/HOLD"]
    allocation = build_allocation(buy_rows, capital * 0.5, metadata, latest_date)

    if not allocation:
        st.info("Nessun segnale BUY/HOLD questo mese.")
    for r, alloc, w in allocation:
        meta = metadata.get(r["item_id"], {})
        link = get_cardmarket_deep_link(r["name"], franchise=meta.get("franchise", "pokemon"),
                                         language=meta.get("language", "en"))
        img_url = get_product_image(meta.get("game_slug"), meta.get("item_slug"))
        img_tag = f'<img class="signal-card-thumb" src="{img_url}" />' if img_url else '<div class="signal-card-thumb"></div>'
        st.markdown(f"""
        <div class="signal-card signal-card-buy">
            {img_tag}
            <div class="signal-card-body">
            <strong>{r['name']}</strong> &nbsp; <span style="color:#10b981;">+{r['trailing_12m_return_pct']:.0f}% (12m)</span>
            &nbsp;·&nbsp; {r['current_price_eur']:.0f}€ (PriceCharting) &nbsp;·&nbsp; peso età {w:.2f}
            <br><span style="font-family:'JetBrains Mono',monospace; font-size:15px; color:#f8fafc;">{alloc:,.0f}€</span>
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
    st.caption("⚠️ Da comprare: la carta GIÀ GRADATA Grade 9 (uno slab, non la carta raw, non PSA10) — il "
               "fattore lavora solo su questa serie di PriceCharting, non confronta mai tra gradi diversi. "
               "Confronta sempre col prezzo reale su Cardmarket e verifica il grado dell'inserzione a mano: "
               "il link di ricerca è testuale, non un filtro reale per grado. \"Sconto vs. pari\" è quanto la "
               "carta costa in meno (%) rispetto a quanto la sua rarità/età/set implicherebbero rispetto alle "
               "sue pari — più negativo, più sottovalutata secondo il modello. \"Segnale da\" è da quanti mesi "
               "consecutivi la carta è nel quantile BUY: solo le carte entrate negli ultimi 3 mesi (la cadenza "
               "di ribilanciamento validata nel backtest) sono mostrate — oltre, comprarla oggi non è ciò che "
               "è stato testato, è un possibile *value trap* (sconto persistente che il mercato non corregge). "
               "Prime 15 con grafico, le altre in tabella sotto.")
    singles_rows, singles_latest_date = get_singles_signal()
    singles_prices_full = get_singles_prices_full()
    singles_allocation = build_equal_allocation(singles_rows, capital * 0.5)

    if not singles_allocation:
        st.info("Nessuna carta nel quantile BUY questo mese.")
    for r, alloc in singles_allocation[:15]:
        meta = {"franchise": r.get("franchise", "pokemon"), "language": r.get("language", "en")}
        link = get_cardmarket_deep_link(r["name"], franchise=meta["franchise"], language=meta["language"], item_type="single")
        start = r["signal_start_date"]
        start_str = start.strftime("%Y-%m") if hasattr(start, "strftime") else str(start)
        full_meta = metadata.get(r["item_id"], {})
        img_url = get_product_image(full_meta.get("game_slug"), full_meta.get("item_slug"))
        img_tag = f'<img class="signal-card-thumb" src="{img_url}" />' if img_url else '<div class="signal-card-thumb"></div>'
        st.markdown(f"""
        <div class="signal-card signal-card-buy">
            {img_tag}
            <div class="signal-card-body">
            <strong>{r['name']}</strong> &nbsp; <span style="color:#94a3b8;">{r['rarity']}</span>
            &nbsp;·&nbsp; {r['current_price_eur']:.2f}€ <span style="color:#fbbf24;">[Grade 9]</span> (PriceCharting) &nbsp;·&nbsp; sconto vs. pari {r['discount_pct']:+.0f}%
            &nbsp;·&nbsp; <span style="color:#94a3b8;">segnale da {start_str} ({r['months_in_signal']}m)</span>
            <br><span style="font-family:'JetBrains Mono',monospace; font-size:15px; color:#f8fafc;">{alloc:,.0f}€</span>
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
                {"Carta": r["name"], "Rarità": r["rarity"], "Grado": "Grade 9",
                 "Prezzo (€)": r["current_price_eur"], "Sconto vs. pari (%)": r["discount_pct"],
                 "Segnale da": r["signal_start_date"].strftime("%Y-%m") if hasattr(r["signal_start_date"], "strftime") else str(r["signal_start_date"]),
                 "Allocazione (€)": alloc}
                for r, alloc in singles_allocation[15:]
            ])
            st.dataframe(rest_df, use_container_width=True, hide_index=True,
                         column_config={
                             "Prezzo (€)": st.column_config.NumberColumn(format="%.2f €"),
                             "Sconto vs. pari (%)": st.column_config.NumberColumn(format="%+.1f%%"),
                             "Allocazione (€)": st.column_config.NumberColumn(format="%.0f €"),
                         })

    # --- USCITE/AVOID: SINGOLE SOPRAVVALUTATE (specchio del BUY) ---
    avoid_rows, _ = get_singles_avoid_signal()
    if avoid_rows:
        st.markdown('<div class="section-title">🔴 Singole da evitare/vendere — sopravvalutate vs pari</div>', unsafe_allow_html=True)
        st.caption("⚠️ Specchio del quantile BUY (stesso modello, residuo più positivo): la carta costa più di "
                   "quanto la sua rarità/età/set implicherebbero rispetto alle pari. Molte di queste sono chase "
                   "iconiche (Charizard, Lugia, carte ★) — il modello non cattura il premio da fama/desiderabilità, "
                   "solo rarità/età/franchise, quindi un sovrapprezzo enorme spesso riflette un premio reale, non "
                   "un errore di prezzo. A differenza del quantile BUY, qui NON è stato validato un backtest di "
                   "vendita/short — è informativo (come le Uscite dei box), non una strategia a sé testata.")
        for r in avoid_rows[:15]:
            full_meta = metadata.get(r["item_id"], {})
            img_url = get_product_image(full_meta.get("game_slug"), full_meta.get("item_slug"))
            img_tag = f'<img class="signal-card-thumb" src="{img_url}" />' if img_url else '<div class="signal-card-thumb"></div>'
            st.markdown(f"""
            <div class="signal-card signal-card-sell">
                {img_tag}
                <div class="signal-card-body">
                <strong>{r['name']}</strong> &nbsp; <span style="color:#94a3b8;">{r['rarity']}</span>
                &nbsp;·&nbsp; {r['current_price_eur']:.2f}€ <span style="color:#fbbf24;">[Grade 9]</span> (PriceCharting)
                &nbsp;·&nbsp; sovrapprezzo vs. pari {r['discount_pct']:+.0f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        if len(avoid_rows) > 15:
            with st.expander(f"Altre {len(avoid_rows) - 15} carte sopravvalutate"):
                avoid_df = pd.DataFrame([
                    {"Carta": r["name"], "Rarità": r["rarity"], "Grado": "Grade 9",
                     "Prezzo (€)": r["current_price_eur"], "Sovrapprezzo vs. pari (%)": r["discount_pct"]}
                    for r in avoid_rows[15:]
                ])
                st.dataframe(avoid_df, use_container_width=True, hide_index=True,
                             column_config={
                                 "Prezzo (€)": st.column_config.NumberColumn(format="%.2f €"),
                                 "Sovrapprezzo vs. pari (%)": st.column_config.NumberColumn(format="%+.1f%%"),
                             })

    # --- EQUITY CURVE (box, singole, blend) ---
    st.markdown('<div class="section-title">📈 Backtest 2020-2026 — Box, Singole, Blend</div>', unsafe_allow_html=True)
    res, n_universe = get_backtest_results()
    res_singles, n_universe_singles = get_singles_backtest_results()

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
               "10.000€ propri, poi combinata come media dei rendimenti mensili (equivalente a un ribilanciamento "
               "50/50 mensile) — frizioni reali incluse in entrambe (Cardmarket 5%+0,60€, spedizione, slippage, "
               "costo di custodia).")

    # --- GIORNALE DEI TRADE CHIUSI (BOX) ---
    st.markdown('<div class="section-title">📜 Giornale dei trade chiusi — Box (backtest)</div>', unsafe_allow_html=True)
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
        st.caption("P&L e ROI sono netti di commissioni Cardmarket (5%+0,60€), spedizione e costo di custodia — "
                   "vedi la sezione Metriche di Validazione per CAGR/Sharpe/MaxDD aggregati sull'intero backtest.")

    # --- GIORNALE DEI TRADE CHIUSI (SINGOLE) ---
    st.markdown('<div class="section-title">📜 Giornale dei trade chiusi — Singole (backtest)</div>', unsafe_allow_html=True)
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
        st.caption("P&L e ROI sono netti di commissioni Cardmarket (5%+0,60€), spedizione e costo di custodia — "
                   "vedi la sezione Metriche di Validazione per CAGR/Sharpe/MaxDD aggregati sull'intero backtest.")

    st.markdown("---")
    st.caption("PokeQuant · Blend box+singole scelto per correlazione bassa (0,34), non per rendimento massimo · "
                "box sotto soglia istituzionale dopo l'audit sull'intera sessione, singole sopra (vedi avviso in alto) · "
                "[Runbook Italia](https://github.com/davbenx/pokequant/blob/main/OPERATIONS_ITALIA.md) · "
                "Rivalidare con `scripts/optimize_and_falsify.py` e `scripts/scarcity_value_singles_test.py` ogni 6 mesi.")


if __name__ == "__main__":
    main()
