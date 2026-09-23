"""
app.py — PokeQuant: Dashboard della strategia validata (TS Momentum, box sigillati).

Mostra SOLO cio' che ha superato la validazione istituzionale di questa sessione
(DSR 0,913, PBO 28,6%, bootstrap, walk-forward H1/H2 senza inversione di segno - vedi
scripts/optimize_and_falsify.py). Tutto il resto (Slabs Radar, desk discrezionale a
tier, OptimalSealedStrategy/SealedAccumulatorStrategy/ChaseDipBuyerStrategy, l'audit
a slider) e' stato rimosso per direttiva esplicita - "tieni solo quello che abbiamo
validato" - non e' sparito: resta nel codice/git history per ricerca futura, solo non
piu' mostrato come se fosse pronto per capitale reale.

Azionabilita' per l'Italia: link diretti a Cardmarket (mercato primario per chi opera
dall'Italia, vedi OPERATIONS_ITALIA.md) su ogni posizione BUY/HOLD.
"""

from __future__ import annotations
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))

from poke_quant.data.storage import load_price_matrix, load_metadata
from poke_quant.data.cardmarket_bridge import get_cardmarket_deep_link
from poke_quant.engine.backtester import Backtester
from poke_quant.engine.strategies.time_series_momentum import TimeSeriesMomentumStrategy
from poke_quant.engine.position_sizing import age_weight
from scripts.generate_monthly_signal import compute_signal_rows, MODERN_ERA_CUTOFF

# =============================================================================
# NUMERI VALIDATI (scripts/optimize_and_falsify.py, lookback=12m, n_trials=5)
# Fissi, non ricalcolati a ogni caricamento pagina - una strategia si rivalida
# ogni 6 mesi (vedi OPERATIONS_ITALIA.md), non ogni refresh del browser.
# =============================================================================
VALIDATED = {
    "dsr": 0.913, "pbo": 0.286, "sharpe": 1.10, "cagr": 23.54, "max_dd": -13.40,
    "bootstrap_cagr_p_pos": 100, "bootstrap_sharpe_p_pos": 100,
    "h1_sharpe": -0.10, "h2_sharpe": 1.29,
}

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
.signal-card { background: rgba(15,23,42,0.65); border-radius: 10px; padding: 12px 14px; margin-bottom: 8px; border-left: 3px solid; border-top: 1px solid rgba(255,255,255,0.05); border-right: 1px solid rgba(255,255,255,0.05); border-bottom: 1px solid rgba(255,255,255,0.05); }
.signal-card-buy { border-left-color: #10b981; }
.signal-card-sell { border-left-color: #f43f5e; }
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
def get_backtest_equity_curve():
    metadata = load_metadata()
    prices_full = load_price_matrix()
    sealed_ids = [
        k for k, v in metadata.items()
        if v.get("type") == "sealed" and v.get("data_quality") != "thin_unreliable"
        and k in prices_full.columns and v.get("release_date") and v["release_date"] >= MODERN_ERA_CUTOFF
    ]
    meta_sub = {k: v for k, v in metadata.items() if k in sealed_ids}
    prices_sub = prices_full[sealed_ids]
    strat = TimeSeriesMomentumStrategy(prices_sub, lookback_months=12)
    bt = Backtester(strat, prices_sub, meta_sub, initial_cash=10000.0, platform="cardmarket",
                     apply_liquidity_slippage=True, apply_holding_cost=True)
    res = bt.run()
    return res.nav_history, len(sealed_ids)


def build_allocation(buy_rows: list, capital: float, metadata: dict, latest_date: str):
    latest_dt = pd.to_datetime(latest_date)
    weighted = []
    for r in buy_rows:
        rel_dt = metadata.get(r["item_id"], {}).get("release_date")
        age_m = None
        if rel_dt:
            rd = pd.to_datetime(rel_dt)
            age_m = (latest_dt.year - rd.year) * 12 + (latest_dt.month - rd.month)
        weighted.append((r, age_weight(age_m)))
    total_w = sum(w for _, w in weighted) or 1.0
    return [(r, capital * (w / total_w), w) for r, w in sorted(weighted, key=lambda x: -x[1])]


def main():
    metadata = load_metadata()
    sig_rows, latest_date = get_signal()
    n_buy = sum(1 for r in sig_rows if r["signal"] == "BUY/HOLD")
    n_sell = sum(1 for r in sig_rows if r["signal"] == "AVOID/SELL")
    n_verify = len(sig_rows) - n_buy - n_sell

    st.markdown(f"""
    <div class="nav-header">
        <div>
            <span class="nav-title">⚡ PokeQuant</span>
            <span style="color:#64748b; font-size:12px; margin-left:8px;">TS Momentum · Box Sigillati Era Moderna (2019+)</span>
        </div>
        <div>
            <span class="pill-tag pill-emerald">✅ Validato (DSR {VALIDATED['dsr']:.3f})</span>
            <span class="pill-tag pill-blue">Segnale {latest_date[:7]}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- SIDEBAR: capitale ---
    with st.sidebar:
        st.markdown("### 💰 Capitale")
        capital = st.number_input("Capitale dedicato (€)", min_value=100.0, max_value=1_000_000.0,
                                   value=10000.0, step=500.0)
        st.caption("Cap 12% del capitale per singola posizione, pesato per età (0,4x sotto i 18 mesi "
                   "dalla release, 1,0x dopo — vedi poke_quant/engine/position_sizing.py).")
        st.markdown("---")
        st.markdown("### 🇮🇹 Esecuzione dall'Italia")
        st.caption("1. Cardmarket — priorità assoluta (fee 5%, no dogana intra-UE)\n\n"
                   "2. eBay.it / eBay.de — box USA/JP con meno offerta su Cardmarket\n\n"
                   "3. TCGplayer — solo se il differenziale supera nettamente dogana+spedizione")
        st.markdown("---")
        st.caption("⚠️ Nessuna verifica di liquidità reale integrata. Controlla sempre il prezzo "
                   "reale su Cardmarket prima di comprare — il modello non sa se il box è disponibile.")

    # --- METRICHE VALIDATE ---
    st.markdown('<div class="section-title">📊 Metriche di Validazione</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Numeri fissi da scripts/optimize_and_falsify.py — non ricalcolati a ogni refresh. Rivalidare ogni 6 mesi.</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card"><div class="kpi-label">DSR</div><div class="kpi-value">{VALIDATED['dsr']:.3f}</div><div class="kpi-sub kpi-sub-emerald">Soglia istituzionale 0,95</div></div>
        <div class="kpi-card"><div class="kpi-label">Sharpe</div><div class="kpi-value">{VALIDATED['sharpe']:.2f}</div><div class="kpi-sub kpi-sub-emerald">CAGR +{VALIDATED['cagr']:.1f}%</div></div>
        <div class="kpi-card"><div class="kpi-label">PBO (8 split)</div><div class="kpi-value">{VALIDATED['pbo']*100:.1f}%</div><div class="kpi-sub kpi-sub-amber">Sopra fascia comfort (&lt;20-25%)</div></div>
        <div class="kpi-card"><div class="kpi-label">Max Drawdown</div><div class="kpi-value">{VALIDATED['max_dd']:.1f}%</div><div class="kpi-sub kpi-sub-emerald">Bootstrap P(&gt;0)={VALIDATED['bootstrap_cagr_p_pos']}%</div></div>
        <div class="kpi-card"><div class="kpi-label">Walk-forward H1</div><div class="kpi-value">{VALIDATED['h1_sharpe']:.2f}</div><div class="kpi-sub kpi-sub-amber">Sharpe 2020-12→2023-10</div></div>
        <div class="kpi-card"><div class="kpi-label">Walk-forward H2</div><div class="kpi-value">{VALIDATED['h2_sharpe']:.2f}</div><div class="kpi-sub kpi-sub-emerald">Sharpe 2023-11→2026-09, nessuna inversione di segno</div></div>
    </div>
    """, unsafe_allow_html=True)

    # --- AZIONE: BUY/HOLD con allocazione e link Cardmarket ---
    st.markdown('<div class="section-title">🟢 Posizioni da aprire/mantenere</div>', unsafe_allow_html=True)
    buy_rows = [r for r in sig_rows if r["signal"] == "BUY/HOLD"]
    allocation = build_allocation(buy_rows, capital, metadata, latest_date)

    if not allocation:
        st.info("Nessun segnale BUY/HOLD questo mese.")
    for r, alloc, w in allocation:
        meta = metadata.get(r["item_id"], {})
        link = get_cardmarket_deep_link(r["name"], franchise=meta.get("franchise", "pokemon"),
                                         language=meta.get("language", "en"))
        st.markdown(f"""
        <div class="signal-card signal-card-buy">
            <strong>{r['name']}</strong> &nbsp; <span style="color:#10b981;">+{r['trailing_12m_return_pct']:.0f}% (12m)</span>
            &nbsp;·&nbsp; {r['current_price_eur']:.0f}€ &nbsp;·&nbsp; peso età {w:.2f}
            <br><span style="font-family:'JetBrains Mono',monospace; font-size:15px; color:#f8fafc;">{alloc:,.0f}€</span>
            &nbsp; <a class="cm-btn" href="{link}" target="_blank">🛒 Verifica su Cardmarket</a>
        </div>
        """, unsafe_allow_html=True)

    # --- ROTAZIONE: AVOID/SELL ---
    sell_rows = [r for r in sig_rows if r["signal"] == "AVOID/SELL"]
    if sell_rows:
        st.markdown('<div class="section-title">🔴 Uscite (momentum invertito)</div>', unsafe_allow_html=True)
        for r in sell_rows:
            st.markdown(f"""
            <div class="signal-card signal-card-sell">
                <strong>{r['name']}</strong> &nbsp; <span style="color:#f43f5e;">{r['trailing_12m_return_pct']:.0f}% (12m)</span>
                &nbsp;·&nbsp; {r['current_price_eur']:.0f}€
            </div>
            """, unsafe_allow_html=True)

    if n_verify:
        with st.expander(f"⚠️ Da verificare a mano ({n_verify}) — rendimento implausibile, mercato troppo sottile"):
            for r in sig_rows:
                if "VERIFICARE" in r["signal"]:
                    st.markdown(f"- **{r['name']}** — {r['trailing_12m_return_pct']:+.0f}% (12m), {r['current_price_eur']:.0f}€")

    # --- EQUITY CURVE ---
    st.markdown('<div class="section-title">📈 Backtest 2020-2026</div>', unsafe_allow_html=True)
    nav_df, n_universe = get_backtest_equity_curve()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3],
                         subplot_titles=("NAV (€)", "Drawdown (%)"))
    fig.add_trace(go.Scatter(x=nav_df.index, y=nav_df["nav"], mode="lines", name="TS Momentum",
                              line=dict(color="#38bdf8", width=2.5)), row=1, col=1)
    peak = nav_df["nav"].cummax()
    dd = (nav_df["nav"] - peak) / peak * 100.0
    fig.add_trace(go.Scatter(x=nav_df.index, y=dd, mode="lines", fill="tozeroy", name="Drawdown",
                              line=dict(color="#f43f5e", width=1), fillcolor="rgba(244,63,94,0.15)"), row=2, col=1)
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(15,23,42,0.4)", plot_bgcolor="rgba(15,23,42,0.4)",
                       height=380, margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Universo: {n_universe} box/ETB era 2019+. Capitale iniziale 10.000€, frizioni reali incluse "
               "(Cardmarket 5%+0,60€, spedizione, slippage, costo di custodia).")

    st.markdown("---")
    st.caption("PokeQuant · Solo strategie validate a livello istituzionale · "
                "[Runbook Italia](https://github.com/davbenx/pokequant/blob/main/OPERATIONS_ITALIA.md) · "
                "Rivalidare con `scripts/optimize_and_falsify.py` ogni 6 mesi.")


if __name__ == "__main__":
    main()
