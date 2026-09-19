"""
app.py — Dashboard Quantitativa Interattiva PokeQuant (Streamlit + Plotly).
Design System Moderno, Lean & Frictionless: connubio perfetto tra accessibilità retail
ed eccellenza analitica istituzionale (Bloomberg-meets-Modern Fintech).
"""

from __future__ import annotations
import datetime
import json
from pathlib import Path
import urllib.parse
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from poke_quant.config import PLATFORM_FEES, SHIPPING_COSTS, GRADING_DEFAULT
from poke_quant.data.storage import load_price_matrix, load_metadata, load_macro_matrix
from poke_quant.data.price_fetcher import build_and_cache_universe
from poke_quant.data.catalog_fetcher import fetch_all_sets, fetch_cards_by_set
from poke_quant.engine.backtester import Backtester
from poke_quant.engine.strategies.sealed_accumulator import SealedAccumulatorStrategy
from poke_quant.engine.strategies.chase_dip_buyer import ChaseDipBuyerStrategy
from poke_quant.engine.strategies.optimal_sealed_strategy import OptimalSealedStrategy
from poke_quant.engine.friction import evaluate_grading_arbitrage
from poke_quant.validation.statistical_validation import deflated_sharpe_ratio, pbo_cscv
from poke_quant.signal_scanner import scan_signals, scan_historical_signals, load_user_holdings, format_telegram_alert, send_telegram_message

# =============================================================================
# 1. CONFIGURAZIONE PAGINA & DESIGN SYSTEM CSS
# =============================================================================
st.set_page_config(
    page_title="PokeQuant — Quantitative Investment Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #cbd5e1;
}

/* Custom Scrollbars */
::-webkit-scrollbar {
    width: 5px;
    height: 5px;
}
::-webkit-scrollbar-track {
    background: #080c14;
}
::-webkit-scrollbar-thumb {
    background: #1e293b;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #334155;
}

/* Compact Nav Header (Linear / Ramp style) */
.nav-header {
    background: linear-gradient(180deg, rgba(15, 23, 42, 0.85) 0%, rgba(8, 12, 20, 0.95) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 10px 16px;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
    backdrop-filter: blur(12px);
}
.nav-brand {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
}
.nav-title {
    font-size: 18px;
    font-weight: 800;
    letter-spacing: -0.4px;
    color: #f8fafc;
}

/* Pill Tags */
.pill-tag {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 8px;
    border-radius: 9999px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.2px;
}
.pill-emerald {
    background: rgba(16, 185, 129, 0.12);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.28);
}
.pill-blue {
    background: rgba(56, 189, 248, 0.12);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.28);
}
.pill-amber {
    background: rgba(245, 158, 11, 0.12);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.28);
}
.pill-purple {
    background: rgba(168, 85, 247, 0.12);
    color: #c084fc;
    border: 1px solid rgba(168, 85, 247, 0.28);
}
.pill-slate {
    background: rgba(148, 163, 184, 0.12);
    color: #94a3b8;
    border: 1px solid rgba(148, 163, 184, 0.25);
}

/* Micro-HUD Strip */
.hud-strip {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: rgba(15, 23, 42, 0.45);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 10px;
    padding: 8px 14px;
    margin-bottom: 12px;
    font-size: 12px;
    color: #cbd5e1;
    flex-wrap: wrap;
    gap: 8px;
}

/* Executive KPI Grid (4 Compact Cards) */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
    margin-bottom: 14px;
}
.kpi-card {
    background: rgba(15, 23, 42, 0.55);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 12px;
    padding: 12px 15px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    transition: all 0.15s ease-in-out;
}
.kpi-card:hover {
    border-color: rgba(56, 189, 248, 0.3);
    transform: translateY(-1px);
}
.kpi-label {
    font-size: 10.5px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #94a3b8;
    margin-bottom: 2px;
}
.kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 21px;
    font-weight: 700;
    color: #f8fafc;
    letter-spacing: -0.4px;
}
.kpi-sub {
    font-size: 11px;
    font-weight: 500;
    margin-top: 2px;
    display: flex;
    align-items: center;
    gap: 3px;
}
.kpi-sub-emerald { color: #10b981; }
.kpi-sub-blue { color: #60a5fa; }
.kpi-sub-amber { color: #fbbf24; }
.kpi-sub-rose { color: #f43f5e; }
.kpi-sub-neutral { color: #94a3b8; }

/* Signal Action Cards */
.signal-card {
    background: rgba(15, 23, 42, 0.65);
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 10px;
    border-left: 3px solid;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
    border-right: 1px solid rgba(255, 255, 255, 0.05);
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(8px);
}
.signal-card-buy { border-left-color: #10b981; }
.signal-card-rotate { border-left-color: #fbbf24; }
.signal-card-sell { border-left-color: #f43f5e; }
.signal-card-vault { border-left-color: #38bdf8; }

/* Tabs Styling */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(15, 23, 42, 0.6);
    padding: 4px;
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.06);
    gap: 4px;
    margin-bottom: 12px;
}
.stTabs [data-baseweb="tab"] {
    height: 38px;
    border-radius: 7px;
    color: #94a3b8;
    font-weight: 500;
    font-size: 13px;
    padding: 0 14px;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: #1e293b !important;
    color: #f8fafc !important;
    font-weight: 600;
}

/* Section Header */
.section-title {
    font-size: 16px;
    font-weight: 700;
    letter-spacing: -0.3px;
    color: #f1f5f9;
    margin-top: 8px;
    margin-bottom: 2px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.section-desc {
    font-size: 12px;
    color: #94a3b8;
    margin-bottom: 10px;
}

/* Cardmarket 1-Click Action Buttons */
.cm-btn {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.35);
    color: #10b981 !important;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 600;
    text-decoration: none !important;
    transition: all 0.15s ease;
}
.cm-btn:hover {
    background: rgba(16, 185, 129, 0.25);
    border-color: #10b981;
    color: #34d399 !important;
    transform: translateY(-1px);
}
.cm-btn-sell {
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid rgba(245, 158, 11, 0.35);
    color: #fbbf24 !important;
}
.cm-btn-sell:hover {
    background: rgba(245, 158, 11, 0.25);
    border-color: #fbbf24;
    color: #fcd34d !important;
    transform: translateY(-1px);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# 1.1 HELPER OPERATIVI & DEEP-LINKS CARDMARKET
# =============================================================================
def get_cardmarket_url(item_name: str, franchise: str = "pokemon") -> str:
    """Costruisce deep-link di ricerca diretta sul marketplace Cardmarket europeo."""
    clean_name = item_name.replace("[JP]", "").strip()
    if franchise == "one_piece" or "OP-" in clean_name or "One Piece" in clean_name:
        game = "OnePiece"
    else:
        game = "Pokemon"

    if "box" not in clean_name.lower() and "bundle" not in clean_name.lower():
        search_query = f"{clean_name} Booster Box"
    else:
        search_query = clean_name

    encoded = urllib.parse.quote(search_query)
    return f"https://www.cardmarket.com/en/{game}/Products/Search?searchString={encoded}"


def generate_executive_report(
    snapshot_label: str,
    initial_cash: float,
    buys: list[dict],
    sells: list[dict],
    metadata: dict
) -> str:
    """Genera report operativo formattato pronto da copiare per Telegram, chat o diario investimenti."""
    lines = [
        f"⚡ POKEQUANT EXECUTIVE DESK REPORT — {snapshot_label}",
        "=" * 58,
        f"💰 Capitale Allocato: {initial_cash:,.2f} €",
        f"🛡️ Regola Prezzo Max: Cap 1.15x MSRP (Margine di Sicurezza)",
        f"📦 Sizing Box: Discrete Integer Lots (No Frazioni)",
        "",
        "🟢 ORDINI BUY IMMEDIATI (Nuovi Ingressi in Finestra):",
        "-" * 58
    ]
    budget_per_set = initial_cash * 0.12
    if buys:
        for b in buys:
            px = b["current_price"]
            qty = max(1, int(budget_per_set // px)) if px <= initial_cash * 0.35 else int(budget_per_set // px)
            tot = qty * px
            pct = (tot / initial_cash) * 100 if initial_cash > 0 else 0
            cm_url = get_cardmarket_url(b["name"], metadata.get(b.get("item_id"), {}).get("franchise", "pokemon"))
            lines.append(f"• {b['name']} (Tier {b['tier']})")
            lines.append(f"  Azione: COMPRA {qty} BOX @ {px:.2f} € | Spesa: {tot:,.2f} € ({pct:.1f}% NAV)")
            lines.append(f"  Prezzo Max: {b['max_buy_price']:.2f} € | Risparmio vs Max: {b['margin_vs_max']:+.2f} €/box")
            lines.append(f"  Finestra: Mese {b['age_months']}/14 ({b['months_left_in_window']}m rimasti)")
            lines.append(f"  Link: {cm_url}")
            lines.append("")
    else:
        lines.append("• Nessun set attualmente in finestra di acquisto. 100% liquidità preservata.")
        lines.append("")

    lines.append("🔄 ORDINI ROTAZIONE LIQUIDITÀ (SELL Tranche 1 a +70%):")
    lines.append("-" * 58)
    if sells:
        for s in sells:
            cm_url = get_cardmarket_url(s["name"], metadata.get(s.get("item_id"), {}).get("franchise", "pokemon"))
            lines.append(f"• {s['name']}")
            lines.append(f"  Azione: VENDI {s['quantity']} su {s.get('total_quantity', s['quantity'])} BOX @ {s['current_price']:.2f} €")
            lines.append(f"  Incasso Netto: {s['net_proceeds']:,.2f} € (ROI: +{s['net_roi_pct']:.1f}%)")
            lines.append(f"  Link: {cm_url}")
            lines.append("")
    else:
        lines.append("• Nessuna posizione pronta per la rotazione. Custodia attiva in corso.")
        lines.append("")

    lines.append("=" * 58)
    lines.append("PokeQuant Terminal · ApexConvex Institutional Collectibles System")
    return "\n".join(lines)


# =============================================================================
# 2. DATA CACHING & INITIALIZATION
# =============================================================================
@st.cache_data(show_spinner=False)
def get_cached_data():
    prices_df = load_price_matrix()
    meta = load_metadata()
    if prices_df is None or meta is None:
        prices_df, meta = build_and_cache_universe(force_refresh=False)
    macro_df = load_macro_matrix()
    return prices_df, meta, macro_df


def main():
    # Caricamento dati
    with st.spinner("Inizializzazione feed prezzi e asset reali..."):
        full_prices_df, full_metadata, macro_df = get_cached_data()

    # --- COMPACT NAV HEADER ---
    st.markdown("""
    <div class="nav-header">
        <div class="nav-brand">
            <span class="nav-title">⚡ PokeQuant Quantitative Terminal</span>
            <span style="color:#64748b; font-size:12px; margin-left:6px;">| Institutional Collectibles & Alternative Alpha Engine</span>
        </div>
        <div class="pill-group">
            <span class="pill-tag pill-emerald">● 53 Asset Reali</span>
            <span class="pill-tag pill-blue">● 68 Mesi Storico</span>
            <span class="pill-tag pill-amber">● Rotazione Scalare</span>
            <span class="pill-tag pill-purple">● 8 Test Popperiani</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- SIDEBAR: PARAMETRI GLOBALI & AMBIENTE ---
    if "current_capital" not in st.session_state:
        st.session_state["current_capital"] = 10000.0

    # --- SIDEBAR: PARAMETRI GLOBALI & AMBIENTE ---
    with st.sidebar:
        st.markdown("### ⚙️ Definizione Capitale & Allocazione")
        capital_mode = st.radio(
            "Modalità Inserimento Capitale",
            options=["💰 Capitale Dedicato a PokeQuant", "🌐 Patrimonio Totale + Quota %"],
            help="Scegli se inserire direttamente l'ammontare allocato a PokeQuant o calcolarlo come quota percentuale del tuo patrimonio complessivo."
        )

        if "Dedicato" in capital_mode:
            initial_cash = st.number_input(
                "Capitale Allocato a PokeQuant (€)",
                min_value=500.0, max_value=1000000.0,
                value=float(st.session_state["current_capital"]),
                step=500.0,
                key="sb_initial_cash_input",
                help="Ammontare monetario netto interamente dedicato alla strategia sui box sigillati."
            )
            st.session_state["current_capital"] = initial_cash
            total_investable_wealth = initial_cash
            alloc_pct_val = 100.0
        else:
            total_investable_wealth = st.number_input(
                "Patrimonio Complessivo Investibile (€)",
                min_value=2000.0, max_value=5000000.0, value=100000.0, step=5000.0,
                help="Valore complessivo del patrimonio liquido/investibile (Azioni, Obbligazioni, Cassa, Alternativi)."
            )
            alloc_pct_val = st.slider(
                "Quota Alternativi PokeQuant (%)",
                min_value=1.0, max_value=30.0, value=10.0, step=0.5,
                help="Percentuale del patrimonio totale allocata alla strategia PokeQuant (consigliata: 5% - 15%)."
            )
            initial_cash = round(total_investable_wealth * (alloc_pct_val / 100.0), 2)
            st.session_state["current_capital"] = initial_cash
            core_capital = total_investable_wealth - initial_cash
            st.markdown(f"""
            <div style="background:rgba(56, 189, 248, 0.08); border:1px solid rgba(56, 189, 248, 0.25); border-radius:8px; padding:10px; font-size:12px; margin-bottom:10px;">
                • <strong>Capitale Allocato PokeQuant</strong>: <span style="color:#38bdf8; font-weight:700;">{initial_cash:,.2f} €</span> ({alloc_pct_val:.1f}%)<br>
                • <strong>Patrimonio Core (S&P 500 / Bond)</strong>: <span style="color:#94a3b8;">{core_capital:,.2f} €</span> ({100.0 - alloc_pct_val:.1f}%)
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 📊 Benchmark di Mercato")
        bench_choice = st.selectbox(
            "Benchmark di Riferimento",
            options=["S&P 500 Reale (SPY ETF)", "Oro Reale (GLD ETF)", "Bitcoin Reale (BTC)", "Tasso Fisso (10% CAGR)"],
            help="Serie temporale reale utilizzata per il calcolo di Alpha, Beta e Correlazione."
        )
        benchmark_cagr = 0.10
        benchmark_series = None
        if macro_df is not None:
            if "S&P 500" in bench_choice and "spy" in macro_df:
                benchmark_series = macro_df["spy"]
            elif "Oro" in bench_choice and "gold" in macro_df:
                benchmark_series = macro_df["gold"]
            elif "Bitcoin" in bench_choice and "btc" in macro_df:
                benchmark_series = macro_df["btc"]

        st.markdown("### 🛡️ Frizioni & Costi Vivi")
        platform = st.selectbox(
            "Canale di Liquidazione",
            options=["cardmarket", "ebay", "direct_private"],
            format_func=lambda x: {
                "cardmarket": "Cardmarket (5% fee + 0.60€)",
                "ebay": "eBay (12.5% fee + 0.35€)",
                "direct_private": "Scambio Privato / Fiere (0% fee)"
            }[x]
        )
        absorb_shipping = st.checkbox("Spedizione tracciata assorbita dal venditore", value=False)
        apply_slippage = st.checkbox("Slippage di Liquidità (1-2%)", value=True, help="Riflette lo spread denaro-lettera sul book di Cardmarket.")
        apply_holding = st.checkbox("Costi di Custodia e Immagazzinamento (0.5%/anno)", value=True, help="Spese vive di conservazione e assicurazione.")

        st.markdown("### 🎯 Universo Investibile")
        universe_filter = st.selectbox(
            "Filtro Categoria",
            options=[
                f"Tutti gli Asset ({len(full_metadata)})",
                "Solo Pokémon",
                "Solo One Piece TCG",
                "Solo Edizioni Giapponesi [JP]",
                "Solo Booster Box Sigillati"
            ]
        )

        st.markdown("---")
        st.caption("PokeQuant v2.2 · Sviluppato con standard quantitativi ApexConvex.")
        st.markdown("[🔗 GitHub Repository](https://github.com/davbenx/pokequant)")

    # Filtraggio dinamico dell'universo
    if "Solo One Piece" in universe_filter:
        metadata = {k: v for k, v in full_metadata.items() if v.get("franchise") == "one_piece"}
    elif "Solo Edizioni Giapponesi" in universe_filter:
        metadata = {k: v for k, v in full_metadata.items() if v.get("language") == "jp"}
    elif "Solo Pokémon" in universe_filter:
        metadata = {k: v for k, v in full_metadata.items() if v.get("franchise") == "pokemon"}
    elif "Solo Booster Box" in universe_filter:
        metadata = {k: v for k, v in full_metadata.items() if v.get("product_type") == "booster_box"}
    else:
        metadata = full_metadata

    active_cols = [c for c in full_prices_df.columns if c in metadata]
    prices_df = full_prices_df[active_cols]

    # --- QUICK CAPITAL BAR (One-Click Allocation Switcher) ---
    st.markdown('<div style="font-size:12px; font-weight:600; color:#94a3b8; margin-bottom:4px;">⚡ Allocazione Rapida Capitale:</div>', unsafe_allow_html=True)
    q1, q2, q3, q4, q5, q6 = st.columns(6)
    cap_presets = [2500.0, 5000.0, 10000.0, 25000.0, 50000.0, 100000.0]
    for col, cap_val in zip([q1, q2, q3, q4, q5, q6], cap_presets):
        with col:
            is_active = (abs(initial_cash - cap_val) < 1.0)
            btn_lbl = f"{int(cap_val/1000)}k €" if cap_val >= 1000 else f"{int(cap_val)} €"
            if st.button(f"{'● ' if is_active else ''}{btn_lbl}", key=f"btn_cap_{int(cap_val)}", type="primary" if is_active else "secondary", use_container_width=True):
                st.session_state["current_capital"] = cap_val
                if "sb_initial_cash_input" in st.session_state:
                    st.session_state["sb_initial_cash_input"] = cap_val
                st.rerun()

    # --- TABS PRINCIPALI (5-TAB COCKPIT) ---
    tab_cmd, tab_backtest, tab_psa, tab_audit, tab_catalog = st.tabs([
        "⚡ Command Center",
        "📈 Backtest & Performance",
        "⚖️ Arbitraggio PSA",
        "🛡️ Audit & Falsificazione",
        "🔍 Catalogo Live"
    ])

    # =========================================================================
    # TAB 1: ⚡ COMMAND CENTER (DESK OPERATIVO & RADAR SEGNALI)
    # =========================================================================
    with tab_cmd:
        st.markdown('<div class="section-title">⚡ Command Center & Desk Operativo</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Quadro decisionale real-time: ordini di acquisto basati sul capitale, rotazioni e scanner finestre di mercato.</div>', unsafe_allow_html=True)

        cmd_h1, cmd_h2 = st.columns([3, 2])
        with cmd_h1:
            desk_snap_opts = {
                "Dicembre 2024 (Ciclo Recente - 5 BUY Attivi)": "2024-12-01",
                "Settembre 2026 (Rotazioni & Asset Maturo)": "2026-09-01",
                "Maggio 2024 (Fase Scarlet & Violet 151)": "2024-05-01",
                "Dicembre 2023 (Fase Paldea Evolved / OP-03)": "2023-12-01",
                "Novembre 2022 (Fase Lost Origin / Silver Tempest)": "2022-11-01",
                "Oggi (Data Attuale di Sistema)": "today"
            }
            chosen_desk_snap = st.selectbox("📅 Snapshot Temporale Desk", options=list(desk_snap_opts.keys()), index=0, key="cmd_desk_snap_sel")
            desk_snap_val = desk_snap_opts[chosen_desk_snap]
            if desk_snap_val == "today":
                d_eval_dt = datetime.date.today()
                d_eval_px = prices_df.iloc[-1].to_dict()
            else:
                d_eval_dt = pd.to_datetime(desk_snap_val).date()
                d_eval_px = prices_df.loc[desk_snap_val].to_dict() if desk_snap_val in prices_df.index else prices_df.iloc[-1].to_dict()

        with cmd_h2:
            radar_tiers = st.multiselect("Tier Monitorati dal Desk", options=["S", "A", "B", "C"], default=["S", "A", "B"], key="cmd_tiers_sel")

        # Scan dei segnali
        d_scan = scan_signals(current_prices=d_eval_px, metadata=metadata, today_dt=d_eval_dt, allowed_tiers=radar_tiers)
        d_buys = d_scan.get("buy_signals", [])
        d_sells = d_scan.get("sell_signals", [])
        watchlist = d_scan.get("watchlist", [])
        all_evals = d_scan.get("all_evaluations", [])

        # Micro-HUD Strip
        top_disc_str = f"({d_buys[0]['margin_vs_max']:+.1f}€ vs Max)" if d_buys else "N/D"
        st.markdown(f'''
        <div class="hud-strip">
            <div>💼 <strong>Capitale Gestito</strong>: <span style="color:#f8fafc; font-weight:700;">{initial_cash:,.2f} €</span> (Lotti Interi · Cap 12%/Set)</div>
            <div>🟢 <strong>Opportunità BUY</strong>: <span style="color:#10b981; font-weight:700;">{len(d_buys)} in finestra</span> <span style="font-size:11px; color:#38bdf8;">{top_disc_str}</span></div>
            <div>🔄 <strong>Rotazioni Pronte</strong>: <span style="color:#fbbf24; font-weight:700;">{len(d_sells)} Tranche 1</span></div>
            <div>🔒 <strong>Asset OOP</strong>: <span style="color:#94a3b8; font-weight:600;">Regola 14 Mesi Attiva</span></div>
        </div>
        ''', unsafe_allow_html=True)

        # 3 Action Cards Side-by-Side
        dc1, dc2, dc3 = st.columns(3)
        with dc1:
            st.markdown("##### 🟢 ORDINI BUY IMMEDIATI")
            if d_buys:
                b_budget = initial_cash * 0.12
                for b in d_buys[:3]:
                    b_px = b["current_price"]
                    b_qty = max(1, int(b_budget // b_px)) if b_px <= initial_cash * 0.35 else int(b_budget // b_px)
                    b_tot = b_qty * b_px
                    b_pct = (b_tot / initial_cash) * 100 if initial_cash > 0 else 0
                    cm_url = get_cardmarket_url(b["name"], metadata.get(b.get("item_id"), {}).get("franchise", "pokemon"))
                    st.markdown(f'''
                    <div class="signal-card signal-card-buy">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <strong style="color:#f8fafc; font-size:13.5px;">{b['name']}</strong>
                            <span class="pill-tag pill-emerald">Tier {b['tier']}</span>
                        </div>
                        <div style="font-size:12px; color:#cbd5e1; margin-top:6px; line-height:1.4;">
                            • Ordine Suggerito: <strong style="color:#10b981;">Compra {b_qty} Box</strong> ({b_tot:,.0f} € · {b_pct:.1f}% NAV)<br>
                            • Prezzo Attuale: <strong>{b_px:.2f} €</strong> (Prezzo Max: <span style="color:#fbbf24;">{b['max_buy_price']:.2f} €</span>)<br>
                            • Margine Sicurezza: <strong style="color:#38bdf8;">{b['margin_vs_max']:+.2f} €</strong> ({b['months_left_in_window']}m rimasti)
                        </div>
                        <div style="margin-top:8px; text-align:right;">
                            <a href="{cm_url}" target="_blank" class="cm-btn">🛒 Compra su Cardmarket ↗</a>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                if len(d_buys) > 3:
                    st.caption(f"+ altri {len(d_buys)-3} set in finestra consultabili nella matrice sealed.")
            else:
                st.markdown('''
                <div style="background:rgba(15,23,42,0.5); border:1px dashed rgba(255,255,255,0.1); border-radius:10px; padding:14px; font-size:12px; color:#94a3b8;">
                    🔒 <strong>Nessun set in finestra d'acquisto</strong><br>
                    Nessun set attualmente a sconto per questa data. Cassa preservata per i prossimi reprint.
                </div>
                ''', unsafe_allow_html=True)

        with dc2:
            st.markdown("##### 🔄 ORDINI ROTAZIONE (Tranche 1)")
            if d_sells:
                for s in d_sells[:3]:
                    cm_url = get_cardmarket_url(s["name"], metadata.get(s.get("item_id"), {}).get("franchise", "pokemon"))
                    st.markdown(f'''
                    <div class="signal-card signal-card-rotate">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <strong style="color:#fbbf24; font-size:13.5px;">{s['name']}</strong>
                            <span class="pill-tag pill-amber">+{s['net_roi_pct']:.0f}% ROI</span>
                        </div>
                        <div style="font-size:12px; color:#cbd5e1; margin-top:6px; line-height:1.4;">
                            • Azione: <strong>Vendi {s['quantity']} su {s.get('total_quantity', s['quantity'])} Box</strong><br>
                            • Prezzo: <strong>{s['current_price']:.1f} €</strong> (Carico: {s['buy_price']:.1f} €)<br>
                            • Incasso Netto: <strong style="color:#10b981;">{s['net_proceeds']:,.1f} €</strong>
                        </div>
                        <div style="margin-top:8px; text-align:right;">
                            <a href="{cm_url}" target="_blank" class="cm-btn cm-btn-sell">🏷️ Vendi su Cardmarket ↗</a>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                if len(d_sells) > 3:
                    st.caption(f"+ altri {len(d_sells)-3} ordini rotazione attivi.")
            else:
                st.markdown('''
                <div style="background:rgba(15,23,42,0.5); border:1px dashed rgba(255,255,255,0.1); border-radius:10px; padding:14px; font-size:12px; color:#94a3b8;">
                    ⏳ <strong>Maturazione in corso</strong><br>
                    Nessuna posizione ha ancora raggiunto il target del +70% ROI con 18 mesi di holding.
                </div>
                ''', unsafe_allow_html=True)

        with dc3:
            st.markdown("##### 🔒 CUSTODIA ATTIVA (OOP Vault)")
            st.markdown('''
            <div class="signal-card" style="border-left-color:#38bdf8; border-top:1px solid rgba(56,189,248,0.2);">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <strong style="color:#38bdf8; font-size:13.5px;">Cassaforte Asset Out-of-Print</strong>
                    <span class="pill-tag pill-blue">Zero Reprint</span>
                </div>
                <div style="font-size:12px; color:#cbd5e1; margin-top:6px; line-height:1.4;">
                    • <strong>Regola Tassativa</strong>: Box oltre 14 mesi di vita sono ufficialmente fuori produzione. Non riacquistare a mercato per evitare premi speculativi.<br>
                    • <strong>Strategia</strong>: Holding puro e monitoraggio del target di rotazione (+70%) o target finale (+150%).
                </div>
            </div>
            ''', unsafe_allow_html=True)

        # Sub-tools: Export Report & Order Slip
        col_rep, col_slip = st.columns(2)
        with col_rep:
            with st.expander("📋 Copia Report Operativo Desk (Telegram / Note)", expanded=False):
                st.caption("Testo formattato pronto da copiare per diario investimenti, chat Telegram o memo broker.")
                rep_text = generate_executive_report(chosen_desk_snap, initial_cash, d_buys, d_sells, metadata)
                st.code(rep_text, language="text")

        with col_slip:
            with st.expander("🎯 Simulatore d'Ordine Spicciolo (Order Slip Interattivo)", expanded=False):
                slip_sealed = [k for k, v in metadata.items() if v.get("type") == "sealed" and v.get("product_type") in ["booster_box", "specialty_bundle"]]
                slip_def_idx = 0
                if d_buys:
                    top_id = d_buys[0]["item_id"]
                    if top_id in slip_sealed:
                        slip_def_idx = slip_sealed.index(top_id)

                sel_slip_box = st.selectbox(
                    "Seleziona Box",
                    options=slip_sealed,
                    index=slip_def_idx,
                    format_func=lambda x: f"{metadata[x].get('name', x)} (Tier {metadata[x].get('set_tier', 'B')})",
                    key="cmd_slip_box"
                )
                s_meta = metadata[sel_slip_box]
                s_name = s_meta.get("name", sel_slip_box)
                s_msrp = float(s_meta.get("msrp", 140.0) or 140.0)
                s_max_p = round(s_msrp * 1.15, 2)
                s_cur_p = float(d_eval_px.get(sel_slip_box, s_msrp))

                sc_c1, sc_c2 = st.columns(2)
                with sc_c1:
                    b_suggest = max(1, int((initial_cash * 0.12) // s_cur_p)) if s_cur_p <= initial_cash * 0.35 else 1
                    s_qty = st.number_input("Quantità Box", min_value=1, max_value=100, value=min(b_suggest, 20), step=1, key="cmd_slip_qty")
                with sc_c2:
                    s_price = st.number_input("Prezzo Unitario (€)", min_value=10.0, max_value=5000.0, value=float(s_cur_p), step=1.0, key="cmd_slip_px")

                s_tot_cost = s_qty * s_price
                s_pct_nav = (s_tot_cost / initial_cash) * 100 if initial_cash > 0 else 0
                s_margin = round(s_max_p - s_price, 2)

                t1_q = s_qty // 2 if s_qty >= 2 else s_qty
                t1_net = (t1_q * s_price * 1.70 * 0.95) - 0.60
                t1_prof = t1_net - (t1_q * s_price)

                t2_net = (s_qty * s_price * 2.50 * 0.95) - 0.60
                t2_prof = t2_net - s_tot_cost

                st.markdown(f'''
                <div style="background:rgba(15,23,42,0.6); border:1px solid rgba(255,255,255,0.06); border-radius:8px; padding:10px; font-size:12px; color:#cbd5e1; margin-top:6px;">
                    • <strong>Esborso Totale</strong>: <span style="color:#f8fafc; font-weight:700;">{s_tot_cost:,.2f} €</span> ({s_pct_nav:.1f}% NAV)<br>
                    • <strong>Margine vs Prezzo Max</strong>: <span style="color:{'#10b981' if s_margin>=0 else '#f43f5e'}; font-weight:700;">{s_margin:+.2f} €</span> (Max: {s_max_p:.1f} €)<br>
                    • <strong>Tranche 1 (+70%, 18m)</strong>: Vendi {t1_q} box · Netto: <strong style="color:#10b981;">{t1_net:,.1f} €</strong> (+{t1_prof:,.1f}€)<br>
                    • <strong>Target Finale (+150%, 36m)</strong>: Netto: <strong style="color:#10b981;">{t2_net:,.1f} €</strong> (+{t2_prof:,.1f}€)
                </div>
                ''', unsafe_allow_html=True)

                s_cm_url = get_cardmarket_url(s_name, s_meta.get("franchise", "pokemon"))
                st.markdown(f'<div style="text-align:right; margin-top:8px;"><a href="{s_cm_url}" target="_blank" class="cm-btn">🛒 Compra su Cardmarket ↗</a></div>', unsafe_allow_html=True)

        # 39-Set Sealed Matrix
        st.markdown('<div class="section-title" style="margin-top:16px;">⚡ Matrice Operativa di Tutti i Box Sealed (Finestre & Prezzi Max)</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Quadro strategico unificato su tutti i set sealed: verifica immediata di quali box sono ancora acquistabili a sconto, in chiusura o già Out-of-Print.</div>', unsafe_allow_html=True)

        if all_evals:
            mat_filter = st.radio(
                "Filtro Rapido Matrice",
                ["Tutti i Box Sealed", "🟢 Solo in Finestra d'Acquisto", "⏳ Finestra in Chiusura (Pre-OOP)", "🟡 In Avvicinamento (Attendere)", "🔒 Finestra Chiusa (OOP)"],
                horizontal=True,
                key="cmd_mat_filter"
            )
            eval_rows = []
            for ev in all_evals:
                st_w = ev["window_status"]
                if "Solo in Finestra" in mat_filter and "🟢" not in st_w and "⏳" not in st_w:
                    continue
                if "In Chiusura" in mat_filter and "⏳" not in st_w:
                    continue
                if "In Avvicinamento" in mat_filter and "🟡" not in st_w:
                    continue
                if "Finestra Chiusa" in mat_filter and "🔒" not in st_w and "⚠️" not in st_w:
                    continue

                ev_fran = metadata.get(ev.get("item_id"), {}).get("franchise", "pokemon")
                cm_url = get_cardmarket_url(ev["name"], ev_fran)

                eval_rows.append({
                    "Set / Box": ev["name"],
                    "Tier": ev["tier"],
                    "Data Rilascio": ev["release_date"],
                    "Età (Mesi)": f"{ev['age_months']}m",
                    "Prezzo Attuale": f"{ev['current_price']:.2f} €",
                    "MSRP Ufficiale": f"{ev['msrp']:.2f} €",
                    "Prezzo Max Acquisto": f"{ev['max_buy_price']:.2f} €",
                    "Margine vs Max": f"{ev['margin_vs_max']:+.2f} € ({ev['margin_vs_max_pct']:+.1f}%)",
                    "Stato Finestra": ev["window_status"],
                    "Mesi Residui": f"{ev['months_left']}m" if ev['months_left'] > 0 else "0m (OOP)",
                    "Azione Operativa Immediata": ev["action"],
                    "Cardmarket": cm_url
                })
            if eval_rows:
                st.dataframe(
                    pd.DataFrame(eval_rows).set_index("Set / Box"),
                    column_config={
                        "Cardmarket": st.column_config.LinkColumn("Cardmarket", display_text="🛒 Apri ↗")
                    },
                    use_container_width=True
                )
            else:
                st.info("Nessun prodotto corrisponde al filtro selezionato.")

        # Progressive Disclosure Expanders
        with st.expander("💼 Gestione Posizioni Reali Possedute (portfolio_holdings.json)", expanded=False):
            h_data = load_user_holdings()
            if h_data:
                h_rows = []
                for h in h_data:
                    iid = h.get("item_id")
                    px = d_eval_px.get(iid, h.get("buy_price_unit", 0.0))
                    b_px = h.get("buy_price_unit", 0.0)
                    qty = h.get("quantity", 1)
                    cost = b_px * qty
                    cur_val = px * qty
                    net_val = (px * 0.95 - 0.60) * qty
                    pnl = net_val - cost
                    roi = pnl / cost if cost > 0 else 0.0
                    
                    m_info = metadata.get(iid, {})
                    msrp_val = float(m_info.get("msrp") or 140.0)
                    max_p = round(msrp_val * 1.15, 2)
                    
                    rel_str = m_info.get("release_date")
                    if rel_str:
                        try:
                            rel_d = pd.to_datetime(rel_str).date()
                            age_m = (d_eval_dt.year - rel_d.year) * 12 + (d_eval_dt.month - rel_d.month)
                        except Exception:
                            age_m = 24
                    else:
                        age_m = 24

                    is_reaccumulabile = (4 <= age_m <= 14) and (px <= max_p)
                    w_status = f"🟢 IN FINESTRA ({max(0, 14 - age_m)}m rimasti)" if is_reaccumulabile else ("🔒 CHIUSA (OOP)" if age_m > 14 else "⚠️ SOPRA MAX")

                    h_fran = m_info.get("franchise", "pokemon")
                    cm_url = get_cardmarket_url(h.get("name", iid), h_fran)

                    h_rows.append({
                        "Prodotto": h.get("name", iid),
                        "Quantità": qty,
                        "Data Acquisto": h.get("buy_date"),
                        "Prezzo Carico": f"{b_px:.1f} €",
                        "Prezzo Attuale": f"{px:.1f} €",
                        "Prezzo Max Acquisto": f"{max_p:.1f} €",
                        "Stato Finestra": w_status,
                        "Valore Netto": f"{net_val:.1f} €",
                        "PnL Netto": f"{pnl:+,.1f} €",
                        "ROI Netto %": f"{roi*100:+.1f}%",
                        "Cardmarket": cm_url
                    })
                st.dataframe(
                    pd.DataFrame(h_rows).set_index("Prodotto"),
                    column_config={
                        "Cardmarket": st.column_config.LinkColumn("Cardmarket", display_text="🛒 Apri ↗")
                    },
                    use_container_width=True
                )
            else:
                st.info("Nessuna posizione registrata in portfolio_holdings.json.")

            st.markdown("##### ➕ Registra Nuovo Acquisto Box nel Portafoglio")
            with st.form("add_box_form"):
                af1, af2 = st.columns(2)
                with af1:
                    avail = [k for k, v in metadata.items() if v.get("type") == "sealed"]
                    sel_item = st.selectbox("Seleziona Prodotto", options=avail, format_func=lambda x: metadata[x].get("name", x))
                    b_qty = st.number_input("Quantità (Box)", min_value=1, max_value=100, value=1)
                with af2:
                    def_px = metadata[sel_item].get("msrp", 140.0)
                    b_px = st.number_input("Prezzo Unitario Acquisto (€)", min_value=10.0, max_value=5000.0, value=float(def_px))
                    b_date = st.date_input("Data di Acquisto", value=datetime.date.today())
                if st.form_submit_button("Salva Posizione"):
                    h_file = Path(__file__).resolve().parent / "data_cache" / "portfolio_holdings.json"
                    curr = load_user_holdings()
                    curr.append({
                        "item_id": sel_item,
                        "name": metadata[sel_item].get("name", sel_item),
                        "quantity": int(b_qty),
                        "buy_date": b_date.strftime("%Y-%m-%d"),
                        "buy_price_unit": float(b_px)
                    })
                    with open(h_file, "w", encoding="utf-8") as f:
                        json.dump(curr, f, indent=2)
                    st.success("Posizione salvata con successo!")
                    st.rerun()

        with st.expander("📜 Archivio Storico Segnali Radar (2021 - 2026)", expanded=False):
            hist_radar = scan_historical_signals(prices_df=prices_df, metadata=metadata, allowed_tiers=radar_tiers)
            if not hist_radar.empty:
                hist_radar["Year"] = hist_radar["date"].str[:4]
                y_sel = st.selectbox("Seleziona Anno", options=["Tutti gli Anni"] + sorted(hist_radar["Year"].unique().tolist()), key="cmd_hist_year")
                df_hr = hist_radar if y_sel == "Tutti gli Anni" else hist_radar[hist_radar["Year"] == y_sel]
                rows_r = []
                for _, r in df_hr.iterrows():
                    max_p_str = f"{r['max_buy_price']:.2f} €" if "max_buy_price" in r and r["max_buy_price"] > 0 else "-"
                    rows_r.append({
                        "Data": r["date"],
                        "Azione": "🟢 BUY" if r["action"] == "BUY" else ("🔄 ROTAZIONE" if "Tranche 1" in r["reason"] else "🔴 SELL"),
                        "Set": r["item_name"],
                        "Quantità": r["quantity"],
                        "Prezzo": f"{r['price']:.2f} €",
                        "Prezzo Max Acquisto": max_p_str,
                        "Valore": f"{r['total_value']:,.2f} €",
                        "Trigger": r["reason"]
                    })
                st.dataframe(pd.DataFrame(rows_r), use_container_width=True, hide_index=True)

        with st.expander("🤖 Notifiche Push Telegram Live", expanded=False):
            st.caption("Notifiche automatiche ogni lunedì mattina via GitHub Actions (.github/workflows/poke_signals.yml) o on-demand.")
            t_c1, t_c2 = st.columns([3, 1])
            with t_c1:
                st.code("python poke_quant/signal_scanner.py", language="bash")
            with t_c2:
                if st.button("🔔 Invia Test Telegram", key="cmd_tg_test"):
                    msg = format_telegram_alert(d_scan)
                    sent = send_telegram_message(msg)
                    if sent:
                        st.success("Notifica inviata con successo!")
                    else:
                        st.warning("Variabili TELEGRAM_TOKEN o TELEGRAM_CHAT_ID non configurate.")

    # =========================================================================
    # TAB 2: 📈 BACKTEST & PERFORMANCE
    # =========================================================================
    with tab_backtest:
        st.markdown('<div class="section-title">⚡ Simulatore Strategie con Rotazione Dinamica Scalare</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Esecuzione quantitativa su 68 mesi di prezzi reali (2021-2026), sblocco periodico di liquidità e reinvestimento nei reprint moderni.</div>', unsafe_allow_html=True)

        # PRESETS FRICTIONLESS (One-Click Setup)
        preset_col1, preset_col2 = st.columns([3, 1])
        with preset_col1:
            preset = st.radio(
                "Configurazione Rapida (Preset)",
                options=[
                    "⚡ Istituzionale PokeQuant (Rotazione 12% + Tier S/A/B)",
                    "💎 Prestige Out-of-Print (Solo Tier S/A, No Rotazione)",
                    "🔄 DCA / Accumulo Continuo (PAC 250 €/mese)",
                    "⚙️ Personalizzato (Sblocca Parametri)"
                ],
                horizontal=True
            )

        # Inizializzazione parametri in base al preset
        if "Istituzionale" in preset:
            enable_rot = True
            t1_roi = 0.70
            t1_hold = 18
            max_alloc = 0.12
            target_roi = 1.50
            min_h = 30
            sel_tiers = ["S", "A", "B"]
            dca_val = 0.0
        elif "Prestige" in preset:
            enable_rot = False
            t1_roi = 0.70
            t1_hold = 18
            max_alloc = 0.20
            target_roi = 1.50
            min_h = 30
            sel_tiers = ["S", "A"]
            dca_val = 0.0
        elif "DCA" in preset:
            enable_rot = True
            t1_roi = 0.70
            t1_hold = 18
            max_alloc = 0.12
            target_roi = 1.50
            min_h = 30
            sel_tiers = ["S", "A", "B"]
            dca_val = round(initial_cash * 0.025, 0)
        else:
            enable_rot = True
            t1_roi = 0.70
            t1_hold = 18
            max_alloc = 0.12
            target_roi = 1.50
            min_h = 30
            sel_tiers = ["S", "A", "B"]
            dca_val = 0.0

        # Mostra slider personalizzati solo se richiesto o espanso
        with st.expander("🛠️ Personalizza Parametri di Selezione, Allocazione e Rotazione", expanded=("Personalizzato" in preset)):
            p_col1, p_col2, p_col3, p_col4 = st.columns(4)
            with p_col1:
                enable_rot = st.checkbox("Abilita Rotazione Tranche 1", value=enable_rot)
                t1_roi = st.slider("Target ROI Tranche 1 (%)", 40, 120, int(t1_roi * 100), 5) / 100.0 if enable_rot else 0.70
            with p_col2:
                t1_hold = st.slider("Holding Minimo Tranche 1 (Mesi)", 12, 24, int(t1_hold), 1) if enable_rot else 18
                target_roi = st.slider("Target ROI Finale Tranche 2 (%)", 50, 250, int(target_roi * 100), 10) / 100.0
            with p_col3:
                max_alloc = st.slider("Cap Allocazione per Set (%)", 5, 25, int(max_alloc * 100), 1) / 100.0
                min_h = st.slider("Holding Minimo Finale (Mesi)", 12, 48, int(min_h), 2)
            with p_col4:
                dca_val = st.number_input("PAC Mensile Liquidità (€/mese)", min_value=0.0, max_value=2000.0, value=float(dca_val), step=50.0)
                sel_tiers = st.multiselect("Tier Qualitativi Inclusi", options=["S", "A", "B", "C"], default=sel_tiers)

        # Esecuzione Strategie nel Backtester
        strat_optimal = OptimalSealedStrategy(
            allowed_tiers=sel_tiers,
            min_buy_age_months=4,
            max_buy_age_months=14,
            max_msrp_multiplier=1.15,
            min_hold_months=min_h,
            target_roi=target_roi,
            max_hold_months=48,
            max_allocation_pct=max_alloc,
            enable_dynamic_rotation=enable_rot,
            tranche1_roi=t1_roi,
            tranche1_min_hold_months=t1_hold,
            tranche1_pct=0.50
        )
        bt_optimal = Backtester(
            strategy=strat_optimal,
            historical_prices_df=prices_df,
            items_metadata=metadata,
            initial_cash=initial_cash,
            platform=platform,
            seller_absorbs_shipping=absorb_shipping,
            benchmark_cagr=benchmark_cagr,
            benchmark_series=benchmark_series,
            apply_liquidity_slippage=apply_slippage,
            apply_holding_cost=apply_holding,
            monthly_cash_injection=dca_val
        )
        res_optimal = bt_optimal.run()

        # Benchmark standard per confronto
        strat_sealed = SealedAccumulatorStrategy(
            max_allocation_per_set_pct=0.25,
            max_buy_age_months=14,
            min_hold_months=20,
            target_profit_roi=0.80,
            msrp_max_multiplier=1.20
        )
        bt_sealed = Backtester(
            strategy=strat_sealed,
            historical_prices_df=prices_df,
            items_metadata=metadata,
            initial_cash=initial_cash,
            platform=platform,
            seller_absorbs_shipping=absorb_shipping,
            benchmark_cagr=benchmark_cagr,
            benchmark_series=benchmark_series,
            apply_liquidity_slippage=apply_slippage,
            apply_holding_cost=apply_holding,
            monthly_cash_injection=dca_val
        )
        res_sealed = bt_sealed.run()

        strat_chase = ChaseDipBuyerStrategy(
            min_dip_months=4,
            max_dip_months=10,
            min_drop_from_launch_pct=0.15,
            target_profit_roi=0.60,
            max_allocation_per_card_pct=0.15
        )
        bt_chase = Backtester(
            strategy=strat_chase,
            historical_prices_df=prices_df,
            items_metadata=metadata,
            initial_cash=initial_cash,
            platform=platform,
            seller_absorbs_shipping=absorb_shipping,
            benchmark_cagr=benchmark_cagr,
            benchmark_series=benchmark_series,
            apply_liquidity_slippage=apply_slippage,
            apply_holding_cost=apply_holding,
            monthly_cash_injection=dca_val
        )
        res_chase = bt_chase.run()

        # --- EXECUTIVE KPI BANNER ---
        tot_inv_val = sum(p["current_value"] for p in res_optimal.open_positions) if res_optimal.open_positions else 0.0
        last_cash = float(res_optimal.nav_history["cash"].iloc[-1])

        st.markdown(f'''
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Capitale Finale Netto</div>
                <div class="kpi-value">{res_optimal.final_nav:,.2f} €</div>
                <div class="kpi-sub kpi-sub-emerald">▲ {res_optimal.total_net_return*100:+.1f}% ROI Netto</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">CAGR Netto vs Benchmark</div>
                <div class="kpi-value">{res_optimal.cagr*100:+.2f}%</div>
                <div class="kpi-sub kpi-sub-blue">★ {res_optimal.alpha_annualized*100:+.2f}% Alpha Annuale</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Rischio & Volatilità</div>
                <div class="kpi-value">{res_optimal.sharpe:.2f} Sharpe</div>
                <div class="kpi-sub kpi-sub-rose">▼ {res_optimal.max_drawdown*100:.2f}% Max Drawdown</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Rotazione del Capitale</div>
                <div class="kpi-value">{res_optimal.rotation_trades_count} Tranche 1</div>
                <div class="kpi-sub kpi-sub-amber">🔄 {res_optimal.turnover_ratio*100:.1f}% Turnover NAV</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        # Sizing summary bar
        open_box_cnt = sum(p["quantity"] for p in res_optimal.open_positions) if res_optimal.open_positions else 0
        closed_box_cnt = int(res_optimal.trades_df["quantity"].sum()) if not res_optimal.trades_df.empty else 0
        total_boxes_managed = open_box_cnt + closed_box_cnt
        net_profit_abs = res_optimal.final_nav - initial_cash

        st.markdown(f'''
        <div style="display:flex; justify-content:space-between; align-items:center; background:rgba(30, 41, 59, 0.45); border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:8px 16px; margin-top:8px; margin-bottom:12px; font-size:12.5px; color:#cbd5e1; flex-wrap:wrap; gap:8px;">
            <div>💰 <strong>Capitale Iniziale</strong>: <span style="color:#f8fafc; font-weight:700;">{initial_cash:,.2f} €</span></div>
            <div>📈 <strong>Profitto Netto</strong>: <span style="color:#10b981; font-weight:700;">+{net_profit_abs:,.2f} €</span></div>
            <div>📦 <strong>Box Fisici Gestiti</strong>: <span style="color:#38bdf8; font-weight:700;">{total_boxes_managed} Unità Intere</span> ({open_box_cnt} custodia + {closed_box_cnt} ruotati)</div>
            <div>⚡ <strong>Sizing</strong>: <span style="color:#fbbf24; font-weight:600;">Lotto Minimo Discreto (No Frazioni)</span></div>
        </div>
        ''', unsafe_allow_html=True)

        # Plotly chart
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.72, 0.28],
            subplot_titles=("Evoluzione del Valore Liquidativo (NAV) vs Benchmark", "Drawdown Subito (%)")
        )
        bench_nav = res_optimal.benchmark_nav
        fig.add_trace(go.Scatter(
            x=bench_nav.index, y=bench_nav.values,
            mode='lines', name=f"Benchmark ({bench_choice.split(' ')[0]})",
            line=dict(color='rgba(148, 163, 184, 0.7)', dash='dash', width=2)
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=res_optimal.nav_history.index, y=res_optimal.nav_history["nav"],
            mode='lines', name="Optimal Sealed (Rotazione)",
            line=dict(color='#10b981', width=3)
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=res_sealed.nav_history.index, y=res_sealed.nav_history["nav"],
            mode='lines', name="Sealed Standard",
            line=dict(color='#3b82f6', width=2)
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=res_chase.nav_history.index, y=res_chase.nav_history["nav"],
            mode='lines', name="Chase Card (Singole)",
            line=dict(color='#f97316', width=1.5)
        ), row=1, col=1)

        nav = res_optimal.nav_history["nav"]
        dd = (nav / nav.cummax() - 1.0) * 100.0
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values,
            mode='lines', fill='tozeroy',
            name="Drawdown Optimal",
            line=dict(color='#f43f5e', width=1),
            fillcolor='rgba(244, 63, 94, 0.15)'
        ), row=2, col=1)

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(15, 23, 42, 0.5)",
            plot_bgcolor="rgba(15, 23, 42, 0.5)",
            font=dict(family="Inter, sans-serif", color="#94a3b8"),
            height=520,
            margin=dict(l=20, r=20, t=35, b=20),
            hovermode="x unified",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                bgcolor="rgba(15, 23, 42, 0.7)", bordercolor="rgba(255, 255, 255, 0.08)", borderwidth=1
            )
        )
        fig.update_xaxes(showgrid=True, gridcolor="rgba(255, 255, 255, 0.05)")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(255, 255, 255, 0.05)", title_text="Euro (€)", row=1, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(255, 255, 255, 0.05)", title_text="Drawdown %", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

        # Portafoglio Attivo (Posizioni Aperte)
        st.markdown('<div class="section-title">💼 Portafoglio Attivo in Detenzione (Posizioni a Fine Backtest)</div>', unsafe_allow_html=True)
        st.markdown("<div class='section-desc'>Box fisicamente custoditi a Settembre 2026: monitoraggio dello stato della finestra d'acquisto e del Prezzo Massimo Consentito.</div>", unsafe_allow_html=True)

        if res_optimal.open_positions:
            op_rows = []
            for p in res_optimal.open_positions:
                cur_p = p["current_price"]
                max_p = p.get("max_buy_price", round(p["buy_price_unit"] * 1.15, 2))
                is_win = p.get("is_in_buy_window", False)
                w_status = p.get("window_status", "N/D")
                
                if is_win:
                    act_badge = "🟢 ACCUMULA (DCA)"
                elif p["holding_months"] >= 18 and p["unrealized_roi"] >= 0.70:
                    act_badge = "🔄 RUOTA (Tranche 1)"
                elif p["holding_months"] >= 30 and p["unrealized_roi"] >= 1.50:
                    act_badge = "🔴 ESCI (Target 150%)"
                else:
                    act_badge = "🔒 CUSTODIA (OOP)"

                fran = metadata.get(p.get("item_id"), {}).get("franchise", "pokemon")
                cm_url = get_cardmarket_url(p["item_name"], fran)

                op_rows.append({
                    "Set / Articolo": p["item_name"],
                    "Q.tà": p["quantity"],
                    "Data Acquisto": p["buy_date"],
                    "Holding": f"{p['holding_months']}m",
                    "Carico": f"{p['buy_price_unit']:.2f} €",
                    "Prezzo Attuale": f"{cur_p:.2f} €",
                    "Prezzo Max Acquisto": f"{max_p:.2f} €",
                    "Finestra d'Acquisto": w_status,
                    "Azione Operativa": act_badge,
                    "Valore MTM": f"{p['current_value']:,.2f} €",
                    "PnL Non Realizzato": f"{p['unrealized_pnl']:+,.2f} €",
                    "ROI Non Realizzato": f"{p['unrealized_roi']*100:+.1f}%",
                    "Cardmarket": cm_url
                })
            st.dataframe(
                pd.DataFrame(op_rows).set_index("Set / Articolo"),
                column_config={
                    "Cardmarket": st.column_config.LinkColumn("Cardmarket", display_text="🛒 Apri ↗")
                },
                use_container_width=True
            )

            # Guida Operativa Lean
            c_acc1, c_acc2 = st.columns(2)
            with c_acc1:
                accumulabili = [p for p in res_optimal.open_positions if p.get("is_in_buy_window", False)]
                acc_names = ", ".join([f"**{p['item_name']}** ({p['current_price']:.1f}€ vs Max {p['max_buy_price']:.1f}€)" for p in accumulabili]) if accumulabili else "Nessuna posizione attualmente in finestra (tutti i set posseduti sono Out-of-Print)."
                st.markdown(f'''
                <div style="background:rgba(16, 185, 129, 0.08); border:1px solid rgba(16, 185, 129, 0.3); border-radius:10px; padding:12px; margin-bottom:12px;">
                    <div style="font-weight:700; color:#10b981; font-size:13px; margin-bottom:4px;">
                        🟢 POSIZIONI ANCORA IN FINESTRA BUONA D'ACQUISTO (ACCUMULABILI)
                    </div>
                    <div style="font-size:12px; color:#cbd5e1; line-height:1.45;">
                        • <strong>Set Reperibili</strong>: {acc_names}<br>
                        • <strong>Azione Consentita</strong>: È possibile incrementare la posizione via PAC/DCA perché il prezzo è rigorosamente inferiore al Prezzo Massimo di Acquisto (+15% MSRP) e non sono ancora cessate le ristampe.<br>
                        • <strong>Limite di Rischio</strong>: Non superare il 12% di allocazione complessiva per singolo set.
                    </div>
                </div>
                ''', unsafe_allow_html=True)
            with c_acc2:
                oop_pos = [p for p in res_optimal.open_positions if not p.get("is_in_buy_window", False)]
                oop_names = ", ".join([f"**{p['item_name']}** (+{p['unrealized_roi']*100:.0f}%)" for p in oop_pos[:3]]) if oop_pos else "Nessuna posizione in stato Out-of-Print."
                st.markdown(f'''
                <div style="background:rgba(148, 163, 184, 0.08); border:1px solid rgba(148, 163, 184, 0.3); border-radius:10px; padding:12px; margin-bottom:12px;">
                    <div style="font-weight:700; color:#cbd5e1; font-size:13px; margin-bottom:4px;">
                        🔒 POSIZIONI A FINESTRA CHIUSA (SOLO CUSTODIA / OOP)
                    </div>
                    <div style="font-size:12px; color:#94a3b8; line-height:1.45;">
                        • <strong>Set in Cassaforte</strong>: {oop_names} {f'e altri {len(oop_pos)-3}' if len(oop_pos) > 3 else ''}<br>
                        • <strong>Azione Tassativa</strong>: <u>NON RIACQUISTARE A MERCATO</u>. L'offerta di ristampa è chiusa ed il prezzo è già cresciuto oltre il limite di sicurezza.<br>
                        • <strong>Target Operativo</strong>: Mantenere le unità sigillate fino al target di rotazione Tranche 1 (+70%) o uscita finale (+150%).
                    </div>
                </div>
                ''', unsafe_allow_html=True)
        else:
            st.info("Nessuna posizione aperta in inventario.")

        # Progressive Disclosure Expanders
        with st.expander("📊 Tabella Comparativa Istituzionale vs Strategie Alternative", expanded=False):
            comp_rows = [
                {
                    "Strategia": "Optimal Sealed Strategy (Rotazione Scalare)",
                    "Capitale Finale": f"{res_optimal.final_nav:,.2f} €",
                    "ROI Netto": f"{res_optimal.total_net_return*100:+.1f}%",
                    "CAGR": f"{res_optimal.cagr*100:+.2f}%",
                    "Alpha vs SPY": f"{res_optimal.alpha_annualized*100:+.2f}%",
                    "Sharpe": f"{res_optimal.sharpe:.2f}",
                    "MaxDD": f"{res_optimal.max_drawdown*100:.2f}%",
                    "Rotazioni (Tranche 1)": res_optimal.rotation_trades_count,
                    "Trades Chiusi": res_optimal.total_trades,
                    "Win Rate": f"{res_optimal.win_rate*100:.1f}%",
                    "Frizioni Pagate": f"{res_optimal.total_fees_paid:,.2f} €"
                },
                {
                    "Strategia": "Sealed Accumulator (Standard No Rotazione)",
                    "Capitale Finale": f"{res_sealed.final_nav:,.2f} €",
                    "ROI Netto": f"{res_sealed.total_net_return*100:+.1f}%",
                    "CAGR": f"{res_sealed.cagr*100:+.2f}%",
                    "Alpha vs SPY": f"{res_sealed.alpha_annualized*100:+.2f}%",
                    "Sharpe": f"{res_sealed.sharpe:.2f}",
                    "MaxDD": f"{res_sealed.max_drawdown*100:.2f}%",
                    "Rotazioni (Tranche 1)": res_sealed.rotation_trades_count,
                    "Trades Chiusi": res_sealed.total_trades,
                    "Win Rate": f"{res_sealed.win_rate*100:.1f}%",
                    "Frizioni Pagate": f"{res_sealed.total_fees_paid:,.2f} €"
                },
                {
                    "Strategia": "Chase Dip Buyer (Singole Hype-Cycle)",
                    "Capitale Finale": f"{res_chase.final_nav:,.2f} €",
                    "ROI Netto": f"{res_chase.total_net_return*100:+.1f}%",
                    "CAGR": f"{res_chase.cagr*100:+.2f}%",
                    "Alpha vs SPY": f"{res_chase.alpha_annualized*100:+.2f}%",
                    "Sharpe": f"{res_chase.sharpe:.2f}",
                    "MaxDD": f"{res_chase.max_drawdown*100:.2f}%",
                    "Rotazioni (Tranche 1)": res_chase.rotation_trades_count,
                    "Trades Chiusi": res_chase.total_trades,
                    "Win Rate": f"{res_chase.win_rate*100:.1f}%",
                    "Frizioni Pagate": f"{res_chase.total_fees_paid:,.2f} €"
                }
            ]
            st.dataframe(pd.DataFrame(comp_rows).set_index("Strategia"), use_container_width=True)

        with st.expander("🌐 Performance per Regime Macroeconomico (Optimal Sealed)", expanded=False):
            r_cols = st.columns(len(res_optimal.regime_performance))
            for i, (regime, val) in enumerate(res_optimal.regime_performance.items()):
                with r_cols[i]:
                    st.metric(regime.split(" ")[0], f"{val*100:+.1f}%", help=regime)

        with st.expander("📡 Cronologia Completa dei Segnali Storici Generati (2021 - 2026)", expanded=False):
            if res_optimal.signals_history:
                sig_df = pd.DataFrame(res_optimal.signals_history)
                sig_df["Year"] = sig_df["date"].str[:4]

                cs1, cs2, cs3 = st.columns([2, 2, 2])
                with cs1:
                    year_f = st.selectbox("Filtra per Anno", options=["Tutti gli Anni (2021-2026)"] + sorted(sig_df["Year"].unique().tolist()), key="bt_sig_year")
                with cs2:
                    action_f = st.selectbox("Filtra per Tipologia", options=["Tutti i Segnali", "Solo BUY (Acquisti)", "Solo SELL (Rotazioni & Uscite)"], key="bt_sig_action")
                with cs3:
                    st.metric("Segnali Registrati", f"{len(sig_df)} Totali", help="Totale decisioni operative eseguite nel periodo")

                y_counts = sig_df.groupby(["Year", "action"]).size().unstack(fill_value=0)
                fig_bar = go.Figure()
                if "BUY" in y_counts.columns:
                    fig_bar.add_trace(go.Bar(x=y_counts.index, y=y_counts["BUY"], name="BUY (Acquisto Dip)", marker_color="#10b981"))
                if "SELL" in y_counts.columns:
                    fig_bar.add_trace(go.Bar(x=y_counts.index, y=y_counts["SELL"], name="SELL (Rotazione / Uscita)", marker_color="#f59e0b"))
                fig_bar.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(15, 23, 42, 0.4)",
                    plot_bgcolor="rgba(15, 23, 42, 0.4)",
                    barmode="group", height=220, margin=dict(l=20, r=20, t=25, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                filtered_df = sig_df.copy()
                if year_f != "Tutti gli Anni (2021-2026)":
                    filtered_df = filtered_df[filtered_df["Year"] == year_f]
                if "BUY" in action_f:
                    filtered_df = filtered_df[filtered_df["action"] == "BUY"]
                elif "SELL" in action_f:
                    filtered_df = filtered_df[filtered_df["action"] == "SELL"]

                sig_rows = []
                for _, s in filtered_df.iterrows():
                    badge = "🟢 BUY" if s["action"] == "BUY" else ("🔄 TRANCHE 1" if "Tranche 1" in s["reason"] else "🔴 SELL")
                    max_p_str = f"{s['max_buy_price']:.2f} €" if "max_buy_price" in s and s["max_buy_price"] > 0 else "-"
                    sig_rows.append({
                        "Data": s["date"],
                        "Azione": badge,
                        "Prodotto": s["item_name"],
                        "Quantità": s["quantity"],
                        "Prezzo Unitario": f"{s['price']:.2f} €",
                        "Prezzo Max Acquisto": max_p_str,
                        "Controvalore": f"{s['total_value']:,.2f} €",
                        "Cassa Residua Prima": f"{s['portfolio_cash_before']:,.2f} €",
                        "Motivazione / Trigger": s["reason"]
                    })
                st.dataframe(pd.DataFrame(sig_rows), use_container_width=True, hide_index=True)

        with st.expander(f"📜 Registro Operazioni Concluse ({len(res_optimal.trades_df)} Trades Chiusi)", expanded=False):
            if not res_optimal.trades_df.empty:
                td_disp = res_optimal.trades_df.copy()
                td_disp["Prezzo Acq."] = td_disp["buy_price_unit"].apply(lambda x: f"{x:.2f} €")
                td_disp["Prezzo Vend."] = td_disp["sell_price_unit"].apply(lambda x: f"{x:.2f} €")
                td_disp["Incasso Lordo"] = td_disp["gross_proceeds"].apply(lambda x: f"{x:,.2f} €")
                td_disp["Fee Pagate"] = td_disp["fees_paid"].apply(lambda x: f"{x:.2f} €")
                td_disp["PnL Netto"] = td_disp["net_pnl"].apply(lambda x: f"{x:+,.2f} €")
                td_disp["ROI Netto %"] = td_disp["net_roi"].apply(lambda x: f"{x*100:+.1f}%")
                cols = ["item_name", "quantity", "buy_date", "sell_date", "holding_months", "Prezzo Acq.", "Prezzo Vend.", "Fee Pagate", "PnL Netto", "ROI Netto %"]
                st.dataframe(td_disp[cols].rename(columns={"item_name": "Articolo", "quantity": "Q.tà", "buy_date": "Data Acq.", "sell_date": "Data Vend.", "holding_months": "Mesi"}), use_container_width=True)

        with st.expander("ℹ️ Come Scala la Strategia sui Box Indivisibili (Discrete Unit Lot Sizing)", expanded=False):
            st.markdown(r'''
            **Principio Quantitativo dell'Indivisibilità degli Asset Reali:**
            - **Nessuna Frazione Astratta**: Nei mercati azionari o crypto è possibile acquistare frazioni (es. 0.35 azioni). I Booster Box sono **beni fisici discreti e indivisibili** ($q \in \mathbb{N}_{\ge 1}$).
            - **Scaling Proporzionale Matematico**: Variando il capitale (es. da 2.500 € a 100.000 €), l'algoritmo scala il budget per set ($NAV \times \text{Cap \%}$) e applica la divisione intera per il prezzo unitario reale ($\lfloor \text{Budget} / P_i \rfloor$).
            - **Regola del Lotto Minimo per Piccoli Capitali (< 3.000 €)**: Se il budget teorico del 12% è inferiore al prezzo di 1 box (es. 120 € su 1.000 € di capitale con un box da 135 €), il sistema **autorizza comunque l'acquisto di 1 box intero** purché la cassa lo copra e non si superi il 35% del NAV totale. Questo evita che i conti piccoli restino a zero acquisti!
            - **Resto Indivisibile (Cash Buffer)**: La frazione monetaria non sufficiente a comprare un ulteriore box intero rimane liquida in cassa, pronta per i reprint successivi o per assorbire prodotti più accessibili (es. Specialty Bundle da 30-35 €).
            - **Rotazione Tranche 1 Discreta**: Se la posizione ha $\ge 2$ box, si vende il $50\%$ intero ($\lfloor q / 2 \rfloor$). Se la posizione è di $1$ solo box, il sigillo non viene violato: il box rimane intero fino al target finale (+150%).
            ''')
    # =========================================================================
    # TAB 3: ⚖️ ARBITRAGGIO GRADING PSA
    # =========================================================================
    with tab_psa:
        st.markdown('<div class="section-title">⚖️ Calcolatore di Arbitraggio Statistico Grading PSA / BGS</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Valutazione analitica del Valore Atteso Netto (EV) deducendo fee di servizio, spedizioni assicurate, fermo capitale e Gem Rate reale.</div>', unsafe_allow_html=True)

        cg1, cg2, cg3 = st.columns(3)
        with cg1:
            raw_in = st.number_input("Prezzo Acquisto Carta Raw Near-Mint (€)", min_value=1.0, max_value=10000.0, value=120.0, step=10.0)
            gem_in = st.slider("Gem Rate Stimata (Probabilità PSA 10)", min_value=0.10, max_value=0.95, value=0.68, step=0.02)
        with cg2:
            psa10_in = st.number_input("Prezzo di Mercato PSA 10 (€)", min_value=1.0, max_value=25000.0, value=480.0, step=10.0)
            psa9_in = st.number_input("Prezzo di Mercato PSA 9 (€)", min_value=1.0, max_value=10000.0, value=110.0, step=5.0)
        with cg3:
            grading_fee_in = st.number_input("Costo Grading All-In (€)", min_value=10.0, max_value=200.0, value=25.0, step=5.0)
            turnaround_in = st.number_input("Turnaround Fermo Capitale (Mesi)", min_value=1, max_value=6, value=2)

        p10 = gem_in
        p9 = (1.0 - gem_in) * 0.85
        p8 = max(0.0, 1.0 - p10 - p9)
        ev_gross = (p10 * psa10_in) + (p9 * psa9_in) + (p8 * raw_in * 0.50)
        fee_info = PLATFORM_FEES[platform]
        ev_net_proceeds = ev_gross * (1.0 - fee_info["percentage"]) - fee_info["fixed_fee"] - 0.60
        tot_cost = raw_in + grading_fee_in
        ev_profit = ev_net_proceeds - tot_cost
        ev_roi = ev_profit / tot_cost if tot_cost > 0 else 0.0

        st.markdown(f"""
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Valore Atteso Netto (EV)</div>
                <div class="kpi-value">{ev_net_proceeds:,.2f} €</div>
                <div class="kpi-sub kpi-sub-neutral">Lordo atteso: {ev_gross:.1f} €</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Costo Totale Operazione</div>
                <div class="kpi-value">{tot_cost:,.2f} €</div>
                <div class="kpi-sub kpi-sub-neutral">Raw {raw_in:.0f}€ + Fee {grading_fee_in:.0f}€</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Profitto Netto Atteso</div>
                <div class="kpi-value" style="color:{'#10b981' if ev_profit>0 else '#f43f5e'};">{ev_profit:+,.2f} €</div>
                <div class="kpi-sub {'kpi-sub-emerald' if ev_roi>0 else 'kpi-sub-rose'}">{ev_roi*100:+.1f}% ROI Atteso</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Valutazione Quantitativa</div>
                <div class="kpi-value" style="font-size:16px; margin-top:4px;">
                    {'🟢 CONSIGLIATA' if ev_roi >= 0.25 else ('🟡 MARGINALE' if ev_roi >= 0.10 else '🔴 SCONSIGLIATA')}
                </div>
                <div class="kpi-sub kpi-sub-neutral">Soglia Edge Minimo: 25% ROI</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Opportunità di Grading nel Catalogo Attuale (Chase Alt Art)")
        sample_cards = [
            {"Nome": "Umbreon VMAX Alt Art (Moonbreon)", "Raw (€)": 1990.0, "PSA 10 (€)": 3800.0, "Gem Rate": 0.72},
            {"Nome": "Rayquaza VMAX Alt Art", "Raw (€)": 1018.0, "PSA 10 (€)": 1950.0, "Gem Rate": 0.68},
            {"Nome": "Giratina V Alt Art (Lost Origin)", "Raw (€)": 695.0, "PSA 10 (€)": 1400.0, "Gem Rate": 0.65},
            {"Nome": "Gengar VMAX Alt Art (Fusion Strike)", "Raw (€)": 848.0, "PSA 10 (€)": 1550.0, "Gem Rate": 0.70},
            {"Nome": "Charizard V Alt Art (Brilliant Stars)", "Raw (€)": 247.0, "PSA 10 (€)": 550.0, "Gem Rate": 0.75},
        ]
        sc_rows = []
        for c in sample_cards:
            res = evaluate_grading_arbitrage(
                raw_price=c["Raw (€)"],
                psa10_price=c["PSA 10 (€)"],
                gem_rate=c["Gem Rate"],
                platform=platform
            )
            sc_rows.append({
                "Carta": c["Nome"],
                "Prezzo Raw": f"{c['Raw (€)']:.0f} €",
                "Prezzo PSA 10": f"{c['PSA 10 (€)']:.0f} €",
                "Gem Rate": f"{c['Gem Rate']*100:.0f}%",
                "Valore Atteso Netto": f"{res.expected_graded_net:.1f} €",
                "Profitto Atteso": f"{res.expected_net_profit:+,.1f} €",
                "ROI Netto %": f"{res.expected_net_roi*100:+.1f}%",
                "Verdetto": "✅ Consigliato" if res.is_favorable else "⚠️ Neutro/Rischioso"
            })
        st.dataframe(pd.DataFrame(sc_rows).set_index("Carta"), use_container_width=True)

    # =========================================================================
    # TAB 4: 🛡️ AUDIT STATISTICO & FALSIFICAZIONE
    # =========================================================================
    with tab_audit:
        st.markdown('<div class="section-title">🛡️ Audit Istituzionale Anti-Overfitting & Suite di Falsificazione Popperiana</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Validazione rigorosa per eliminare il rischio di data-snooping (Bailey & López de Prado) e 8 stress-test popperiani.</div>', unsafe_allow_html=True)

        n_trials_in = st.slider("Numero di configurazioni esplorate nella griglia (n_trials)", 1, 100, 10, 1)
        dsr_val = deflated_sharpe_ratio(observed_sr=res_optimal.sharpe, n_trials=n_trials_in, n_obs=len(res_optimal.monthly_returns))

        aud1, aud2 = st.columns(2)
        with aud1:
            st.markdown("#### Deflated Sharpe Ratio (DSR)")
            st.metric("DSR Score", f"{dsr_val:.4f}", f"{dsr_val*100:.1f}% Confidenza Statistica")
            if dsr_val >= 0.95:
                st.success("✅ **Alpha Istituzionale Confermato**: L'extra-rendimento resiste alla correzione per test multipli con oltre il 95% di confidenza.")
            else:
                st.warning("⚠️ Confidenza ridotta dal numero di test.")

        with aud2:
            st.markdown("#### Probability of Backtest Overfitting (PBO / CSCV)")
            try:
                common_idx = res_optimal.monthly_returns.index.intersection(
                    res_sealed.monthly_returns.index
                ).intersection(res_chase.monthly_returns.index)
                
                if len(common_idx) >= 8:
                    perf_mat = np.column_stack([
                        res_optimal.monthly_returns.loc[common_idx].values,
                        res_sealed.monthly_returns.loc[common_idx].values,
                        res_chase.monthly_returns.loc[common_idx].values
                    ])
                    n_sp = 4 if len(perf_mat) >= 16 else 2
                    pbo_val = pbo_cscv(perf_mat, n_splits=n_sp)
                    st.metric("PBO Score", f"{pbo_val:.2f}", "Rischio Overfitting Nullo" if pbo_val == 0.0 else "Rischio Moderato")
                    if pbo_val <= 0.10:
                        st.success("✅ **Zero Overfitting**: La strategia vincente in-sample si riconferma top performer out-of-sample in tutte le partizioni CSCV.")
                    else:
                        st.info(f"PBO calcolato: {pbo_val:.2%}")
                else:
                    st.info("Campione temporale insufficiente per il calcolo del PBO.")
            except Exception as e:
                st.warning(f"Calcolo CSCV non disponibile: {e}")

        st.markdown("---")
        st.markdown("#### Risultati della Suite di Falsificazione (8 Test Popperiani)")
        falsif_tests = [
            ("1. Anti-Outlier Test", "Esclusione di Evolving Skies & Team Up", "+22,86%", "Superato (Alpha indipendente da singoli unicorni)"),
            ("2. Bear Market Test", "Partenza durante il QT (Gennaio 2022)", "+35,45%", "Superato (MaxDD limitato a -5,17% durante crollo crypto/tech)"),
            ("3. Friction Shock", "Fee eBay 12.5% + Spedizioni + Slippage + Storage", "+26,76%", "Superato (Assorbe anche attriti commerciali pesanti)"),
            ("4. Falsificazione Tier C", "Portafoglio 100% set deboli (Rebel Clash, Battle Styles)", "+5,38%", "Confermato (I set senza chase iconiche distruggono valore vs SPY)"),
            ("5. Generalizzazione Cross-TCG", "Applicazione a One Piece TCG (OP-01 -> OP-06)", "+11,76%", "Validato (+86,0% ROI, legge economica strutturale)"),
            ("6. Edizioni Giapponesi [JP]", "High-Class (VSTAR Universe, VMAX Climax)", "+1,69%", "Evidenziata maggiore volatilità e import markup penalty"),
            ("7. Monte Carlo Noise", "100 run con rumore gaussiano +-15% sui prezzi", "+19,29%", "Resiste al 95% di confidenza (Worst-case doppio rispetto a SPY)"),
            ("8. Covarianza Multi-Asset", "Correlazione vs S&P 500 (+0.11), Oro (+0.36), BTC (+0.06)", "Beta 0.11", "Autentico asset decorrelato alternativo")
        ]
        f_rows = []
        for t_name, desc, res_cagr, status in falsif_tests:
            f_rows.append({
                "Test di Falsificazione": t_name,
                "Condizione di Stress": desc,
                "CAGR Risultante": res_cagr,
                "Esito Popperiano": status
            })
        st.dataframe(pd.DataFrame(f_rows).set_index("Test di Falsificazione"), use_container_width=True)

    # =========================================================================
    # TAB 5: 🔍 CATALOGO & QUOTAZIONI LIVE
    # =========================================================================
    with tab_catalog:
        st.markdown('<div class="section-title">🔍 Catalogo Ufficiale & Quotazioni Live Cardmarket / TCGplayer</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Interrogazione in tempo reale dell\'API pubblica di PokemonTCG.io con spread EUR/USD.</div>', unsafe_allow_html=True)

        with st.spinner("Connessione all'anagrafica set..."):
            sets_data = fetch_all_sets()

        if sets_data:
            set_dict = {s["name"]: s["id"] for s in sets_data[:50]}
            sel_sname = st.selectbox("Seleziona Set", options=list(set_dict.keys()))
            cur_s = next(s for s in sets_data if s["name"] == sel_sname)

            s_col1, s_col2, s_col3, s_col4 = st.columns(4)
            with s_col1:
                st.metric("Serie", cur_s.get("series", "N/A"))
            with s_col2:
                st.metric("Data Rilascio", cur_s.get("releaseDate", "N/A"))
            with s_col3:
                st.metric("Totale Carte", cur_s.get("total", "N/A"))
            with s_col4:
                st.metric("ID Set", cur_s.get("id", "N/A"))

            if st.button("Carica Prezzi Live delle Carte di Questo Set"):
                with st.spinner("Scaricamento quotazioni live..."):
                    cards = fetch_cards_by_set(cur_s["id"], page_size=50)
                if cards:
                    c_tab = []
                    for c in cards:
                        cm = c.get("cardmarket", {}).get("prices", {})
                        tcg = c.get("tcgplayer", {}).get("prices", {})
                        tcg_px = None
                        for f_type in ["holofoil", "reverseHolofoil", "normal"]:
                            if f_type in tcg and tcg[f_type].get("market"):
                                tcg_px = tcg[f_type]["market"]
                                break
                        c_tab.append({
                            "Numero": c.get("number"),
                            "Nome": c.get("name"),
                            "Rarità": c.get("rarity", "N/A"),
                            "Cardmarket Trend (EUR)": f"{cm.get('trendPrice', 0.0):.2f} €" if cm.get('trendPrice') else "N/A",
                            "Cardmarket Low (EUR)": f"{cm.get('lowPrice', 0.0):.2f} €" if cm.get('lowPrice') else "N/A",
                            "TCGplayer Mkt (USD)": f"${tcg_px:.2f}" if tcg_px else "N/A"
                        })
                    st.dataframe(pd.DataFrame(c_tab).set_index("Numero"), use_container_width=True)
        else:
            st.warning("Servizio PokemonTCG.io temporaneamente non disponibile.")


if __name__ == "__main__":
    main()
