"""
app.py — Dashboard Quantitativa Interattiva PokeQuant (Streamlit + Plotly).
Consente di simulare strategie, analizzare il backtest su dati storici reali,
valutare l'arbitraggio di grading e verificare le metriche istituzionali di Alpha e Rischio.
"""

import datetime
import json
from pathlib import Path
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

# Configurazione pagina
st.set_page_config(
    page_title="PokeQuant — Motore Quantitativo Pokémon & One Piece TCG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Stile minimale istituzionale
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #e9ecef;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_cached_data():
    prices_df = load_price_matrix()
    meta = load_metadata()
    if prices_df is None or meta is None:
        prices_df, meta = build_and_cache_universe(force_refresh=False)
    macro_df = load_macro_matrix()
    return prices_df, meta, macro_df


def main():
    st.title("⚡ PokeQuant — Quantitative Investment Engine")
    st.caption("Motore istituzionale per il calcolo di convenienza, Edge e backtest di strategie su carte e prodotti sigillati (Pokémon & One Piece TCG).")

    # Caricamento dati
    with st.spinner("Caricamento serie storiche reali..."):
        full_prices_df, full_metadata, macro_df = get_cached_data()

    # --- SIDEBAR: PARAMETRI GLOBALI & FRIZIONI ---
    st.sidebar.header("⚙️ Parametri di Portafoglio")
    initial_cash = st.sidebar.number_input("Capitale Iniziale (€)", min_value=1000.0, max_value=500000.0, value=10000.0, step=1000.0)
    
    st.sidebar.subheader("Benchmark & Covarianza")
    bench_choice = st.sidebar.selectbox(
        "Benchmark di Confronto",
        options=["S&P 500 Reale (SPY ETF)", "Oro Reale (GLD ETF)", "Bitcoin Reale (BTC)", "Tasso Fisso (10% CAGR)"]
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

    st.sidebar.subheader("Frizioni di Mercato & Custodia")
    platform = st.sidebar.selectbox(
        "Piattaforma Principale di Vendita",
        options=["cardmarket", "ebay", "direct_private"],
        format_func=lambda x: {
            "cardmarket": "Cardmarket (5% fee)",
            "ebay": "eBay (12.5% fee + 0.35€)",
            "direct_private": "Scambio Privato / Fiere (0% fee)"
        }[x]
    )
    absorb_shipping = st.sidebar.checkbox("Venditore assorbe spedizione tracciata", value=False)
    apply_slippage = st.sidebar.checkbox("Slippage di Liquidità (1-2%)", value=True)
    apply_holding = st.sidebar.checkbox("Costi di Custodia/Storage (0.5%/anno)", value=True)

    st.sidebar.subheader("Universo Prodotti")
    universe_filter = st.sidebar.selectbox(
        "Filtro Categoria Asset",
        options=[
            f"Tutti gli Asset ({len(full_metadata)})",
            "Solo Pokémon",
            "Solo One Piece TCG",
            "Solo Edizioni Giapponesi [JP]",
            "Solo Sealed Booster Box"
        ]
    )

    # Filtraggio dinamico dell'universo
    if "Solo One Piece" in universe_filter:
        metadata = {k: v for k, v in full_metadata.items() if v.get("franchise") == "one_piece"}
    elif "Solo Edizioni Giapponesi" in universe_filter:
        metadata = {k: v for k, v in full_metadata.items() if v.get("language") == "jp"}
    elif "Solo Pokémon" in universe_filter:
        metadata = {k: v for k, v in full_metadata.items() if v.get("franchise") == "pokemon"}
    elif "Solo Sealed" in universe_filter:
        metadata = {k: v for k, v in full_metadata.items() if v.get("type") == "sealed"}
    else:
        metadata = full_metadata

    active_cols = [c for c in full_prices_df.columns if c in metadata]
    prices_df = full_prices_df[active_cols]

    # --- TABS PRINCIPALI ---
    tab1, tab_radar, tab2, tab3, tab4 = st.tabs([
        "📈 Backtest Strategie & Alpha",
        "📡 Radar Segnali & Automazione",
        "⚖️ Arbitraggio Grading (PSA)",
        "🔍 Catalogo & Dati Real-Time",
        "🛡️ Audit & Validazione Statistica"
    ])

    # =========================================================================
    # TAB 1: BACKTEST STRATEGIE & ROTAZIONE CAPITALE
    # =========================================================================
    with tab1:
        st.subheader("Simulazione Strategie su Prezzi Reali & Rotazione Dinamica del Capitale")
        st.caption("Esegue la strategia su 68 mesi di prezzi reali (2021-2026), gestendo liquidità, ingressi su reprint e rotazioni scalari (Tranche 1 & 2).")

        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
        with col_ctrl1:
            strat_choice = st.selectbox(
                "Strategia da Eseguire",
                options=[
                    "Strategia Ottimale Validata (Tier S/A/B con Rotazione)",
                    "Sealed Accumulator (Standard)",
                    "Chase Card Dip Buyer (Singole)",
                    "Confronto Strategie (Ottimale vs Standard vs Singole)"
                ]
            )
        
        with col_ctrl2:
            target_roi_pct = st.slider("Target ROI Uscita Finale Tranche 2 (%)", min_value=50, max_value=250, value=150, step=10) / 100.0
        with col_ctrl3:
            min_hold = st.slider("Periodo Minimo Detenzione Finale (Mesi)", min_value=12, max_value=48, value=30, step=2)

        # Controlli Rotazione del Capitale & PAC
        col_rot1, col_rot2, col_rot3, col_rot4 = st.columns(4)
        with col_rot1:
            enable_rotation = st.checkbox("🔄 Abilita Rotazione Dinamica (Tranche 1)", value=True, help="Vende parzialmente (50%) le posizioni a +70% ROI dopo 18 mesi (Out-of-Print) per liberare liquidità e comprare i nuovi set a MSRP.")
        with col_rot2:
            tranche1_roi = st.slider("Target ROI Tranche 1 (%)", min_value=40, max_value=120, value=70, step=5) / 100.0 if enable_rotation else 0.70
        with col_rot3:
            tranche1_hold = st.slider("Mesi Minimi Tranche 1 (Hold)", min_value=12, max_value=24, value=18, step=1) if enable_rotation else 18
        with col_rot4:
            max_alloc_pct = st.slider("Cap Allocazione per Singolo Set (%)", min_value=5, max_value=25, value=12, step=1, help="10-12% permette di detenere 8-10 set in contemporanea evitando il blocco della liquidità.") / 100.0

        # Filtri avanzati espandibili
        with st.expander("🛠️ Parametri Avanzati (Tiers, PAC Mensile, Finestra di Ingresso)"):
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                selected_tiers = st.multiselect("Filtro Tier Qualitativo", options=["S", "A", "B", "C"], default=["S", "A", "B"])
            with f_col2:
                monthly_dca = st.number_input("PAC Mensile Liquidità (€ / mese)", min_value=0.0, max_value=2000.0, value=0.0, step=50.0, help="Iniezione di risparmio mensile per accumulo continuo.")
            with f_col3:
                buy_min_m = st.number_input("Mese Minimo dal Lancio (Inizio Finestra)", min_value=0, max_value=12, value=4)
                buy_max_m = st.number_input("Mese Massimo dal Lancio (Fine Finestra)", min_value=6, max_value=24, value=14)

        from poke_quant.engine.strategies.optimal_sealed_strategy import OptimalSealedStrategy

        # 1. Strategia Ottimale Validata con Rotazione
        strat_optimal = OptimalSealedStrategy(
            allowed_tiers=selected_tiers,
            min_buy_age_months=buy_min_m,
            max_buy_age_months=buy_max_m,
            max_msrp_multiplier=1.15,
            min_hold_months=min_hold,
            target_roi=target_roi_pct,
            max_hold_months=48,
            max_allocation_pct=max_alloc_pct,
            enable_dynamic_rotation=enable_rotation,
            tranche1_roi=tranche1_roi,
            tranche1_min_hold_months=tranche1_hold,
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
            monthly_cash_injection=monthly_dca
        )
        res_optimal = bt_optimal.run()

        # 2. Sealed Accumulator Standard
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
            monthly_cash_injection=monthly_dca
        )
        res_sealed = bt_sealed.run()

        # 3. Chase Dip Buyer
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
            monthly_cash_injection=monthly_dca
        )
        res_chase = bt_chase.run()

        # Selezione dei risultati da visualizzare
        if "Ottimale" in strat_choice:
            active_results = [res_optimal]
        elif "Sealed Accumulator" in strat_choice:
            active_results = [res_sealed]
        elif "Chase Card" in strat_choice:
            active_results = [res_chase]
        else:
            active_results = [res_optimal, res_sealed, res_chase]

        # Tabella Metriche Riassuntive
        metric_rows = []
        for r in active_results:
            metric_rows.append({
                "Strategia": r.strategy_name.replace("Strategy", ""),
                "Capitale Finale": f"{r.final_nav:,.2f} €",
                "ROI Netto": f"{r.total_net_return*100:+.1f}%",
                "CAGR Netto": f"{r.cagr*100:+.2f}%",
                "Alpha Netto": f"{r.alpha_annualized*100:+.2f}%",
                "Sharpe": f"{r.sharpe:.2f}",
                "Max Drawdown": f"{r.max_drawdown*100:.2f}%",
                "Rotazioni (Tranche 1)": r.rotation_trades_count,
                "Turnover": f"{r.turnover_ratio*100:.1f}%",
                "Trades Chiusi": r.total_trades,
                "Posizioni Aperte": len(r.open_positions),
                "Win Rate": f"{r.win_rate*100:.1f}%",
                "Frizioni Pagate": f"{r.total_fees_paid:,.2f} €"
            })
        st.dataframe(pd.DataFrame(metric_rows).set_index("Strategia"), use_container_width=True)

        # Regimi Macro
        with st.expander("📊 Rendimento per Regime Macro Economico (Optimal Sealed Strategy)"):
            r_cols = st.columns(len(res_optimal.regime_performance))
            for i, (regime, val) in enumerate(res_optimal.regime_performance.items()):
                with r_cols[i]:
                    st.metric(regime.split(" ")[0], f"{val*100:+.1f}%")

        # Grafico Plotly NAV Curve
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3], subplot_titles=("Evoluzione del Capitale Netto (NAV)", "Drawdown Storico"))
        
        # Benchmark
        bench_df = res_sealed.benchmark_nav
        fig.add_trace(go.Scatter(
            x=bench_df.index, y=bench_df.values,
            mode='lines', name=f"Benchmark ({bench_choice})",
            line=dict(color='gray', dash='dash', width=2)
        ), row=1, col=1)

        colors = {"OptimalSealedStrategy": "#2b8a3e", "SealedAccumulatorStrategy": "#1971c2", "ChaseDipBuyerStrategy": "#e8590c"}
        for r in active_results:
            c = colors.get(r.strategy_name, "#495057")
            fig.add_trace(go.Scatter(
                x=r.nav_history.index, y=r.nav_history["nav"],
                mode='lines', name=r.strategy_name.replace("Strategy", ""),
                line=dict(color=c, width=2.5)
            ), row=1, col=1)

            # Drawdown
            nav = r.nav_history["nav"]
            dd = (nav / nav.cummax() - 1.0) * 100.0
            fig.add_trace(go.Scatter(
                x=dd.index, y=dd.values,
                mode='lines', fill='tozeroy',
                name=f"DD {r.strategy_name.replace('Strategy', '')}",
                line=dict(color=c, width=1)
            ), row=2, col=1)

        fig.update_layout(height=550, margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified")
        fig.update_yaxes(title_text="Euro (€)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

        # SEZIONE 1: PORTAFOGLIO ATTIVO IN DETENZIONE (POSIZIONI APERTE)
        st.subheader("💼 Portafoglio Attivo in Detenzione (Posizioni Aperte a Fine Backtest)")
        st.caption("Asset reali ancora custoditi in inventario al termine del periodo (Mark-to-Market su prezzi di clearing reali).")
        
        primary_res = res_optimal if "Ottimale" in strat_choice or len(active_results) == 1 else active_results[0]
        if primary_res.open_positions:
            tot_inv_val = sum(p["current_value"] for p in primary_res.open_positions)
            tot_inv_cost = sum(p["total_cost"] for p in primary_res.open_positions)
            tot_unrealized_pnl = tot_inv_val - tot_inv_cost
            tot_unrealized_roi = (tot_unrealized_pnl / tot_inv_cost) * 100 if tot_inv_cost > 0 else 0.0
            last_cash = float(primary_res.nav_history["cash"].iloc[-1])

            op_m1, op_m2, op_m3, op_m4, op_m5 = st.columns(5)
            with op_m1:
                st.metric("Box in Inventario", f"{sum(p['quantity'] for p in primary_res.open_positions)} pz", f"{len(primary_res.open_positions)} Set")
            with op_m2:
                st.metric("Valore Attuale Inventario", f"{tot_inv_val:,.2f} €")
            with op_m3:
                st.metric("Capitale di Carico", f"{tot_inv_cost:,.2f} €")
            with op_m4:
                st.metric("PnL Non Realizzato", f"{tot_unrealized_pnl:+,.2f} €", f"{tot_unrealized_roi:+.1f}% ROI")
            with op_m5:
                st.metric("Cassa Residua Libera", f"{last_cash:,.2f} €")

            op_rows = []
            for p in primary_res.open_positions:
                op_rows.append({
                    "Articolo": p["item_name"],
                    "Tipo": p["item_type"],
                    "Q.tà (Box)": p["quantity"],
                    "Data Acquisto": p["buy_date"],
                    "Holding (Mesi)": p["holding_months"],
                    "Carico Unitario": f"{p['buy_price_unit']:.2f} €",
                    "Prezzo Attuale": f"{p['current_price']:.2f} €",
                    "Valore di Mercato": f"{p['current_value']:,.2f} €",
                    "PnL Non Realizzato": f"{p['unrealized_pnl']:+,.2f} €",
                    "ROI Non Realizzato": f"{p['unrealized_roi']*100:+.1f}%"
                })
            st.dataframe(pd.DataFrame(op_rows).set_index("Articolo"), use_container_width=True)
        else:
            st.info("Nessuna posizione aperta in portafoglio a fine periodo (100% liquidità).")

        # SEZIONE 2: CRONOLOGIA COMPLETA DEI SEGNALI STORICI (2021-2026)
        st.subheader("📡 Cronologia Completa dei Segnali Storici Generati (2021 - 2026)")
        st.caption("Registro cronologico completo di ogni decisione generata dal motore di trading: BUY su reprint dip, vendite parziali Tranche 1 e uscite Tranche 2.")
        
        if primary_res.signals_history:
            sig_df = pd.DataFrame(primary_res.signals_history)
            sig_df["Year"] = sig_df["date"].str[:4]

            # Filtri interattivi
            c_sf1, c_sf2, c_sf3 = st.columns([2, 2, 2])
            with c_sf1:
                year_filter = st.selectbox(
                    "Filtra per Anno",
                    options=["Tutti gli Anni (2021-2026)"] + sorted(sig_df["Year"].unique().tolist())
                )
            with c_sf2:
                action_filter = st.selectbox(
                    "Filtra per Tipologia Azione",
                    options=["Tutte le Azioni", "Solo BUY (Acquisti)", "Solo SELL (Vendite & Rotazioni)"]
                )
            with c_sf3:
                st.metric("Totale Segnali Generati", f"{len(sig_df)} Segnali", help="Numero complessivo di eventi operativi eseguiti")

            # Grafico a barre segnali per anno
            yearly_counts = sig_df.groupby(["Year", "action"]).size().unstack(fill_value=0)
            fig_sig = go.Figure()
            if "BUY" in yearly_counts.columns:
                fig_sig.add_trace(go.Bar(x=yearly_counts.index, y=yearly_counts["BUY"], name="BUY (Acquisto)", marker_color="#2b8a3e"))
            if "SELL" in yearly_counts.columns:
                fig_sig.add_trace(go.Bar(x=yearly_counts.index, y=yearly_counts["SELL"], name="SELL (Rotazione / Uscita)", marker_color="#e03131"))
            fig_sig.update_layout(
                title="Distribuzione Temporale dei Segnali Operativi per Anno (2021-2026)",
                barmode="group", height=280, margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_sig, use_container_width=True)

            # Applicazione filtri
            filtered_sigs = sig_df.copy()
            if year_filter != "Tutti gli Anni (2021-2026)":
                filtered_sigs = filtered_sigs[filtered_sigs["Year"] == year_filter]
            if "BUY" in action_filter:
                filtered_sigs = filtered_sigs[filtered_sigs["action"] == "BUY"]
            elif "SELL" in action_filter:
                filtered_sigs = filtered_sigs[filtered_sigs["action"] == "SELL"]

            # Tabella formattata segnali
            sig_disp = []
            for _, s in filtered_sigs.iterrows():
                badge_action = "🟢 BUY" if s["action"] == "BUY" else "🔴 SELL"
                sig_disp.append({
                    "Data": s["date"],
                    "Azione": badge_action,
                    "Prodotto": s["item_name"],
                    "Q.tà": s["quantity"],
                    "Prezzo Unit.": f"{s['price']:.2f} €",
                    "Controvalore": f"{s['total_value']:,.2f} €",
                    "Cassa Prima": f"{s['portfolio_cash_before']:,.2f} €",
                    "Motivazione & Trigger": s["reason"]
                })
            st.dataframe(pd.DataFrame(sig_disp), use_container_width=True, hide_index=True)
        else:
            st.info("Nessun segnale generato con i parametri attuali.")

        # SEZIONE 3: REGISTRO OPERAZIONI CONCLUSE (TRADES REALIZZATI)
        st.subheader("📜 Registro Operazioni Concluse (Trades Realizzati)")
        all_trades = pd.concat([r.trades_df for r in active_results if not r.trades_df.empty])
        if not all_trades.empty:
            disp_trades = all_trades.copy()
            disp_trades["Prezzo Acq. Unit."] = disp_trades["buy_price_unit"].apply(lambda x: f"{x:.2f} €")
            disp_trades["Prezzo Vend. Unit."] = disp_trades["sell_price_unit"].apply(lambda x: f"{x:.2f} €")
            disp_trades["Incasso Lordo"] = disp_trades["gross_proceeds"].apply(lambda x: f"{x:.2f} €")
            disp_trades["Fee & Frizioni"] = disp_trades["fees_paid"].apply(lambda x: f"{x:.2f} €")
            disp_trades["PnL Netto"] = disp_trades["net_pnl"].apply(lambda x: f"{x:+,.2f} €")
            disp_trades["ROI Netto %"] = disp_trades["net_roi"].apply(lambda x: f"{x*100:+.1f}%")
            cols_show = ["item_name", "item_type", "quantity", "buy_date", "sell_date", "holding_months", "Prezzo Acq. Unit.", "Prezzo Vend. Unit.", "Fee & Frizioni", "PnL Netto", "ROI Netto %"]
            st.dataframe(disp_trades[cols_show].rename(columns={"item_name": "Articolo", "item_type": "Tipo", "quantity": "Q.tà", "buy_date": "Data Acq.", "sell_date": "Data Vend.", "holding_months": "Mesi Hold"}), use_container_width=True)
        else:
            st.info("Nessun trade chiuso nel periodo con i parametri correnti (posizioni ancora in hold).")

    # =========================================================================
    # TAB RADAR: SEGNALI LIVE & AUTOMAZIONE
    # =========================================================================
    with tab_radar:
        st.subheader("📡 Radar Segnali Operativi in Tempo Reale & Scanner Storico")
        st.markdown("""
        Questo modulo analizza l'intero catalogo di mercato (Pokémon & One Piece TCG) e le posizioni registrate,
        identificando i segnali di **BUY** (finestra di ristampa a sconto), **Tranche 1 Rotazione** (sblocco liquidità al 50% dopo 18m) e **Tranche 2** (uscita finale).
        """)

        from poke_quant.signal_scanner import scan_signals, scan_historical_signals, format_telegram_alert, send_telegram_message

        current_px_map = prices_df.iloc[-1].to_dict()
        radar_tiers = st.multiselect("Tier Monitorati dal Radar", options=["S", "A", "B", "C"], default=["S", "A", "B"], key="radar_tiers_sel")
        scan_res = scan_signals(current_prices=current_px_map, metadata=metadata, allowed_tiers=radar_tiers)

        buys = scan_res.get("buy_signals", [])
        sells = scan_res.get("sell_signals", [])
        watchlist = scan_res.get("watchlist", [])

        # Sezione BUY
        st.markdown("### 🟢 Segnali di Acquisto di Mercato (BUY)")
        if buys:
            for b in buys:
                st.success(f"""
                **{b['name']}** (Tier {b['tier']})  
                • Prezzo Corrente: **{b['current_price']:.1f} €** (MSRP: {b['msrp']:.1f} € | {b['diff_vs_msrp_pct']:+.1f}%)  
                • Età del Set: **{b['age_months']} mesi** (Piena finestra di ristampa 4-14 mesi)  
                • **Azione**: Acquisto consigliato (allocazione target 10-12% del capitale).
                """)
        else:
            st.info("Nessun set attualmente nella finestra di acquisto (Mesi 4-14) con prezzo <= 1.15x MSRP. I set monitorati rimangono in watchlist.")

        # Sezione SELL (Rotazioni Tranche 1 & Uscite Finali)
        st.markdown("### 🔄 Segnali di Vendita & Rotazione Capitale (SELL)")
        if sells:
            for s in sells:
                sig_t = s.get("signal_type", "SELL")
                if "TRANCHE 1" in sig_t:
                    st.warning(f"""
                    **🔄 {s['name']} — ROTAZIONE TRANCHE 1**  
                    • **Consiglio Operativo**: Vendere **{s['quantity']} su {s['total_quantity']} box** per liberare liquidità.  
                    • Prezzo Unitario Stimato: **{s['current_price']:.1f} €** (Carico: {s['buy_price']:.1f} €)  
                    • Rendimento Netto: **+{s['net_roi_pct']:.1f}%** | Incasso Netto Stimato: **{s['net_proceeds']:.1f} €**  
                    • Holding Period: **{s['holding_months']} mesi** (Fase Out-of-Print attiva).  
                    • **Motivazione**: Sbloccare capitale per reinvestire nei nuovi set di Tier S/A/B in finestra di sconto.
                    """)
                else:
                    st.error(f"""
                    **🔴 {s['name']} — USCITA FINALE (TRANCHE 2)** (Q.tà: {s['quantity']} pz)  
                    • Prezzo Unitario Stimato: **{s['current_price']:.1f} €** (Carico: {s['buy_price']:.1f} €)  
                    • Rendimento Netto: **+{s['net_roi_pct']:.1f}%** | Incasso Netto Stimato: **{s['net_proceeds']:.1f} €**  
                    • Holding Period: **{s['holding_months']} mesi**  
                    • **Trigger**: {s['trigger']}  
                    • **Azione**: Mettere in vendita su Cardmarket.
                    """)
        else:
            st.info("Nessuna posizione in portafoglio ha attualmente raggiunto le soglie di rotazione Tranche 1 (+70% a 18m) o Tranche 2 (+150% a 30m).")

        # Watchlist
        if watchlist:
            with st.expander(f"👀 Watchlist Prodotti Monitorati ({len(watchlist)})"):
                st.dataframe(pd.DataFrame(watchlist)[["name", "tier", "age_months", "current_price", "msrp", "status"]].rename(columns={
                    "name": "Prodotto", "tier": "Tier", "age_months": "Età (Mesi)", "current_price": "Prezzo (€)", "msrp": "MSRP (€)", "status": "Stato Monitoraggio"
                }), use_container_width=True)

        # SEZIONE STORICO SEGNALI RADAR MULTI-ANNO
        with st.expander("📜 Esplora Segnali Storici del Radar (Timeline 2021 - 2026)"):
            st.caption("Visualizza l'attivazione storica degli alert d'acquisto e di vendita per ogni set dal 2021 a oggi.")
            hist_radar_df = scan_historical_signals(prices_df=prices_df, metadata=metadata, allowed_tiers=radar_tiers)
            if not hist_radar_df.empty:
                hist_radar_df["Year"] = hist_radar_df["date"].str[:4]
                yr_rad = st.selectbox("Seleziona Anno Storico", options=["Tutti gli Anni"] + sorted(hist_radar_df["Year"].unique().tolist()), key="yr_rad_sel")
                
                disp_rad = hist_radar_df.copy()
                if yr_rad != "Tutti gli Anni":
                    disp_rad = disp_rad[disp_rad["Year"] == yr_rad]
                
                rows_rad = []
                for _, r in disp_rad.iterrows():
                    rows_rad.append({
                        "Data Segnale": r["date"],
                        "Azione": "🟢 BUY" if r["action"] == "BUY" else "🔴 SELL",
                        "Set / Prodotto": r["item_name"],
                        "Quantità": r["quantity"],
                        "Prezzo (€)": f"{r['price']:.2f} €",
                        "Valore (€)": f"{r['total_value']:,.2f} €",
                        "Trigger / Motivazione": r["reason"]
                    })
                st.dataframe(pd.DataFrame(rows_rad), use_container_width=True, hide_index=True)

        # Sezione Gestione Posizioni Reali del Portafoglio
        from poke_quant.signal_scanner import load_user_holdings
        st.markdown("---")
        st.markdown("### 💼 Gestione Posizioni Possedute (`portfolio_holdings.json`)")
        holdings_data = load_user_holdings()
        if holdings_data:
            holdings_rows = []
            for h in holdings_data:
                iid = h.get("item_id")
                px = current_px_map.get(iid, h.get("buy_price_unit", 0.0))
                b_px = h.get("buy_price_unit", 0.0)
                qty = h.get("quantity", 1)
                cost = b_px * qty
                cur_val = px * qty
                net_val = (px * 0.95 - 0.60) * qty
                pnl = net_val - cost
                roi = pnl / cost if cost > 0 else 0.0
                holdings_rows.append({
                    "ID": iid,
                    "Nome Prodotto": h.get("name", iid),
                    "Q.tà": qty,
                    "Data Acquisto": h.get("buy_date"),
                    "Prezzo Carico": f"{b_px:.1f} €",
                    "Prezzo Attuale": f"{px:.1f} €",
                    "Valore Netto": f"{net_val:.1f} €",
                    "PnL Netto": f"{pnl:+,.1f} €",
                    "ROI Netto %": f"{roi*100:+.1f}%"
                })
            st.dataframe(pd.DataFrame(holdings_rows).set_index("ID"), use_container_width=True)

        with st.expander("➕ Registra Nuovo Acquisto Box nel Portafoglio"):
            with st.form("add_holding_form"):
                ah_col1, ah_col2 = st.columns(2)
                with ah_col1:
                    available_items = [k for k, v in metadata.items() if v.get("type") == "sealed"]
                    selected_item_id = st.selectbox("Seleziona Prodotto", options=available_items, format_func=lambda x: metadata[x].get("name", x))
                    item_qty = st.number_input("Quantità (Box)", min_value=1, max_value=50, value=1)
                with ah_col2:
                    default_buy_px = metadata[selected_item_id].get("msrp", 140.0)
                    buy_price_in = st.number_input("Prezzo Unitario di Acquisto (€)", min_value=10.0, max_value=5000.0, value=float(default_buy_px))
                    buy_date_in = st.date_input("Data di Acquisto", value=datetime.date.today())

                if st.form_submit_button("Salva nel Portafoglio"):
                    holdings_file = Path(__file__).resolve().parent / "data_cache" / "portfolio_holdings.json"
                    curr_list = load_user_holdings()
                    curr_list.append({
                        "item_id": selected_item_id,
                        "name": metadata[selected_item_id].get("name", selected_item_id),
                        "quantity": int(item_qty),
                        "buy_date": buy_date_in.strftime("%Y-%m-%d"),
                        "buy_price_unit": float(buy_price_in)
                    })
                    with open(holdings_file, "w", encoding="utf-8") as f:
                        json.dump(curr_list, f, indent=2)
                    st.success(f"Posizione {metadata[selected_item_id].get('name')} registrata con successo!")
                    st.rerun()

        # Sezione Automazione Notifiche Telegram
        st.markdown("---")
        st.markdown("### 🤖 Configurazione Notifiche Push Telegram")
        st.markdown("""
        Lo scanner può inviare alert automatici sul tuo smartphone via Telegram ogni lunedì mattina,
        utilizzando il workflow programmato in `.github/workflows/poke_signals.yml` o tramite cron locale.
        """)

        t_col1, t_col2 = st.columns([3, 1])
        with t_col1:
            st.code("python poke_quant/signal_scanner.py", language="bash")
        with t_col2:
            if st.button("🔔 Invia Notifica Telegram Adesso"):
                msg = format_telegram_alert(scan_res)
                sent = send_telegram_message(msg)
                if sent:
                    st.success("Notifica inviata con successo su Telegram!")
                else:
                    st.warning("Credenziali TELEGRAM_TOKEN o TELEGRAM_CHAT_ID non trovate nelle variabili d'ambiente.")

    # =========================================================================
    # TAB 2: ARBITRAGGIO GRADING
    # =========================================================================
    with tab2:
        st.subheader("Calcolatore di Arbitraggio Statistico Grading PSA / BGS")
        st.markdown("""
        Valuta se l'acquisto di una carta non gradata (Raw Near Mint) presenta un **Valore Atteso Netto positivo**
        dopo aver contabilizzato il costo del servizio PSA, la spedizione, il fermo capitale e la probabilità reale di prendere 10 (*Gem Rate*).
        """)

        col_g1, col_g2, col_g3 = st.columns(3)
        with col_g1:
            raw_input = st.number_input("Prezzo Acquisto Carta Raw (€)", min_value=1.0, max_value=5000.0, value=80.0, step=5.0)
            gem_rate_input = st.slider("Gem Rate Stimata (Probabilità PSA 10)", min_value=0.10, max_value=0.95, value=0.70, step=0.05)
        with col_g2:
            psa10_input = st.number_input("Prezzo di Mercato PSA 10 (€)", min_value=1.0, max_value=25000.0, value=350.0, step=10.0)
            psa9_input = st.number_input("Prezzo di Mercato PSA 9 (€)", min_value=1.0, max_value=10000.0, value=75.0, step=5.0)
        with col_g3:
            grading_fee_input = st.number_input("Costo Servizio Grading All-In (€)", min_value=10.0, max_value=150.0, value=25.0, step=5.0)
            turnaround_input = st.number_input("Mesi di Fermo Capitale (Turnaround)", min_value=1, max_value=6, value=2)

        # Calcolo EV
        prob_10 = gem_rate_input
        prob_9 = (1.0 - gem_rate_input) * 0.80
        prob_8 = max(0.0, 1.0 - prob_10 - prob_9)
        psa8_val = raw_input * 0.50

        ev_gross = (prob_10 * psa10_input) + (prob_9 * psa9_input) + (prob_8 * psa8_val)
        fee_info = PLATFORM_FEES[platform]
        platform_fee = ev_gross * fee_info["percentage"] + fee_info["fixed_fee"]
        ev_net_proceeds = ev_gross - platform_fee - 0.60
        
        total_cost = raw_input + grading_fee_input
        expected_profit = ev_net_proceeds - total_cost
        expected_roi = expected_profit / total_cost if total_cost > 0 else 0.0

        st.markdown("---")
        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        with res_col1:
            st.metric("Valore Atteso Netto Incasso", f"{ev_net_proceeds:.2f} €")
        with res_col2:
            st.metric("Costo Base Totale", f"{total_cost:.2f} €")
        with res_col3:
            st.metric("Profitto Netto Atteso", f"{expected_profit:+,.2f} €", delta=f"{expected_roi*100:+.1f}% ROI")
        with res_col4:
            if expected_roi >= 0.25:
                st.success("✅ OPERAZIONE CONSIGLIATA\n(Edge sufficiente per il rischio)")
            else:
                st.warning("⚠️ SCONSIGLIATA\n(Margine insufficiente)")

        # Scanner rapido sul nostro universo
        st.subheader("Opportunità di Grading nel Catalogo Attuale")
        sample_cards = [
            {"Nome": "Umbreon VMAX Alt Art", "Raw (€)": 1990.0, "PSA 10 (€)": 3800.0, "Gem Rate": 0.72},
            {"Nome": "Rayquaza VMAX Alt Art", "Raw (€)": 1018.0, "PSA 10 (€)": 1950.0, "Gem Rate": 0.68},
            {"Nome": "Giratina V Alt Art", "Raw (€)": 695.0, "PSA 10 (€)": 1400.0, "Gem Rate": 0.65},
            {"Nome": "Gengar VMAX Alt Art", "Raw (€)": 848.0, "PSA 10 (€)": 1550.0, "Gem Rate": 0.70},
            {"Nome": "Charizard V Alt Art", "Raw (€)": 247.0, "PSA 10 (€)": 550.0, "Gem Rate": 0.75},
        ]
        scanner_rows = []
        for c in sample_cards:
            res = evaluate_grading_arbitrage(
                raw_price=c["Raw (€)"],
                psa10_price=c["PSA 10 (€)"],
                gem_rate=c["Gem Rate"],
                platform=platform
            )
            scanner_rows.append({
                "Carta": c["Nome"],
                "Prezzo Raw": f"{c['Raw (€)']:.0f} €",
                "Prezzo PSA 10": f"{c['PSA 10 (€)']:.0f} €",
                "Gem Rate": f"{c['Gem Rate']*100:.0f}%",
                "Valore Atteso Netto": f"{res.expected_graded_net:.1f} €",
                "Profitto Netto Atteso": f"{res.expected_net_profit:+,.1f} €",
                "ROI Netto %": f"{res.expected_net_roi*100:+.1f}%",
                "Esito": "Consigliato" if res.is_favorable else "Neutro/Rischioso"
            })
        st.dataframe(pd.DataFrame(scanner_rows).set_index("Carta"), use_container_width=True)

    # =========================================================================
    # TAB 3: CATALOGO & DATI LIVE POKEMONTCG
    # =========================================================================
    with tab3:
        st.subheader("Consultazione Catalogo e Prezzi Correnti (Cardmarket EUR & TCGplayer USD)")
        st.markdown("Integrazione diretta con l'API pubblica di PokemonTCG.io.")

        with st.spinner("Caricamento set ufficiali..."):
            sets_data = fetch_all_sets()

        if sets_data:
            set_options = {s["name"]: s["id"] for s in sets_data[:40]}
            selected_set_name = st.selectbox("Seleziona Set Ufficiale", options=list(set_options.keys()))
            selected_set_id = set_options[selected_set_name]

            # Mostra info set
            curr_set = next(s for s in sets_data if s["id"] == selected_set_id)
            st.write(f"**Serie:** {curr_set.get('series')} | **Data Rilascio:** {curr_set.get('releaseDate')} | **Totale Carte:** {curr_set.get('total')}")

            if st.button("Carica Carte e Prezzi Correnti di Questo Set"):
                with st.spinner("Scaricamento quotazioni di mercato..."):
                    cards = fetch_cards_by_set(selected_set_id, page_size=50)
                
                if cards:
                    card_table = []
                    for c in cards:
                        cm = c.get("cardmarket", {}).get("prices", {})
                        tcg = c.get("tcgplayer", {}).get("prices", {})
                        
                        # Prendi prezzo holofoil o normal
                        tcg_mkt = None
                        for finish in ["holofoil", "reverseHolofoil", "normal"]:
                            if finish in tcg and tcg[finish].get("market"):
                                tcg_mkt = tcg[finish]["market"]
                                break

                        card_table.append({
                            "Numero": c.get("number"),
                            "Nome": c.get("name"),
                            "Rarità": c.get("rarity", "N/A"),
                            "Cardmarket Trend (EUR)": f"{cm.get('trendPrice', 0.0):.2f} €" if cm.get('trendPrice') else "N/A",
                            "Cardmarket Low (EUR)": f"{cm.get('lowPrice', 0.0):.2f} €" if cm.get('lowPrice') else "N/A",
                            "Cardmarket 30d Avg (EUR)": f"{cm.get('avg30', 0.0):.2f} €" if cm.get('avg30') else "N/A",
                            "TCGplayer Mkt (USD)": f"${tcg_mkt:.2f}" if tcg_mkt else "N/A",
                        })
                    st.dataframe(pd.DataFrame(card_table).set_index("Numero"), use_container_width=True)
                else:
                    st.warning("Nessuna carta trovata per questo set.")
        else:
            st.warning("Impossibile contattare l'API PokemonTCG.io al momento.")

    # =========================================================================
    # TAB 4: AUDIT & VALIDAZIONE STATISTICA
    # =========================================================================
    with tab4:
        st.subheader("Validazione Quantitativa Istituzionale Anti-Overfitting")
        st.markdown("""
        In finanza quantitativa, selezionare una strategia che ha performato bene nel passato comporta spesso il rischio di **data-snooping** (overfitting).
        Adottiamo le metodologie di **David Bailey e Marcos López de Prado**:
        """)

        st.markdown("""
        1. **Deflated Sharpe Ratio (DSR)**: Corregge lo Sharpe osservato per il numero di test o varianti di parametri provate (`n_trials`).
           Risponde alla domanda: *'Qual è la probabilità che questo Sharpe sia reale e non frutto del caso, dato quante combinazioni abbiamo esplorato?'*
        2. **Probability of Backtest Overfitting (PBO)** via *Combinatorially Symmetric Cross-Validation (CSCV)*:
           Verifica quanto spesso la strategia migliore in-sample finisce sotto la mediana fuori campione.
        """)

        n_trials_input = st.slider("Numero di configurazioni/varianti testate (n_trials)", min_value=1, max_value=100, value=10, step=1)
        
        dsr_s = deflated_sharpe_ratio(observed_sr=res_sealed.sharpe, n_trials=n_trials_input, n_obs=len(res_sealed.monthly_returns))
        dsr_c = deflated_sharpe_ratio(observed_sr=res_chase.sharpe, n_trials=n_trials_input, n_obs=len(res_chase.monthly_returns))

        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.markdown(f"### Sealed Box Accumulator")
            st.metric("Sharpe Ratio Osservato", f"{res_sealed.sharpe:.2f}")
            st.metric("Deflated Sharpe Ratio (DSR)", f"{dsr_s:.4f}", help="> 0.95 indica significatività statistica istituzionale")
            if dsr_s >= 0.95:
                st.success("✅ **Edge Validato Statiticamente**: La sovraperformance dei Booster Box sigillati resiste alla correzione per test multipli.")
            else:
                st.warning("⚠️ Confidenza moderata o ridotta dal numero di test.")

        with col_v2:
            st.markdown(f"### Chase Card Dip Buyer")
            st.metric("Sharpe Ratio Osservato", f"{res_chase.sharpe:.2f}")
            st.metric("Deflated Sharpe Ratio (DSR)", f"{dsr_c:.4f}")
            if dsr_c < 0.50:
                st.error("❌ **Edge Non Significativo**: Il timing delle singole non dimostra un alpha netto robusto dopo aver dedotto commissioni e correzione per tentativi.")
            else:
                st.info("Performance entro i limiti di varianza campionaria.")

        st.markdown("---")
        st.markdown("### Regola di Cautela Istituzionale")
        st.info("""
        **Sintesi dell'Audit**: I dati reali confermano che il vero Alpha nei collezionabili Pokémon risiede nello **shock d'offerta strutturale dei prodotti sigillati (Booster Box Out-of-Print)**, dove le aperture continue riducono l'offerta mondiale.
        Al contrario, il trading attivo di singole carte soffre di attrito elevato (commissioni 5-12%, spese di spedizione per singolo pezzo) e svalutazione post-hype, rendendo il rischio/rendimento sfavorevole rispetto a un indice azionario standard.
        """)


if __name__ == "__main__":
    main()
