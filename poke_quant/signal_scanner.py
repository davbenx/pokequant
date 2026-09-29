"""
poke_quant/signal_scanner.py — Scanner settimanale dei segnali BUY/AVOID di
mercato + segnali SELL sulle posizioni possedute (portfolio_holdings.json).
Invia notifiche push via Telegram (se configurato) o stampa report da riga
di comando.

RICONCILIATO (2026-09-29, richiesta esplicita dell'utente "Riconcilia
signal_scanner.py con la strategia validata"): fino a questa versione questo
scanner girava ogni lunedi' (poke_signals.yml) con una strategia INDIPENDENTE
e MAI validata con DSR/PBO (finestra eta' 4-14 mesi + prezzo <= 1.15x MSRP su
Tier S/A/B, poi rigettata come baseline in poke_quant/falsification_suite.py)
- poteva mandare un "COMPRA SUBITO" reale su Telegram che contraddiceva la
dashboard nella STESSA settimana. Ora i segnali BUY/AVOID di mercato vengono
dalle stesse identiche funzioni gia' usate da run_monthly_production_signal.py
(l'orchestratore mensile autoritativo) e da app.py (la dashboard):
compute_signal_rows() per i box (TS Momentum, DSR 0,778 sull'universo
liquido) e compute_singles_signal_rows()/compute_singles_avoid_rows() per le
singole (Fattore Scarsita', DSR 0,980). Una sola fonte di verita' - questo
scanner e' ora solo un CHECK PIU' FREQUENTE (settimanale) sugli stessi
segnali, non una seconda strategia.

NOTA sulla cadenza: i prezzi PriceCharting si aggiornano una volta al mese
(rebuild_prices_with_real_fx.py gira dentro l'orchestratore mensile, non
qui) - un run settimanale di questo scanner ricalcola quindi lo STESSO
segnale sugli STESSI prezzi per ~3 settimane su 4, e cambia solo quando il
refresh mensile e' passato. E' un comportamento corretto (nessun dato nuovo
= nessun segnale nuovo), non un bug: il valore di girare ogni settimana è
un promemoria/backup del canale Telegram, non una fonte di dati piu' fresca.

Il tracking delle posizioni POSSEDUTE (tranche di rotazione +70%/+150%/
time-stop 48 mesi) resta un blocco distinto e INFORMATIVO, come "Singole da
evitare/vendere" in app.py: nessun backtest valida quelle soglie di uscita,
sono euristiche operative sul portafoglio reale dell'utente, non sostituiscono
i segnali AVOID/SELL validati (box, momentum invertito) o l'informativo
"sopravvalutate" (singole) inclusi qui sopra.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import datetime
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_price_matrix, load_metadata
from poke_quant.engine.friction import calculate_sale_friction
from poke_quant.notify.telegram import send_telegram_message
from scripts.generate_monthly_signal import compute_signal_rows
from scripts.generate_singles_signal import compute_singles_signal_rows, compute_singles_avoid_rows

logger = logging.getLogger(__name__)


def load_user_holdings() -> List[Dict[str, Any]]:
    """Carica il registro delle posizioni reali possedute dall'utente."""
    path = Path(__file__).resolve().parent.parent / "data_cache" / "portfolio_holdings.json"
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Errore caricamento portfolio_holdings.json: {e}")
        return []


def scan_signals(
    current_prices: Optional[Dict[str, float]] = None,
    today_dt: Optional[datetime.date] = None,
    box_rows: Optional[List[Dict[str, Any]]] = None,
    singles_buy_rows: Optional[List[Dict[str, Any]]] = None,
    singles_avoid_rows: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Esegue la scansione completa per generare:
      - Segnali BUY/AVOID di mercato su box e singole, dalla stessa fonte
        validata della dashboard e dell'orchestratore mensile (compute_signal_rows,
        compute_singles_signal_rows, compute_singles_avoid_rows - vedi docstring
        del modulo). box_rows/singles_buy_rows/singles_avoid_rows sono
        iniettabili (usati dai test) e per default vengono calcolati per davvero.
      - Segnali SELL di portafoglio (Tranche 1 a 18m/+70%, Tranche 2 a 30m/+150%,
        Time-Stop 48m) sulle posizioni in portfolio_holdings.json - INFORMATIVO,
        non validato con un backtest (vedi docstring del modulo).
    """
    if today_dt is None:
        today_dt = datetime.date.today()

    if box_rows is None:
        box_rows, _ = compute_signal_rows()
    if singles_buy_rows is None:
        singles_buy_rows, _ = compute_singles_signal_rows()
    if singles_avoid_rows is None:
        singles_avoid_rows, _ = compute_singles_avoid_rows()

    if current_prices is None:
        prices_df = load_price_matrix()
        if prices_df is not None and not prices_df.empty:
            current_prices = prices_df.iloc[-1].to_dict()
        else:
            current_prices = {}

    # SEGNALI DI MERCATO (stessa fonte validata di app.py e run_monthly_production_signal.py)
    box_buy_signals = [r for r in box_rows if r["signal"] == "BUY/HOLD"]
    box_reversal_signals = [r for r in box_rows if r["signal"] == "AVOID/SELL"]
    box_watchlist = [
        r for r in box_rows
        if r["signal"] in ("PREZZO ECCESSIVO (oltre tetto MSRP)", "VERIFICARE A MANO (rendimento implausibile)")
    ]

    # 2. SCANSIONE SEGNALI SELL SU POSIZIONI POSSEDUTE (ROTAZIONE TRANCHE 1 & 2)
    sell_signals = []
    holdings = load_user_holdings()
    for h in holdings:
        item_id = h.get("item_id")
        cur_px = current_prices.get(item_id, 0.0)
        buy_px = h.get("buy_price_unit", 0.0)
        buy_date_str = h.get("buy_date", str(today_dt))
        qty = h.get("quantity", 1)

        b_dt = pd.to_datetime(buy_date_str).date()
        holding_months = (today_dt.year - b_dt.year) * 12 + (today_dt.month - b_dt.month)

        if cur_px > 0 and buy_px > 0:
            friction = calculate_sale_friction(cur_px, item_type="sealed", platform="cardmarket")
            net_proceeds_unit = friction.net_proceeds
            net_pnl_unit = net_proceeds_unit - buy_px
            net_roi = net_pnl_unit / buy_px

            # Condizione Uscita A1: Tranche 1 Rotazione (18+ mesi e ROI Netto >= +70%)
            if 18 <= holding_months < 30 and net_roi >= 0.70:
                trim_qty = max(1, qty // 2) if qty > 1 else 1
                sell_signals.append({
                    "item_id": item_id,
                    "name": h.get("name", item_id),
                    "quantity": trim_qty,
                    "total_quantity": qty,
                    "buy_date": buy_date_str,
                    "holding_months": holding_months,
                    "buy_price": buy_px,
                    "current_price": cur_px,
                    "net_proceeds": net_proceeds_unit * trim_qty,
                    "net_roi_pct": net_roi * 100,
                    "signal_type": "TRANCHE 1 ROTAZIONE",
                    "trigger": f"TRANCHE 1 ROTAZIONE (+{net_roi*100:.1f}% netto dopo {holding_months} mesi). Vendere {trim_qty}/{qty} box per ruotare capitale su nuovi set."
                })
            # Condizione Uscita A2: Tranche 2 Finale (30+ mesi e ROI Netto >= +150%)
            elif holding_months >= 30 and net_roi >= 1.50:
                sell_signals.append({
                    "item_id": item_id,
                    "name": h.get("name", item_id),
                    "quantity": qty,
                    "total_quantity": qty,
                    "buy_date": buy_date_str,
                    "holding_months": holding_months,
                    "buy_price": buy_px,
                    "current_price": cur_px,
                    "net_proceeds": net_proceeds_unit * qty,
                    "net_roi_pct": net_roi * 100,
                    "signal_type": "TRANCHE 2 FINALE",
                    "trigger": f"TARGET PROFIT FINALE RAGGIUNTO (+{net_roi*100:.1f}% netto dopo {holding_months} mesi)"
                })
            # Condizione Uscita B: 48+ mesi (Time-Stop di rotazione)
            elif holding_months >= 48:
                sell_signals.append({
                    "item_id": item_id,
                    "name": h.get("name", item_id),
                    "quantity": qty,
                    "total_quantity": qty,
                    "buy_date": buy_date_str,
                    "holding_months": holding_months,
                    "buy_price": buy_px,
                    "current_price": cur_px,
                    "net_proceeds": net_proceeds_unit * qty,
                    "net_roi_pct": net_roi * 100,
                    "signal_type": "TIME-STOP ROTAZIONE",
                    "trigger": f"TIME-STOP ROTAZIONE (Holding di {holding_months} mesi >= 4 anni)"
                })

    return {
        "timestamp": today_dt.strftime("%Y-%m-%d"),
        "box_buy_signals": box_buy_signals,
        "box_reversal_signals": box_reversal_signals,
        "box_watchlist": box_watchlist,
        "singles_buy_signals": singles_buy_rows,
        "singles_avoid_signals": singles_avoid_rows,
        "holdings_sell_signals": sell_signals,
        "user_holdings_count": len(holdings),
    }


def scan_historical_signals(
    prices_df: Optional[pd.DataFrame] = None,
    metadata: Optional[Dict[str, Any]] = None,
    allowed_tiers: Optional[List[str]] = None,
    initial_cash: float = 10000.0,
    enable_rotation: bool = True
) -> pd.DataFrame:
    """
    Esegue la simulazione della VECCHIA euristica OptimalSealedStrategy (finestra
    eta'/MSRP) su tutta la timeline storica - SUPERSEDUTA dalla riconciliazione
    del 2026-09-29 (vedi docstring del modulo), mai usata dal path live
    (scan_signals/run_scanner_cli). Resta qui solo per confronto/ricerca
    storica - non e' la strategia che genera gli alert Telegram.
    """
    from poke_quant.engine.strategies.optimal_sealed_strategy import OptimalSealedStrategy
    from poke_quant.engine.backtester import Backtester

    if prices_df is None:
        prices_df = load_price_matrix()
    if metadata is None:
        metadata = load_metadata()

    strat = OptimalSealedStrategy(
        allowed_tiers=allowed_tiers or ["S", "A", "B"],
        enable_dynamic_rotation=enable_rotation,
        max_allocation_pct=0.12
    )
    bt = Backtester(
        strategy=strat,
        historical_prices_df=prices_df,
        items_metadata=metadata,
        initial_cash=initial_cash
    )
    res = bt.run()
    if res.signals_history:
        df = pd.DataFrame(res.signals_history)
        return df
    return pd.DataFrame()


def format_telegram_alert(scan_results: Dict[str, Any]) -> str:
    """Formatta il messaggio markdown per Telegram in stile sintetico e istituzionale.
    Stessi campi/fonte dati di app.py e run_monthly_production_signal.py (vedi
    docstring del modulo) - un lettore che confronta questo messaggio con la
    dashboard vede sempre gli stessi segnali BUY, non due strategie diverse."""
    date_str = scan_results.get("timestamp", datetime.date.today().strftime("%d/%m/%Y"))
    box_buys = scan_results.get("box_buy_signals", [])
    singles_buys = scan_results.get("singles_buy_signals", [])
    singles_avoid = scan_results.get("singles_avoid_signals", [])
    holdings_sells = scan_results.get("holdings_sell_signals", [])

    lines = [f"*POKEQUANT · SEGNALI DI MERCATO* ({date_str})\n"]

    if not box_buys and not singles_buys and not holdings_sells:
        lines.append("Nessun segnale operativo attivo oggi.")
        lines.append("Tutti i parametri monitorati rimangono in fase di attesa.")
        return "\n".join(lines)

    if box_buys:
        lines.append(f"🟢 *BOX SIGILLATI — BUY/HOLD (TS Momentum)* [{len(box_buys)}]")
        for b in box_buys:
            max_px = b.get("max_price_eur")
            max_px_str = f" | Max: *{max_px:.0f} €*" if max_px else ""
            lines.append(
                f"• *{b['name']}* ({b.get('tier', '?')})\n"
                f"  Prezzo: *{b['current_price_eur']:.0f} €*{max_px_str} · Momentum 12m: *{b['trailing_12m_return_pct']:+.0f}%*\n"
            )

    if singles_buys:
        lines.append(f"🟢 *SINGOLE — BUY (Fattore Scarsità, Grade 9)* [{len(singles_buys)}]")
        for s in singles_buys[:15]:
            lines.append(
                f"• *{s['name']}* [{s.get('set_name') or '?'}]\n"
                f"  Prezzo: *{s['current_price_eur']:.2f} €* · Sconto vs. pari: *{s['discount_pct']:+.0f}%* · Target: {s.get('target_grade', '?')}\n"
            )
        if len(singles_buys) > 15:
            lines.append(f"  … e altre {len(singles_buys) - 15}")

    if singles_avoid:
        lines.append(f"🔴 *SINGOLE SOPRAVVALUTATE (informativo, non un segnale di vendita validato)* [{len(singles_avoid)}]")
        for s in singles_avoid[:5]:
            lines.append(f"• {s['name']}: sovrapprezzo {s['discount_pct']:+.0f}% @ {s['current_price_eur']:.2f}€")

    if holdings_sells:
        lines.append(f"🔴 *POSIZIONI POSSEDUTE — SELL (tranche di rotazione, informativo)* [{len(holdings_sells)}]")
        for s in holdings_sells:
            lines.append(
                f"• *{s['name']}* (Q.tà: {s['quantity']})\n"
                f"  Prezzo Vendita: *{s['current_price']:.1f} €* (Carico: {s['buy_price']:.1f} €)\n"
                f"  ROI Netto: *+{s['net_roi_pct']:.1f}%* dopo {s['holding_months']} mesi\n"
                f"  Trigger: {s['trigger']}\n"
                f"  Azione: Mettere in vendita su Cardmarket\n"
            )

    return "\n".join(lines)


def run_scanner_cli():
    print("\n" + "=" * 70)
    print("  POKEQUANT — SCANNER SETTIMANALE SEGNALI DI INVESTIMENTO")
    print("=" * 70)

    res = scan_signals()
    msg = format_telegram_alert(res)
    print(msg)
    print("=" * 70 + "\n")

    sent = send_telegram_message(msg)
    if sent:
        print("✅ Notifica inviata con successo su Telegram.")
    else:
        print("❌ Notifica non inviata (vedi log sopra: credenziali mancanti o errore di invio).")


if __name__ == "__main__":
    run_scanner_cli()
