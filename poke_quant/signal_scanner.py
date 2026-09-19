"""
poke_quant/signal_scanner.py — Scanner automatico dei segnali BUY / SELL su dati di mercato live.
Monitora le finestre d'acquisto (Mesi 4-14, Prezzo <= 1.15x MSRP su Tier S/A) e i target di vendita
(+150% netto a 30+ mesi) per le posizioni possedute in portfolio_holdings.json.
Invia notifiche push via Telegram (se configurato) o stampa report da riga di comando.
"""

from __future__ import annotations
import os
import json
import logging
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Dict, List, Any, Optional
import datetime
import pandas as pd

from poke_quant.config import PLATFORM_FEES, DEFAULT_EUR_USD
from poke_quant.data.storage import load_price_matrix, load_metadata
from poke_quant.engine.friction import calculate_sale_friction

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
    metadata: Optional[Dict[str, Any]] = None,
    today_dt: Optional[datetime.date] = None
) -> Dict[str, Any]:
    """
    Esegue la scansione completa di mercato per generare:
      - Segnali BUY di mercato (prodotti Tier S/A in finestra Mesi 4-14 a sconto)
      - Segnali SELL di portafoglio (posizioni possedute con target raggiunto o time-stop)
      - Alert di watchlist (prodotti prossimi all'ingresso nella finestra d'acquisto)
    """
    if today_dt is None:
        today_dt = datetime.date.today()

    if metadata is None:
        metadata = load_metadata() or {}

    if current_prices is None:
        prices_df = load_price_matrix()
        if prices_df is not None and not prices_df.empty:
            current_prices = prices_df.iloc[-1].to_dict()
        else:
            current_prices = {}

    buy_signals = []
    watchlist_items = []
    sell_signals = []

    # 1. SCANSIONE SEGNALI BUY DI MERCATO
    for item_id, meta in metadata.items():
        if meta.get("type") != "sealed":
            continue
        p_type = meta.get("product_type", "booster_box")
        if p_type not in ["booster_box", "specialty_bundle"]:
            continue

        tier = meta.get("set_tier", "B")
        if tier not in ["S", "A"]:
            continue

        rel_str = meta.get("release_date")
        if not rel_str:
            continue

        rel_dt = pd.to_datetime(rel_str).date()
        age_months = (today_dt.year - rel_dt.year) * 12 + (today_dt.month - rel_dt.month)

        cur_px = current_prices.get(item_id, 0.0)
        msrp = meta.get("msrp", 140.0)
        max_allowed_buy_px = msrp * 1.15

        # Caso A: Prodotto nella finestra ottimale di acquisto (Mesi 4-14)
        if 4 <= age_months <= 14:
            discount_or_premium = ((cur_px - msrp) / msrp) * 100 if msrp > 0 else 0.0
            if cur_px > 0 and cur_px <= max_allowed_buy_px:
                buy_signals.append({
                    "item_id": item_id,
                    "name": meta.get("name", item_id),
                    "tier": tier,
                    "age_months": age_months,
                    "current_price": cur_px,
                    "msrp": msrp,
                    "diff_vs_msrp_pct": discount_or_premium,
                    "reason": f"Tier {tier} in finestra ottimale ({age_months} mesi) a {cur_px:.1f}€ (MSRP: {msrp:.1f}€, {discount_or_premium:+.1f}%)"
                })
            else:
                watchlist_items.append({
                    "item_id": item_id,
                    "name": meta.get("name", item_id),
                    "tier": tier,
                    "age_months": age_months,
                    "current_price": cur_px,
                    "msrp": msrp,
                    "status": f"In finestra ({age_months}m) ma prezzo alto ({cur_px:.1f}€ vs limite {max_allowed_buy_px:.1f}€)"
                })
        # Caso B: Prodotto in avvicinamento alla finestra (Mesi 1-3)
        elif 1 <= age_months < 4:
            watchlist_items.append({
                "item_id": item_id,
                "name": meta.get("name", item_id),
                "tier": tier,
                "age_months": age_months,
                "current_price": cur_px,
                "msrp": msrp,
                "status": f"Nuovo set in avvicinamento (tra {4 - age_months} mesi inizia la finestra ristampa)"
            })

    # 2. SCANSIONE SEGNALI SELL SU POSIZIONI POSSEDUTE
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
            # Calcolo netto Cardmarket
            friction = calculate_sale_friction(cur_px, item_type="sealed", platform="cardmarket")
            net_proceeds_unit = friction.net_proceeds
            net_pnl_unit = net_proceeds_unit - buy_px
            net_roi = net_pnl_unit / buy_px

            # Condizione Uscita A: 30+ mesi e ROI Netto >= +150%
            if holding_months >= 30 and net_roi >= 1.50:
                sell_signals.append({
                    "item_id": item_id,
                    "name": h.get("name", item_id),
                    "quantity": qty,
                    "buy_date": buy_date_str,
                    "holding_months": holding_months,
                    "buy_price": buy_px,
                    "current_price": cur_px,
                    "net_proceeds": net_proceeds_unit * qty,
                    "net_roi_pct": net_roi * 100,
                    "trigger": f"TARGET PROFIT RAGGIUNTO (+{net_roi*100:.1f}% netto dopo {holding_months} mesi)"
                })
            # Condizione Uscita B: 48+ mesi (Time-Stop di rotazione)
            elif holding_months >= 48:
                sell_signals.append({
                    "item_id": item_id,
                    "name": h.get("name", item_id),
                    "quantity": qty,
                    "buy_date": buy_date_str,
                    "holding_months": holding_months,
                    "buy_price": buy_px,
                    "current_price": cur_px,
                    "net_proceeds": net_proceeds_unit * qty,
                    "net_roi_pct": net_roi * 100,
                    "trigger": f"TIME-STOP ROTAZIONE (Holding di {holding_months} mesi >= 4 anni)"
                })

    return {
        "timestamp": today_dt.strftime("%Y-%m-%d"),
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "watchlist": watchlist_items,
        "total_monitored_items": len(metadata),
        "user_holdings_count": len(holdings)
    }


def format_telegram_alert(scan_results: Dict[str, Any]) -> str:
    """Formatta il messaggio markdown per Telegram in stile sintetico e istituzionale."""
    date_str = scan_results.get("timestamp", datetime.date.today().strftime("%d/%m/%Y"))
    buys = scan_results.get("buy_signals", [])
    sells = scan_results.get("sell_signals", [])

    lines = [f"*POKEQUANT · SEGNALI DI MERCATO* ({date_str})\n"]

    if not buys and not sells:
        lines.append("Nessun segnale operativo attivo oggi.")
        lines.append("Tutti i parametri monitorati rimangono in fase di attesa.")
        return "\n".join(lines)

    if buys:
        lines.append(f"🟢 *SEGNALI DI ACQUISTO (BUY)* [{len(buys)}]")
        for b in buys:
            lines.append(
                f"• *{b['name']}* (Tier {b['tier']})\n"
                f"  Prezzo: *{b['current_price']:.1f} €* (MSRP: {b['msrp']:.1f} €, {b['diff_vs_msrp_pct']:+.1f}%)\n"
                f"  Età Set: {b['age_months']} mesi (Finestra Ristampa Attiva)\n"
                f"  Azione: Comprare (Max 15-20% del capitale)\n"
            )

    if sells:
        lines.append(f"🔴 *SEGNALI DI VENDITA (SELL)* [{len(sells)}]")
        for s in sells:
            lines.append(
                f"• *{s['name']}* (Q.tà: {s['quantity']})\n"
                f"  Prezzo Vendita: *{s['current_price']:.1f} €* (Carico: {s['buy_price']:.1f} €)\n"
                f"  ROI Netto: *+{s['net_roi_pct']:.1f}%* dopo {s['holding_months']} mesi\n"
                f"  Trigger: {s['trigger']}\n"
                f"  Azione: Mettere in vendita su Cardmarket\n"
            )

    return "\n".join(lines)


def send_telegram_message(message: str, token: Optional[str] = None, chat_id: Optional[str] = None) -> bool:
    """Invia notifica Telegram se token e chat_id sono configurati."""
    bot_token = token or os.environ.get("TELEGRAM_TOKEN")
    target_chat = chat_id or os.environ.get("TELEGRAM_CHAT_ID")

    if not bot_token or not target_chat:
        logger.info("Credenziali Telegram non presenti. Salto invio notifica.")
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = urllib.parse.urlencode({
        "chat_id": target_chat,
        "text": message,
        "parse_mode": "Markdown"
    }).encode("utf-8")

    req = urllib.request.Request(url, data=payload)
    try:
        urllib.request.urlopen(req, timeout=15)
        return True
    except Exception as e:
        logger.error(f"Errore invio messaggio Telegram: {e}")
        return False


def run_scanner_cli():
    print("\n" + "=" * 70)
    print("  POKEQUANT — SCANNER AUTOMATICO SEGNALI DI INVESTIMENTO")
    print("=" * 70)

    res = scan_signals()
    msg = format_telegram_alert(res)
    print(msg)
    print("=" * 70 + "\n")

    # Invia su Telegram se configurato
    if os.environ.get("TELEGRAM_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"):
        sent = send_telegram_message(msg)
        if sent:
            print("✅ Notifica inviata con successo su Telegram.")
        else:
            print("❌ Errore durante l'invio su Telegram.")


if __name__ == "__main__":
    run_scanner_cli()
