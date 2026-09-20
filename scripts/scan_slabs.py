#!/usr/bin/env python3
"""
scripts/scan_slabs.py — CLI Scanner per Carte Gradate (PSA, BGS, CGC).
Esegue la scansione degli Edge matematici (BUY, SELL, ROTATE) e invia
notifica push Telegram se configurato.
"""

import sys
import os
from pathlib import Path

# Assicura importazione di poke_quant
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.slabs import scan_slabs_market, print_slabs_cli_summary, format_slabs_telegram_alert
from poke_quant.signal_scanner import send_telegram_message


def main():
    print("Avvio scansione quantitativa mercato Slabs...")
    results = scan_slabs_market()
    print_slabs_cli_summary(results)

    # Invia notifica Telegram se configurato
    if os.environ.get("TELEGRAM_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"):
        msg = format_slabs_telegram_alert(results)
        sent = send_telegram_message(msg)
        if sent:
            print("✅ Notifica Slabs inviata con successo su Telegram.")
        else:
            print("❌ Errore durante l'invio della notifica su Telegram.")


if __name__ == "__main__":
    main()
