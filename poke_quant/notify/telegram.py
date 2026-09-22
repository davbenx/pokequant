"""
poke_quant/notify/telegram.py — Invio notifiche Telegram, stesso schema già in
uso in apex-engine (convex_telegram_reminder.py) e in questo repo
(slabs/slab_notifier.py formatta, ma non invia): credenziali da env var
TELEGRAM_TOKEN/TELEGRAM_CHAT_ID, skip silenzioso se non configurate (mai un
errore bloccante se l'utente non ha ancora impostato i secret).
"""

from __future__ import annotations
import os
from typing import Optional
import requests


def send_telegram_message(text: str, token: Optional[str] = None, chat_id: Optional[str] = None) -> bool:
    token = token or os.environ.get("TELEGRAM_TOKEN")
    chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("[-] Credenziali Telegram non configurate (TELEGRAM_TOKEN/TELEGRAM_CHAT_ID). Skip invio.")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    # Telegram limita ~4096 caratteri per messaggio.
    chunks = [text[i:i + 3800] for i in range(0, len(text), 3800)] or [text]
    ok = True
    for chunk in chunks:
        try:
            resp = requests.post(url, data={"chat_id": chat_id, "text": chunk}, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            print(f"[-] Invio Telegram fallito: {e}")
            ok = False
    return ok
