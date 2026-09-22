#!/usr/bin/env python3
"""
scripts/run_monthly_production_signal.py — Orchestratore mensile di produzione.

Sequenza:
  1. Ricostruisce i pannelli prezzi con tasso FX reale (rebuild_prices_with_real_fx.py)
  2. Ri-applica il filtro di attendibilità (flag_unreliable_assets.py)
  3. Genera il segnale TS Momentum (box sigillati, era moderna)
  4. Genera il segnale Carry/Scarsità (singole gradate Grade 9)
  5. Compone un messaggio unico e lo invia via Telegram (se configurato)

Pensato per essere lanciato da una GitHub Action mensile (vedi
.github/workflows/monthly_signal.yml) — nessun intervento manuale richiesto,
a parte impostare i secret TELEGRAM_TOKEN/TELEGRAM_CHAT_ID una volta.

NON verifica liquidità reale. Il messaggio lo ripete esplicitamente ogni volta,
apposta, perché è il gate che manca ancora.
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_monthly_signal import compute_signal_rows
from scripts.generate_carry_signal_singles import compute_top_ranked
from poke_quant.notify.telegram import send_telegram_message

ROOT = Path(__file__).resolve().parent.parent


def run_step(description: str, args: list):
    print(f"\n--- {description} ---")
    result = subprocess.run([sys.executable] + args, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"[ATTENZIONE] step fallito: {description} (codice {result.returncode})")


def build_message() -> str:
    sealed_rows, sealed_date = compute_signal_rows()
    top_singles, ranked_singles, singles_date, singles_meta = compute_top_ranked()

    lines = []
    lines.append(f"📊 PokeQuant — Segnale mensile {sealed_date.strftime('%Y-%m')}")
    lines.append("⚠️ Nessuna verifica di liquidità reale: controllare disponibilità/prezzo prima di agire.")

    lines.append("\n— TS MOMENTUM (box sigillati, era 2019+) —")
    n_buy = sum(1 for r in sealed_rows if r["signal"] == "BUY/HOLD")
    n_verify = sum(1 for r in sealed_rows if "VERIFICARE" in r["signal"])
    lines.append(f"BUY/HOLD: {n_buy} | AVOID/SELL: {len(sealed_rows) - n_buy - n_verify} | Da verificare: {n_verify}")
    for r in sealed_rows:
        if r["signal"] == "BUY/HOLD":
            lines.append(f"  + {r['name']}: {r['trailing_12m_return_pct']:+.0f}% @ {r['current_price_eur']:.0f}€")

    lines.append("\n— CARRY/SCARSITÀ (singole gradate Grade 9, top 30% età) —")
    lines.append(f"Universo eleggibile: {len(ranked_singles)} | Nel quantile top: {len(top_singles)}")
    for item_id, age_m, price in top_singles[:20]:
        name = singles_meta[item_id].get("name", item_id)
        flag = " [VERIFICARE]" if price > 3000 else ""
        lines.append(f"  + {name}: {age_m}m @ {price:.0f}€{flag}")
    if len(top_singles) > 20:
        lines.append(f"  ... e altre {len(top_singles) - 20} carte (vedi repo per l'elenco completo)")

    return "\n".join(lines)


def main():
    run_step("Ricostruzione prezzi con FX reale", ["scripts/rebuild_prices_with_real_fx.py"])
    run_step("Aggiornamento filtro attendibilità", ["scripts/flag_unreliable_assets.py"])

    message = build_message()
    print("\n" + "=" * 100)
    print(message)
    print("=" * 100)

    sent = send_telegram_message(message)
    print(f"\nNotifica Telegram: {'inviata' if sent else 'non inviata (vedi sopra)'}")


if __name__ == "__main__":
    main()
