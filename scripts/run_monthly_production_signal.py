#!/usr/bin/env python3
"""
scripts/run_monthly_production_signal.py — Orchestratore mensile di produzione.

Sequenza:
  1. Ricostruisce i pannelli prezzi con tasso FX reale (rebuild_prices_with_real_fx.py)
  2. Ri-applica il filtro di attendibilità (flag_unreliable_assets.py)
  3. Genera il segnale TS Momentum (box sigillati, era moderna) - UNICO segnale operativo
  4. Compone un messaggio e lo invia via Telegram (se configurato)

Pensato per essere lanciato da una GitHub Action mensile (vedi
.github/workflows/monthly_signal.yml) — nessun intervento manuale richiesto,
a parte impostare i secret TELEGRAM_TOKEN/TELEGRAM_CHAT_ID una volta.

SINGOLE GRADATE (Carry/Scarsità e ogni altro fattore testato) SOSPESE, non incluse
nel messaggio operativo: scripts/optimize_and_falsify.py::section_singles_factor_search
ha testato 5 famiglie di fattori (eta'/carry, TS momentum, cross-sectional momentum,
dip mean-reversion, rarita' ex-ante) sull'universo reale e bias-auditato (928 carte) -
nessuno supera la soglia istituzionale (il migliore, dip mean-reversion, ha PBO=0.514
e si inverte di segno tra prima e seconda meta' del campione). scripts/generate_carry_
signal_singles.py resta disponibile per ricerca, ma NON va rimesso in questo messaggio
finche' un fattore non passa DSR/PBO/bootstrap/walk-forward. Vedi i docstring in
poke_quant/engine/strategies/{carry_scarcity_factor,dip_mean_reversion,
rarity_tier_factor,cross_sectional_momentum}.py per lo stato di ciascuno.

NON verifica liquidità reale. Il messaggio lo ripete esplicitamente ogni volta,
apposta, perché è il gate che manca ancora.
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_monthly_signal import compute_signal_rows
from poke_quant.notify.telegram import send_telegram_message

ROOT = Path(__file__).resolve().parent.parent


def run_step(description: str, args: list):
    print(f"\n--- {description} ---")
    result = subprocess.run([sys.executable] + args, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"[ATTENZIONE] step fallito: {description} (codice {result.returncode})")


def build_message() -> str:
    sealed_rows, sealed_date = compute_signal_rows()

    lines = []
    lines.append(f"📊 PokeQuant — Segnale mensile {sealed_date.strftime('%Y-%m')}")
    lines.append("⚠️ Nessuna verifica di liquidità reale: controllare disponibilità/prezzo prima di agire.")
    lines.append("ℹ️ Solo box sigillati: le singole gradate sono sospese (nessun fattore validato, vedi repo).")

    lines.append("\n— TS MOMENTUM (box sigillati, era 2019+) —")
    n_buy = sum(1 for r in sealed_rows if r["signal"] == "BUY/HOLD")
    n_verify = sum(1 for r in sealed_rows if "VERIFICARE" in r["signal"])
    lines.append(f"BUY/HOLD: {n_buy} | AVOID/SELL: {len(sealed_rows) - n_buy - n_verify} | Da verificare: {n_verify}")
    for r in sealed_rows:
        if r["signal"] == "BUY/HOLD":
            lines.append(f"  + {r['name']}: {r['trailing_12m_return_pct']:+.0f}% @ {r['current_price_eur']:.0f}€")

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
