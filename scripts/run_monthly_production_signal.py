#!/usr/bin/env python3
"""
scripts/run_monthly_production_signal.py — Orchestratore mensile di produzione.

Sequenza:
  1. Scopre nuovi set sealed (discover_sealed_universe.py) - se PriceCharting ha
     una pagina booster-box per un set pokemontcg.io non ancora in metadata, lo
     aggiunge (msrp=None esplicito per i vintage, nessun dato inventato). Idempotente:
     se non c'e' nulla di nuovo non fa nulla (verificato: 0 nuovi al momento della
     scrittura, ma il controllo va rifatto ogni mese perche' escono nuovi set).
  2. Ricostruisce TUTTI i pannelli prezzi (sealed + graded singles) con tasso FX
     reale (rebuild_prices_with_real_fx.py) - rifetcha ogni item gia' in metadata,
     quindi include automaticamente il mese appena chiuso.
  3. Ri-applica il filtro di attendibilità (flag_unreliable_assets.py).
  4. Genera ENTRAMBI i segnali di produzione - TS Momentum sui box (universo
     allargato via liquidity_filter.is_liquid_sealed) e fattore Scarsita' sulle
     singole (BUY fresco + AVOID sopravvalutate, stessa logica della dashboard) -
     e invia un messaggio via Telegram (se configurato).

Pensato per essere lanciato da una GitHub Action mensile (vedi
.github/workflows/monthly_signal.yml) — nessun intervento manuale richiesto,
a parte impostare i secret TELEGRAM_TOKEN/TELEGRAM_CHAT_ID una volta.

AGGIORNAMENTO: le singole NON sono piu' sospese. Il fattore scarsita'
(poke_quant/engine/strategies/scarcity_value_factor.py) ha superato la soglia
istituzionale (DSR 0,980 corretto per l'intera ricerca sulle singole) dopo che
questo messaggio era stato scritto - i 5 fattori citati nella versione
precedente di questo docstring (carry/eta', TS momentum, cross-sectional
momentum, dip mean-reversion, rarita' ex-ante) restano non validati, ma non
sono piu' l'ultima parola sulle singole.

LIMITE CONOSCIUTO, non silenziato: il passo 1 scopre nuovi SEALED automaticamente,
ma le SINGOLE chase/controllo di un set appena uscito richiedono ancora
un'aggiunta manuale a SET_IDS in scripts/discover_chase_cards.py - quella mappa
lega uno slug PriceCharting all'ID set ufficiale pokemontcg.io (es. "sv9"), che
non e' derivabile automaticamente dal nome senza rischiare falsi appaiamenti.

NON verifica liquidità reale. Il messaggio lo ripete esplicitamente ogni volta,
apposta, perché è il gate che manca ancora.
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.data.liquidity_filter import liquid_sealed_ids
from scripts.generate_monthly_signal import compute_signal_rows
from scripts.generate_singles_signal import compute_singles_signal_rows, compute_singles_avoid_rows
from poke_quant.notify.telegram import send_telegram_message

ROOT = Path(__file__).resolve().parent.parent
MAX_LINES_PER_LIST = 15


def run_step(description: str, args: list):
    print(f"\n--- {description} ---")
    result = subprocess.run([sys.executable] + args, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"[ATTENZIONE] step fallito: {description} (codice {result.returncode})")


def _capped(lines: list) -> list:
    if len(lines) > MAX_LINES_PER_LIST:
        return lines[:MAX_LINES_PER_LIST] + [f"  … e altre {len(lines) - MAX_LINES_PER_LIST}"]
    return lines


def build_message() -> str:
    metadata = load_metadata()
    prices_full = load_price_matrix()
    n_universe_sealed = len(liquid_sealed_ids(metadata, prices_full))

    sealed_rows, sealed_date = compute_signal_rows()
    singles_buy_rows, singles_date = compute_singles_signal_rows()
    singles_avoid_rows, _ = compute_singles_avoid_rows()

    lines = []
    lines.append(f"📊 PokeQuant — Segnale mensile {sealed_date.strftime('%Y-%m')}")
    lines.append("⚠️ Nessuna verifica di liquidità reale: controllare disponibilità/prezzo prima di agire.")

    lines.append(f"\n— BOX SIGILLATI — TS Momentum (universo: {n_universe_sealed}, DSR 0,778) —")
    n_buy = sum(1 for r in sealed_rows if r["signal"] == "BUY/HOLD")
    n_verify = sum(1 for r in sealed_rows if "VERIFICARE" in r["signal"])
    lines.append(f"BUY/HOLD: {n_buy} | AVOID/SELL: {len(sealed_rows) - n_buy - n_verify} | Da verificare: {n_verify}")
    lines.extend(_capped([
        f"  🟢 {r['name']}: {r['trailing_12m_return_pct']:+.0f}% @ {r['current_price_eur']:.0f}€"
        for r in sealed_rows if r["signal"] == "BUY/HOLD"
    ]))

    lines.append(f"\n— SINGOLE (Grade 9) — Fattore Scarsità (DSR 0,980) — segnale {singles_date.strftime('%Y-%m')} —")
    lines.append(f"BUY (fresco, <=3 mesi nel quantile): {len(singles_buy_rows)} | AVOID (sopravvalutate): {len(singles_avoid_rows)}")
    lines.extend(_capped([
        f"  🟢 {r['name']}: sconto {r['discount_pct']:+.0f}% @ {r['current_price_eur']:.2f}€"
        for r in singles_buy_rows
    ]))
    if singles_avoid_rows:
        lines.append("  Sopravvalutate (informativo, non un segnale di vendita validato a se'):")
        lines.extend(_capped([
            f"  🔴 {r['name']}: sovrapprezzo {r['discount_pct']:+.0f}% @ {r['current_price_eur']:.2f}€"
            for r in singles_avoid_rows[:5]
        ]))

    return "\n".join(lines)


def main():
    run_step("Scoperta nuovi set sealed", ["scripts/discover_sealed_universe.py"])
    run_step("Ricostruzione prezzi con FX reale (sealed + graded singles)", ["scripts/rebuild_prices_with_real_fx.py"])
    run_step("Aggiornamento filtro attendibilità", ["scripts/flag_unreliable_assets.py"])

    message = build_message()
    print("\n" + "=" * 100)
    print(message)
    print("=" * 100)

    sent = send_telegram_message(message)
    print(f"\nNotifica Telegram: {'inviata' if sent else 'non inviata (vedi sopra)'}")


if __name__ == "__main__":
    main()
