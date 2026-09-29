#!/usr/bin/env python3
"""
scripts/run_monthly_production_signal.py — Orchestratore mensile di produzione.

Sequenza:
  1. Scopre nuovi set sealed (discover_sealed_universe.py) - se PriceCharting ha
     una pagina booster-box per un set pokemontcg.io non ancora in metadata, lo
     aggiunge (msrp=None esplicito per i vintage, nessun dato inventato). Idempotente:
     se non c'e' nulla di nuovo non fa nulla (verificato: 0 nuovi al momento della
     scrittura, ma il controllo va rifatto ogni mese perche' escono nuovi set).
  2. Scopre nuove SINGOLE chase (discover_chase_cards.py) + controllo casuale
     (discover_random_control_singles.py) - entrambi ora usano
     discover_chase_cards.build_set_ids(), che si ricostruisce da solo leggendo
     quali era hanno gia' un box sealed in metadata (incluso quello appena
     trovato al passo 1, nella STESSA run) invece di una mappa scritta a mano:
     un set nuovo scoperto al passo 1 ha automaticamente le sue singole
     scoperte qui, senza intervento manuale.
  3. Ricostruisce TUTTI i pannelli prezzi (sealed + graded singles) con tasso FX
     reale (rebuild_prices_with_real_fx.py) - rifetcha ogni item gia' in metadata,
     quindi include automaticamente il mese appena chiuso.
  4. Ri-applica il filtro di attendibilità (flag_unreliable_assets.py).
  5. Genera ENTRAMBI i segnali di produzione - TS Momentum sui box (universo
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

LIMITE RESIDUO, non silenziato: build_set_ids() trova un set SOLO se
pokemontcg.io usa lo stesso nome (via era_id()) del box sealed gia' in
metadata - verificato empiricamente 97/98 casi storici corrispondono esatti.
I set giapponesi esclusivi (VMAX Climax, VSTAR Universe, ecc.) restano fuori
perche' pokemontcg.io copre solo la stampa inglese - nessun modo automatico
di risolverlo con questa fonte dati.

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


def run_step(description: str, args: list) -> bool:
    print(f"\n--- {description} ---")
    result = subprocess.run([sys.executable] + args, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"[ATTENZIONE] step fallito: {description} (codice {result.returncode})")
        return False
    return True


def _capped(lines: list) -> list:
    if len(lines) > MAX_LINES_PER_LIST:
        return lines[:MAX_LINES_PER_LIST] + [f"  … e altre {len(lines) - MAX_LINES_PER_LIST}"]
    return lines


def _prepend_failure_warning(message: str, failed_steps: list) -> str:
    if not failed_steps:
        return message
    warning = (
        f"🚨 ATTENZIONE: {len(failed_steps)} step falliti in questa run "
        f"(dati sottostanti potenzialmente non aggiornati):\n"
        + "\n".join(f"  - {s}" for s in failed_steps)
    )
    return warning + "\n\n" + message


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
    # BUG TROVATO (audit performance/bug richiesto dall'utente, 2026-09-29):
    # run_step() stampava solo un warning su stdout (nessuna GitHub Action lo
    # legge di routine) e main() proseguiva comunque attraverso TUTTI gli step
    # successivi indipendentemente dall'esito - se rebuild_prices_with_real_fx.py
    # falliva (es. timeout di rete), il pannello prezzi restava quello del mese
    # precedente ma build_message()/l'invio Telegram procedevano IDENTICI,
    # presentando dati vecchi come se fossero freschi, senza nessuna traccia
    # visibile all'utente. Ora ogni fallimento viene raccolto e (a) antepone un
    # avviso esplicito al messaggio Telegram stesso, (b) fa uscire con codice
    # non-zero cosi' la GitHub Action mensile risulta visibilmente rossa.
    steps = [
        ("Scoperta nuovi set sealed", ["scripts/discover_sealed_universe.py"]),
        # Aggiunti 2026-09-29 (richiesta esplicita dell'utente: "fai in modo che il
        # motore scarichi dati gia' su One piece e pokemon china"): fino a qui solo
        # Pokemon EN aveva una scoperta ricorrente - One Piece era fermo a 2 anni fa
        # (ultimo set curato a mano: 2024-03-15) e Pokemon Cinese non aveva alcuna
        # scoperta automatica dopo la scansione una tantum di Fase 1.3. Entrambi gli
        # script sotto ri-scaricano dal vivo la propria categoria PriceCharting a
        # ogni run (non un elenco congelato) e sono idempotenti (solo nuovi item).
        ("Scoperta nuovi box One Piece", ["scripts/discover_one_piece_sealed_universe.py"]),
        ("Scoperta nuovi box Pokemon Cinese", ["scripts/discover_pokemon_chinese_sealed_universe.py"]),
        ("Scoperta nuove singole chase", ["scripts/discover_chase_cards.py"]),
        ("Scoperta nuove singole di controllo", ["scripts/discover_random_control_singles.py"]),
        ("Ricostruzione prezzi con FX reale (sealed + graded singles)", ["scripts/rebuild_prices_with_real_fx.py"]),
        ("Aggiornamento filtro attendibilità", ["scripts/flag_unreliable_assets.py"]),
    ]
    failed_steps = [description for description, args in steps if not run_step(description, args)]

    message = _prepend_failure_warning(build_message(), failed_steps)
    print("\n" + "=" * 100)
    print(message)
    print("=" * 100)

    sent = send_telegram_message(message)
    print(f"\nNotifica Telegram: {'inviata' if sent else 'non inviata (vedi sopra)'}")

    if failed_steps:
        sys.exit(1)


if __name__ == "__main__":
    main()
