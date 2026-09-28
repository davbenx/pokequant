#!/usr/bin/env python3
"""
scripts/generate_monthly_signal.py — Genera il segnale operativo attuale della
strategia raccomandata (Time-Series Momentum, 12 mesi, box sigillati) sull'ultimo
mese disponibile. Output pensato per essere automatizzabile end-to-end: nessun
giudizio soggettivo, solo prezzo storico reale -> rendimento trailing -> segnale.

Esclude per costruzione:
  - asset con data_quality="thin_unreliable" (poke_quant/data/liquidity_filter.py)
  - asset single (perimetro: solo sealed box, per ora)

NON verifica la liquidità reale (nessuna inserzione attiva, nessun prezzo
eseguibile) — quello resta un gate manuale separato via
scripts/log_execution_price.py finché non c'è un'API Cardmarket ufficiale.
Il segnale qui sotto dice COSA comprare/tenere secondo il modello, non se è
davvero disponibile ora a quel prezzo.

Uso: python scripts/generate_monthly_signal.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.data.liquidity_filter import liquid_sealed_ids, MAX_PRICE_TO_MSRP_RATIO
from poke_quant.engine.strategies.time_series_momentum import TimeSeriesMomentumStrategy

LOOKBACK_MONTHS = 12
# Sopra questa soglia il rendimento a 12m è più probabile rumore da mercato sottile
# (visto empiricamente: box vintage con salti cumulati "morbidi" mese per mese che
# compongono comunque +300%+ in un anno) che un vero segnale azionabile. Non è una
# statistica di mercato — è un cap di plausibilità sul SEGNALE, per non presentare
# rumore come edge.
PLAUSIBILITY_CAP_PCT = 80.0

# Anche sotto il cap di rendimento, molti box vintage mostrano LIVELLI di prezzo
# assurdi in assoluto (es. Team Rocket Returns a 58.235€, Power Keepers a 17.472€ -
# set di media fama, non pezzi da museo): il problema non e' il rendimento, e' che
# il volume di vendita reale e' troppo basso per qualsiasi prezzo mensile attendibile,
# indipendentemente dal filtro statistico. MODERN_ERA_CUTOFF resta il taglio di
# base, ma liquid_sealed_ids() (poke_quant/data/liquidity_filter.py) recupera anche
# i box PIU' VECCHI il cui rapporto prezzo/MSRP reale resta dentro il range gia'
# osservato nell'universo moderno (1,5x-21,6x) - si sono apprezzati come un box
# "normale", non come un pezzo da museo. Validato in
# scripts/sealed_universe_expansion_test.py: 36->40 box, Sharpe 1,10->1,31,
# MaxDD -13,4%->-10,6%, DSR full-session 0,675->0,778 su 47 trial cumulativi (grazie a H1,
# il periodo debole, che passa da Sharpe -0,10 a +0,68). Nessun MSRP viene
# inventato per i box senza questo dato: restano esclusi se piu' vecchi del 2019.
MODERN_ERA_CUTOFF = "2019-01-01"


def compute_signal_rows():
    """Ritorna (rows, latest_date). Riutilizzabile da altri script/orchestratori."""
    metadata = load_metadata()
    prices_df = load_price_matrix()

    sealed_ids = liquid_sealed_ids(metadata, prices_df, modern_era_cutoff=MODERN_ERA_CUTOFF)
    prices_sealed = prices_df[sealed_ids]
    latest_date = prices_sealed.index[-1]

    strat = TimeSeriesMomentumStrategy(prices_sealed, lookback_months=LOOKBACK_MONTHS)

    rows = []
    for item_id in sealed_ids:
        mom = strat._trailing_return(item_id, latest_date)
        if mom is None:
            continue
        cur_price = prices_sealed[item_id].dropna().iloc[-1]
        ret_pct = mom * 100.0
        msrp = metadata[item_id].get("msrp")
        # Spesa massima TOTALE (oggetto + spedizione) da non superare - stesso
        # numero validato in scripts/box_max_price_ratio_test.py. Richiesto
        # esplicitamente dall'utente di NON sottrarre qui una nostra stima di
        # spedizione (10EUR/box e' una media, non il costo reale di QUESTA
        # inserzione): la spedizione reale la verifica l'utente stesso
        # sull'inserzione Cardmarket, confrontando oggetto+spedizione reali
        # con questo numero.
        max_price_eur = msrp * MAX_PRICE_TO_MSRP_RATIO if msrp else None
        if abs(ret_pct) > PLAUSIBILITY_CAP_PCT:
            signal = "VERIFICARE A MANO (rendimento implausibile)"
        elif mom > 0 and max_price_eur is not None and cur_price > max_price_eur:
            # "impedire di comprare sopra un prezzo che rompe l'edge" (richiesto
            # esplicitamente): stesso rapporto prezzo/MSRP gia' usato per
            # ammettere un box vintage nell'universo (poke_quant/data/
            # liquidity_filter.py::MAX_PRICE_TO_MSRP_RATIO, 21,6x), qui applicato
            # anche a un box gia' dentro l'universo - il momentum dice compra, ma
            # il prezzo e' gia' oltre il confine di plausibilita' del modello.
            signal = "PREZZO ECCESSIVO (oltre tetto MSRP)"
        else:
            signal = "BUY/HOLD" if mom > 0 else "AVOID/SELL"
        info = metadata.get(item_id, {})
        rows.append({
            "item_id": item_id,
            "name": info.get("name", item_id),
            "current_price_eur": cur_price,
            "trailing_12m_return_pct": ret_pct,
            "max_price_eur": max_price_eur,
            "signal": signal,
            "franchise": info.get("franchise", "pokemon"),
            "language": info.get("language", "en"),
            "era": info.get("era", "modern"),
        })

    rows.sort(key=lambda r: -r["trailing_12m_return_pct"])

    # Assegnazione gerarchica Tier per franchise/language:
    # - Vault: box >= 500€ o set_tier 'Grail' (pezzi da collezione/museo)
    # - Core: i primi 8 box con momentum positivo (<500€) che assorbono la cassa
    # - Bench: i successivi box ad alto momentum (alternative/riserve)
    core_counts: dict[tuple[str, str], int] = {}
    for r in rows:
        if r["signal"] == "BUY/HOLD":
            set_tier = metadata.get(r["item_id"], {}).get("set_tier")
            if r["current_price_eur"] >= 500.0 or set_tier == "Grail":
                r["tier"] = "vault"
            else:
                f_key = (r["franchise"], r["language"])
                cnt = core_counts.get(f_key, 0)
                if cnt < 8:
                    r["tier"] = "core"
                    core_counts[f_key] = cnt + 1
                else:
                    r["tier"] = "bench"
        elif "PREZZO ECCESSIVO" in r["signal"]:
            r["tier"] = "excessive"
        elif "VERIFICARE" in r["signal"]:
            r["tier"] = "verify"
        else:
            r["tier"] = "avoid"

    return rows, latest_date


def main():
    rows, latest_date = compute_signal_rows()

    print("=" * 100)
    print(f"  SEGNALE TS MOMENTUM (12m) — {latest_date.strftime('%Y-%m')} | Universo: {len(rows)} box sigillati")
    print("  NON verifica disponibilità/prezzo eseguibile reale — vedi scripts/log_execution_price.py")
    print("=" * 100)
    n_buy = sum(1 for r in rows if r["signal"] == "BUY/HOLD")
    n_verify = sum(1 for r in rows if "VERIFICARE" in r["signal"])
    n_excessive = sum(1 for r in rows if "PREZZO ECCESSIVO" in r["signal"])
    n_sell = len(rows) - n_buy - n_verify - n_excessive
    n_core = sum(1 for r in rows if r.get("tier") == "core")
    n_bench = sum(1 for r in rows if r.get("tier") == "bench")
    n_vault = sum(1 for r in rows if r.get("tier") == "vault")
    print(f"\nBUY/HOLD: {n_buy} (💎 Core: {n_core} | 🛡️ Panchina: {n_bench} | 🏛️ Vault: {n_vault}) | AVOID/SELL: {n_sell} | PREZZO ECCESSIVO: {n_excessive} | DA VERIFICARE: {n_verify}\n")

    print(f"{'Tier':10s} {'Segnale':28s} {'Rend.12m':>9s}  {'Prezzo':>10s}  {'Franchise':12s} Nome")
    print("-" * 115)
    for r in rows:
        tier_label = r.get("tier", "").upper()
        f_label = f"{r.get('franchise', '')}-{r.get('language', '')}"
        print(f"{tier_label:10s} {r['signal']:28s} {r['trailing_12m_return_pct']:>+8.1f}%  {r['current_price_eur']:>9.2f}€  {f_label:12s} {r['name']}")


if __name__ == "__main__":
    main()
