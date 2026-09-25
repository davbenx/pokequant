"""
scripts/grading_company_crossover_arbitrage_test.py — Ha senso comprare uno slab di una
compagnia "debole" (CGC/SGC/GRAAD...) a sconto, romperlo (crack-out) e rimandarlo a
gradare a PSA per incassare lo spread PSA vs. altre compagnie?

Domanda dell'utente: "Considerando tutti i costi, ha senso sfruttare in qualche modo
lo spread tra le varie case di gradazione? Fai dei test." (segue la scoperta, nella
sessione precedente, che PSA 10 vale 2.6x-3.8x CGC 10 sulla stessa carta).

METODO — dati REALI, non simulati:
1. Prezzo per-compagnia: PriceCharting spezza il prezzo per compagnia SOLO al grado 10
   ("Ungraded/Grade 1.../Grade 9/Grade 9.5/PSA 10/CGC 10/SGC 10/BGS 10/..." - verificato
   via fetch live il 2026-09-25 su 3 carte). Dal grado 1 al 9.5 il prezzo è un blend
   cross-company: NON esiste alcun dato PriceCharting sul prezzo "PSA 9" vs "CGC 9"
   separato - il caso reale dell'utente (due carte Grado 9) NON è testabile con questi
   dati. Questo script testa quindi solo la versione al grado 10 (l'unica per cui esiste
   un prezzo reale per compagnia), e lo dice esplicitamente nel verdetto finale.
2. Costi di gradazione 2026 REALI (fonti in fondo al file), non gli assunti datatici
   (fee_per_card_eur=25, turnaround 2 mesi) già presenti in poke_quant/config.py per la
   strategia raw->PSA (GradingArbitrageStrategy) - risultano STALE: PSA ha sospeso i tier
   economici (Value $29-59) il 2 giugno 2026 per arretrato, lasciando Standard $59.99
   (90-100 giorni lavorativi, ~4.5-5 mesi) come tier più economico disponibile oggi.
3. Rischio di "non incrociare": CGC offre un servizio Crossover con opzione "same grade
   or better" (verifica preliminare prima di rompere lo slab originale - se non
   incrocerebbe al grado richiesto, la carta torna nello slab originale, ma la fee resta
   dovuta); PSA non offre un servizio equivalente per le carte in ingresso da altre
   compagnie - si manda come sottomissione raw normale, nessuna garanzia, fee dovuta a
   prescindere dal voto. Non esiste una probabilità di "incrocio riuscito" misurabile con
   i dati disponibili (population report PSA bloccati da Cloudflare, vedi
   poke_quant/slabs/edge_calculator.py) - questo script quindi risolve l'EQUAZIONE AL
   CONTRARIO: qual è la probabilità minima di successo (breakeven) richiesta perché
   l'operazione non perda soldi, dati i costi e lo spread REALI, lasciando alla persona
   valutare se quella soglia è plausibile.

TROVATO IN CORSO DI QUESTA RICERCA (secondario ma rilevante): poke_quant/slabs/ contiene
un modulo di "arbitraggio cross-grading" completo (edge_calculator.py, slab_backtester.py,
slab_universe.py) con Sharpe/CAGR calcolati - MA edge_calculator.py stesso segnala in testa
che HISTORICAL_GRADE_RATIOS non è calibrato su dati reali (priors scritti a mano). Peggio:
slab_universe.py (24 carte "grail") NON porta lo stesso avviso e presenta prezzi/listing
come "VERIFIED_AVAILABLE" con paese/unità specifiche - non esiste in questo repo alcuno
script che abbia fatto quella verifica live, sono numeri inventati con falsa precisione.
Il modulo NON è agganciato a app.py (isolato, mai in produzione) - ma i suoi risultati
NON vanno citati come evidenza. Va segnalato all'utente, non silenziato.
"""
import numpy as np
import sys
sys.path.insert(0, '.')
from poke_quant.engine.friction import calculate_sale_friction

EUR_USD = 1.08  # stesso assunto già usato in poke_quant/slabs/edge_calculator.py (calc_geo_dislocation default)

# --- 1. Dati REALI, verificati via fetch live PriceCharting il 2026-09-25 (grado 10 - unico
#     grado dove il sito spezza il prezzo per compagnia) ---
REAL_GRADE10_DATA_USD = {
    "Charizard #4 (Base Set)": dict(era="vintage", cgc10=4707.23, sgc10=7365.00, psa10=12275.00,
                                     bgs10=15958.00, grade9_blend=2976.69, grade8_blend=1311.59),
    "Blastoise #2 (Base Set)": dict(era="vintage", cgc10=1825.00, sgc10=4138.00, psa10=6897.39,
                                     bgs10=8967.00, grade9_blend=950.00, grade8_blend=372.00),
    "Umbreon VMAX #215 (Evolving Skies)": dict(era="modern", cgc10=2903.52, sgc10=1136.00, psa10=4125.00,
                                                bgs10=4850.00, grade9_blend=2098.10, grade8_blend=1875.00),
}

# --- 2. Costi REALI di gradazione 2026 (fonti in fondo al file) ---
PSA_STANDARD_FEE_USD = 59.99       # tier più economico disponibile oggi (Value sospeso 2/6/2026)
PSA_HANDLING_FEE_USD = 10.00
PSA_RETURN_SHIP_INSURED_USD = 15.00  # minimo indicativo, per carte da migliaia di $ sarebbe più alto
PSA_TURNAROUND_MONTHS = 4.5          # 90-100 giorni lavorativi

SUBMIT_SHIPPING_EUR = 15.00   # spedizione assicurata IT->hub gradazione (stima conservativa-bassa)
SELL_PLATFORM = "cardmarket"


def net_sale_eur(gross_eur: float) -> float:
    return calculate_sale_friction(gross_eur, item_type="single", platform=SELL_PLATFORM,
                                    seller_absorbs_shipping=True).net_proceeds


def breakeven_success_probability(buy_price_eur: float, target_price_eur: float, fallback_price_eur: float,
                                   grading_cost_eur: float) -> float:
    """Prob. minima di ottenere il voto target (altrimenti si recupera fallback_price_eur)
    perché il trade non perda soldi in media, dati i costi REALI."""
    net_target = net_sale_eur(target_price_eur)
    net_fallback = net_sale_eur(fallback_price_eur)
    total_cost = buy_price_eur + grading_cost_eur
    denom = net_target - net_fallback
    if denom <= 0:
        return float("inf")
    return (total_cost - net_fallback) / denom


def main():
    grading_cost_eur = (PSA_STANDARD_FEE_USD + PSA_HANDLING_FEE_USD + PSA_RETURN_SHIP_INSURED_USD) / EUR_USD \
        + 2 * SUBMIT_SHIPPING_EUR  # andata (a gradare) + ritorno (per la vendita) del CGC-slab originale
    print(f"Costo gradazione stimato (fee PSA Standard + handling + return + spedizioni IT, REALE 2026): "
          f"{grading_cost_eur:.0f}€ | Fermo capitale: ~{PSA_TURNAROUND_MONTHS} mesi\n")

    print(f"{'Carta':40s} | {'Era':8s} | {'Buy CGC10':>10s} | {'Target PSA10':>13s} | "
          f"{'Fallback G9':>12s} | {'Breakeven P(succ.)':>18s}")
    for name, d in REAL_GRADE10_DATA_USD.items():
        buy_eur = d["cgc10"] / EUR_USD
        target_eur = d["psa10"] / EUR_USD
        fallback_eur = d["grade9_blend"] / EUR_USD
        p_be = breakeven_success_probability(buy_eur, target_eur, fallback_eur, grading_cost_eur)
        print(f"{name:40s} | {d['era']:8s} | {buy_eur:9.0f}€ | {target_eur:12.0f}€ | "
              f"{fallback_eur:11.0f}€ | {p_be*100:16.1f}%")

    print("\n--- Sensitivity: fallback pessimistico (recupero a Grado 8, non 9) ---")
    for name, d in REAL_GRADE10_DATA_USD.items():
        buy_eur = d["cgc10"] / EUR_USD
        target_eur = d["psa10"] / EUR_USD
        fallback_eur = d["grade8_blend"] / EUR_USD
        p_be = breakeven_success_probability(buy_eur, target_eur, fallback_eur, grading_cost_eur)
        print(f"{name:40s} | breakeven P(successo) = {p_be*100:.1f}% (fallback Grado 8 = {fallback_eur:.0f}€)")


if __name__ == "__main__":
    main()

"""
ESITO: NON VALIDATO come strategia sistematica — e non nel senso statistico usuale
(niente DSR/PBO/walk-forward qui: non esiste una serie storica di "trade di cross-grading"
da testare, è un'attività carta-per-carta, non un fattore rotazionale su un paniere). Il
verdetto qui è un onesto conto dei costi/probabilità, con questi limiti dichiarati:

1. Il caso REALE dell'utente (CGC-9 vs GRAAD-9) NON è testabile: PriceCharting non spezza
   il prezzo per compagnia sotto il grado 10 — a Grado 9 non sappiamo quanto valga
   "PSA 9" vs "CGC 9" vs "GRAAD 9" separatamente, quindi non si può calcolare né lo
   spread né la convenienza per le sue due carte specifiche con dati reali. Le sue carte
   NON sono il caso che questo script prova ad arbitrare (che richiede di partire da uno
   slab di grado 10, l'unico dove i dati esistono).

2. Sul caso testabile (comprare un CGC 10 vintage a sconto, romperlo, tentare PSA 10):
   con costi 2026 reali (109€ di gradazione+spedizioni, ~4.5 mesi di fermo) la
   probabilità minima di successo per andare in pari varia molto per carta (fallback
   grado 9 / grado 8): dal 18.6%/25.8% su Blastoise Base Set e 22.7%/34.4% su Charizard
   Base Set (spread enorme, CGC10 vale solo 0.38x/0.26x PSA10) al 53.8%/58.4% su Umbreon
   VMAX (spread modesto - 0.70x - mercato moderno più liquido su tutte le compagnie,
   inclusa SGC il cui prezzo qui è probabilmente un outlier per pochissime vendite: SGC10
   $1.136 è sotto persino il Grado 9.5 blended $2.939, segno di un dato troppo sottile per
   fare testo). QUESTA VARIANZA È IL PUNTO: lo spread non è una costante strutturale
   sfruttabile in modo sistematico,
   è quasi certamente dominato dalla scarsità di comp/vendite reali per le compagnie
   minori sulle carte vintage più rare (esattamente il pattern "mercato sottile, dato
   rumoroso" già trovato più volte in questa ricerca su altri fattori, es. Mantine #64
   thin_unreliable) — non un'inefficienza pulita e ripetibile.

3. Non esiste, in nessuna fonte disponibile in questo momento, un tasso di successo
   misurato per il crossover CGC->PSA o GRAAD->PSA sulle carte Pokémon — CGC stessa non
   garantisce l'esito del proprio servizio Crossover ("same grade or better" è solo un
   pre-check, non una garanzia; PSA non ha nemmeno un servizio equivalente per le
   sottomissioni provenienti da altre compagnie). Senza questo numero, qualunque EV
   puntuale sarebbe una precisione finta. Conoscenza generale del settore (non
   verificabile con i dati qui disponibili) suggerisce che PSA è tipicamente PIÙ severo
   di CGC su centratura/angoli — quindi la probabilità di successo per una carta
   "appena" CGC-9/10 è verosimilmente sotto il 50%, non sopra: la soglia di breakeven
   (18.6%-58.4% secondo la carta) NON è quindi chiaramente sotto una probabilità di
   successo plausibile né chiaramente sopra — è proprio nella zona grigia dove il verdetto
   dipende dalla perizia fisica reale della singola carta, non da un numero assumibile
   a priori.

4. Costo aggiornato (già corretto in poke_quant/config.py, GRADING_DEFAULT, in questo
   stesso commit): la sospensione dei tier PSA economici (2 giugno 2026) e il turnaround
   reale di 90-100 giorni lavorativi rendevano l'ipotesi fee_per_card_eur=25€/turnaround
   2 mesi (usata da GradingArbitrageStrategy, raw->PSA, anch'essa già SOLO SEGNALE
   ESPLORATIVO, mai eseguita) OBSOLETA — aggiornata a 95€/5 mesi con fonti citate nel
   commento del config.

CONCLUSIONE PRATICA (confidenza: alta sui costi/spread misurati, bassa sulla probabilità
di successo che nessuna fonte fornisce): non c'è una strategia SISTEMATICA da eseguire
qui — il costo/fermo capitale reale è concreto e misurato, il lato dei ricavi richiede una
stima di probabilità che nessun dato disponibile permette di calibrare, e il campione di
sole 3 carte mostra una variabilità abbastanza ampia da escludere un edge stabile e
ripetibile. Resta, al massimo, un'attività IDIOSINCRATICA caso-per-caso (perizia fisica
di una singola carta), non un fattore da aggiungere alla strategia validata.

FONTI (ricerca web 2026-09-25):
- PSA fee 2026 (Standard $59.99, Value sospeso 2/6/2026, turnaround 90-100gg lavorativi):
  allvintagecards.com/psa-grading-costs/, cardgrade.io/psa-grading
- CGC fee 2026 (Bulk $17, Economy $20, Standard $55): cgccards.com/news/article/14993/,
  cgccards.com/submit/services-fees/cgc-grading/
- CGC Crossover "same grade or better" (non garantito, fee dovuta comunque):
  cgccards.com/about/help-center-faqs/cgc-cards-grading/other-grading-topics/
- PSA Europe (Frankfurt, dealer/partner, dal luglio/agosto 2026): cardpulse.club,
  cardsense.pro/guides/psa-europe-2026-frankfurt-opening
"""
