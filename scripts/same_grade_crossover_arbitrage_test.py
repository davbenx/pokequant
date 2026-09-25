"""
scripts/same_grade_crossover_arbitrage_test.py — Follow-up alla domanda sull'arbitraggio
di gradazione: "Io intendo se ha senso rompere slab GRAAD o CGC per tentare di gradare con
PSA o BGS, SENZA tentare di alzare la gradazione, ma rimanendo sempre 9."

Diversa dal test precedente (scripts/grading_company_crossover_arbitrage_test.py, che
puntava al 10): qui la domanda è se lo spread "stesso voto, compagnia diversa" (non il
premio di rarità 9->10) copre il rischio di ricaduta a voto 8 più il costo di gradazione.

DATO NUOVO REALE trovato in questa ricerca (PriceCharting non lo spezza sotto il grado
10, ma pokeprice.gg/guides/pokemon-card-grading-guide-psa-bgs-cgc, 2026-05-21, lo dice
esplicitamente per lo STESSO voto): "CGC slabs trade at a small discount to PSA in the
resale market right now — typically 10-20% less for the same grade — purely because PSA
has the bigger buyer pool. The card inside isn't any worse; it's a liquidity discount."
Lo stesso articolo classifica "CGC 9 Mint" e "BGS 9 (Mint)" come "comparable to PSA 9" -
le tre compagnie usano fasce comparabili per lo stesso voto (a differenza del salto
9->10, dove popolazioni/rarità differiscono molto tra compagnie). Nessuna fonte
equivalente esiste per GRAAD - vedi punto 2 nell'ESITO, è la differenza cruciale tra le
due carte dell'utente.

RISULTATO CENTRALE (contro-intuitivo): con un premio così piccolo (10-20%), la soglia di
probabilità di successo per andare in pari NON è bassa come nel test "punta al 10" - è
ALTA, spesso 75-93%, e SENZA COSTO FISSO (vale per qualunque prezzo della carta). Questo
perché il downside (ricaduta a voto 8) non è "nessun danno" come nel caso 10 (dove il
fallback è comunque un buon Grado 9) - qui il fallback è una perdita reale e il premio in
caso di successo è piccolo: il valore atteso puro (ignorando la fee di gradazione) è
NEGATIVO a meno che la probabilità di successo sia già alta.

Fonti aggiuntive di questo file: cardgrade.io/blog/bgs-grading-cost-breakdown,
pregradecards.com/bgs-grading-cost-calculator (BGS Economy $14.95/Standard $34.95 CHIUSI
alle sottomissioni online dall'8/8/2026, riapertura annunciata per il 15/9/2026 - non
verificabile con certezza se già riaperti oggi 25/9/2026; il tier certo disponibile e'
Express $79.95, 5-10 giorni).
"""
import sys
sys.path.insert(0, '.')
from poke_quant.engine.friction import calculate_sale_friction
from poke_quant.config import GRADING_DEFAULT

CARDMARKET_NET_FRACTION = 0.95  # ~5% fee Cardmarket, ignorando la piccola quota fissa


def net_eur(gross_eur: float) -> float:
    return calculate_sale_friction(gross_eur, item_type="single", platform="cardmarket",
                                    seller_absorbs_shipping=True).net_proceeds


def pure_ratio_threshold(discount: float, downgrade_ratio: float, fee_fraction: float = CARDMARKET_NET_FRACTION) -> float:
    """Probabilità minima di successo (restare voto 9) perché il valore atteso PURO
    (ignorando il costo fisso di gradazione, valido a QUALUNQUE prezzo della carta) non
    sia negativo. target_coef = 1/(1-discount) [prezzo PSA9 relativo al CGC9 comprato],
    fallback_coef = downgrade_ratio [prezzo se ricade a voto 8]."""
    target_coef = 1.0 / (1.0 - discount)
    return (1.0 / fee_fraction - downgrade_ratio) / (target_coef - downgrade_ratio)


def breakeven_buy_price(discount: float, downgrade_ratio: float, success_prob: float,
                         grading_cost_eur: float) -> float:
    """Prezzo minimo di acquisto (slab CGC voto 9) sopra il quale, DATA una probabilità
    di successo success_prob, l'operazione e' EV positiva includendo il costo fisso.
    Se success_prob e' sotto la soglia pura (nessun prezzo la rende profittevole),
    ritorna float('inf')."""
    if success_prob <= pure_ratio_threshold(discount, downgrade_ratio):
        return float("inf")
    lo, hi = 1.0, 200000.0
    for _ in range(60):
        mid = (lo + hi) / 2
        target = mid / (1 - discount)
        fallback = mid * downgrade_ratio
        ev = success_prob * net_eur(target) + (1 - success_prob) * net_eur(fallback) - mid - grading_cost_eur
        if ev > 0:
            hi = mid
        else:
            lo = mid
    return hi


def main():
    grading_cost_eur = GRADING_DEFAULT.fee_per_card_eur  # 95€, PSA Standard reale 2026
    print(f"Costo gradazione PSA usato (reale 2026, GRADING_DEFAULT): {grading_cost_eur:.0f}€, "
          f"fermo capitale ~{GRADING_DEFAULT.turnaround_months} mesi\n")

    print("1. SOGLIA PURA (senza costo fisso, vale a QUALUNQUE prezzo della carta): probabilità")
    print("   minima di restare voto 9 perché il valore atteso non sia già negativo:\n")
    print(f"{'Sconto stesso voto':>20s} | {'downgrade=0.25':>15s} | {'downgrade=0.35':>15s} | "
          f"{'downgrade=0.45':>15s} | {'downgrade=0.55':>15s}")
    for d in [0.10, 0.15, 0.20]:
        row = [pure_ratio_threshold(d, r) * 100 for r in [0.25, 0.35, 0.45, 0.55]]
        print(f"{d*100:18.0f}% | {row[0]:13.1f}% | {row[1]:13.1f}% | {row[2]:13.1f}% | {row[3]:13.1f}%")

    print("\n2. Sopra soglia pura, prezzo MINIMO della carta per assorbire anche il costo fisso")
    print(f"   (sconto 15%, downgrade 0.45 - scenario centrale):\n")
    for p in [0.85, 0.90, 0.95, 0.99]:
        be = breakeven_buy_price(discount=0.15, downgrade_ratio=0.45, success_prob=p, grading_cost_eur=grading_cost_eur)
        be_str = f"{be:.0f}€" if be < 200000 else "MAI (sotto soglia pura)"
        print(f"   p={p*100:.0f}% -> prezzo minimo carta: {be_str}")


if __name__ == "__main__":
    main()

"""
ESITO: risposta diversa, e più netta, di quella del test "punta al 10" - qui il verdetto
NON dipende principalmente dal costo fisso o dal valore della carta, ma da una soglia di
probabilità di successo strutturalmente ALTA (75-93% secondo lo scenario, vedi tabella 1),
valida a QUALUNQUE prezzo: con un premio piccolo (10-20%) e un downside reale (ricaduta a
voto 8, non "nessun danno" come nel caso 10), il valore atteso puro è NEGATIVO a meno che
la probabilità di restare un 9 sia già alta di per sé - il costo fisso di gradazione (95€,
~5 mesi di fermo) si aggiunge SOPRA questa soglia, alzando ulteriormente il prezzo minimo
della carta richiesto (tabella 2: appena sopra la soglia pura - es. p=85% con soglia
83.0% - il prezzo minimo esplode, ~7.260€; a p=90% scende a ~2.110€, a p=95% ~1.230€, a
p=99% ~930€ - resta comunque un'operazione per carte di valore reale (centinaia/migliaia
di euro), non per carte comuni, ed E' comunque servita una probabilità di successo molto
alta e verificabile, non assunta).

1. Per una CGC-9: pokeprice.gg classifica CGC 9 come "comparable to PSA 9" (stessa fascia
   di severità, non standard radicalmente diversi) - un argomento a favore di una
   probabilità di successo relativamente alta, ma NON una misura, e senza sapere se supera
   l'83-93% richiesto (tabella 1) non si può concludere che l'operazione sia EV positiva.
   Resta un giudizio da fare carta per carta (perizia fisica reale: centratura, angoli,
   superficie rispetto ai criteri PSA specificamente), non un numero da assumere.

2. Per una GRAAD-9: qui la differenza tra le due carte dell'utente è CRUCIALE. Il 10-20%
   di sconto "stesso voto" e la classificazione "comparable to PSA 9" sono misurati/dette
   SOLO per CGC e BGS - GRAAD non compare in NESSUN confronto tra compagnie di gradazione
   trovato in questa ricerca (ne' nella ricerca precedente, ne' in questa). Non c'è alcuna
   base per assumere che un "9" GRAAD abbia probabilità di successo comparabile a quella
   di un CGC 9 - potrebbe essere molto più vicina a quella di una carta raw di qualità
   incerta (il caso già modellato, mai eseguito, da GradingArbitrageStrategy). Trattare la
   carta GRAAD come "quasi certamente un 9 vero" in questi conti sarebbe l'errore esatto
   che questa ricerca ha già punito altrove (vedi promo/illustrator/rarità premium) -
   assumere il risultato migliore perché è quello che serve alla tesi.

CONCLUSIONE PRATICA (confidenza: alta sulla direzione - il premio piccolo rende la soglia
di successo richiesta strutturalmente alta, indipendentemente dai dettagli dei parametri
usati; media sui numeri esatti di soglia - dipendono da un dato di sconto secondario,
seppure con fonti multiple concordanti; bassa/assente sulla probabilità di successo vera,
che nessuna fonte disponibile misura): romperla per restare un 9 ha senso solo se sei
CONFIDENTE (in modo verificabile con perizia fisica, non assunto) che il crossover terrà,
E la carta vale già qualche centinaio di euro o più. Su una carta comune, o senza quella
confidenza, il conto è quasi certamente sfavorevole - il costo fisso è secondario rispetto
al fatto che il premio stesso (10-20%) è troppo piccolo per compensare anche un rischio di
ricaduta moderato. Per la carta GRAAD specificamente, mancando qualunque base per stimare
la probabilità di successo, questo conto non è nemmeno impostabile in modo responsabile.
"""
