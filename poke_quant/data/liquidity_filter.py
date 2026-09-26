"""
poke_quant/data/liquidity_filter.py — Filtro di attendibilità/liquidità sulle serie prezzo.

Trovato espandendo l'universo a 389 asset: molti box sigillati vintage (e alcune
singole ultra-rare/promo) hanno volumi di vendita reali così bassi che il prezzo
guida di PriceCharting NON è rumore accettabile ma dato inutilizzabile — es.
neo_destiny_bb passa da ~13.000€ a 53,7€ in un mese, skyridge_bb tocca 182.915€.
Non è un dato "reale ma volatile": è un artefatto di mercato troppo sottile per
avere un prezzo mensile significativo (spesso una sola vendita anomala per mese).

Questo modulo NON cancella quei dati (restano in historical_prices.csv per
trasparenza) — li FLAGGA, cosicché ogni script di validazione possa escluderli
esplicitamente dall'universo investibile invece di lasciare che dominino
silenziosamente i risultati di una strategia.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any
import pandas as pd

DEFAULT_MAX_MONTHLY_JUMP = 2.0   # +-200% in un mese singolo
DEFAULT_MAX_RANGE_RATIO = 15.0   # max/min sull'intera serie


def compute_reliability_flags(
    prices_df: pd.DataFrame,
    max_monthly_jump: float = DEFAULT_MAX_MONTHLY_JUMP,
    max_range_ratio: float = DEFAULT_MAX_RANGE_RATIO,
    min_observations: int = 6,
) -> Dict[str, Tuple[bool, str]]:
    """Ritorna {item_id: (is_reliable, motivo)} per ogni colonna di prices_df."""
    flags: Dict[str, Tuple[bool, str]] = {}
    for col in prices_df.columns:
        s = prices_df[col].dropna()
        s = s[s > 0]
        if len(s) < min_observations:
            flags[col] = (False, f"serie troppo corta ({len(s)} osservazioni)")
            continue
        mom_ret = s.pct_change().dropna()
        max_jump = float(mom_ret.abs().max()) if not mom_ret.empty else 0.0
        ratio = float(s.max() / s.min())
        if max_jump > max_monthly_jump:
            flags[col] = (False, f"salto mensile {max_jump*100:.0f}% (soglia {max_monthly_jump*100:.0f}%)")
        elif ratio > max_range_ratio:
            flags[col] = (False, f"range max/min {ratio:.1f}x (soglia {max_range_ratio:.1f}x)")
        else:
            flags[col] = (True, "")
    return flags


def get_reliable_columns(prices_df: pd.DataFrame, **kwargs) -> List[str]:
    flags = compute_reliability_flags(prices_df, **kwargs)
    return [col for col, (ok, _) in flags.items() if ok]


def filter_reliable(prices_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Ritorna prices_df con solo le colonne che passano il filtro di attendibilità."""
    return prices_df[get_reliable_columns(prices_df, **kwargs)]


# Rapporto prezzo attuale / MSRP piu' alto osservato nell'universo sealed "era
# moderna" (2019+) gia' in produzione - calcolato UNA VOLTA sui 30 box moderni
# con MSRP noto, PRIMA di guardare l'effetto sul backtest (altrimenti sarebbe
# overfitting della definizione stessa di universo, non delle sue regole).
# Vedi scripts/sealed_universe_expansion_test.py per il calcolo e l'esito.
MAX_PRICE_TO_MSRP_RATIO = 21.6


def is_liquid_sealed(item_id: str, meta: dict, prices_df: pd.DataFrame,
                      modern_era_cutoff: str = "2019-01-01",
                      max_price_msrp_ratio: float = MAX_PRICE_TO_MSRP_RATIO) -> bool:
    """Criterio di liquidita' per un box sealed, PIU' AMPIO del taglio per anno
    usato finora (MODERN_ERA_CUTOFF da solo): un box e' incluso se e' di era
    moderna (2019+, come prima), OPPURE se e' piu' vecchio ma il suo rapporto
    prezzo/MSRP resta DENTRO il range gia' osservato nell'universo moderno
    validato - cioe' si e' apprezzato in modo comparabile a un box "normale",
    non come un pezzo da museo a volume di scambio quasi nullo (es. Team Rocket
    Returns a 485x il MSRP). Se il box vintage non ha un MSRP reale in metadata,
    resta escluso - NON si fabbrica un numero storico che non conosciamo (stessa
    regola di scripts/discover_sealed_universe.py)."""
    if meta.get("data_quality") == "thin_unreliable":
        return False
    if item_id not in prices_df.columns:
        return False
    release = meta.get("release_date")
    if not release:
        return False
    if release >= modern_era_cutoff:
        return True
    msrp = meta.get("msrp")
    if not msrp:
        return False
    s = prices_df[item_id].dropna()
    s = s[s > 0]
    if s.empty:
        return False
    ratio = float(s.iloc[-1]) / float(msrp)
    return ratio <= max_price_msrp_ratio


def liquid_sealed_ids(metadata: dict, prices_df: pd.DataFrame, **kwargs) -> List[str]:
    return [
        k for k, v in metadata.items()
        if v.get("type") == "sealed" and is_liquid_sealed(k, v, prices_df, **kwargs)
    ]


# Sotto questa percentile della propria coorte d'eta' (+-3 anni di uscita, coorte
# richiesta >=20 carte) e' un outlier statistico vero sul rapporto grade9/raw, non
# solo "un po' sotto la mediana" - vedi scripts/graded_raw_ratio_reliability_test.py.
GRADE_RAW_RATIO_PERCENTILE_CUTOFF = 0.10
GRADE_RAW_RATIO_MIN_COHORT = 20
GRADE_RAW_RATIO_COHORT_WINDOW_YEARS = 3


# Trovato verificando un prezzo reale (l'utente: "Mantine e' consigliata a
# 11EUR, ma la sola procedura di gradazione costa di piu'"): stima
# conservativa del costo minimo reale per portare una carta a PSA/CGC Grade 9
# anche nella fascia bulk piu' economica (tariffa di sottomissione + carta +
# spedizione/assicurazione) - stima ragionata da conoscenza generale delle
# tariffe correnti, NON un listino verificato in tempo reale: va corretta se
# emergono numeri piu' precisi. Sotto questa soglia, nessuno gradirebbe oggi
# una nuova copia (perdita garantita) - l'offerta e' un pool fisso e non
# rinnovabile di slab gia' gradati in passato (tipicamente durante il boom
# PSA 2020-21) e ora svenduti sotto costo da speculatori delusi, non un
# mercato normale legato alla scarsita' della carta. Testato empiricamente
# (scripts/grading_cost_floor_test.py): escludere queste carte NON peggiora
# Sharpe/CAGR/DSR del fattore scarsita' (1,57->1,58 con questa soglia) -
# coerente con l'ipotesi che il loro "sconto" sia un artefatto della
# regressione log-lineare compressa vicino allo zero, non un segnale reale.
MIN_SINGLES_MEDIAN_PRICE_EUR = 20.0

# BUG TROVATO (l'utente: "Vedo ancora Sandslash #42, che e' sotto il prezzo
# da gradazione"): la prima versione usava la mediana su TUTTA la storia
# della carta - Sandslash #42 valeva 21-24EUR a fine 2025, e' sceso a ~14EUR
# a marzo 2026 e ci resta da 7 mesi consecutivi, ma la mediana storica
# (gonfiata dai prezzi vecchi, piu' alti) resta a 20,32EUR - appena sopra
# soglia, quindi passa il filtro nonostante il prezzo REALE di oggi sia ben
# sotto il costo di gradazione. Non un caso isolato: verificato che altre 9
# carte hanno lo stesso problema (mediana storica >=20EUR ma mediana degli
# ultimi 12 mesi <20EUR). Corretto usando la mediana sui SOLI ultimi 3 mesi
# (stessa finestra di "freschezza" del ribilanciamento in produzione,
# rebalance_every_months=3 - non una scelta arbitraria) invece di tutta la
# storia: abbastanza reattiva da cogliere un calo sostenuto come questo,
# abbastanza smussata da non far entrare/uscire una carta ogni mese per un
# singolo dato rumoroso.
MIN_SINGLES_PRICE_WINDOW_MONTHS = 3

# BUG TROVATO (l'utente, due carte reali: "Unown [K] #58" e "Dark Golduck #37" -
# "ancora prezzi che e' impossibile trovare gradate"): un crollo di un SOLO mese
# recente puo' restare mascherato dalla mediana a 3 mesi se gli altri 2 mesi della
# finestra erano ancora sopra soglia - Unown K #58 e' stabile a ~22EUR per 11 mesi
# poi crolla a 14,63EUR nell'ultimo mese: mediana degli ultimi 3 mesi (22,55 /
# 22,38 / 14,63) = 22,38EUR, sopra soglia, ma il prezzo ATTUALE (quello che vedi e
# a cui compreresti oggi) e' 14,63EUR, sotto costo di gradazione. Diverso dal bug
# Sandslash (calo SOSTENUTO per mesi mascherato dalla mediana su tutta la storia):
# qui il calo e' improvviso e recentissimo, e la mediana-di-finestra lo diluisce
# con 2 mesi ancora "vecchi" e piu' alti. Corretto richiedendo che ANCHE l'ultimo
# prezzo disponibile (non solo la mediana della finestra) sia sopra soglia - una
# carta deve essere consistentemente sopra costo, non solo "in maggioranza" nella
# finestra, per essere considerata economicamente gradabile al prezzo mostrato
# oggi. Non riapre il caso Sandslash (un calo sostenuto fallisce comunque
# entrambi i controlli) e non esclude un recupero genuino iniziato prima
# dell'ultimo mese (se gli ultimi mesi sono giA' saliti sopra soglia, sia la
# mediana che l'ultimo prezzo la superano).


def liquid_singles_ids(
    metadata: Dict[str, Any],
    grade9_prices_df: pd.DataFrame,
    min_median_price_eur: float = MIN_SINGLES_MEDIAN_PRICE_EUR,
    price_window_months: int = MIN_SINGLES_PRICE_WINDOW_MONTHS,
) -> List[str]:
    """Universo singole investibile: esclude le carte gia' flaggate
    data_quality="thin_unreliable" (compute_reliability_flags, gia' applicato
    in ogni backtest ma MAI wired nel segnale live prima di questa funzione -
    vedi scripts/generate_singles_signal.py) e quelle sotto il pavimento di
    costo di gradazione MIN_SINGLES_MEDIAN_PRICE_EUR, valutato SIA sulla
    mediana degli ultimi price_window_months mesi (non tutta la storia - un
    prezzo calato e rimasto basso per mesi non deve restare "invisibile"
    dietro prezzi vecchi piu' alti, vedi commento sopra) SIA sull'ultimo
    prezzo disponibile (un crollo di un solo mese recente non deve restare
    mascherato da una mediana ancora alta per gli altri 2 mesi della
    finestra - vedi il bug Unown K #58/Dark Golduck #37 nel commento sopra).
    Non sul solo mese corrente da solo (per non far entrare/uscire una carta
    ogni mese per rumore) - la carta deve superare entrambi i controlli.

    Progetto pilota Magic: The Gathering (2026-09-26): il pavimento di costo
    di gradazione NON si applica alle carte franchise="magic" - il pannello
    prezzo passato qui per quegli item e' RAW/ungraded (scelta confermata
    dall'utente: la gradazione MTG e' quasi assente su PriceCharting fuori
    da poche carte vintage iconiche, vedi discover_mtg_chase_singles.py), e
    il pavimento esiste solo perche' e' antieconomico gradare una carta che
    costa meno della gradazione stessa - un concetto che non si applica a
    una carta che non si intende gradare. Il controllo di attendibilita'
    (salti di prezzo estremi) resta applicato a tutte le franchise."""
    ids = []
    for item_id, info in metadata.items():
        if info.get("type") != "single":
            continue
        if info.get("data_quality") == "thin_unreliable":
            continue
        if item_id not in grade9_prices_df.columns:
            continue
        s = grade9_prices_df[item_id].dropna()
        s = s[s > 0]
        if s.empty:
            continue
        if info.get("franchise") == "magic":
            ids.append(item_id)
            continue
        window = s.tail(price_window_months)
        if window.median() < min_median_price_eur or window.iloc[-1] < min_median_price_eur:
            continue
        ids.append(item_id)
    return ids


def compute_grade_raw_ratio_flags(
    metadata: Dict[str, Any],
    grade9_prices_df: pd.DataFrame,
    percentile_cutoff: float = GRADE_RAW_RATIO_PERCENTILE_CUTOFF,
    min_cohort: int = GRADE_RAW_RATIO_MIN_COHORT,
    cohort_window_years: int = GRADE_RAW_RATIO_COHORT_WINDOW_YEARS,
) -> Dict[str, Tuple[bool, str]]:
    """Flagga singole gradate il cui rapporto grade9/raw (pannello PriceCharting
    Grade 9 vs "cardmarket_ref_price_eur" in metadata, gia' presente ma non
    usato in nessun'altra pipeline) e' un outlier basso rispetto alla coorte di
    carte della stessa era. Trovato verificando un prezzo reale (l'utente ha
    trovato un Raichu #14 [Fossil 1999] a 250EUR tutto compreso contro 107,60EUR
    mostrati in dashboard): il filtro di attendibilita' esistente
    (compute_reliability_flags) controlla solo salti/volatilita' della serie -
    una serie liscia ma persistentemente troppo bassa rispetto al livello reale
    (grade9 sottostimato per scarsita' di vendite PSA9 storiche su carte vintage
    poco liquide) non viene vista per costruzione. Testato empiricamente
    (scripts/graded_raw_ratio_reliability_test.py): escludere questi outlier
    migliora Sharpe/CAGR/MaxDD/DSR del fattore scarsita', non solo protegge -
    coerente con l'ipotesi che siano falsi positivi da dato sottile, non alfa
    reale. Ritorna SOLO le carte flaggate (non ok); tutte le altre (incluse
    quelle senza cardmarket_ref_price_eur, dato insufficiente) sono is_reliable.
    """
    rows = []
    for item_id, info in metadata.items():
        if info.get("type") != "single":
            continue
        ref = info.get("cardmarket_ref_price_eur")
        rel = info.get("release_date")
        if not ref or ref <= 0 or not rel or item_id not in grade9_prices_df.columns:
            continue
        s = grade9_prices_df[item_id].dropna()
        s = s[s > 0]
        if s.empty:
            continue
        ratio = float(s.iloc[-1]) / float(ref)
        if not (0.3 < ratio < 30):  # artefatto grossolano (raw~0 o dato corrotto), non l'oggetto di questo filtro
            continue
        rows.append({"item_id": item_id, "year": pd.to_datetime(rel).year, "ratio": ratio})
    df = pd.DataFrame(rows)

    flags: Dict[str, Tuple[bool, str]] = {}
    if df.empty:
        return flags
    for _, row in df.iterrows():
        cohort = df[(df["year"] >= row["year"] - cohort_window_years) & (df["year"] <= row["year"] + cohort_window_years)]
        if len(cohort) < min_cohort:
            continue
        cutoff = cohort["ratio"].quantile(percentile_cutoff)
        if row["ratio"] < cutoff:
            flags[row["item_id"]] = (
                False,
                f"rapporto grade9/raw {row['ratio']:.2f} sotto il {percentile_cutoff*100:.0f}° percentile "
                f"della coorte d'eta' (soglia {cutoff:.2f})",
            )
    return flags
