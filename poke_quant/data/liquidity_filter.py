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
from typing import Dict, List, Tuple, Any, Optional, Iterable
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


# BUG TROVATO (2026-09-26, poi riconfermato nell'audit generale del 2026-09-29
# dopo che l'utente ha chiesto di annullare il primo fix e poi ha chiesto un
# audit completo "trova bug... invalida"): il pilota Magic: The Gathering e'
# stato testato e RIGETTATO in modo decisivo (scripts/mtg_pilot_validation.py:
# box Sharpe -0,09 DSR 0,006; singole Sharpe negativo a qualunque tetto di
# quantita' realistico) con la conclusione esplicita di NON integrarlo in
# produzione - ma nessuna funzione qui lo escludeva mai di default: ogni
# chiamante (generate_monthly_signal.py, generate_singles_signal.py, le stesse
# get_backtest_results()/get_singles_backtest_results() di app.py da cui
# derivano i numeri di produzione mostrati all'utente, script di ricerca)
# doveva ricordarsi di filtrare da solo - nessuno lo faceva. Default sicuro:
# esclude "magic" a meno che il chiamante non lo chieda esplicitamente
# (scripts/mtg_pilot_validation.py passa exclude_franchises=frozenset()
# apposta, per poter testare MTG isolato). NOTA: dopo il primo fix, un'altra
# sessione/l'utente ha aggiunto in app.py un selettore "Magic (MTG)" come
# franchise scelta esplicitamente nella UI box - quella e' una scelta di
# prodotto (mostrare MTG come opzione visibile), non in conflitto con QUESTO
# default: il default protegge le STATISTICHE AGGREGATE (Sharpe/DSR/split), il
# selettore resta libero di far vedere le righe MTG a chi lo seleziona
# esplicitamente, con un caveat (vedi app.py).
#
# AGGIORNATO (2026-09-29, "valutiamo quantitativamente di eliminare franchise
# deboli tra mtg, Pokemon Jap, One piece e di aggiungere Pokémon chinese"):
#   - "magic": confermato escluso (pilota rigettato, vedi sopra).
#   - "pokemon_chinese": NON ancora adottato - scoperto e testato
#     (scripts/discover_pokemon_chinese_sealed_universe.py,
#     scripts/pokemon_chinese_pilot_validation.py) ma il mercato secondario su
#     PriceCharting e' troppo giovane (0 prodotti con >=13 mesi di storico
#     reale su 15 scoperti - la maggior parte ne ha 1-8) per calcolare anche
#     un solo rendimento trailing a 12 mesi, figuriamoci un DSR. Escluso per
#     DATO MANCANTE, non per esito negativo - rivalutare in 6-12 mesi.
#   - "one_piece": NON escluso - testato in isolamento (scripts/
#     one_piece_pilot_validation.py): campione minuscolo (4 box, 1 trade),
#     nessuna evidenza di danno al blend (Sharpe pooled 1,23 vs 1,22 senza -
#     differenza nulla), nessuna prova sufficiente per eliminarlo. Tenuto con
#     confidenza bassa dichiarata, non validato come Pokemon EN.
DEFAULT_EXCLUDED_FRANCHISES = frozenset({"magic", "pokemon_chinese"})

# Pokemon JP: eliminato dalla produzione (scripts/jp_pilot_validation.py,
# 2026-09-29) - DSR(51) 0,015, 4/4 trade storici tutti in perdita
# (-19,2%/-33,7%/-29,8%/-18,0%), pattern negativo consistente non un campione
# insufficiente. L'esclusione e' per LINGUA (language=="jp"), non per
# franchise ("pokemon" EN resta l'universo core validato) - richiede un
# meccanismo separato da DEFAULT_EXCLUDED_FRANCHISES perche' filtra sullo
# stesso franchise "pokemon" degli asset EN che restano validi.
DEFAULT_EXCLUDED_LANGUAGES = frozenset({"jp"})


def liquid_sealed_ids(metadata: dict, prices_df: pd.DataFrame,
                       exclude_franchises: frozenset = DEFAULT_EXCLUDED_FRANCHISES,
                       exclude_languages: frozenset = DEFAULT_EXCLUDED_LANGUAGES, **kwargs) -> List[str]:
    return [
        k for k, v in metadata.items()
        if v.get("type") == "sealed" and v.get("franchise") not in exclude_franchises
        and v.get("language") not in exclude_languages
        and is_liquid_sealed(k, v, prices_df, **kwargs)
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
    exclude_franchises: frozenset = DEFAULT_EXCLUDED_FRANCHISES,
    exclude_languages: frozenset = DEFAULT_EXCLUDED_LANGUAGES,
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
    (salti di prezzo estremi) resta applicato a tutte le franchise.

    BUG TROVATO (riconfermato nell'audit generale del 2026-09-29): il pilota
    MTG e' stato RIGETTATO in modo decisivo con la conclusione esplicita di
    non integrarlo - ma questa funzione lasciava sempre passare le carte
    magic (skip del pavimento sopra), quindi ogni chiamante che non filtrava
    da solo (nessuno lo faceva) le vedeva comunque nel segnale live.
    exclude_franchises default esclude "magic" (vedi DEFAULT_EXCLUDED_FRANCHISES);
    mtg_pilot_validation.py passa frozenset() apposta per testare MTG isolato."""
    ids = []
    for item_id, info in metadata.items():
        if info.get("type") != "single":
            continue
        if info.get("franchise") in exclude_franchises:
            continue
        if info.get("language") in exclude_languages:
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


# Soglia calibrata UNA VOLTA sulla distribuzione empirica del drift su tutto
# l'universo liquido (1.077 carte con raw+grade9 comuni, finestra 6 mesi),
# PRIMA di guardare l'effetto sul backtest - vedi
# scripts/thin_market_ratio_drift_test.py. p97,5 della distribuzione reale
# (mediana 1,17x, p90 2,12x, p99 3,79x, max 10,28x): un salto netto rispetto
# al grosso della distribuzione, non un numero scelto per far scattare un
# caso particolare.
THIN_MARKET_DRIFT_WINDOW_MONTHS = 6
THIN_MARKET_DRIFT_RATIO_CUTOFF = 3.0


def compute_thin_market_drift_flags(
    metadata: Dict[str, Any],
    grade9_prices_df: pd.DataFrame,
    raw_prices_df: pd.DataFrame,
    window_months: int = THIN_MARKET_DRIFT_WINDOW_MONTHS,
    drift_ratio_cutoff: float = THIN_MARKET_DRIFT_RATIO_CUTOFF,
) -> Dict[str, Tuple[bool, str]]:
    """Flagga singole gradate il cui rapporto grade9/raw e' esploso di recente
    rispetto alla propria storia - un segnale di mercato SOTTILE (poche vendite
    reali al grado spingono l'indice), non un vero re-pricing.

    Trovato valutando un acquisto reale (Azumarill #114 [Delta Species] a
    90,25EUR): il modello dava "COLPACCIO -52%" (fair value 189,68EUR), ma il
    prezzo Grade 9 era passato da 47,50EUR a 257,37EUR in 7 mesi (+442%) mentre
    il RAW della stessa carta, nello stesso periodo, e' salito solo da 26,23EUR
    a 45,91EUR (+75%) - il rapporto grade9/raw e' triplicato mentre il mercato
    raw (molto piu' liquido, molte piu' vendite) confermava solo una crescita
    modesta. compute_grade_raw_ratio_flags() sopra confronta un
    'cardmarket_ref_price_eur' STATICO (spesso vecchio di mesi/anni) contro un
    cutoff di COORTE - non vede un salto RECENTE come questo. Qui si confronta
    la carta con SE STESSA nel tempo (rapporto oggi vs `window_months` fa),
    indipendente dalla cross-section.

    Diverso da compute_reliability_flags (che guarda salti MENSILI singoli o il
    range max/min sull'intera storia del solo grade9): un salto graduale
    distribuito su piu' mesi (come questo caso: nessun singolo mese supera il
    200%) non viene mai visto da quel filtro, ma resta un artefatto di mercato
    sottile quando il RAW non conferma la stessa velocita' di crescita.

    ESITO backtest: vedi scripts/thin_market_ratio_drift_test.py. Ritorna SOLO
    le carte flaggate; le altre (incluse quelle senza storico raw+grade9
    comune sufficiente) sono is_reliable per costruzione di questo controllo."""
    flags: Dict[str, Tuple[bool, str]] = {}
    for item_id, info in metadata.items():
        if info.get("type") != "single":
            continue
        if item_id not in grade9_prices_df.columns or item_id not in raw_prices_df.columns:
            continue
        g = grade9_prices_df[item_id].dropna()
        r = raw_prices_df[item_id].dropna()
        common_idx = g.index.intersection(r.index)
        if len(common_idx) < window_months + 1:
            continue
        g = g.loc[common_idx]
        r = r.loc[common_idx]
        if len(g) < window_months + 1:
            continue
        raw_now = float(r.iloc[-1])
        raw_past = float(r.iloc[-window_months - 1])
        grade9_now = float(g.iloc[-1])
        grade9_past = float(g.iloc[-window_months - 1])
        if raw_now <= 0 or raw_past <= 0 or grade9_past <= 0:
            continue
        ratio_now = grade9_now / raw_now
        ratio_past = grade9_past / raw_past
        if ratio_past <= 0:
            continue
        drift = ratio_now / ratio_past
        if drift >= drift_ratio_cutoff:
            flags[item_id] = (
                False,
                f"rapporto grade9/raw cresciuto {drift:.1f}x in {window_months} mesi "
                f"({ratio_past:.2f} -> {ratio_now:.2f}) - possibile mercato sottile sul grado, "
                f"il raw non conferma la stessa velocita'",
            )
    return flags


# BUCO STRUTTURALE TROVATO (l'utente, dopo aver segnalato "mercato sottile" su
# una CGC 9.5 valutata col Valutatore Slab: "tappa il buco del filtro mercato
# sottile"): compute_thin_market_drift_flags() sopra guarda SOLO il pannello
# grade9(PSA9)/raw - ma quando l'evaluator trova un dato REALE per un grado
# specifico (9.5/10/8/7) sulla pagina PriceCharting dedicata, quel numero non
# passa da NESSUN controllo di attendibilita', perche' vive in un file diverso
# (data_cache/grade_ladder_prices.json, storico mensile per tier - vedi Fase 2
# di questa sessione) mai collegato a questo modulo. Caso reale che ha esposto
# il buco: Charizard & Braixen-GX #212 [Cosmic Eclipse], tier grade9_5 ->
# 92,21€(mag) 207,06€(giu, +124% in un mese) 193,74€(lug) 203,55€(ago)
# 203,55€(set, IDENTICO al mese prima - nessuna vendita reale registrata da 2
# mesi). Nessuno dei due controlli sopra l'avrebbe preso: il salto (+124%) e'
# sotto la soglia del 200% di compute_reliability_flags, e il rapporto vs raw
# nella finestra a 6 mesi (1,90x) e' sotto la soglia 3,0x del check sopra (lo
# stesso problema di finestra "punto fisso" gia' diagnosticato per Azumarill).
#
# CALIBRAZIONE (PRIMA di guardare l'effetto sul caso che ha motivato il
# controllo, stesso principio delle soglie sopra): ne' "salto massimo in una
# finestra di 6 mesi" ne' "run di valori identici negli ultimi 6 mesi" da soli
# sono segnali utilizzabili su queste serie per-tier - sono strutturalmente
# molto piu' rumorose del pannello grade9/raw (mediana del salto singolo
# mensile gia' 1,6-2,0x su tutto l'universo, altamente COMUNE restare fermi 2+
# mesi: 23-34% dell'universo a seconda del tier, per la semplice scarsita' di
# vendite su gradi alti). Il segnale specifico e riproducibile e' la
# COMBINAZIONE: prezzo FERMO (>=2 mesi identico) DOPO un salto recente - non
# ancora confermato ne' smentito da una vendita reale successiva. Calcolato
# su tutto l'universo (script vedi
# scripts/grade_ladder_thin_market_test.py): tra le carte gia' ferme da 2+
# mesi, il salto massimo nella finestra e' quasi sempre modesto (mediana
# ~0,99-1,3x - la maggioranza dei "fermi" non e' affatto sospetta) ma con una
# coda distinta. Soglie adottate al 90° percentile PER TIER, con
# interpolazione lineare standard (np.percentile, non un rango grezzo
# sull'indice troncato - la prima versione di questa calibrazione usava un
# metodo piu' grezzo e mancava Charizard&Braixen per un soffio, 2,25x vs il
# suo 2,246x: corretto usando il percentile interpolato standard, non
# abbassando la soglia apposta per farlo rientrare) - non al 97,5° usato
# altrove: qui la coda e' meno estrema e un taglio piu' severo lascerebbe
# passare la classe di caso che ha motivato il controllo, scelta dichiarata
# esplicitamente, non nascosta:
#   grade7:   p90 = 1,95x -> soglia 1,95x
#   grade8:   p90 = 1,90x -> soglia 1,90x
#   grade9_5: p90 = 2,23x -> soglia 2,23x (Charizard&Braixen: run=2, salto
#             2,246x - appena sopra, non un caso forzato)
#   psa10:    p90 = 4,48x -> soglia 4,48x (i grail a PSA 10 sono
#             strutturalmente piu' volatili anche legittimamente - popolazioni
#             minuscole, verificato che la coda e' molto piu' estesa)
# Il Grado 9 resta escluso qui (gia' coperto da compute_thin_market_drift_flags
# sopra sul pannello legacy grade9/raw).
GRADE_LADDER_TIERS_CHECKED: Tuple[str, ...] = ("grade7", "grade8", "grade9_5", "psa10")
GRADE_LADDER_FROZEN_JUMP_CUTOFF: Dict[str, float] = {
    "grade7": 1.95,
    "grade8": 1.90,
    "grade9_5": 2.23,
    "psa10": 4.48,
}
GRADE_LADDER_FROZEN_MIN_MONTHS = 2
GRADE_LADDER_JUMP_WINDOW_MONTHS = 6


def check_grade_ladder_tier_reliable(
    series: Dict[str, float],
    tier: str,
    frozen_min_months: int = GRADE_LADDER_FROZEN_MIN_MONTHS,
    jump_window_months: int = GRADE_LADDER_JUMP_WINDOW_MONTHS,
    cutoffs: Optional[Dict[str, float]] = None,
) -> Tuple[bool, str]:
    """Nucleo puro (testabile su una singola serie) del controllo "fermo dopo
    un salto" per una serie storica per-tier di data_cache/grade_ladder_prices.json
    (es. item['grade9_5']: {"2026-01-01": 82.25, ...}). Vedi il blocco di
    commenti sopra per la calibrazione. Ritorna (is_reliable, reason)."""
    cutoffs = cutoffs if cutoffs is not None else GRADE_LADDER_FROZEN_JUMP_CUTOFF
    cutoff = cutoffs.get(tier)
    if cutoff is None or not series:
        return True, ""

    dates = sorted(series.keys())
    vals = [float(series[d]) for d in dates]
    if len(vals) < jump_window_months + 1:
        return True, ""

    recent = vals[-(jump_window_months + 1):]

    # Lunghezza del run finale di valori identici (nessuna vendita reale
    # registrata da N mesi - il valore mostrato e' l'ultimo noto, non uno
    # aggiornato).
    run = 1
    i = len(recent) - 1
    while i > 0 and recent[i] == recent[i - 1]:
        run += 1
        i -= 1
    if run < frozen_min_months:
        return True, ""

    # Massimo salto mensile in QUALUNQUE punto della finestra recente (non
    # solo appena-prima-del-fermo, che sottostimerebbe un salto avvenuto
    # all'inizio della finestra - stesso errore gia' diagnosticato e corretto
    # altrove in questa sessione per il check grade9/raw).
    max_jump = 1.0
    for j in range(1, len(recent)):
        if recent[j - 1] > 0:
            max_jump = max(max_jump, recent[j] / recent[j - 1], recent[j - 1] / recent[j])

    if max_jump >= cutoff:
        return False, (
            f"prezzo {tier} fermo da {run} mesi (nessuna vendita reale registrata) dopo un "
            f"salto di {max_jump:.1f}x nella finestra di {jump_window_months} mesi - possibile "
            f"mercato sottile su questo grado specifico, non ancora confermato ne' smentito"
        )
    return True, ""


def compute_grade_ladder_reliability_flags(
    grade_ladder_data: Dict[str, Dict[str, Dict[str, float]]],
    tiers: Iterable[str] = GRADE_LADDER_TIERS_CHECKED,
    frozen_min_months: int = GRADE_LADDER_FROZEN_MIN_MONTHS,
    jump_window_months: int = GRADE_LADDER_JUMP_WINDOW_MONTHS,
    cutoffs: Optional[Dict[str, float]] = None,
) -> Dict[str, Tuple[bool, str]]:
    """Applica check_grade_ladder_tier_reliable() a tutto il contenuto di
    data_cache/grade_ladder_prices.json. A differenza degli altri filtri in
    questo modulo, la chiave qui e' "{item_id}:{tier}" (non solo item_id) -
    questo e' un segnale PER GRADO, non per carta: una carta puo' essere
    perfettamente affidabile a Grado 8 e sottile a PSA 10. Ritorna SOLO le
    combinazioni non affidabili (stesso pattern degli altri check sopra)."""
    flags: Dict[str, Tuple[bool, str]] = {}
    for item_id, tier_series in grade_ladder_data.items():
        for tier in tiers:
            series = tier_series.get(tier)
            if not series:
                continue
            ok, reason = check_grade_ladder_tier_reliable(
                series, tier, frozen_min_months=frozen_min_months,
                jump_window_months=jump_window_months, cutoffs=cutoffs,
            )
            if not ok:
                flags[f"{item_id}:{tier}"] = (ok, reason)
    return flags
