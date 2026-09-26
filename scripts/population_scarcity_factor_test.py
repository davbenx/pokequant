#!/usr/bin/env python3
"""
scripts/population_scarcity_factor_test.py — L'utente: "Vedendo le popolazioni
di gradate di vari gradi e RAW, e' possibile identificare inefficienze di
mercato, carte sottovalutate rispetto alle pari? Verifica."

Il fattore di produzione (ScarcityValueFactorStrategy) usa la scarsita' TIPOGRAFICA
(EXPECTED_COPIES_PER_BOX - una tabella generica per fascia di rarita', stima
community, non un conteggio reale) come proxy di quanto sia difficile trovare
una carta. Ora abbiamo popolazione REALE per grado (data_cache/population_history.csv,
PriceCharting, PSA+CGC) - una misura DIRETTA della scarsita' realizzata, non un
proxy ex-ante. Domanda: la popolazione reale spiega variazione di prezzo che la
rarita' tipografica NON spiega già? Se si', e' un candidato per arricchire il
fattore; se la popolazione e' solo un'altra faccia della stessa rarita' (le carte
piu' rare tipograficamente hanno anche popolazione minore, ovviamente), non
aggiunge informazione nuova.

LIMITE METODOLOGICO DICHIARATO (perche' questo e' un test CROSS-SEZIONALE di OGGI,
non un backtest storico validato con DSR/PBO/walk-forward): abbiamo un SOLO
snapshot di popolazione (25/9/2026, vedi la ricerca su "possiamo trovare dati per
pop report" - nessuna fonte gratuita ha storico retroattivo 2021-2026). Non si può
quindi testare se la popolazione ha PREVISTO i rendimenti passati in modo
backtestato - solo se oggi spiega prezzo attuale al netto di quanto già spiegato
da eta'/rarita'/set. Un'eventuale backtest retroattivo userebbe la popolazione DI
OGGI come proxy per la popolazione di anni fa - ragionevole per carte vintage
(poche nuove sottomissioni ogni anno su una carta del 1999), MA introdurrebbe un
lookahead bias serio su carte moderne (popolazione crescuta rapidamente insieme
al prezzo, se la carta e' diventata di moda di recente) - per questo qui il test
retroattivo, se eseguito, e' ristretto alle sole carte vintage (ante-2017).

ESITO: NO, la popolazione reale non identifica inefficienze che eta'/rarita'/set
non catturino gia' - testato con DUE metriche di popolazione indipendenti,
risultato coerente su entrambe:

  Popolazione grado 9 (log(1/pop), n=469 carte): R^2 0,6053 -> 0,6066 aggiungendo
  la popolazione (+0,13 punti percentuali, ininfluente). Coefficiente -0,039 (SEGNO
  SBAGLIATO - popolazione minore -> prezzo PIU' BASSO nel modello, l'opposto di
  quanto un premio di scarsita' predirebbe). Correlazione tra il residuo del
  modello SENZA popolazione e la scarsita' di popolazione: -0,041 (praticamente
  zero).

  Gem rate al grado 10 (pop_grado10/pop_totale, n=464 carte): stesso schema - R^2
  0,5996 -> 0,6031 (+0,35pp), coefficiente -0,085 (di nuovo segno sbagliato),
  correlazione residuo -0,073.

INTERPRETAZIONE: la popolazione reale non e' informazione INDIPENDENTE - e' in
larga parte un riflesso a valle della STESSA rarita' tipografica gia' nel
modello (una carta rara per stampa ha anche popolazione gradata minore, per
costruzione: meno esemplari esistono, meno ne arrivano a gradazione). Il
"segnale nuovo" che ci si aspetterebbe da un dato di popolazione REALE invece
di un proxy tipografico non emerge - a parita' di rarita'/eta'/set, la
popolazione residua non prevede il prezzo. La lista delle carte "più
sottovalutate" (primo quantile 20%) cambia solo per il 4% (89/93 in comune)
aggiungendo la popolazione - nessun impatto pratico sul segnale live.

NON e' stato eseguito il backtest retroattivo (vintage-only, con l'assunzione di
popolazione stabile) descritto sopra: il test cross-sezionale odierno mostra
gia' che il segnale non porta informazione aggiuntiva, quindi un backtest
costruito sulla stessa variabile non avrebbe nulla da mostrare che non sia già
confutato qui - non ha senso spendere il rischio di lookahead bias su un
segnale che risulta nullo anche nella forma piu' favorevole (nessun ritardo,
dato di oggi usato su se stesso).

CONCLUSIONE PRATICA: non aggiunto al fattore di produzione. Non e' uno scarto
per motivi statistici di validazione (DSR/PBO/walk-forward, come per rarita'
premium/illustratore/promo) - e' piu' semplice: il dato non porta contenuto
informativo misurabile, a monte di qualunque backtest. Resta un'ipotesi
plausibile in linea di principio (la popolazione reale DOVREBBE contenere
informazione che un proxy tipografico non ha) ma il test odierno non la
confirma - da rivalutare se emergono metriche di popolazione più fini (es.
tasso di crescita nel tempo, quando avremo storico sufficiente) invece del
solo livello assoluto testato qui.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.data.liquidity_filter import liquid_singles_ids
from poke_quant.engine.strategies.scarcity_value_factor import RARITY_RANK, _N_TIERS, EXCLUDED_RARITIES


def load_grade9_population() -> dict:
    pop = pd.read_csv("data_cache/population_history.csv")
    pop = pop[pop["grade"].astype(str) == "9"].dropna(subset=["total_pop"])
    return pop.set_index("item_id")["total_pop"].to_dict()


def load_gem_rate() -> dict:
    """pop_grado10 / pop_totale (tutti i gradi) - quanto e' rara una copia PERFETTA
    di questa carta, indipendentemente da quante copie a grado 9 esistano."""
    pop = pd.read_csv("data_cache/population_history.csv")
    piv = pop.pivot_table(index="item_id", columns="grade", values="total_pop", aggfunc="first")
    piv.columns = [str(c) for c in piv.columns]
    rates = {}
    for item_id, row in piv.iterrows():
        total = row.dropna().sum()
        g10 = row.get("10", np.nan)
        if total > 0 and not pd.isna(g10) and g10 > 0:
            rates[item_id] = g10 / total
    return rates


def build_features(item_ids, metadata, pop_metric_by_item, latest_prices, cur_dt):
    rows, ids = [], []
    for item_id in item_ids:
        info = metadata[item_id]
        rarity = info.get("rarity")
        if rarity in EXCLUDED_RARITIES or rarity not in RARITY_RANK:
            continue
        rel_dt_raw = info.get("release_date")
        if not rel_dt_raw or item_id not in pop_metric_by_item:
            continue
        rel_dt = pd.to_datetime(rel_dt_raw)
        age_m = (cur_dt.year - rel_dt.year) * 12 + (cur_dt.month - rel_dt.month)
        if age_m < 6:
            continue
        price = latest_prices.get(item_id)
        if price is None or pd.isna(price) or price <= 0:
            continue
        metric = pop_metric_by_item[item_id]
        if metric <= 0:
            continue
        rarity_feat = float(RARITY_RANK[rarity]) / _N_TIERS
        is_op = 1.0 if info.get("franchise") == "one_piece" else 0.0
        is_jp = 1.0 if info.get("language") == "jp" else 0.0
        is_chase = 1.0 if info.get("selection_method") == "chase_price_filter_survivorship_biased" else 0.0
        log_age = np.log(age_m + 1.0)
        pop_feat = np.log(metric)
        rows.append([1.0, rarity_feat, is_op, is_jp, log_age, is_chase, pop_feat, np.log(price)])
        ids.append(item_id)
    return rows, ids


def run_test(label, pop_metric_by_item, metadata, universe, latest_prices, cur_dt, expect_sign):
    rows, ids = build_features(universe, metadata, pop_metric_by_item, latest_prices, cur_dt)
    arr = np.array(rows)
    X_a, y = arr[:, :6], arr[:, -1]
    coef_a, _, _, _ = np.linalg.lstsq(X_a, y, rcond=None)
    resid_a = y - X_a @ coef_a
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2_a = 1 - float(np.sum(resid_a ** 2)) / ss_tot

    X_b = arr[:, :7]
    coef_b, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)
    resid_b = y - X_b @ coef_b
    r2_b = 1 - float(np.sum(resid_b ** 2)) / ss_tot

    corr = float(np.corrcoef(resid_a, arr[:, 6])[0, 1])
    print(f"\n=== {label} (n={len(ids)}) ===")
    print(f"  R^2 senza popolazione: {r2_a:.4f} | R^2 con popolazione: {r2_b:.4f} (delta {r2_b-r2_a:+.4f})")
    print(f"  Coefficiente sulla popolazione: {coef_b[-1]:+.4f} (segno atteso se informativo: {expect_sign})")
    print(f"  Correlazione residuo-senza-pop vs metrica popolazione: {corr:+.4f}")

    df = pd.DataFrame({"item_id": ids, "resid_a": resid_a, "resid_b": resid_b})
    n_top = max(1, int(len(df) * 0.20))
    top_a = set(df.nsmallest(n_top, "resid_a")["item_id"])
    top_b = set(df.nsmallest(n_top, "resid_b")["item_id"])
    overlap = len(top_a & top_b)
    print(f"  Overlap top 20% piu' sottovalutate (con vs senza popolazione): {overlap}/{n_top} "
          f"({overlap/n_top*100:.0f}%)")


def main():
    metadata = load_metadata()
    prices = load_price_matrix("historical_prices_graded_singles_grade9.csv")
    universe = liquid_singles_ids(metadata, prices)
    cur_dt = prices.index[-1]
    latest_prices = prices.loc[cur_dt].to_dict()

    grade9_pop = load_grade9_population()
    inv_grade9_pop = {k: 1.0 / v for k, v in grade9_pop.items() if v > 0}
    run_test("Popolazione grado 9 (scarsita' = 1/pop)", inv_grade9_pop, metadata, universe,
              latest_prices, cur_dt, expect_sign="positivo (meno copie -> prezzo piu' alto)")

    gem_rate = load_gem_rate()
    run_test("Gem rate al grado 10 (pop10/pop_totale)", gem_rate, metadata, universe,
              latest_prices, cur_dt, expect_sign="negativo (gem rate piu' basso -> prezzo piu' alto)")


if __name__ == "__main__":
    main()
