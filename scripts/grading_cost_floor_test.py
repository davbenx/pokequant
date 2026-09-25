#!/usr/bin/env python3
"""
scripts/grading_cost_floor_test.py — L'utente ha verificato un caso reale:
Mantine #64 (Neo Genesis, Common) mostrato in BUY a 11,48EUR Grade 9 - ma la
sola gradazione PSA/CGC (anche nella fascia bulk piu' economica) costa di
piu' di cosi', quindi "e' impossibile trovarla a quei prezzi" nel senso che
NESSUNO gradirebbe oggi una copia a queste condizioni (perdita garantita) -
l'intera offerta possibile e' un pool fisso e non rinnovabile di slab gia'
gradati in passato da speculatori bulk (boom PSA 2020-21) e ora svenduti
sotto costo, non un mercato normale.

Verificato: NON e' un errore dati (serie storica di 12 mesi stabile,
9-14EUR, nessun salto). NON e' catturato dal filtro esistente
(compute_grade_raw_ratio_flags): quel filtro flagga rapporti grade9/raw
BASSI rispetto alla coorte (grade9 troppo depresso vs raw) - qui il
rapporto e' ALTO (~22x, tipico per una common), quindi passa il filtro
senza problemi. E' un fallimento di un tipo diverso: un pavimento
ASSOLUTO legato al costo reale di gradazione, non un controllo RELATIVO
alla coorte.

Trovato anche: 216/869 carte (24,9% dell'intero universo singole) hanno un
prezzo Grade9 MEDIANO storico sotto una stima conservativa di 20EUR (fascia
bulk PSA/CGC + carta + spedizione/assicurazione - stima ragionata, non un
listino verificato in tempo reale, l'utente puo' correggerla se ha numeri
più precisi). Di queste, 208/216 sono carte "random_control" (campione
casuale aggiunto per testare survivorship bias nella selezione "chase", non
carte scelte perche' ritenute buoni investimenti - vedi
scripts/discover_random_control_singles.py). Nella lista BUY live di oggi,
6 carte su 15 mostrate (40%) sono sotto questo pavimento, inclusa Mantine.

Domanda testata qui: escludere queste carte dall'universo cambia
l'edge misurato (Sharpe/CAGR/MaxDD/DSR)? Se il fattore "scarsita'" ha
un vero potere predittivo, rimuovere carte strutturalmente non comprabili
a profitto (l'intera "sottovalutazione" e' un artefatto della regressione
log-lineare compressa vicino allo zero, non un segnale reale) non dovrebbe
peggiorare le cose - se le peggiora significativamente, vuol dire che una
parte non piccola dell'edge misurato viene proprio da questi outlier di
prezzo, non da un vero effetto scarsita'/eta'/rarita'.

ESITO:

  pavimento=nessuno (produzione)   | universo  869 | Sharpe   1.57 | CAGR  +26.73% | MaxDD  -12.03% | Trade 299 | DSR(72) 0.869
  pavimento=10EUR                  | universo  828 | Sharpe   1.58 | CAGR  +27.79% | MaxDD  -12.10% | Trade 286 | DSR(72) 0.875
  pavimento=15EUR                  | universo  694 | Sharpe   1.62 | CAGR  +28.42% | MaxDD  -12.15% | Trade 284 | DSR(72) 0.889
  pavimento=20EUR                  | universo  653 | Sharpe   1.58 | CAGR  +27.66% | MaxDD  -14.60% | Trade 275 | DSR(72) 0.875
  pavimento=25EUR                  | universo  623 | Sharpe   1.53 | CAGR  +26.02% | MaxDD  -13.40% | Trade 259 | DSR(72) 0.852

CONCLUSIONE PRATICA: escludere le carte sotto il pavimento di costo di
gradazione NON peggiora l'edge misurato (resta 1,5-1,6 di Sharpe su tutta la
griglia, DSR resta alto 0,85-0,89) - anzi migliora leggermente in un punto
della griglia (15EUR). Coerente con l'ipotesi che il "sconto" di queste carte
fosse un artefatto della regressione, non alfa reale. Adottato 20EUR (la
stima pre-registrata PRIMA di vedere la griglia, non il punto migliore della
griglia - 15EUR ha DSR piu' alto ma sceglierlo ora sarebbe in-sample tuning).
Wired in poke_quant/data/liquidity_filter.py::liquid_singles_ids(), usato ora
sia nel backtest (app.py::get_singles_backtest_results) che nel segnale live
(scripts/generate_singles_signal.py) - PRIMA di questo fix, il segnale live
non applicava NESSUN filtro di attendibilita' (nemmeno data_quality=
thin_unreliable, gia' validato ma mai wired li'), un bug distinto trovato
indagando lo stesso caso Mantine (vedi VALIDATED_SINGLES in app.py).

INTERAZIONE COL TETTO DI QUANTITA' (scripts/max_quantity_per_trade_test.py,
rieseguito su questo universo pulito): il pavimento riduce ANCHE la
dipendenza dal comprare piu' copie identiche, perche' erano proprio le carte
quasi-senza-valore a rendere possibile qty=95 in un mese (bastano pochi euro
per comprarne decine). Col tetto realistico di 1 copia: Sharpe 0,30->0,64,
DSR 0,04->0,18 (ancora debole, ma molto meno). Con 2 copie: Sharpe 0,89->1,14,
DSR 0,37->0,59 (piu' vicino alla soglia). Non risolve il problema del tetto
di quantita' (vedi commit 823930d), ma ne attenua una parte non piccola -
i due filtri interagiscono, non sono indipendenti.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.engine.backtester import Backtester
from poke_quant.engine.strategies.scarcity_value_factor import ScarcityValueFactorStrategy
from poke_quant.validation.statistical_validation import deflated_sharpe_ratio
from scripts.generate_singles_signal import PRODUCTION_PARAMS

FLOOR_GRID = [0.0, 10.0, 15.0, 20.0, 25.0]  # 0.0 = nessun pavimento (produzione attuale)


def run_bt(strat, prices_df, metadata):
    bt = Backtester(strat, prices_df, metadata, initial_cash=10000.0, platform="cardmarket",
                     apply_liquidity_slippage=True, apply_holding_cost=True, apply_buy_side_shipping=True)
    return bt.run()


def main():
    metadata = load_metadata()
    grade9_prices = load_price_matrix("historical_prices_graded_singles_grade9.csv")
    base_ids = [k for k, v in metadata.items() if v.get("type") == "single"
                and v.get("data_quality") != "thin_unreliable" and k in grade9_prices.columns]

    for floor in FLOOR_GRID:
        if floor <= 0:
            ids = base_ids
        else:
            ids = [k for k in base_ids if grade9_prices[k].dropna().median() >= floor]
        meta_sub = {k: metadata[k] for k in ids}
        prices_sub = grade9_prices[ids]
        strat = ScarcityValueFactorStrategy(**PRODUCTION_PARAMS)
        res = run_bt(strat, prices_sub, meta_sub)
        dsr = deflated_sharpe_ratio(observed_sr=res.sharpe / np.sqrt(12), n_trials=72, n_obs=len(res.monthly_returns))
        label = "nessuno (produzione)" if floor <= 0 else f"{floor:.0f}EUR"
        print(f"  pavimento={label:22s} | universo {len(ids):4d} | Sharpe {res.sharpe:6.2f} | "
              f"CAGR {res.cagr*100:+7.2f}% | MaxDD {res.max_drawdown*100:7.2f}% | Trade {res.total_trades:3d} | DSR(72) {dsr:.3f}")


if __name__ == "__main__":
    main()
